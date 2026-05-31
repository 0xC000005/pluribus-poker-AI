"""Tianshou PPO control for RLCard no-limit Hold'em.

This is an environment-native public-reference control. It trains and evaluates
inside RLCard's 50bb/5-action no-limit Hold'em surface; it does not adapt a
native 9-action Slumbot-facing checkpoint into RLCard.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from poker_ai.research.alphanlholdem_benchmark import PolicyDecision
from poker_ai.research.native_nfsp import resolve_device


N_RLCARD_FEATURES = 54
N_RLCARD_ACTIONS = 5


def _enum_value(value: Any) -> int:
    return int(getattr(value, "value", value))


def _current_rlcard_state(observation: Any) -> dict[str, Any]:
    if isinstance(observation, (tuple, list)) and observation:
        state = observation[0]
    else:
        state = observation
    if not isinstance(state, dict):
        raise TypeError(f"expected RLCard state dict, got {type(state).__name__}")
    return state


def _legal_actions_from_state(state: dict[str, Any]) -> list[int]:
    legal = sorted(_enum_value(action) for action in state.get("legal_actions", {}).keys())
    return [action for action in legal if 0 <= action < N_RLCARD_ACTIONS]


def _legal_mask_from_state(state: dict[str, Any]) -> np.ndarray:
    mask = np.zeros((N_RLCARD_ACTIONS,), dtype=np.bool_)
    legal = _legal_actions_from_state(state)
    if legal:
        mask[legal] = True
    return mask


def _extract_obs_and_mask(obs) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None]:
    if isinstance(obs, dict):
        return obs["obs"], obs.get("mask")
    if hasattr(obs, "obs"):
        return obs.obs, getattr(obs, "mask", None)
    return obs, None


class RLCardNoLimitHoldemSingleAgentEnv(gym.Env):
    """Single-seat Gymnasium wrapper over RLCard HUNL against random policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int = 20260528,
        learner_seat: int = 0,
        max_steps_per_hand: int = 256,
        opponent_checkpoint: str | Path | None = None,
        opponent_device: str = "cpu",
        observation_format: str = "tianshou",
    ) -> None:
        from gymnasium import spaces
        import rlcard

        self.learner_seat = int(learner_seat)
        self.max_steps_per_hand = int(max_steps_per_hand)
        self.observation_format = str(observation_format)
        if self.observation_format not in {"tianshou", "rllib"}:
            raise ValueError("observation_format must be one of: tianshou, rllib")
        self.opponent_checkpoint = None if opponent_checkpoint is None else str(opponent_checkpoint)
        self.opponent_device = str(opponent_device)
        self._opponent_policy = (
            RLCardPPOPolicy.from_checkpoint(self.opponent_checkpoint, device=self.opponent_device)
            if self.opponent_checkpoint is not None
            else None
        )
        self.rng = np.random.default_rng(seed)
        self._rlcard_seed = int(seed)
        self._rl_env = rlcard.make(
            "no-limit-holdem",
            config={"game_num_players": 2, "seed": int(seed)},
        )
        self.action_space = spaces.Discrete(N_RLCARD_ACTIONS)
        if self.observation_format == "rllib":
            self.observation_space = spaces.Dict(
                {
                    "observations": spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=(N_RLCARD_FEATURES,),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.Box(
                        0.0,
                        1.0,
                        shape=(N_RLCARD_ACTIONS,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "obs": spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=(N_RLCARD_FEATURES,),
                        dtype=np.float32,
                    ),
                    "mask": spaces.Box(0, 1, shape=(N_RLCARD_ACTIONS,), dtype=np.bool_),
                }
            )
        self._last_observation: Any = None
        self.n_steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            random.seed(int(seed))
            np.random.seed(int(seed))
            self._rlcard_seed = int(seed)
        base_seed = int(self._rlcard_seed)
        for attempt in range(32):
            if attempt > 0 or seed is not None:
                self._rl_env.seed(base_seed + attempt)
            self._last_observation = self._rl_env.reset()
            self.n_steps = 0
            self._advance_opponent_until_learner()
            if not self._rl_env.game.is_over():
                break
        else:
            raise RuntimeError("RLCard reset repeatedly produced terminal hands before learner action")
        return self._observation(), self._info()

    def step(self, action: int):
        if self._last_observation is None:
            raise RuntimeError("reset must be called before step")
        if self._rl_env.game.is_over():
            return (
                self._observation(),
                self._terminal_reward(),
                True,
                False,
                {"already_terminal": True, "illegal_action": False, **self._info()},
            )
        state = _current_rlcard_state(self._last_observation)
        legal_mask = _legal_mask_from_state(state)
        action = int(action)
        if action < 0 or action >= N_RLCARD_ACTIONS or not bool(legal_mask[action]):
            return (
                self._observation(),
                -1.0,
                True,
                False,
                {"illegal_action": True, **self._info()},
            )

        self._last_observation = self._rl_env.step(action)
        self.n_steps += 1
        self._advance_opponent_until_learner()
        terminated = bool(self._rl_env.game.is_over())
        truncated = bool(self.n_steps >= self.max_steps_per_hand and not terminated)
        reward = self._terminal_reward() if terminated else 0.0
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            {"illegal_action": False, **self._info()},
        )

    def _advance_opponent_until_learner(self) -> None:
        while (
            self._last_observation is not None
            and not self._rl_env.game.is_over()
            and self._current_player() != self.learner_seat
            and self.n_steps < self.max_steps_per_hand
        ):
            state = _current_rlcard_state(self._last_observation)
            legal_actions = _legal_actions_from_state(state)
            if not legal_actions:
                return
            if self._opponent_policy is None:
                action = int(self.rng.choice(legal_actions))
            else:
                action = int(self._opponent_policy.act_rlcard_state(state).action)
            self._last_observation = self._rl_env.step(action)
            self.n_steps += 1

    def _current_player(self) -> int:
        if isinstance(self._last_observation, (tuple, list)) and len(self._last_observation) >= 2:
            return int(self._last_observation[1])
        state = _current_rlcard_state(self._last_observation)
        raw_obs = state.get("raw_obs", {})
        return int(raw_obs.get("current_player", self.learner_seat))

    def _terminal_reward(self) -> float:
        return float(self._rl_env.get_payoffs()[self.learner_seat])

    def _observation(self) -> dict[str, np.ndarray]:
        if self._last_observation is None or self._rl_env.game.is_over():
            return self._format_observation(
                np.zeros((N_RLCARD_FEATURES,), dtype=np.float32),
                np.ones((N_RLCARD_ACTIONS,), dtype=np.bool_),
            )
        state = _current_rlcard_state(self._last_observation)
        return self._format_observation(
            np.asarray(state["obs"], dtype=np.float32),
            _legal_mask_from_state(state),
        )

    def _format_observation(self, features: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        if self.observation_format == "rllib":
            return {
                "observations": np.asarray(features, dtype=np.float32),
                "action_mask": np.asarray(mask, dtype=np.float32),
            }
        return {
            "obs": np.asarray(features, dtype=np.float32),
            "mask": np.asarray(mask, dtype=np.bool_),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "current_player": int(self.learner_seat if self._last_observation is None else self._current_player()),
            "seat": int(self.learner_seat),
            "environment": "rlcard:no-limit-holdem",
            "num_actions": N_RLCARD_ACTIONS,
            "opponent_kind": "rlcard-ppo" if self._opponent_policy is not None else "random",
            "opponent_checkpoint": self.opponent_checkpoint,
            "trained_environment_native": True,
            "native_action_projection": False,
        }


class _RLCardMaskedActor(nn.Module):
    def __init__(self, *, hidden_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(N_RLCARD_FEATURES, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), N_RLCARD_ACTIONS),
        )

    def forward(self, obs, state=None, info=None):
        features, mask = _extract_obs_and_mask(obs)
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.net(x)
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0)
            logits = logits.masked_fill(~mask_t, -1e8)
        return logits, state


class _RLCardCritic(nn.Module):
    def __init__(self, *, hidden_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(N_RLCARD_FEATURES, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, obs):
        features, _mask = _extract_obs_and_mask(obs)
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def _build_ppo_policy_from_actor(
    actor: _RLCardMaskedActor,
    *,
    action_space,
    observation_space,
    deterministic_eval: bool = True,
):
    from tianshou.algorithm.modelfree.ppo import ProbabilisticActorPolicy
    from tianshou.algorithm.modelfree.reinforce import dist_fn_categorical_from_logits

    return ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn_categorical_from_logits,
        deterministic_eval=bool(deterministic_eval),
        action_space=action_space,
        observation_space=observation_space,
        action_scaling=False,
        action_bound_method=None,
    )


@dataclass
class RLCardPPOPolicy:
    actor: _RLCardMaskedActor
    device: torch.device
    policy_name: str = "rlcard_tianshou_ppo"

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> "RLCardPPOPolicy":
        resolved_device = torch.device(device)
        payload = torch.load(path, map_location=resolved_device, weights_only=False)
        if int(payload.get("num_actions", -1)) != N_RLCARD_ACTIONS:
            raise ValueError("RLCard PPO checkpoint action count does not match RLCard contract")
        if int(payload.get("num_features", -1)) != N_RLCARD_FEATURES:
            raise ValueError("RLCard PPO checkpoint feature count does not match RLCard contract")
        hidden_dim = int(payload.get("hidden_dim", 128))
        actor = _RLCardMaskedActor(hidden_dim=hidden_dim, device=resolved_device).to(resolved_device)
        actor.load_state_dict(payload["actor_state_dict"])
        actor.eval()
        return cls(
            actor=actor,
            device=resolved_device,
            policy_name=str(payload.get("algorithm", "rlcard_tianshou_ppo")),
        )

    def act_rlcard_state(self, state: dict[str, Any]) -> PolicyDecision:
        features = np.asarray(state["obs"], dtype=np.float32)
        legal_actions = _legal_actions_from_state(state)
        if not legal_actions:
            raise ValueError("RLCard state has no legal actions")
        mask = np.zeros((N_RLCARD_ACTIONS,), dtype=np.bool_)
        mask[legal_actions] = True
        with torch.no_grad():
            logits_t, _ = self.actor({"obs": features[None, :], "mask": mask[None, :]})
        masked_logits = logits_t[0].detach().cpu().numpy().astype(np.float32)
        masked_logits[~mask] = -np.inf
        logits = masked_logits.copy()
        action = int(np.argmax(masked_logits))
        return PolicyDecision(
            action=action,
            legal_actions=legal_actions,
            logits=logits,
            masked_logits=masked_logits,
            policy_name=self.policy_name,
        )


def save_untrained_ppo_checkpoint(
    path: str | Path,
    *,
    hidden_dim: int = 128,
    device: str = "cpu",
) -> None:
    resolved_device = torch.device(device)
    actor = _RLCardMaskedActor(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
    critic = _RLCardCritic(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": "tianshou_ppo_rlcard",
            "environment": "rlcard:no-limit-holdem",
            "num_actions": N_RLCARD_ACTIONS,
            "num_features": N_RLCARD_FEATURES,
            "hidden_dim": int(hidden_dim),
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "trained_environment_native": True,
            "native_action_projection": False,
        },
        out,
    )


def _summary_mean(value) -> float | None:
    if value is None:
        return None
    for attr in ("mean", "avg"):
        if hasattr(value, attr):
            raw = getattr(value, attr)
            return float(raw() if callable(raw) else raw)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluate_policy(
    policy,
    *,
    eval_games: int,
    seed: int,
    max_steps_per_hand: int,
    opponent_checkpoint: str | Path | None = None,
    opponent_device: str = "cpu",
) -> dict[str, Any]:
    from tianshou.data import Batch

    rewards: list[float] = []
    steps = 0
    for game_i in range(int(eval_games)):
        env = RLCardNoLimitHoldemSingleAgentEnv(
            seed=int(seed) + int(game_i),
            max_steps_per_hand=int(max_steps_per_hand),
            opponent_checkpoint=opponent_checkpoint,
            opponent_device=opponent_device,
        )
        obs, _info = env.reset(seed=int(seed) + int(game_i))
        terminated = False
        truncated = False
        while not terminated and not truncated:
            batch = Batch(
                obs=Batch(obs=obs["obs"][None, :], mask=obs["mask"][None, :]),
                info={},
            )
            with torch.no_grad():
                action = int(policy(batch).act[0])
            obs, reward, terminated, truncated, _info = env.step(action)
            steps += 1
        rewards.append(float(reward))
    opponent_kind = "rlcard-ppo" if opponent_checkpoint is not None else "random"
    return {
        "eval_games": int(eval_games),
        "eval_steps": int(steps),
        "eval_opponent_kind": opponent_kind,
        "mean_eval_payoff_p0_vs_random": (
            float(np.mean(rewards)) if rewards and opponent_checkpoint is None else None
        ),
        "mean_eval_payoff_p0_vs_opponent": (
            float(np.mean(rewards)) if rewards and opponent_checkpoint is not None else None
        ),
    }


def run_control(
    *,
    rollout_steps: int = 1024,
    updates: int = 10,
    repeat: int = 2,
    batch_size: int = 256,
    hidden_dim: int = 128,
    replay_size: int = 20_000,
    eval_games: int = 100,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    seed: int = 20260528,
    device: str = "auto",
    max_steps_per_hand: int = 256,
    opponent_checkpoint: str | Path | None = None,
    opponent_device: str = "auto",
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    from tianshou.algorithm.algorithm_base import policy_within_training_step
    from tianshou.algorithm.modelfree.ppo import PPO
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, ReplayBuffer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    if opponent_checkpoint is None:
        effective_opponent_device = "cpu"
    elif str(opponent_device) == "auto":
        effective_opponent_device = str(resolved_device)
    else:
        effective_opponent_device = str(resolve_device(str(opponent_device))["resolved_device"])

    env = RLCardNoLimitHoldemSingleAgentEnv(
        seed=int(seed),
        max_steps_per_hand=int(max_steps_per_hand),
        opponent_checkpoint=opponent_checkpoint,
        opponent_device=effective_opponent_device,
    )
    actor = _RLCardMaskedActor(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
    critic = _RLCardCritic(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
    policy = _build_ppo_policy_from_actor(
        actor=actor,
        action_space=env.action_space,
        observation_space=env.observation_space,
    )
    algorithm = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=float(lr)),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        ent_coef=float(ent_coef),
        vf_coef=float(vf_coef),
        max_batchsize=int(batch_size),
    )
    algorithm.to(resolved_device)
    replay = ReplayBuffer(size=int(replay_size))
    collector = Collector(algorithm, env, replay, exploration_noise=True)

    train_start = time.perf_counter()
    collected_steps = 0
    losses: list[float] = []
    for update_i in range(int(updates)):
        collect_result = collector.collect(
            n_step=max(int(rollout_steps), 1),
            reset_before_collect=update_i == 0,
        )
        collected_steps += int(getattr(collect_result, "n_collected_steps", rollout_steps))
        with policy_within_training_step(algorithm.policy):
            stats = algorithm.update(replay, batch_size=int(batch_size), repeat=int(repeat))
        loss_mean = _summary_mean(getattr(stats, "loss", None))
        if loss_mean is not None:
            losses.append(loss_mean)
        replay.reset()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    eval_metrics = _evaluate_policy(
        policy,
        eval_games=int(eval_games),
        seed=int(seed) + 100_000,
        max_steps_per_hand=int(max_steps_per_hand),
        opponent_checkpoint=opponent_checkpoint,
        opponent_device=effective_opponent_device,
    )
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    metrics: dict[str, Any] = {
        "algorithm": "tianshou_ppo_rlcard",
        "role": "public_reference_framework_control",
        "environment": "rlcard:no-limit-holdem",
        "warning": (
            "RLCard no-limit Hold'em is a 5-action public reference surface; "
            "train a separate native 9-action model for Slumbot-facing gates."
        ),
        **device_info,
        "seed": int(seed),
        "num_actions": N_RLCARD_ACTIONS,
        "num_features": N_RLCARD_FEATURES,
        "rollout_steps": int(rollout_steps),
        "train_steps": int(collected_steps),
        "updates": int(updates),
        "repeat": int(repeat),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "replay_size": int(replay_size),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "opponent_kind": "rlcard-ppo" if opponent_checkpoint is not None else "random",
        "opponent_checkpoint": None if opponent_checkpoint is None else str(opponent_checkpoint),
        "opponent_device_requested": str(opponent_device),
        "opponent_device": effective_opponent_device if opponent_checkpoint is not None else None,
        "population_training": bool(opponent_checkpoint is not None),
        "train_seconds": float(train_seconds),
        "train_steps_per_second": float(collected_steps / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "eval_seconds": float(eval_seconds),
        **eval_metrics,
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "league_eligible": False,
    }
    if checkpoint_out is not None:
        checkpoint_path = Path(checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "environment": metrics["environment"],
                "num_actions": N_RLCARD_ACTIONS,
                "num_features": N_RLCARD_FEATURES,
                "hidden_dim": int(hidden_dim),
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "metrics": metrics,
                "trained_environment_native": True,
                "native_action_projection": False,
            },
            checkpoint_path,
        )
        metrics["checkpoint_path"] = str(checkpoint_path)

    text = json.dumps(metrics, indent=2, sort_keys=True)
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return metrics
