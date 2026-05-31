#!/usr/bin/env python3
"""Run a Tianshou PPO control on the native 9-action poker state.

This is a plug-in RL control baseline, not a promoted poker agent. PPO is
provided by Tianshou; this file only adapts the repo's native full-deck state,
legal-action mask, vectorized collectors, and reporting gates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, new_game  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from scripts.run_tianshou_rainbow_native_control import NativeRainbowPokerEnv  # noqa: E402


class _MaskedActor(nn.Module):
    def __init__(self, *, hidden_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
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


class _Critic(nn.Module):
    def __init__(self, *, hidden_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        features, _mask = _extract_obs_and_mask(obs)
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def _extract_obs_and_mask(obs) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None]:
    if isinstance(obs, dict):
        return obs["obs"], obs.get("mask")
    if hasattr(obs, "obs"):
        return obs.obs, getattr(obs, "mask", None)
    return obs, None


def _make_env(
    *,
    seed: int,
    initial_chips: int,
    max_steps_per_hand: int,
    num_envs: int,
    vector_env_backend: str,
    opponent_kind: str = "random",
    opponent_checkpoint: str | Path | Sequence[str | Path] | None = None,
    opponent_device: str = "cpu",
):
    from tianshou.data import ReplayBuffer, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv

    base_env = NativeRainbowPokerEnv(
        seed=seed,
        initial_chips=initial_chips,
        max_steps_per_hand=max_steps_per_hand,
        opponent_kind=opponent_kind,
        opponent_checkpoint=opponent_checkpoint,
        opponent_device=opponent_device,
    )
    if num_envs == 1:
        return base_env, ReplayBuffer
    env_fns = [
        partial(
            NativeRainbowPokerEnv,
            seed=int(seed) + env_i,
            initial_chips=int(initial_chips),
            max_steps_per_hand=int(max_steps_per_hand),
            opponent_kind=opponent_kind,
            opponent_checkpoint=opponent_checkpoint,
            opponent_device=opponent_device,
        )
        for env_i in range(num_envs)
    ]
    vector_cls = DummyVectorEnv if vector_env_backend == "dummy" else SubprocVectorEnv
    if vector_cls is SubprocVectorEnv and opponent_kind != "random":
        return vector_cls(env_fns, context="spawn"), VectorReplayBuffer
    return vector_cls(env_fns), VectorReplayBuffer


def _evaluate_policy(
    policy,
    *,
    eval_games: int,
    seed: int,
    initial_chips: int,
    max_steps_per_hand: int,
    opponent_kind: str = "random",
    opponent_checkpoint: str | Path | Sequence[str | Path] | None = None,
    opponent_device: str = "cpu",
) -> dict:
    from tianshou.data import Batch

    rewards: list[float] = []
    steps = 0
    for game_i in range(int(eval_games)):
        env = NativeRainbowPokerEnv(
            seed=int(seed) + int(game_i),
            initial_chips=initial_chips,
            max_steps_per_hand=max_steps_per_hand,
            opponent_kind=opponent_kind,
            opponent_checkpoint=opponent_checkpoint,
            opponent_device=opponent_device,
        )
        obs, _info = env.reset(seed=int(seed) + int(game_i))
        terminated = False
        truncated = False
        while not terminated and not truncated:
            batch = Batch(obs=Batch(obs=obs["obs"][None, :], mask=obs["mask"][None, :]), info={})
            with torch.no_grad():
                action = int(policy(batch).act[0])
            obs, reward, terminated, truncated, _info = env.step(action)
            steps += 1
        rewards.append(float(reward))
    return {
        "eval_games": int(eval_games),
        "eval_steps": int(steps),
        "eval_opponent_kind": str(opponent_kind),
        "mean_eval_payoff_p0_vs_random": (
            float(np.mean(rewards)) if rewards and opponent_kind == "random" else None
        ),
        "mean_eval_payoff_p0_vs_opponent": (
            float(np.mean(rewards)) if rewards and opponent_kind != "random" else None
        ),
    }


def _build_ppo_policy_from_actor(
    actor: _MaskedActor,
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


def _load_ppo_checkpoint_policy(
    checkpoint_path: str,
    resolved_device: torch.device | str,
    *,
    deterministic_eval: bool = True,
):
    resolved_device = torch.device(resolved_device)
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("PPO checkpoint action count does not match native full-deck contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("PPO checkpoint feature count does not match native full-deck contract")
    hidden_dim = int(payload.get("hidden_dim", 128))
    env = NativeRainbowPokerEnv()
    actor = _MaskedActor(hidden_dim=hidden_dim, device=resolved_device).to(resolved_device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    policy = _build_ppo_policy_from_actor(
        actor,
        action_space=env.action_space,
        observation_space=env.observation_space,
        deterministic_eval=deterministic_eval,
    )
    policy.to(resolved_device)
    policy.eval()
    return payload, policy


def _ppo_greedy_action(policy, features: np.ndarray, legal_mask: np.ndarray) -> int:
    from tianshou.data import Batch

    batch = Batch(
        obs=Batch(
            obs=np.asarray(features, dtype=np.float32)[None, :],
            mask=np.asarray(legal_mask, dtype=bool)[None, :],
        ),
        info={},
    )
    with torch.no_grad():
        return int(policy(batch).act[0])


def _ppo_action_probs(policy, features: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    from tianshou.data import Batch

    batch_obs = Batch(
        obs=np.asarray(features, dtype=np.float32)[None, :],
        mask=np.asarray(legal_mask, dtype=bool)[None, :],
    )
    with torch.no_grad():
        logits, _state = policy.actor(batch_obs)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float32)
    masked = probs * np.asarray(legal_mask, dtype=np.float32)
    total = float(masked.sum())
    if total <= 1e-8:
        legal = np.asarray(legal_mask, dtype=np.float32)
        legal_total = float(legal.sum())
        return legal / legal_total if legal_total > 0.0 else np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32)
    return masked / total


def evaluate_ppo_checkpoint_vs_native_nfsp(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260526,
    deterministic_eval: bool = True,
) -> dict:
    """Duplicate-swapped H2H: PPO greedy policy versus native NFSP average policy."""
    from poker_ai.research.native_nfsp import (
        _load_native_checkpoint_networks,
        _network_probs,
        get_legal_mask,
        select_action,
    )

    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_payload, candidate_policy = _load_ppo_checkpoint_policy(
        candidate_checkpoint,
        resolved_device,
        deterministic_eval=deterministic_eval,
    )
    baseline_payload, _baseline_q, baseline_avg = _load_native_checkpoint_networks(
        baseline_checkpoint,
        resolved_device,
    )
    baseline_config = baseline_payload.get("config", {})
    initial_chips = int(baseline_config.get("initial_chips", 1000))
    max_steps_per_hand = int(baseline_config.get("max_steps_per_hand", 256))

    candidate_payoffs: list[float] = []
    candidate_pair_payoffs: list[float] = []
    total_steps = 0
    eval_start = time.perf_counter()
    pair_i = 0
    while len(candidate_payoffs) < int(n_games):
        game_seed = int(seed) + pair_i
        action_seed = int(seed) + 1_000_000 + pair_i
        pair_payoffs: list[float] = []
        for candidate_seat in (0, 1):
            if len(candidate_payoffs) >= int(n_games):
                break
            random.seed(game_seed)
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)
            rng = np.random.default_rng(action_seed)
            state = new_game(2, initial_chips=initial_chips)
            n_steps = 0
            while not state.is_terminal and n_steps < max_steps_per_hand:
                features = state.to_feature_vector().astype(np.float32, copy=False)
                legal_mask = get_legal_mask(state)
                if int(state.player_i) == int(candidate_seat):
                    action_idx = _ppo_greedy_action(candidate_policy, features, legal_mask)
                else:
                    probs = _network_probs(baseline_avg, features, legal_mask, resolved_device)
                    action_idx = select_action(probs, legal_mask, rng=rng)
                state = state.apply_action(INDEX_TO_ACTION[action_idx])
                n_steps += 1
            total_steps += n_steps
            payoff = float(state.payout.get(candidate_seat, 0)) / float(initial_chips)
            candidate_payoffs.append(payoff)
            pair_payoffs.append(payoff)
        if pair_payoffs:
            candidate_pair_payoffs.append(float(np.mean(pair_payoffs)))
        pair_i += 1
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start
    payoff_mean = float(np.mean(candidate_pair_payoffs)) if candidate_pair_payoffs else 0.0
    payoff_std = (
        float(np.std(candidate_pair_payoffs, ddof=1))
        if len(candidate_pair_payoffs) > 1
        else 0.0
    )
    payoff_se = (
        payoff_std / float(np.sqrt(len(candidate_pair_payoffs)))
        if candidate_pair_payoffs
        else 0.0
    )
    return {
        "algorithm": "tianshou_ppo_vs_native_nfsp_h2h",
        "role": "rl_control_baseline_local_league",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_algorithm": str(candidate_payload.get("algorithm", "tianshou_ppo")),
        "candidate_eval_mode": "deterministic" if deterministic_eval else "stochastic",
        "baseline_algorithm": str(baseline_payload.get("algorithm", "native_nfsp_dqn")),
        **device_info,
        "num_actions": N_ACTIONS,
        "n_games": int(n_games),
        "n_pairs": int(len(candidate_pair_payoffs)),
        "eval_seconds": float(eval_seconds),
        "eval_steps": int(total_steps),
        "eval_games_per_second": float(n_games / max(eval_seconds, 1e-9)),
        "mean_candidate_payoff": payoff_mean,
        "std_candidate_payoff": payoff_std,
        "lower95_candidate_payoff": payoff_mean - 1.96 * payoff_se,
        "upper95_candidate_payoff": payoff_mean + 1.96 * payoff_se,
        "promotion": False,
    }


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


def run_control(
    *,
    rollout_steps: int = 1024,
    updates: int = 10,
    repeat: int = 2,
    batch_size: int = 256,
    hidden_dim: int = 128,
    replay_size: int = 20_000,
    num_envs: int = 1,
    vector_env_backend: str = "dummy",
    eval_games: int = 100,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    opponent_kind: str = "random",
    opponent_checkpoint: str | Path | Sequence[str | Path] | None = None,
    opponent_device: str = "auto",
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    seed: int = 20260526,
    device: str = "auto",
    output_json: str | None = None,
    checkpoint_out: str | None = None,
) -> dict:
    from tianshou.algorithm.algorithm_base import policy_within_training_step
    from tianshou.algorithm.modelfree.ppo import PPO
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    num_envs = max(1, int(num_envs))
    vector_env_backend = str(vector_env_backend)
    if vector_env_backend not in {"dummy", "subproc"}:
        raise ValueError("vector_env_backend must be one of: dummy, subproc")
    opponent_kind = str(opponent_kind)
    if opponent_kind not in {"random", "native-nfsp", "rainbow"}:
        raise ValueError("opponent_kind must be one of: random, native-nfsp, rainbow")
    opponent_checkpoints = []
    if opponent_checkpoint is None:
        opponent_checkpoints = []
    elif isinstance(opponent_checkpoint, (str, Path)):
        opponent_checkpoints = [str(opponent_checkpoint)]
    else:
        opponent_checkpoints = [str(path) for path in opponent_checkpoint if str(path)]
    if opponent_kind != "random" and not opponent_checkpoints:
        raise ValueError(f"opponent_checkpoint is required for {opponent_kind} opponent")
    if opponent_kind == "random":
        effective_opponent_device = "cpu"
    elif str(opponent_device) == "auto":
        if num_envs > 1 and vector_env_backend == "subproc":
            effective_opponent_device = "cpu"
        else:
            effective_opponent_device = str(resolved_device)
    else:
        effective_opponent_device = str(resolve_device(str(opponent_device))["resolved_device"])

    env, replay_cls = _make_env(
        seed=int(seed),
        initial_chips=int(initial_chips),
        max_steps_per_hand=int(max_steps_per_hand),
        num_envs=num_envs,
        vector_env_backend=vector_env_backend,
        opponent_kind=opponent_kind,
        opponent_checkpoint=opponent_checkpoints,
        opponent_device=effective_opponent_device,
    )
    base_env = NativeRainbowPokerEnv(
        seed=seed,
        initial_chips=initial_chips,
        max_steps_per_hand=max_steps_per_hand,
    )
    actor = _MaskedActor(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
    critic = _Critic(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
    policy = _build_ppo_policy_from_actor(
        actor=actor,
        action_space=base_env.action_space,
        observation_space=base_env.observation_space,
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
    if replay_cls is ReplayBuffer:
        replay = replay_cls(size=int(replay_size))
    elif replay_cls is VectorReplayBuffer:
        replay = replay_cls(total_size=int(replay_size), buffer_num=num_envs)
    else:
        raise TypeError("unexpected replay buffer class")
    collector = Collector(algorithm, env, replay, exploration_noise=True)

    train_start = time.perf_counter()
    collected_steps = 0
    losses: list[float] = []
    for update_i in range(int(updates)):
        collector.collect(
            n_step=max(int(rollout_steps), 1),
            reset_before_collect=update_i == 0,
        )
        collected_steps += int(rollout_steps)
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
        initial_chips=int(initial_chips),
        max_steps_per_hand=int(max_steps_per_hand),
        opponent_kind=opponent_kind,
        opponent_checkpoint=opponent_checkpoints,
        opponent_device=effective_opponent_device,
    )
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    metrics = {
        "algorithm": "tianshou_ppo",
        "role": "rl_control_baseline",
        "environment": "poker_ai:full_deck_hu_nlhe_single_agent_control",
        "warning": (
            "PPO is a single-agent policy-gradient control against local "
            "opponents, not an equilibrium method or promotion signal."
        ),
        **device_info,
        "num_actions": N_ACTIONS,
        "rollout_steps": int(rollout_steps),
        "train_steps": int(collected_steps),
        "updates": int(updates),
        "repeat": int(repeat),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "num_envs": int(num_envs),
        "vector_env_backend": vector_env_backend if num_envs > 1 else "single",
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "opponent_kind": opponent_kind,
        "opponent_checkpoint": opponent_checkpoints[0] if len(opponent_checkpoints) == 1 else None,
        "opponent_checkpoints": opponent_checkpoints,
        "opponent_device_requested": str(opponent_device),
        "opponent_device": effective_opponent_device if opponent_kind != "random" else None,
        "train_seconds": float(train_seconds),
        "train_steps_per_second": float(collected_steps / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "eval_seconds": float(eval_seconds),
        **eval_metrics,
        "promotion": False,
        "league_eligible": False,
    }
    if checkpoint_out:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "environment": metrics["environment"],
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        metrics["checkpoint_path"] = str(path)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vector-env-backend", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--opponent-kind", choices=("random", "native-nfsp", "rainbow"), default="random")
    parser.add_argument(
        "--opponent-checkpoint",
        action="append",
        help="Opponent checkpoint path. Repeat to sample opponents from a checkpoint league.",
    )
    parser.add_argument(
        "--opponent-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for fixed checkpoint opponents. auto uses learner device for single-env runs and CPU for vector workers.",
    )
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Sample PPO actions during H2H instead of using the modal action.",
    )
    args = parser.parse_args(argv)
    if args.checkpoint_in and args.baseline_checkpoint:
        metrics = evaluate_ppo_checkpoint_vs_native_nfsp(
            args.checkpoint_in,
            args.baseline_checkpoint,
            n_games=args.eval_games,
            device=args.device,
            seed=args.seed,
            deterministic_eval=not args.stochastic_eval,
        )
        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        metrics = run_control(
            rollout_steps=args.rollout_steps,
            updates=args.updates,
            repeat=args.repeat,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            replay_size=args.replay_size,
            num_envs=args.num_envs,
            vector_env_backend=args.vector_env_backend,
            eval_games=args.eval_games,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            opponent_kind=args.opponent_kind,
            opponent_checkpoint=args.opponent_checkpoint,
            opponent_device=args.opponent_device,
            initial_chips=args.initial_chips,
            max_steps_per_hand=args.max_steps_per_hand,
            seed=args.seed,
            device=args.device,
            output_json=args.output_json,
            checkpoint_out=args.checkpoint_out,
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
