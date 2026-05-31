#!/usr/bin/env python3
"""Run a Tianshou Rainbow DQN control on the native 9-action poker state.

This is an RL control baseline, not a promoted poker agent. It trains a
single-seat learner against a random local opponent using the repo's full-deck
state and legal-action mask. The result is useful for falsifying "plain
value-learning is enough"; it is not an equilibrium or Slumbot-parity method.
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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, new_game  # noqa: E402
from poker_ai.deep_cfr.fast_state import new_fast_game  # noqa: E402
from poker_ai.research.native_nfsp import get_legal_mask, masked_uniform, resolve_device, select_action  # noqa: E402
from poker_ai.research.native_rollout_substrate import fast_state_from_full_deck_state  # noqa: E402


def _normalize_opponent_checkpoints(opponent_checkpoint: str | Path | Sequence[str | Path] | None) -> list[str]:
    if opponent_checkpoint is None:
        return []
    if isinstance(opponent_checkpoint, (str, Path)):
        return [str(opponent_checkpoint)]
    return [str(path) for path in opponent_checkpoint if str(path)]


class NativeRainbowPokerEnv(gym.Env):
    """Single-agent Gymnasium wrapper over local HU NLHE against random policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int = 20260526,
        initial_chips: int = 1000,
        max_steps_per_hand: int = 256,
        learner_seat: int = 0,
        opponent_kind: str = "random",
        opponent_checkpoint: str | Path | Sequence[str | Path] | None = None,
        opponent_device: str = "cpu",
        state_backend: str = "full-deck",
    ) -> None:
        from gymnasium import spaces

        self.initial_chips = int(initial_chips)
        self.max_steps_per_hand = int(max_steps_per_hand)
        self.learner_seat = int(learner_seat)
        self.opponent_kind = str(opponent_kind)
        self.state_backend = str(state_backend)
        if self.state_backend not in {"full-deck", "fast-state"}:
            raise ValueError("state_backend must be one of: full-deck, fast-state")
        self.opponent_checkpoints = _normalize_opponent_checkpoints(opponent_checkpoint)
        self.opponent_checkpoint = self.opponent_checkpoints[0] if self.opponent_checkpoints else None
        self.active_opponent_checkpoint = self.opponent_checkpoint
        self.opponent_device_info = resolve_device(str(opponent_device))
        self._opponent_device = torch.device(self.opponent_device_info["resolved_device"])
        self._opponent_avg_by_checkpoint = {}
        self._opponent_rainbow_policy_by_checkpoint = {}
        self._opponent_payload_by_checkpoint = {}
        if self.opponent_kind not in {"random", "native-nfsp", "rainbow"}:
            raise ValueError("opponent_kind must be one of: random, native-nfsp, rainbow")
        if self.opponent_kind in {"native-nfsp", "rainbow"} and not self.opponent_checkpoints:
            raise ValueError(f"opponent_checkpoint is required for {self.opponent_kind} opponent")
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(-np.inf, np.inf, shape=(N_FEATURES,), dtype=np.float32),
                "mask": spaces.Box(0, 1, shape=(N_ACTIONS,), dtype=np.bool_),
            }
        )
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.n_steps = 0
        if self.active_opponent_checkpoint is not None:
            self._load_opponent_if_needed()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
            np.random.seed(seed)
        if self.opponent_checkpoints:
            checkpoint_i = int(self.rng.integers(0, len(self.opponent_checkpoints)))
            self.active_opponent_checkpoint = self.opponent_checkpoints[checkpoint_i]
            self.opponent_checkpoint = self.active_opponent_checkpoint
        self.state = self._new_state()
        self.n_steps = 0
        self._advance_random_opponent()
        return self._observation(), self._info()

    def _new_state(self):
        if self.state_backend == "fast-state":
            return new_fast_game(2, initial_chips=self.initial_chips)
        return new_game(2, initial_chips=self.initial_chips)

    def _legal_mask(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(N_ACTIONS, dtype=np.float32)
        if self.state_backend == "fast-state":
            return self.state.get_legal_mask()
        return get_legal_mask(self.state)

    def _feature_vector(self) -> np.ndarray:
        if self.state is None:
            return np.zeros(N_FEATURES, dtype=np.float32)
        return self.state.to_feature_vector().astype(np.float32, copy=False)

    def _apply_action_idx(self, action: int) -> None:
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        if self.state_backend == "fast-state":
            self.state.apply_action(int(action))
        else:
            self.state = self.state.apply_action(INDEX_TO_ACTION[int(action)])

    def _current_player_i(self) -> int:
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        if self.state_backend == "fast-state":
            return int(self.state.current_player_i)
        return int(self.state.player_i)

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        legal_mask = self._legal_mask().astype(bool)
        action = int(action)
        if action < 0 or action >= N_ACTIONS or not bool(legal_mask[action]):
            return (
                self._observation(),
                -1.0,
                True,
                False,
                {
                    "illegal_action": True,
                    **self._info(),
                },
            )
        self._apply_action_idx(action)
        self.n_steps += 1
        self._advance_random_opponent()
        terminated = bool(self.state.is_terminal)
        truncated = bool(self.n_steps >= self.max_steps_per_hand and not terminated)
        reward = self._terminal_reward() if terminated else 0.0
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            {
                "illegal_action": False,
                **self._info(),
            },
        )

    def _load_opponent_if_needed(self) -> None:
        if self.opponent_kind not in {"native-nfsp", "rainbow"}:
            return

        checkpoint = self.active_opponent_checkpoint or self.opponent_checkpoint
        assert checkpoint is not None
        if self.opponent_kind == "native-nfsp":
            if checkpoint in self._opponent_avg_by_checkpoint:
                return
            from poker_ai.research.native_nfsp import _load_native_checkpoint_networks

            payload, _q_net, avg_net = _load_native_checkpoint_networks(
                checkpoint,
                self._opponent_device,
            )
            self._opponent_payload_by_checkpoint[checkpoint] = payload
            self._opponent_avg_by_checkpoint[checkpoint] = avg_net
        elif self.opponent_kind == "rainbow":
            if checkpoint in self._opponent_rainbow_policy_by_checkpoint:
                return
            payload, policy = _load_rainbow_checkpoint_policy(
                checkpoint,
                self._opponent_device,
            )
            self._opponent_payload_by_checkpoint[checkpoint] = payload
            self._opponent_rainbow_policy_by_checkpoint[checkpoint] = policy

    def _select_opponent_action(self, legal_mask: np.ndarray) -> int:
        if self.opponent_kind == "random":
            return select_action(masked_uniform(legal_mask), legal_mask, rng=self.rng)
        if self.opponent_kind == "native-nfsp":
            from poker_ai.research.native_nfsp import _network_probs

            self._load_opponent_if_needed()
            checkpoint = self.active_opponent_checkpoint or self.opponent_checkpoint
            avg_net = self._opponent_avg_by_checkpoint.get(checkpoint)
            if self.state is None or avg_net is None:
                return select_action(masked_uniform(legal_mask), legal_mask, rng=self.rng)
            features = self.state.to_feature_vector().astype(np.float32, copy=False)
            probs = _network_probs(
                avg_net,
                features,
                legal_mask,
                self._opponent_device,
            )
            return select_action(probs, legal_mask, rng=self.rng)
        if self.opponent_kind == "rainbow":
            self._load_opponent_if_needed()
            checkpoint = self.active_opponent_checkpoint or self.opponent_checkpoint
            policy = self._opponent_rainbow_policy_by_checkpoint.get(checkpoint)
            if self.state is None or policy is None:
                return select_action(masked_uniform(legal_mask), legal_mask, rng=self.rng)
            features = self.state.to_feature_vector().astype(np.float32, copy=False)
            return _rainbow_greedy_action(policy, features, legal_mask)
        raise ValueError(f"unsupported opponent_kind: {self.opponent_kind}")

    def _info(self) -> dict[str, Any]:
        return {
            "seat": self.learner_seat,
            "opponent_kind": self.opponent_kind,
            "opponent_checkpoint": self.active_opponent_checkpoint,
            "active_opponent_checkpoint": self.active_opponent_checkpoint,
            "opponent_device": str(self._opponent_device),
            "opponent_device_requested": str(self.opponent_device_info["requested_device"]),
            "state_backend": self.state_backend,
        }

    def _advance_random_opponent(self) -> None:
        while (
            self.state is not None
            and not self.state.is_terminal
            and self._current_player_i() != self.learner_seat
            and self.n_steps < self.max_steps_per_hand
        ):
            legal_mask = self._legal_mask()
            action_idx = self._select_opponent_action(legal_mask)
            self._apply_action_idx(action_idx)
            self.n_steps += 1

    def _terminal_reward(self) -> float:
        assert self.state is not None
        return float(self.state.payout.get(self.learner_seat, 0)) / float(self.initial_chips)

    def _observation(self) -> dict[str, np.ndarray]:
        if self.state is None:
            features = np.zeros(N_FEATURES, dtype=np.float32)
            mask = np.ones(N_ACTIONS, dtype=np.bool_)
        elif self.state.is_terminal:
            features = np.zeros(N_FEATURES, dtype=np.float32)
            mask = np.ones(N_ACTIONS, dtype=np.bool_)
        else:
            features = self._feature_vector()
            mask = self._legal_mask().astype(np.bool_)
        return {"obs": features, "mask": mask}


class _RainbowDistributionNet(nn.Module):
    def __init__(self, *, hidden_dim: int, num_atoms: int, device: torch.device):
        super().__init__()
        self.num_atoms = int(num_atoms)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS * int(num_atoms)),
        )

    def forward(self, obs, state=None, info=None):
        if isinstance(obs, dict):
            obs = obs["obs"]
        elif hasattr(obs, "obs"):
            obs = obs.obs
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.net(x).view(-1, N_ACTIONS, self.num_atoms)
        return torch.softmax(logits, dim=-1), state


def _evaluate_greedy_policy(
    policy,
    *,
    eval_games: int,
    seed: int,
    initial_chips: int,
    max_steps_per_hand: int,
    opponent_kind: str = "random",
    opponent_checkpoint: str | Path | Sequence[str | Path] | None = None,
    opponent_device: str = "cpu",
    state_backend: str = "full-deck",
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
            state_backend=state_backend,
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
    mean_payoff = float(np.mean(rewards)) if rewards else 0.0
    metrics = {
        "eval_games": int(eval_games),
        "eval_steps": int(steps),
        "eval_opponent_kind": str(opponent_kind),
        "mean_eval_payoff_p0": mean_payoff,
    }
    if opponent_kind == "random":
        metrics["mean_eval_payoff_p0_vs_random"] = mean_payoff
    else:
        metrics["mean_eval_payoff_p0_vs_opponent"] = mean_payoff
    return metrics


def _validate_eval_state_backend(eval_state_backend: str) -> str:
    backend = str(eval_state_backend)
    if backend not in {"full-deck", "fast-state", "fast-state-canonical-deal"}:
        raise ValueError(
            "eval_state_backend must be one of: full-deck, fast-state, "
            "fast-state-canonical-deal"
        )
    return backend


def _new_h2h_state(*, initial_chips: int, eval_state_backend: str):
    if eval_state_backend == "fast-state":
        return new_fast_game(2, initial_chips=int(initial_chips))
    if eval_state_backend == "fast-state-canonical-deal":
        return fast_state_from_full_deck_state(
            new_game(2, initial_chips=int(initial_chips)),
            future_deal_mode="random",
        )
    return new_game(2, initial_chips=int(initial_chips))


def _h2h_current_player_i(state, *, eval_state_backend: str) -> int:
    if eval_state_backend.startswith("fast-state"):
        return int(state.current_player_i)
    return int(state.player_i)


def _h2h_legal_mask(state, *, eval_state_backend: str) -> np.ndarray:
    if eval_state_backend.startswith("fast-state"):
        return state.get_legal_mask()
    return get_legal_mask(state)


def _h2h_apply_action(state, action_idx: int, *, eval_state_backend: str):
    if eval_state_backend.startswith("fast-state"):
        state.apply_action(int(action_idx))
        return state
    return state.apply_action(INDEX_TO_ACTION[int(action_idx)])


def _rainbow_greedy_action(policy, features: np.ndarray, legal_mask: np.ndarray) -> int:
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


def _rainbow_state_dict_from_payload(payload: dict) -> dict:
    return _rainbow_state_dicts_by_seat_from_payload(payload)[0]


def _rainbow_state_dicts_by_seat_from_payload(payload: dict) -> dict[int, dict]:
    if payload.get("shared_model_state_dict") is not None:
        shared = payload["shared_model_state_dict"]
        return {0: shared, 1: shared}
    if "agent_model_state_dicts" in payload:
        agent_state_dicts = payload["agent_model_state_dicts"]
        fallback_key = "player_0" if "player_0" in agent_state_dicts else sorted(agent_state_dicts)[0]
        return {
            seat: agent_state_dicts.get(f"player_{seat}", agent_state_dicts[fallback_key])
            for seat in (0, 1)
        }
    if "model_state_dict" in payload:
        shared = payload["model_state_dict"]
        return {0: shared, 1: shared}
    raise ValueError("Rainbow checkpoint is missing a loadable model state dict")


def _load_rainbow_checkpoint_policy_map(checkpoint_path: str, resolved_device: torch.device):
    from tianshou.algorithm.modelfree.c51 import C51Policy

    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("Rainbow checkpoint action count does not match native full-deck contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("Rainbow checkpoint feature count does not match native full-deck contract")
    hidden_dim = int(payload.get("hidden_dim", 128))
    num_atoms = int(payload.get("num_atoms", 51))
    env = NativeRainbowPokerEnv()
    policies = {}
    for seat, state_dict in _rainbow_state_dicts_by_seat_from_payload(payload).items():
        model = _RainbowDistributionNet(
            hidden_dim=hidden_dim,
            num_atoms=num_atoms,
            device=resolved_device,
        ).to(resolved_device)
        model.load_state_dict(state_dict)
        model.eval()
        policy = C51Policy(
            model=model,
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_atoms=num_atoms,
            v_min=-1.0,
            v_max=1.0,
            eps_training=0.0,
            eps_inference=0.0,
        )
        policy.to(resolved_device)
        policy.eval()
        policies[int(seat)] = policy
    payload = dict(payload)
    payload["loaded_agent_ids"] = [f"player_{seat}" for seat in sorted(policies)]
    return payload, policies


def _load_rainbow_checkpoint_policy(checkpoint_path: str, resolved_device: torch.device):
    payload, policies = _load_rainbow_checkpoint_policy_map(checkpoint_path, resolved_device)
    return payload, policies[0]


def evaluate_rainbow_checkpoint_vs_native_nfsp(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260526,
    eval_state_backend: str = "full-deck",
) -> dict:
    """Duplicate-swapped H2H: Rainbow greedy policy versus native NFSP average policy."""
    from poker_ai.research.native_nfsp import _load_native_checkpoint_networks, _network_probs

    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_payload, candidate_policy_map = _load_rainbow_checkpoint_policy_map(
        candidate_checkpoint,
        resolved_device,
    )
    baseline_payload, _baseline_q, baseline_avg = _load_native_checkpoint_networks(
        baseline_checkpoint,
        resolved_device,
    )
    baseline_config = baseline_payload.get("config", {})
    initial_chips = int(baseline_config.get("initial_chips", 1000))
    max_steps_per_hand = int(baseline_config.get("max_steps_per_hand", 256))
    eval_state_backend = _validate_eval_state_backend(eval_state_backend)

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
            state = _new_h2h_state(
                initial_chips=initial_chips,
                eval_state_backend=eval_state_backend,
            )
            n_steps = 0
            while not state.is_terminal and n_steps < max_steps_per_hand:
                features = state.to_feature_vector().astype(np.float32, copy=False)
                legal_mask = _h2h_legal_mask(state, eval_state_backend=eval_state_backend)
                current_player_i = _h2h_current_player_i(
                    state,
                    eval_state_backend=eval_state_backend,
                )
                if current_player_i == int(candidate_seat):
                    action_idx = _rainbow_greedy_action(
                        candidate_policy_map.get(current_player_i, candidate_policy_map[0]),
                        features,
                        legal_mask,
                    )
                else:
                    probs = _network_probs(baseline_avg, features, legal_mask, resolved_device)
                    action_idx = select_action(probs, legal_mask, rng=rng)
                state = _h2h_apply_action(
                    state,
                    action_idx,
                    eval_state_backend=eval_state_backend,
                )
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
    payoff_std = float(np.std(candidate_pair_payoffs, ddof=1)) if len(candidate_pair_payoffs) > 1 else 0.0
    payoff_se = payoff_std / float(np.sqrt(len(candidate_pair_payoffs))) if candidate_pair_payoffs else 0.0
    return {
        "algorithm": "tianshou_rainbow_vs_native_nfsp_h2h",
        "role": "rl_control_baseline_local_league",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_algorithm": str(candidate_payload.get("algorithm", "tianshou_rainbow_dqn")),
        "baseline_algorithm": str(baseline_payload.get("algorithm", "native_nfsp_dqn")),
        **device_info,
        "eval_state_backend": eval_state_backend,
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


def evaluate_rainbow_checkpoint_vs_neural_policy_iteration(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260526,
    eval_state_backend: str = "full-deck",
) -> dict:
    """Duplicate-swapped H2H: Rainbow greedy policy versus NPI stochastic policy."""
    from poker_ai.research.neural_policy_iteration import (
        _load_policy_iteration_checkpoint,
        _network_policy,
    )

    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_payload, candidate_policy_map = _load_rainbow_checkpoint_policy_map(
        candidate_checkpoint,
        resolved_device,
    )
    baseline_policy, baseline_payload = _load_policy_iteration_checkpoint(
        baseline_checkpoint,
        resolved_device,
    )
    baseline_config = dict(baseline_payload.get("config", {}))
    initial_chips = int(baseline_config.get("initial_chips", 1000))
    max_steps_per_hand = int(baseline_config.get("max_steps_per_hand", 256))
    eval_state_backend = _validate_eval_state_backend(eval_state_backend)

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
            state = _new_h2h_state(
                initial_chips=initial_chips,
                eval_state_backend=eval_state_backend,
            )
            n_steps = 0
            while not state.is_terminal and n_steps < max_steps_per_hand:
                features = state.to_feature_vector().astype(np.float32, copy=False)
                legal_mask = _h2h_legal_mask(state, eval_state_backend=eval_state_backend)
                current_player_i = _h2h_current_player_i(
                    state,
                    eval_state_backend=eval_state_backend,
                )
                if current_player_i == int(candidate_seat):
                    action_idx = _rainbow_greedy_action(
                        candidate_policy_map.get(current_player_i, candidate_policy_map[0]),
                        features,
                        legal_mask,
                    )
                else:
                    probs = _network_policy(
                        baseline_policy,
                        features,
                        legal_mask,
                        resolved_device,
                    )
                    action_idx = select_action(probs, legal_mask, rng=rng)
                state = _h2h_apply_action(
                    state,
                    action_idx,
                    eval_state_backend=eval_state_backend,
                )
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
    payoff_std = float(np.std(candidate_pair_payoffs, ddof=1)) if len(candidate_pair_payoffs) > 1 else 0.0
    payoff_se = payoff_std / float(np.sqrt(len(candidate_pair_payoffs))) if candidate_pair_payoffs else 0.0
    return {
        "algorithm": "tianshou_rainbow_vs_neural_policy_iteration_h2h",
        "role": "rl_control_vs_alpha_zero_style_npi_local_league",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_algorithm": str(candidate_payload.get("algorithm", "tianshou_rainbow_dqn")),
        "baseline_algorithm": str(baseline_payload.get("algorithm", "neural_policy_iteration")),
        **device_info,
        "eval_state_backend": eval_state_backend,
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


def evaluate_rainbow_checkpoint_vs_rainbow(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260526,
    eval_state_backend: str = "full-deck",
) -> dict:
    """Duplicate-swapped H2H: Rainbow greedy policy versus Rainbow greedy policy."""
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_payload, candidate_policy_map = _load_rainbow_checkpoint_policy_map(
        candidate_checkpoint,
        resolved_device,
    )
    baseline_payload, baseline_policy_map = _load_rainbow_checkpoint_policy_map(
        baseline_checkpoint,
        resolved_device,
    )
    initial_chips = int(
        baseline_payload.get("metrics", {}).get(
            "initial_chips",
            candidate_payload.get("metrics", {}).get("initial_chips", 1000),
        )
    )
    max_steps_per_hand = int(
        baseline_payload.get("metrics", {}).get(
            "max_steps_per_hand",
            candidate_payload.get("metrics", {}).get("max_steps_per_hand", 256),
        )
    )
    eval_state_backend = _validate_eval_state_backend(eval_state_backend)

    candidate_payoffs: list[float] = []
    candidate_pair_payoffs: list[float] = []
    total_steps = 0
    eval_start = time.perf_counter()
    pair_i = 0
    while len(candidate_payoffs) < int(n_games):
        game_seed = int(seed) + pair_i
        pair_payoffs: list[float] = []
        for candidate_seat in (0, 1):
            if len(candidate_payoffs) >= int(n_games):
                break
            random.seed(game_seed)
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)
            state = _new_h2h_state(
                initial_chips=initial_chips,
                eval_state_backend=eval_state_backend,
            )
            n_steps = 0
            while not state.is_terminal and n_steps < max_steps_per_hand:
                features = state.to_feature_vector().astype(np.float32, copy=False)
                legal_mask = _h2h_legal_mask(state, eval_state_backend=eval_state_backend)
                current_player_i = _h2h_current_player_i(
                    state,
                    eval_state_backend=eval_state_backend,
                )
                policy_map = (
                    candidate_policy_map
                    if current_player_i == int(candidate_seat)
                    else baseline_policy_map
                )
                policy = policy_map.get(current_player_i, policy_map[0])
                action_idx = _rainbow_greedy_action(policy, features, legal_mask)
                state = _h2h_apply_action(
                    state,
                    action_idx,
                    eval_state_backend=eval_state_backend,
                )
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
    payoff_std = float(np.std(candidate_pair_payoffs, ddof=1)) if len(candidate_pair_payoffs) > 1 else 0.0
    payoff_se = payoff_std / float(np.sqrt(len(candidate_pair_payoffs))) if candidate_pair_payoffs else 0.0
    return {
        "algorithm": "tianshou_rainbow_vs_tianshou_rainbow_h2h",
        "role": "rl_control_checkpoint_league",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_algorithm": str(candidate_payload.get("algorithm", "tianshou_rainbow_dqn")),
        "baseline_algorithm": str(baseline_payload.get("algorithm", "tianshou_rainbow_dqn")),
        **device_info,
        "eval_state_backend": eval_state_backend,
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


def run_control(
    *,
    train_steps: int = 200,
    updates: int = 10,
    updates_per_collect: int = 1,
    batch_size: int = 32,
    hidden_dim: int = 128,
    num_atoms: int = 51,
    warmup_steps: int | None = None,
    eval_games: int = 100,
    lr: float = 1e-3,
    gamma: float = 0.99,
    n_step: int = 1,
    target_update_freq: int = 50,
    replay_size: int = 10_000,
    num_envs: int = 1,
    vector_env_backend: str = "dummy",
    opponent_kind: str = "random",
    opponent_checkpoint: str | Path | Sequence[str | Path] | None = None,
    opponent_device: str = "auto",
    state_backend: str = "full-deck",
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    seed: int = 20260526,
    device: str = "auto",
    output_json: str | None = None,
    checkpoint_in: str | None = None,
    checkpoint_out: str | None = None,
) -> dict:
    overall_start = time.perf_counter()
    import gymnasium  # noqa: F401
    from tianshou.algorithm.algorithm_base import policy_within_training_step
    from tianshou.algorithm.modelfree.c51 import C51Policy
    from tianshou.algorithm.modelfree.rainbow import RainbowDQN
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    num_envs = max(1, int(num_envs))
    vector_env_backend = str(vector_env_backend)
    if vector_env_backend not in {"dummy", "subproc"}:
        raise ValueError("vector_env_backend must be one of: dummy, subproc")
    state_backend = str(state_backend)
    if state_backend not in {"full-deck", "fast-state"}:
        raise ValueError("state_backend must be one of: full-deck, fast-state")
    opponent_kind = str(opponent_kind)
    if opponent_kind not in {"random", "native-nfsp", "rainbow"}:
        raise ValueError("opponent_kind must be one of: random, native-nfsp, rainbow")
    opponent_checkpoints = _normalize_opponent_checkpoints(opponent_checkpoint)
    if opponent_kind != "random":
        if not opponent_checkpoints:
            raise ValueError(f"opponent_checkpoint is required for {opponent_kind} opponent")
        missing = [path for path in opponent_checkpoints if not Path(path).exists()]
        if missing:
            raise ValueError(f"opponent_checkpoint does not exist: {missing[0]}")
    checkpoint_payload = None
    if checkpoint_in:
        if not Path(checkpoint_in).exists():
            raise ValueError(f"checkpoint_in does not exist: {checkpoint_in}")
        checkpoint_payload = torch.load(checkpoint_in, map_location=resolved_device, weights_only=False)
        if int(checkpoint_payload.get("num_actions", -1)) != N_ACTIONS:
            raise ValueError("checkpoint_in action count does not match native full-deck contract")
        if int(checkpoint_payload.get("num_features", -1)) != N_FEATURES:
            raise ValueError("checkpoint_in feature count does not match native full-deck contract")
    if opponent_kind == "random":
        effective_opponent_device = "cpu"
    elif str(opponent_device) == "auto":
        if num_envs > 1 and vector_env_backend == "subproc":
            effective_opponent_device = "cpu"
        else:
            effective_opponent_device = str(resolved_device)
    else:
        effective_opponent_device = str(resolve_device(str(opponent_device))["resolved_device"])
    base_env = NativeRainbowPokerEnv(
        seed=seed,
        initial_chips=initial_chips,
        max_steps_per_hand=max_steps_per_hand,
        # For vector training this object is used only for spaces. Keep it free
        # of checkpoint opponents so SubprocVectorEnv workers load policies
        # after process creation instead of inheriting parent Torch state.
        opponent_kind=opponent_kind if num_envs == 1 else "random",
        opponent_checkpoint=opponent_checkpoints if num_envs == 1 else None,
        opponent_device=effective_opponent_device if num_envs == 1 else "cpu",
        state_backend=state_backend,
    )
    if num_envs == 1:
        env = base_env
        replay = ReplayBuffer(size=int(replay_size))
        vector_env_context = None
    else:
        env_fns = [
            partial(
                NativeRainbowPokerEnv,
                seed=int(seed) + env_i,
                initial_chips=int(initial_chips),
                max_steps_per_hand=int(max_steps_per_hand),
                opponent_kind=opponent_kind,
                opponent_checkpoint=opponent_checkpoints,
                opponent_device=effective_opponent_device,
                state_backend=state_backend,
            )
            for env_i in range(num_envs)
        ]
        vector_cls = DummyVectorEnv if vector_env_backend == "dummy" else SubprocVectorEnv
        if vector_cls is SubprocVectorEnv and opponent_kind != "random":
            vector_env_context = "spawn"
            env = vector_cls(env_fns, context="spawn")
        else:
            vector_env_context = None
            env = vector_cls(env_fns)
        replay = VectorReplayBuffer(total_size=int(replay_size), buffer_num=num_envs)
    model = _RainbowDistributionNet(
        hidden_dim=int(hidden_dim),
        num_atoms=int(num_atoms),
        device=resolved_device,
    ).to(resolved_device)
    if checkpoint_payload is not None:
        model.load_state_dict(_rainbow_state_dict_from_payload(checkpoint_payload))
    policy = C51Policy(
        model=model,
        action_space=base_env.action_space,
        observation_space=base_env.observation_space,
        num_atoms=int(num_atoms),
        v_min=-1.0,
        v_max=1.0,
        eps_training=0.1,
        eps_inference=0.0,
    )
    policy.to(resolved_device)
    algorithm = RainbowDQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=float(lr)),
        gamma=float(gamma),
        n_step_return_horizon=int(n_step),
        target_update_freq=int(target_update_freq),
    )
    collector = Collector(algorithm, env, replay, exploration_noise=True)

    train_start = time.perf_counter()
    warmup = int(warmup_steps if warmup_steps is not None else max(batch_size, train_steps))
    collector.collect(n_step=max(warmup, 1), reset_before_collect=True)
    collected_steps = int(warmup)
    losses: list[float] = []
    learner_updates = 0
    updates_per_collect = max(1, int(updates_per_collect))
    for _ in range(int(updates)):
        collector.collect(n_step=max(int(train_steps), 1))
        collected_steps += int(train_steps)
        for _update_i in range(updates_per_collect):
            with policy_within_training_step(algorithm.policy):
                stats = algorithm.update(replay, sample_size=min(int(batch_size), len(replay)))
            learner_updates += 1
            loss = getattr(stats, "loss", None)
            if loss is not None:
                losses.append(float(loss))
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    eval_metrics = _evaluate_greedy_policy(
        policy,
        eval_games=int(eval_games),
        seed=int(seed) + 100_000,
        initial_chips=int(initial_chips),
        max_steps_per_hand=int(max_steps_per_hand),
        opponent_kind=opponent_kind,
        opponent_checkpoint=opponent_checkpoints,
        opponent_device=effective_opponent_device,
        state_backend=state_backend,
    )
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start
    total_seconds = time.perf_counter() - overall_start

    metrics = {
        "algorithm": "tianshou_rainbow_dqn",
        "role": "rl_control_baseline",
        "environment": "poker_ai:full_deck_hu_nlhe_single_agent_control",
        "warning": (
            "Rainbow DQN is a single-agent value-learning control on a local "
            "opponent wrapper, not an equilibrium method or Slumbot candidate."
        ),
        **device_info,
        "num_actions": N_ACTIONS,
        "train_steps": int(collected_steps),
        "updates": int(updates),
        "collect_iterations": int(updates),
        "updates_per_collect": int(updates_per_collect),
        "learner_updates": int(learner_updates),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "num_atoms": int(num_atoms),
        "num_envs": int(num_envs),
        "vector_env_backend": vector_env_backend if num_envs > 1 else "single",
        "vector_env_context": vector_env_context,
        "state_backend": state_backend,
        "opponent_kind": opponent_kind,
        "opponent_checkpoint": opponent_checkpoints[0] if len(opponent_checkpoints) == 1 else None,
        "opponent_checkpoints": opponent_checkpoints,
        "opponent_device_requested": str(opponent_device),
        "opponent_device": effective_opponent_device if opponent_kind != "random" else None,
        "checkpoint_in": str(checkpoint_in) if checkpoint_in else None,
        "checkpoint_in_algorithm": (
            str(checkpoint_payload.get("algorithm")) if checkpoint_payload is not None else None
        ),
        "n_step": int(n_step),
        "target_update_freq": int(target_update_freq),
        "train_seconds": float(train_seconds),
        "setup_seconds": float(max(0.0, train_start - overall_start)),
        "total_seconds": float(total_seconds),
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
                "num_atoms": int(num_atoms),
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "parent_checkpoint": str(checkpoint_in) if checkpoint_in else None,
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
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--updates-per-collect", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=50)
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vector-env-backend", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--state-backend", choices=("full-deck", "fast-state"), default="full-deck")
    parser.add_argument("--opponent-kind", choices=("random", "native-nfsp", "rainbow"), default="random")
    parser.add_argument(
        "--opponent-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Device for fixed checkpoint opponents. auto uses the learner device "
            "for in-process runs and CPU for subproc workers."
        ),
    )
    parser.add_argument(
        "--opponent-checkpoint",
        action="append",
        help="Opponent checkpoint path. Repeat to sample opponents from a checkpoint league.",
    )
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--eval-state-backend",
        choices=("full-deck", "fast-state", "fast-state-canonical-deal"),
        default="full-deck",
        help="State backend for checkpoint H2H evaluation. Training still uses --state-backend.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument(
        "--baseline-kind",
        choices=("native-nfsp", "npi", "rainbow"),
        default="native-nfsp",
        help="Checkpoint family for --baseline-checkpoint H2H evaluation.",
    )
    args = parser.parse_args(argv)
    if args.checkpoint_in and args.baseline_checkpoint:
        if args.baseline_kind == "rainbow":
            metrics = evaluate_rainbow_checkpoint_vs_rainbow(
                args.checkpoint_in,
                args.baseline_checkpoint,
                n_games=args.eval_games,
                device=args.device,
                seed=args.seed,
                eval_state_backend=args.eval_state_backend,
            )
        elif args.baseline_kind == "npi":
            metrics = evaluate_rainbow_checkpoint_vs_neural_policy_iteration(
                args.checkpoint_in,
                args.baseline_checkpoint,
                n_games=args.eval_games,
                device=args.device,
                seed=args.seed,
                eval_state_backend=args.eval_state_backend,
            )
        else:
            metrics = evaluate_rainbow_checkpoint_vs_native_nfsp(
                args.checkpoint_in,
                args.baseline_checkpoint,
                n_games=args.eval_games,
                device=args.device,
                seed=args.seed,
                eval_state_backend=args.eval_state_backend,
            )
        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        metrics = run_control(
            train_steps=args.train_steps,
            updates=args.updates,
            updates_per_collect=args.updates_per_collect,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_atoms=args.num_atoms,
            warmup_steps=args.warmup_steps,
            eval_games=args.eval_games,
            lr=args.lr,
            gamma=args.gamma,
            n_step=args.n_step,
            target_update_freq=args.target_update_freq,
            replay_size=args.replay_size,
            num_envs=args.num_envs,
            vector_env_backend=args.vector_env_backend,
            opponent_kind=args.opponent_kind,
            opponent_checkpoint=args.opponent_checkpoint,
            opponent_device=args.opponent_device,
            state_backend=args.state_backend,
            initial_chips=args.initial_chips,
            max_steps_per_hand=args.max_steps_per_hand,
            seed=args.seed,
            device=args.device,
            output_json=args.output_json,
            checkpoint_in=args.checkpoint_in,
            checkpoint_out=args.checkpoint_out,
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
