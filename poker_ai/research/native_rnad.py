"""Native full-deck R-NaD trajectory collection over the compiled fast-state batch.

This module is a scaling bridge, not strength evidence: it lets the existing
game-agnostic R-NaD learner consume native 9-action self-play trajectories
generated from the local simulator only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import FastPokerState, N_ACTIONS, N_FEATURES
from poker_ai.research.compiled_fast_rollout import (
    CompiledFastStateBatch,
    compiled_apply_actions,
    compiled_feature_vectors,
    compiled_legal_masks,
)
from poker_ai.research.native_rollout_substrate import _new_seeded_fast_state
from poker_ai.rnad.collector import Trajectory
from poker_ai.rnad.solver import RNaDConfig, RNaDSolver


def _normalize_legal_policy(policy: np.ndarray, legal: np.ndarray) -> np.ndarray:
    pi = np.asarray(policy, dtype=np.float64)
    mask = np.asarray(legal, dtype=np.float64) > 0.0
    pi = np.where(mask, np.maximum(pi, 0.0), 0.0)
    row_sum = pi.sum(axis=1, keepdims=True)
    legal_count = mask.sum(axis=1, keepdims=True).clip(min=1)
    fallback = mask.astype(np.float64) / legal_count
    pi = np.where(row_sum > 0.0, pi / np.maximum(row_sum, 1e-300), fallback)
    return pi.astype(np.float32, copy=False)


def _sample_actions(policy: np.ndarray, rng) -> np.ndarray:
    cdf = np.cumsum(np.asarray(policy, dtype=np.float64), axis=1)
    cdf[:, -1] = 1.0
    draws = rng.random((policy.shape[0], 1))
    return (draws < cdf).argmax(axis=1).astype(np.int16, copy=False)


@dataclass
class CompiledNativeRNaDCollector:
    """R-NaD collector for local full-deck heads-up no-limit trajectories."""

    collector_batch_size: int = 64
    initial_chips: int = 1000
    seed: int = 20260601
    device: str = "cpu"
    is_gpu: bool = False
    n_actions: int = N_ACTIONS
    n_players: int = 2
    obs_dim: int = N_FEATURES
    last_metrics: dict[str, Any] = field(default_factory=dict)

    def collect(self, policy_fn, batch_size: int, trajectory_max: int, rng) -> Trajectory:
        """Collect one padded [T, B] R-NaD trajectory batch from compiled native self-play."""
        n_games = int(batch_size)
        max_steps = int(trajectory_max)
        if n_games <= 0:
            raise ValueError("batch_size must be positive")
        if max_steps <= 0:
            raise ValueError("trajectory_max must be positive")
        batch_size_for_inference = max(1, int(self.collector_batch_size))
        seed = int(self.seed) + int(rng.randint(0, 2**30 - 1))
        states = [
            _new_seeded_fast_state(seed, game_i, int(self.initial_chips))
            for game_i in range(n_games)
        ]
        compiled = CompiledFastStateBatch.from_fast_states(states)
        steps_per_game = np.zeros(n_games, dtype=np.int32)
        records: list[tuple[int, int, int, np.ndarray, np.ndarray, int, np.ndarray]] = []
        forward_calls = 0
        needs_python_showdown = 0
        started = time.perf_counter()

        while True:
            live_indices = [
                game_i
                for game_i in range(n_games)
                if int(compiled.stage[game_i]) < FastPokerState.SHOWDOWN
                and int(steps_per_game[game_i]) < max_steps
            ]
            if not live_indices:
                break
            all_features = compiled_feature_vectors(compiled)
            all_masks = compiled_legal_masks(compiled)
            current_players = compiled.current_players()
            action_array = np.full(n_games, -1, dtype=np.int16)
            for start in range(0, len(live_indices), batch_size_for_inference):
                group = live_indices[start : start + batch_size_for_inference]
                features = all_features[group].astype(np.float32, copy=False)
                legal = all_masks[group].astype(np.float32, copy=False)
                policy = _normalize_legal_policy(policy_fn(features, legal), legal)
                actions = _sample_actions(policy, rng)
                forward_calls += 1
                for row_i, game_i in enumerate(group):
                    action_idx = int(actions[row_i])
                    action_array[int(game_i)] = action_idx
                    records.append(
                        (
                            int(game_i),
                            int(steps_per_game[int(game_i)]),
                            int(current_players[int(game_i)]),
                            features[row_i].astype(np.float32, copy=False),
                            legal[row_i].astype(np.float32, copy=False),
                            action_idx,
                            policy[row_i].astype(np.float32, copy=False),
                        )
                    )
                    steps_per_game[int(game_i)] += 1
            result = compiled_apply_actions(compiled, action_array)
            needs_python_showdown += int(result["needs_python_showdown"])

        seconds = time.perf_counter() - started
        payoffs = (
            compiled.chips.astype(np.float32) - float(self.initial_chips)
        ) / float(self.initial_chips)
        terminal = compiled.stage >= FastPokerState.SHOWDOWN
        truncated_games = int(np.sum(~terminal))
        payoffs[~terminal, :] = 0.0
        traj = self._pack(records, payoffs, n_games=n_games, trajectory_max=max_steps)
        illegal_records = sum(
            1
            for _game_i, _step_i, _player_i, _features, legal, action_idx, _policy in records
            if action_idx < 0 or action_idx >= self.n_actions or legal[action_idx] <= 0.0
        )
        self.last_metrics = {
            "collector": "compiled_native_rnad_collector",
            "backend": "compiled-fast-state",
            "n_games": n_games,
            "collector_batch_size": batch_size_for_inference,
            "trajectory_max": max_steps,
            "steps": int(len(records)),
            "seconds": float(seconds),
            "steps_per_second": float(len(records) / max(seconds, 1e-12)),
            "policy_forward_calls": int(forward_calls),
            "mean_decisions_per_forward": float(len(records) / max(forward_calls, 1)),
            "truncated_games": int(truncated_games),
            "needs_python_showdown": int(needs_python_showdown),
            "illegal_records": int(illegal_records),
        }
        return traj

    def _pack(
        self,
        records: list[tuple[int, int, int, np.ndarray, np.ndarray, int, np.ndarray]],
        payoffs: np.ndarray,
        *,
        n_games: int,
        trajectory_max: int,
    ) -> Trajectory:
        T, B, A, D, P = (
            int(trajectory_max),
            int(n_games),
            int(self.n_actions),
            int(self.obs_dim),
            int(self.n_players),
        )
        obs = np.zeros((T, B, D), dtype=np.float32)
        legal = np.ones((T, B, A), dtype=np.float32)
        player_id = np.zeros((T, B), dtype=np.float32)
        valid = np.zeros((T, B), dtype=np.float32)
        rewards = np.zeros((T, B, P), dtype=np.float32)
        action_oh = np.zeros((T, B, A), dtype=np.float32)
        action_oh[:, :, 0] = 1.0
        policy = np.full((T, B, A), 1.0 / float(A), dtype=np.float32)

        grouped: dict[int, list[tuple[int, int, int, np.ndarray, np.ndarray, int, np.ndarray]]] = {}
        for record in records:
            grouped.setdefault(int(record[0]), []).append(record)
        for game_i, game_records in grouped.items():
            ordered = sorted(game_records, key=lambda r: int(r[1]))[:T]
            if not ordered:
                continue
            for t, record in enumerate(ordered):
                _game_i, _step_i, actor, features, legal_mask, action_idx, behavior_policy = record
                obs[t, game_i, :] = features
                legal[t, game_i, :] = legal_mask
                player_id[t, game_i] = float(actor)
                valid[t, game_i] = 1.0
                action_oh[t, game_i, :] = 0.0
                action_oh[t, game_i, int(action_idx)] = 1.0
                policy[t, game_i, :] = behavior_policy
            last_t = len(ordered) - 1
            rewards[last_t, game_i, : min(P, payoffs.shape[1])] = payoffs[game_i, : min(P, payoffs.shape[1])]

        dev = torch.device(self.device)
        return Trajectory(
            obs=torch.from_numpy(obs).to(dev),
            legal=torch.from_numpy(legal).to(dev),
            player_id=torch.from_numpy(player_id).to(dev),
            valid=torch.from_numpy(valid).to(dev),
            rewards=torch.from_numpy(rewards).to(dev),
            action_oh=torch.from_numpy(action_oh).to(dev),
            policy=torch.from_numpy(policy).to(dev),
        )


def run_compiled_native_rnad_smoke(
    *,
    n_games: int = 32,
    collector_batch_size: int = 16,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    updates: int = 1,
    hidden_dim: int = 64,
    lr: float = 5e-5,
    seed: int = 20260601,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a finite-update smoke for R-NaD on native compiled full-deck rollouts."""
    requested_device = str(device).strip().lower()
    if requested_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        resolved_device = "cuda"
    elif requested_device == "cpu":
        resolved_device = "cpu"
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    collector = CompiledNativeRNaDCollector(
        collector_batch_size=int(collector_batch_size),
        initial_chips=int(initial_chips),
        seed=int(seed),
        device="cpu",
    )
    config = RNaDConfig(
        trajectory_max=int(max_steps_per_game),
        batch_size=int(n_games),
        policy_network_layers=(int(hidden_dim), int(hidden_dim)),
        learning_rate=float(lr),
        entropy_schedule_size=(max(1, int(updates)),),
        entropy_schedule_repeats=(1,),
        seed=int(seed),
    )
    solver = RNaDSolver(config, collector, device=resolved_device)
    losses: list[float] = []
    alphas: list[float] = []
    collector_metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    for _ in range(int(updates)):
        step = solver.step()
        losses.append(float(step["loss"]))
        alphas.append(float(step["alpha"]))
        collector_metrics.append(dict(collector.last_metrics))
    if torch.device(resolved_device).type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    total_samples = int(sum(int(m.get("steps", 0)) for m in collector_metrics))
    compiled_showdown = int(sum(int(m.get("needs_python_showdown", 0)) for m in collector_metrics))
    illegal_records = int(sum(int(m.get("illegal_records", 0)) for m in collector_metrics))
    finite_loss = bool(losses and np.all(np.isfinite(losses)))
    return {
        "algorithm": "rnad_compiled_native_smoke",
        "role": "native_full_deck_rnad_trajectory_contract_smoke",
        "warning": "This is R-NaD native trajectory plumbing, not policy-strength evidence.",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "collector_backend": "compiled-fast-state",
        "trajectory_contract": "poker_ai.rnad.collector.Trajectory",
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "seed": int(seed),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "n_games": int(n_games),
        "collector_batch_size": int(collector_batch_size),
        "max_steps_per_game": int(max_steps_per_game),
        "initial_chips": int(initial_chips),
        "updates": int(updates),
        "hidden_dim": int(hidden_dim),
        "lr": float(lr),
        "losses": losses,
        "last_loss": float(losses[-1]) if losses else None,
        "alphas": alphas,
        "rnad_loss_is_finite": finite_loss,
        "collector_metrics": collector_metrics,
        "compiled_needs_python_showdown": compiled_showdown,
        "illegal_records": illegal_records,
        "n_samples": total_samples,
        "n_trajectories": int(n_games) * int(updates),
        "train_seconds": float(seconds),
        "samples_per_second": float(total_samples / max(seconds, 1e-12)),
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "passed": bool(
            finite_loss
            and total_samples > 0
            and compiled_showdown == 0
            and illegal_records == 0
        ),
    }
