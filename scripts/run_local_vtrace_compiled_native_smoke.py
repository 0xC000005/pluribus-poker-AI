#!/usr/bin/env python3
"""Tiny local V-trace smoke on compiled native self-play trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES  # noqa: E402
from poker_ai.research.local_vtrace import vtrace_policy_value_loss  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.native_rollout_substrate import (  # noqa: E402
    _collect_compiled_fast_policy_gradient_rollout,
)


class _NativeActorCritic(nn.Module):
    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_FEATURES, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.policy = nn.Linear(int(hidden_dim), N_ACTIONS)
        self.value = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(features)
        return self.policy(x), self.value(x).squeeze(-1)


class _PolicyView(nn.Module):
    def __init__(self, actor_critic: _NativeActorCritic) -> None:
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits, _values = self.actor_critic(features)
        return logits


def _write_metrics(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _trajectory_indices(batch: dict[str, np.ndarray]) -> list[list[int]]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for record_i, (game_i, player_i) in enumerate(
        zip(batch["game_indices"], batch["players"], strict=False)
    ):
        grouped.setdefault((int(game_i), int(player_i)), []).append(int(record_i))
    return [
        sorted(indices, key=lambda i: int(batch["step_indices"][i]))
        for indices in grouped.values()
        if indices
    ]


def _pack_trajectories(
    batch: dict[str, np.ndarray],
    trajectories: list[list[int]],
    *,
    gamma: float,
) -> dict[str, np.ndarray]:
    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    max_len = max(len(indices) for indices in trajectories)
    n_traj = len(trajectories)
    features = np.zeros((max_len, n_traj, N_FEATURES), dtype=np.float32)
    legal_masks = np.zeros((max_len, n_traj, N_ACTIONS), dtype=np.bool_)
    legal_masks[:, :, 0] = True
    actions = np.zeros((max_len, n_traj), dtype=np.int64)
    behavior_log_probs = np.zeros((max_len, n_traj), dtype=np.float32)
    rewards = np.zeros((max_len, n_traj), dtype=np.float32)
    discounts = np.zeros((max_len, n_traj), dtype=np.float32)
    valid_mask = np.zeros((max_len, n_traj), dtype=np.bool_)
    for col, indices in enumerate(trajectories):
        length = len(indices)
        features[:length, col, :] = batch["features"][indices]
        legal_masks[:length, col, :] = batch["legal_masks"][indices] > 0
        actions[:length, col] = batch["actions"][indices]
        behavior_log_probs[:length, col] = batch["old_log_probs"][indices]
        discounts[:length, col] = float(gamma)
        discounts[length - 1, col] = 0.0
        rewards[length - 1, col] = float(batch["rewards"][indices[-1]])
        valid_mask[:length, col] = True
    return {
        "features": features,
        "legal_masks": legal_masks,
        "actions": actions,
        "behavior_log_probs": behavior_log_probs,
        "rewards": rewards,
        "discounts": discounts,
        "valid_mask": valid_mask,
    }


def run_smoke(
    *,
    n_games: int = 32,
    collector_batch_size: int = 16,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    updates: int = 1,
    hidden_dim: int = 64,
    lr: float = 3e-4,
    gamma: float = 1.0,
    seed: int = 20260528,
    device: str = "auto",
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    model = _NativeActorCritic(hidden_dim=int(hidden_dim)).to(resolved_device)
    policy_view = _PolicyView(model).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    losses: list[float] = []
    illegal_probabilities: list[float] = []
    n_samples = 0
    n_trajectories = 0
    collector_steps = 0
    compiled_needs_python_showdown = 0
    collector_seconds = 0.0
    start = time.perf_counter()
    for update_i in range(int(updates)):
        batch, collector_metrics = _collect_compiled_fast_policy_gradient_rollout(
            policy_view,
            n_games=int(n_games),
            batch_size=int(collector_batch_size),
            max_steps_per_game=int(max_steps_per_game),
            initial_chips=int(initial_chips),
            device=resolved_device,
            seed=int(seed) + update_i * 10_000,
        )
        collector_steps += int(collector_metrics.get("steps", 0))
        collector_seconds += float(collector_metrics.get("seconds", 0.0))
        compiled_needs_python_showdown += int(collector_metrics.get("needs_python_showdown", 0))
        trajectories = _trajectory_indices(batch)
        if not trajectories:
            continue
        packed = _pack_trajectories(batch, trajectories, gamma=float(gamma))
        features_t = torch.as_tensor(packed["features"], dtype=torch.float32, device=resolved_device)
        legal_masks_t = torch.as_tensor(packed["legal_masks"], dtype=torch.bool, device=resolved_device)
        actions_t = torch.as_tensor(packed["actions"], dtype=torch.long, device=resolved_device)
        behavior_log_probs_t = torch.as_tensor(
            packed["behavior_log_probs"],
            dtype=torch.float32,
            device=resolved_device,
        )
        rewards_t = torch.as_tensor(packed["rewards"], dtype=torch.float32, device=resolved_device)
        discounts_t = torch.as_tensor(packed["discounts"], dtype=torch.float32, device=resolved_device)
        valid_mask_t = torch.as_tensor(packed["valid_mask"], dtype=torch.bool, device=resolved_device)
        flat_features = features_t.reshape(-1, N_FEATURES)
        logits_flat, values_flat = model(flat_features)
        logits = logits_flat.reshape(features_t.shape[0], features_t.shape[1], N_ACTIONS)
        values = values_flat.reshape(features_t.shape[0], features_t.shape[1])
        bootstrap = torch.zeros((features_t.shape[1],), dtype=torch.float32, device=resolved_device)
        loss, stats = vtrace_policy_value_loss(
            logits=logits,
            values=values,
            actions=actions_t,
            legal_mask=legal_masks_t,
            behavior_action_log_probs=behavior_log_probs_t,
            rewards=rewards_t,
            discounts=discounts_t,
            bootstrap_value=bootstrap,
            valid_mask=valid_mask_t,
        )
        illegal_probabilities.append(float(stats["illegal_action_probability"]))
        n_samples += int(valid_mask_t.sum().detach().cpu())
        n_trajectories += len(trajectories)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    max_illegal_probability = float(max(illegal_probabilities) if illegal_probabilities else 0.0)
    metrics: dict[str, Any] = {
        "algorithm": "local_vtrace_compiled_native_smoke",
        "role": "local_vtrace_compiled_native_learner_smoke",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "This is a compiled-trajectory learner smoke, not policy-strength evidence.",
        **device_info,
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
        "gamma": float(gamma),
        "collector_backend": "compiled-fast-state",
        "trajectory_packing": "padded_vectorized",
        "collector_steps": int(collector_steps),
        "collector_seconds": float(collector_seconds),
        "compiled_needs_python_showdown": int(compiled_needs_python_showdown),
        "n_trajectories": int(n_trajectories),
        "n_samples": int(n_samples),
        "train_seconds": float(train_seconds),
        "samples_per_second": float(n_samples / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "loss_is_finite": bool(losses and np.isfinite(losses[-1])),
        "illegal_action_probability": max_illegal_probability,
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "passed": bool(
            losses
            and np.isfinite(losses[-1])
            and max_illegal_probability == 0.0
            and compiled_needs_python_showdown == 0
            and n_samples > 0
        ),
    }
    return _write_metrics(metrics, output_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-games", type=int, default=32)
    parser.add_argument("--collector-batch-size", type=int, default=16)
    parser.add_argument("--max-steps-per-game", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_smoke(
        n_games=args.n_games,
        collector_batch_size=args.collector_batch_size,
        max_steps_per_game=args.max_steps_per_game,
        initial_chips=args.initial_chips,
        updates=args.updates,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
