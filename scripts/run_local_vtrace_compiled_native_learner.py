#!/usr/bin/env python3
"""Train a native 9-action checkpoint with compiled self-play V-trace updates."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES  # noqa: E402
from poker_ai.research.local_vtrace import vtrace_policy_value_loss  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.native_ppo_policy import _load_policy_network, _PolicyMLP, _ValueMLP  # noqa: E402
from poker_ai.research.native_rollout_substrate import (  # noqa: E402
    _collect_compiled_fast_policy_gradient_rollout,
)
from scripts.run_local_vtrace_compiled_native_smoke import (  # noqa: E402
    _pack_trajectories,
    _trajectory_indices,
)


def _write_metrics(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _load_native_ppo_opponents(
    checkpoints: list[str | Path] | None,
    device: torch.device,
) -> list[torch.nn.Module]:
    opponents: list[torch.nn.Module] = []
    for checkpoint in checkpoints or []:
        _payload, policy, feature_mode = _load_policy_network(
            str(checkpoint),
            device,
            strategy_source="auto",
        )
        if feature_mode != "flat":
            raise ValueError("compiled native V-trace opponents must use flat features")
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        opponents.append(policy)
    return opponents


def _train_on_batch(
    *,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    gamma: float,
    device: torch.device,
) -> tuple[float, dict[str, float], int, int]:
    trajectories = _trajectory_indices(batch)
    if not trajectories:
        return 0.0, {"illegal_action_probability": 0.0}, 0, 0
    packed = _pack_trajectories(batch, trajectories, gamma=float(gamma))
    features_t = torch.as_tensor(packed["features"], dtype=torch.float32, device=device)
    legal_masks_t = torch.as_tensor(packed["legal_masks"], dtype=torch.bool, device=device)
    actions_t = torch.as_tensor(packed["actions"], dtype=torch.long, device=device)
    behavior_log_probs_t = torch.as_tensor(
        packed["behavior_log_probs"],
        dtype=torch.float32,
        device=device,
    )
    rewards_t = torch.as_tensor(packed["rewards"], dtype=torch.float32, device=device)
    discounts_t = torch.as_tensor(packed["discounts"], dtype=torch.float32, device=device)
    valid_mask_t = torch.as_tensor(packed["valid_mask"], dtype=torch.bool, device=device)
    flat_features = features_t.reshape(-1, N_FEATURES)
    logits = policy_net(flat_features).reshape(features_t.shape[0], features_t.shape[1], N_ACTIONS)
    values = value_net(flat_features).reshape(features_t.shape[0], features_t.shape[1])
    bootstrap = torch.zeros((features_t.shape[1],), dtype=torch.float32, device=device)
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
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return (
        float(loss.detach().cpu()),
        {name: float(value) for name, value in stats.items()},
        int(valid_mask_t.sum().detach().cpu()),
        int(len(trajectories)),
    )


def run_learner(
    *,
    train_iterations: int = 4,
    games_per_iteration: int = 512,
    collector_batch_size: int = 128,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    gamma: float = 1.0,
    seed: int = 20260559,
    device: str = "auto",
    opponent_checkpoints: list[str | Path] | None = None,
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Train a small native V-trace actor/value checkpoint.

    This is a native full-deck HUNL learner bridge. It deliberately writes a
    9-action checkpoint that is compatible with native H2H evaluators and is
    not a candidate for the RLCard/AlphaNLHoldem 5-action public-reference gate.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    policy_net = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    value_net = _ValueMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    opponent_policies = _load_native_ppo_opponents(opponent_checkpoints, resolved_device)
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=float(lr),
    )

    losses: list[float] = []
    illegal_probabilities: list[float] = []
    n_samples = 0
    n_trajectories = 0
    collector_steps = 0
    collector_seconds = 0.0
    compiled_needs_python_showdown = 0
    train_start = time.perf_counter()
    for iteration_i in range(int(train_iterations)):
        batch, collector_metrics = _collect_compiled_fast_policy_gradient_rollout(
            policy_net,
            n_games=int(games_per_iteration),
            batch_size=int(collector_batch_size),
            max_steps_per_game=int(max_steps_per_game),
            initial_chips=int(initial_chips),
            device=resolved_device,
            seed=int(seed) + iteration_i * 10_000,
            opponent_policies=opponent_policies if opponent_policies else None,
        )
        collector_steps += int(collector_metrics.get("steps", 0))
        collector_seconds += float(collector_metrics.get("seconds", 0.0))
        compiled_needs_python_showdown += int(collector_metrics.get("needs_python_showdown", 0))
        loss_value, stats, batch_samples, batch_trajectories = _train_on_batch(
            policy_net=policy_net,
            value_net=value_net,
            optimizer=optimizer,
            batch=batch,
            gamma=float(gamma),
            device=resolved_device,
        )
        if batch_samples > 0:
            losses.append(loss_value)
            illegal_probabilities.append(float(stats.get("illegal_action_probability", 0.0)))
            n_samples += int(batch_samples)
            n_trajectories += int(batch_trajectories)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    max_illegal_probability = float(max(illegal_probabilities) if illegal_probabilities else 0.0)
    checkpoint_path = None if checkpoint_out is None else str(Path(checkpoint_out))
    metrics: dict[str, Any] = {
        "algorithm": "local_vtrace_compiled_native",
        "role": "native_full_deck_vtrace_actor_learner",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Native 9-action learner checkpoint; not RLCard/AlphaNLHoldem strength evidence.",
        **device_info,
        "seed": int(seed),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "train_iterations": int(train_iterations),
        "games_per_iteration": int(games_per_iteration),
        "collector_batch_size": int(collector_batch_size),
        "max_steps_per_game": int(max_steps_per_game),
        "initial_chips": int(initial_chips),
        "hidden_dim": int(hidden_dim),
        "lr": float(lr),
        "gamma": float(gamma),
        "population_training": bool(opponent_policies),
        "opponent_kind": "native-ppo" if opponent_policies else None,
        "opponent_population_size": int(len(opponent_policies)),
        "opponent_checkpoints": [str(path) for path in (opponent_checkpoints or [])],
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
        "rlcard_candidate": False,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "checkpoint_path": checkpoint_path,
    }
    metrics["passed"] = bool(
        losses
        and np.isfinite(losses[-1])
        and max_illegal_probability == 0.0
        and compiled_needs_python_showdown == 0
        and n_samples > 0
    )

    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "local_vtrace_compiled_native",
                "environment": "poker_ai:full_deck_hu_nlhe",
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "policy_net_state_dict": policy_net.state_dict(),
                "value_net_state_dict": value_net.state_dict(),
                "config": {
                    "feature_mode": "flat",
                    "hidden_dim": int(hidden_dim),
                    "initial_chips": int(initial_chips),
                    "max_steps_per_hand": int(max_steps_per_game),
                    "rollout_backend": "compiled-fast-state",
                    "fsp_average_policy": False,
                    "train_environment": "poker_ai:full_deck_hu_nlhe",
                    "opponent_population_size": int(len(opponent_policies)),
                },
                "metrics": metrics,
                "trained_environment_native": True,
                "native_action_projection": False,
                "rlcard_candidate": False,
                "uses_slumbot_data": False,
                "uses_alphanlholdem_training_data": False,
            },
            path,
        )
    return _write_metrics(metrics, output_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iterations", type=int, default=4)
    parser.add_argument("--games-per-iteration", type=int, default=512)
    parser.add_argument("--collector-batch-size", type=int, default=128)
    parser.add_argument("--max-steps-per-game", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260559)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--opponent-checkpoint",
        action="append",
        dest="opponent_checkpoints",
        help="Frozen native-ppo-compatible opponent checkpoint. May be supplied multiple times.",
    )
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_learner(
        train_iterations=args.train_iterations,
        games_per_iteration=args.games_per_iteration,
        collector_batch_size=args.collector_batch_size,
        max_steps_per_game=args.max_steps_per_game,
        initial_chips=args.initial_chips,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        opponent_checkpoints=args.opponent_checkpoints,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
