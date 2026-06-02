#!/usr/bin/env python3
"""Train a native 9-action neural NashPG/MMD-style learner from local self-play.

This is the native translation of the small-NLHE neural NashPG truth gate:
compiled stochastic self-play, terminal-return policy gradient with a learned
value baseline, entropy, and reference-policy KL. It uses only the local
full-deck simulator and is not Slumbot or solver-label training.
"""

from __future__ import annotations

import argparse
import copy
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
from poker_ai.research.local_vtrace import masked_log_probs  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.native_ppo_policy import _PolicyMLP, _ValueMLP  # noqa: E402
from poker_ai.research.native_rollout_substrate import (  # noqa: E402
    _collect_compiled_fast_policy_gradient_rollout,
)
from scripts.run_local_vtrace_compiled_native_learner import (  # noqa: E402
    _load_compiled_rollout_opponents,
    _load_reference_policy,
    _summarize_opponent_kinds,
)


def _write_metrics(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _load_checkpoint_in(
    *,
    checkpoint_in: str | Path | None,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint_in is None:
        return None
    payload = torch.load(str(checkpoint_in), map_location=device, weights_only=False)
    if str(payload.get("environment", "")) != "poker_ai:full_deck_hu_nlhe":
        raise ValueError("checkpoint_in must be trained in the native full-deck environment")
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("checkpoint_in action count does not match native contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("checkpoint_in feature count does not match native contract")
    if "policy_net_state_dict" not in payload:
        raise ValueError("checkpoint_in does not contain a policy_net_state_dict")
    policy_net.load_state_dict(payload["policy_net_state_dict"])
    if "value_net_state_dict" in payload:
        value_net.load_state_dict(payload["value_net_state_dict"])
    return dict(payload)


def _policy_value_reference_loss(
    *,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    reference_policy: torch.nn.Module | None,
    batch: dict[str, np.ndarray],
    device: torch.device,
    value_weight: float,
    entropy_weight: float,
    reference_kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float], int]:
    if int(batch["actions"].shape[0]) <= 0:
        return torch.zeros((), dtype=torch.float32, device=device), {}, 0

    features = torch.as_tensor(batch["features"], dtype=torch.float32, device=device)
    legal_masks = torch.as_tensor(batch["legal_masks"] > 0, dtype=torch.bool, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    returns = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)

    logits = policy_net(features)
    log_probs = masked_log_probs(logits, legal_masks)
    probs = torch.exp(log_probs)
    values = value_net(features).reshape(-1)
    action_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)

    advantages = returns - values.detach()
    if int(advantages.numel()) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
    policy_loss = -(action_log_probs * advantages).mean()
    value_loss = torch.mean(torch.square(values - returns))
    entropy = -torch.sum(probs * torch.nan_to_num(log_probs, neginf=0.0), dim=1).mean()

    reference_kl = torch.zeros((), dtype=torch.float32, device=device)
    if reference_policy is not None and float(reference_kl_weight) > 0.0:
        with torch.no_grad():
            reference_logits = reference_policy(features)
            reference_log_probs = masked_log_probs(reference_logits, legal_masks)
        kl_terms = probs * (
            torch.nan_to_num(log_probs, neginf=0.0)
            - torch.nan_to_num(reference_log_probs, neginf=0.0)
        )
        kl_terms = torch.where(legal_masks, kl_terms, torch.zeros_like(kl_terms))
        reference_kl = torch.sum(kl_terms, dim=1).mean()

    illegal_probability = torch.where(
        legal_masks,
        torch.zeros_like(probs),
        probs,
    ).sum(dim=1).max()
    loss = (
        policy_loss
        + float(value_weight) * value_loss
        + float(reference_kl_weight) * reference_kl
        - float(entropy_weight) * entropy
    )
    stats = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "reference_kl": float(reference_kl.detach().cpu()),
        "reference_kl_weight": float(reference_kl_weight),
        "illegal_action_probability": float(illegal_probability.detach().cpu()),
        "mean_return": float(returns.detach().mean().cpu()),
        "std_return": float(returns.detach().std(unbiased=False).cpu()),
    }
    return loss, stats, int(actions.numel())


def run_learner(
    *,
    train_iterations: int = 8,
    games_per_iteration: int = 512,
    collector_batch_size: int = 128,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    value_weight: float = 0.5,
    entropy_weight: float = 0.02,
    reference_kl_weight: float = 0.05,
    reference_update_every: int = 4,
    seed: int = 20260661,
    device: str = "auto",
    checkpoint_in: str | Path | None = None,
    opponent_checkpoints: list[str | Path] | None = None,
    reference_policy_checkpoint: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    if int(train_iterations) <= 0:
        raise ValueError("train_iterations must be positive")
    if int(games_per_iteration) <= 0:
        raise ValueError("games_per_iteration must be positive")
    if int(reference_update_every) <= 0:
        raise ValueError("reference_update_every must be positive")
    if float(reference_kl_weight) < 0.0:
        raise ValueError("reference_kl_weight must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    policy_net = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    value_net = _ValueMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    checkpoint_payload = _load_checkpoint_in(
        checkpoint_in=checkpoint_in,
        policy_net=policy_net,
        value_net=value_net,
        device=resolved_device,
    )
    opponent_policies, opponent_kinds = _load_compiled_rollout_opponents(
        opponent_checkpoints,
        resolved_device,
    )

    loaded_reference, reference_kind = _load_reference_policy(
        reference_policy_checkpoint,
        resolved_device,
    )
    moving_reference = loaded_reference is None and float(reference_kl_weight) > 0.0
    if moving_reference:
        reference_policy = copy.deepcopy(policy_net).to(resolved_device)
        reference_kind = "moving-self"
    else:
        reference_policy = loaded_reference
    if reference_policy is not None:
        reference_policy.eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=float(lr),
    )

    losses: list[float] = []
    entropy_values: list[float] = []
    reference_kls: list[float] = []
    illegal_probabilities: list[float] = []
    n_samples = 0
    collector_steps = 0
    collector_total_env_steps = 0
    collector_seconds = 0.0
    compiled_needs_python_showdown = 0
    reference_updates = 0
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
        collector_total_env_steps += int(collector_metrics.get("total_env_steps", 0))
        collector_seconds += float(collector_metrics.get("seconds", 0.0))
        compiled_needs_python_showdown += int(collector_metrics.get("needs_python_showdown", 0))

        loss, stats, batch_samples = _policy_value_reference_loss(
            policy_net=policy_net,
            value_net=value_net,
            reference_policy=reference_policy,
            batch=batch,
            device=resolved_device,
            value_weight=float(value_weight),
            entropy_weight=float(entropy_weight),
            reference_kl_weight=float(reference_kl_weight),
        )
        if batch_samples <= 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy_net.parameters()) + list(value_net.parameters()),
            10.0,
        )
        optimizer.step()

        losses.append(float(stats["loss"]))
        entropy_values.append(float(stats["entropy"]))
        reference_kls.append(float(stats["reference_kl"]))
        illegal_probabilities.append(float(stats["illegal_action_probability"]))
        n_samples += int(batch_samples)

        if moving_reference and (iteration_i + 1) % int(reference_update_every) == 0:
            reference_policy.load_state_dict(policy_net.state_dict())
            reference_updates += 1

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    max_illegal_probability = float(max(illegal_probabilities) if illegal_probabilities else 0.0)
    checkpoint_path = None if checkpoint_out is None else str(Path(checkpoint_out))
    metrics: dict[str, Any] = {
        "algorithm": "native_neural_nashpg_compiled",
        "role": "native_full_deck_reference_regularized_policy_gradient",
        "gate": "native_neural_nashpg_compiled_smoke",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Native 9-action tabula-rasa learner smoke; not Slumbot strength evidence.",
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
        "value_weight": float(value_weight),
        "entropy_weight": float(entropy_weight),
        "reference_kl_weight": float(reference_kl_weight),
        "reference_update_every": int(reference_update_every),
        "moving_reference": bool(moving_reference),
        "reference_updates": int(reference_updates),
        "reference_policy_checkpoint": (
            str(reference_policy_checkpoint) if reference_policy_checkpoint is not None else None
        ),
        "reference_policy_kind": reference_kind,
        "checkpoint_in": str(checkpoint_in) if checkpoint_in is not None else None,
        "checkpoint_in_algorithm": (
            str(checkpoint_payload.get("algorithm", "")) if checkpoint_payload is not None else None
        ),
        "fresh_initial_policy": checkpoint_payload is None,
        "population_training": bool(opponent_policies),
        "opponent_kind": _summarize_opponent_kinds(opponent_kinds),
        "opponent_kinds": list(opponent_kinds),
        "opponent_population_size": int(len(opponent_policies)),
        "opponent_checkpoints": [str(path) for path in (opponent_checkpoints or [])],
        "collector_backend": "compiled-fast-state",
        "collector_steps": int(collector_steps),
        "collector_total_env_steps": int(collector_total_env_steps),
        "collector_seconds": float(collector_seconds),
        "compiled_needs_python_showdown": int(compiled_needs_python_showdown),
        "n_samples": int(n_samples),
        "train_seconds": float(train_seconds),
        "samples_per_second": float(n_samples / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "loss_is_finite": bool(losses and np.isfinite(losses[-1])),
        "mean_entropy": float(np.mean(entropy_values)) if entropy_values else None,
        "last_entropy": float(entropy_values[-1]) if entropy_values else None,
        "mean_reference_kl": float(np.mean(reference_kls)) if reference_kls else None,
        "last_reference_kl": float(reference_kls[-1]) if reference_kls else None,
        "illegal_action_probability": max_illegal_probability,
        "trained_environment_native": True,
        "native_action_projection": False,
        "rlcard_candidate": False,
        "uses_slumbot_data": False,
        "uses_slumbot_training_data": False,
        "uses_alphanlholdem_training_data": False,
        "uses_solver_labels": False,
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
                "algorithm": "native_neural_nashpg_compiled",
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
                    "reference_policy_checkpoint": (
                        str(reference_policy_checkpoint)
                        if reference_policy_checkpoint is not None
                        else None
                    ),
                    "reference_policy_kind": reference_kind,
                    "reference_kl_weight": float(reference_kl_weight),
                    "reference_update_every": int(reference_update_every),
                    "moving_reference": bool(moving_reference),
                    "opponent_population_size": int(len(opponent_policies)),
                    "opponent_kinds": list(opponent_kinds),
                },
                "metrics": metrics,
                "trained_environment_native": True,
                "native_action_projection": False,
                "rlcard_candidate": False,
                "uses_slumbot_data": False,
                "uses_slumbot_training_data": False,
                "uses_alphanlholdem_training_data": False,
                "uses_solver_labels": False,
            },
            path,
        )
    return _write_metrics(metrics, output_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iterations", type=int, default=8)
    parser.add_argument("--games-per-iteration", type=int, default=512)
    parser.add_argument("--collector-batch-size", type=int, default=128)
    parser.add_argument("--max-steps-per-game", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument("--reference-update-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260661)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument(
        "--opponent-checkpoint",
        action="append",
        dest="opponent_checkpoints",
        help="Frozen native-ppo/Rainbow opponent checkpoint. May be supplied multiple times.",
    )
    parser.add_argument(
        "--reference-policy-checkpoint",
        type=Path,
        help="Frozen local policy checkpoint used as the reference policy.",
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
        value_weight=args.value_weight,
        entropy_weight=args.entropy_weight,
        reference_kl_weight=args.reference_kl_weight,
        reference_update_every=args.reference_update_every,
        seed=args.seed,
        device=args.device,
        checkpoint_in=args.checkpoint_in,
        opponent_checkpoints=args.opponent_checkpoints,
        reference_policy_checkpoint=args.reference_policy_checkpoint,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
