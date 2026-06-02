#!/usr/bin/env python3
"""Train a native stochastic policy on local all-action rollout targets.

This is a learner-integration gate for the tabula-rasa workflow. Targets are
computed by forcing every legal action in the local simulator and rolling out
continuations; they are not solver labels, Slumbot traces, or human tactics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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
from poker_ai.research.native_ppo_policy import _PolicyMLP  # noqa: E402
from scripts.build_native_all_action_counterfactual_targets import (  # noqa: E402
    _sample_decision_states,
    estimate_all_action_rollout_values,
)


def target_policy_from_values(
    values: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    legal = np.asarray(legal_mask, dtype=np.float32) > 0
    raw_values = np.asarray(values, dtype=np.float32)
    finite_legal = legal & np.isfinite(raw_values)
    if not np.any(finite_legal):
        raise ValueError("at least one legal action value must be finite")
    logits = np.full(N_ACTIONS, -np.inf, dtype=np.float32)
    centered = raw_values[finite_legal] - float(np.max(raw_values[finite_legal]))
    logits[finite_legal] = centered / float(temperature)
    probs = np.zeros(N_ACTIONS, dtype=np.float32)
    exp_logits = np.exp(logits[finite_legal]).astype(np.float32)
    probs[finite_legal] = exp_logits / float(exp_logits.sum())
    return probs


def _build_dataset(
    *,
    n_states: int,
    rollouts_per_action: int,
    max_steps_per_rollout: int,
    initial_chips: int,
    seed: int,
    target_temperature: float,
) -> dict[str, np.ndarray]:
    states = _sample_decision_states(
        n_states=int(n_states),
        initial_chips=int(initial_chips),
        seed=int(seed),
        max_steps_per_hand=max(int(max_steps_per_rollout), 1),
    )
    features: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    values: list[np.ndarray] = []
    top_actions: list[int] = []
    for state_i, state in enumerate(states):
        estimate = estimate_all_action_rollout_values(
            state,
            n_rollouts_per_action=int(rollouts_per_action),
            max_steps_per_rollout=int(max_steps_per_rollout),
            seed=int(seed) + 50_000 + state_i,
            paired_rollout_seeds=True,
        )
        target = target_policy_from_values(
            estimate["values"],
            estimate["legal_mask"],
            temperature=float(target_temperature),
        )
        features.append(state.to_feature_vector().astype(np.float32, copy=False))
        legal_masks.append(np.asarray(estimate["legal_mask"], dtype=np.float32))
        targets.append(target.astype(np.float32, copy=False))
        values.append(np.asarray(estimate["values"], dtype=np.float32))
        top_actions.append(int(np.argmax(target)))
    return {
        "features": np.stack(features).astype(np.float32, copy=False),
        "legal_masks": np.stack(legal_masks).astype(np.float32, copy=False),
        "targets": np.stack(targets).astype(np.float32, copy=False),
        "values": np.stack(values).astype(np.float32, copy=False),
        "top_actions": np.asarray(top_actions, dtype=np.int64),
    }


def _cross_entropy_to_targets(
    policy_net: torch.nn.Module,
    features: torch.Tensor,
    legal_masks: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    log_probs = masked_log_probs(policy_net(features), legal_masks > 0)
    safe_log_probs = torch.where(targets > 0, log_probs, torch.zeros_like(log_probs))
    return -torch.sum(targets * safe_log_probs, dim=1).mean()


@torch.no_grad()
def _evaluate_policy_targets(
    policy_net: torch.nn.Module,
    dataset: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    features = torch.as_tensor(dataset["features"], dtype=torch.float32, device=device)
    legal_masks = torch.as_tensor(dataset["legal_masks"], dtype=torch.float32, device=device)
    targets = torch.as_tensor(dataset["targets"], dtype=torch.float32, device=device)
    log_probs = masked_log_probs(policy_net(features), legal_masks > 0)
    safe_log_probs = torch.where(targets > 0, log_probs, torch.zeros_like(log_probs))
    ce = -torch.sum(targets * safe_log_probs, dim=1)
    legal_counts = torch.sum(legal_masks > 0, dim=1).to(dtype=torch.float32)
    uniform_ce = torch.log(legal_counts.clamp_min(1.0))
    pred_top = torch.argmax(
        torch.where(legal_masks > 0, torch.exp(log_probs), torch.zeros_like(log_probs)),
        dim=1,
    )
    target_top = torch.as_tensor(dataset["top_actions"], dtype=torch.long, device=device)
    agreement = (pred_top == target_top).to(dtype=torch.float32).mean()
    return {
        "cross_entropy": float(ce.mean().detach().cpu()),
        "uniform_cross_entropy": float(uniform_ce.mean().detach().cpu()),
        "top_action_agreement": float(agreement.detach().cpu()),
    }


def run_training_gate(
    *,
    n_train_states: int = 64,
    n_eval_states: int = 32,
    rollouts_per_action: int = 16,
    max_steps_per_rollout: int = 64,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    n_steps: int = 400,
    batch_size: int = 64,
    lr: float = 1e-3,
    target_temperature: float = 0.1,
    seed: int = 20260720,
    device: str = "auto",
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    if int(n_train_states) <= 0 or int(n_eval_states) <= 0:
        raise ValueError("n_train_states and n_eval_states must be positive")
    if int(rollouts_per_action) <= 0:
        raise ValueError("rollouts_per_action must be positive")
    if int(n_steps) <= 0:
        raise ValueError("n_steps must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    train = _build_dataset(
        n_states=int(n_train_states),
        rollouts_per_action=int(rollouts_per_action),
        max_steps_per_rollout=int(max_steps_per_rollout),
        initial_chips=int(initial_chips),
        seed=int(seed),
        target_temperature=float(target_temperature),
    )
    eval_data = _build_dataset(
        n_states=int(n_eval_states),
        rollouts_per_action=int(rollouts_per_action),
        max_steps_per_rollout=int(max_steps_per_rollout),
        initial_chips=int(initial_chips),
        seed=int(seed) + 100_000,
        target_temperature=float(target_temperature),
    )
    policy_net = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=float(lr))
    train_features = torch.as_tensor(train["features"], dtype=torch.float32, device=resolved_device)
    train_masks = torch.as_tensor(train["legal_masks"], dtype=torch.float32, device=resolved_device)
    train_targets = torch.as_tensor(train["targets"], dtype=torch.float32, device=resolved_device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    losses: list[float] = []

    for _step_i in range(int(n_steps)):
        if int(batch_size) >= int(n_train_states):
            indices = torch.arange(int(n_train_states), device=resolved_device)
        else:
            indices = torch.randint(
                0,
                int(n_train_states),
                (int(batch_size),),
                generator=generator,
                device=resolved_device,
            )
        loss = _cross_entropy_to_targets(
            policy_net,
            train_features[indices],
            train_masks[indices],
            train_targets[indices],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    train_eval = _evaluate_policy_targets(policy_net, train, resolved_device)
    holdout_eval = _evaluate_policy_targets(policy_net, eval_data, resolved_device)
    improvement = float(holdout_eval["uniform_cross_entropy"] - holdout_eval["cross_entropy"])
    checkpoint_path = None if checkpoint_out is None else str(Path(checkpoint_out))
    metrics: dict[str, Any] = {
        "algorithm": "native_all_action_target_policy",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "gate": "native_all_action_target_policy_root_preference",
        "warning": "Policy trained on local all-action rollout targets; not H2H or Slumbot strength evidence.",
        **device_info,
        "seed": int(seed),
        "n_train_states": int(n_train_states),
        "n_eval_states": int(n_eval_states),
        "rollouts_per_action": int(rollouts_per_action),
        "max_steps_per_rollout": int(max_steps_per_rollout),
        "initial_chips": int(initial_chips),
        "hidden_dim": int(hidden_dim),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "target_temperature": float(target_temperature),
        "last_train_loss": float(losses[-1]) if losses else None,
        "train_cross_entropy": float(train_eval["cross_entropy"]),
        "train_uniform_cross_entropy": float(train_eval["uniform_cross_entropy"]),
        "train_top_action_agreement": float(train_eval["top_action_agreement"]),
        "eval_cross_entropy": float(holdout_eval["cross_entropy"]),
        "uniform_eval_cross_entropy": float(holdout_eval["uniform_cross_entropy"]),
        "target_kl_improvement_over_uniform": improvement,
        "eval_top_action_agreement": float(holdout_eval["top_action_agreement"]),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_training_data": False,
        "uses_solver_labels": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "checkpoint_path": checkpoint_path,
    }
    metrics["passed"] = bool(np.isfinite(improvement) and improvement > 0.0)

    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "native_all_action_target_policy",
                "environment": "poker_ai:full_deck_hu_nlhe",
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "policy_net_state_dict": policy_net.state_dict(),
                "config": {
                    "feature_mode": "flat",
                    "hidden_dim": int(hidden_dim),
                    "initial_chips": int(initial_chips),
                    "max_steps_per_hand": int(max_steps_per_rollout),
                    "fsp_average_policy": False,
                    "train_environment": "poker_ai:full_deck_hu_nlhe",
                    "target_source": "local_all_action_rollout",
                    "target_temperature": float(target_temperature),
                },
                "metrics": metrics,
                "trained_environment_native": True,
                "native_action_projection": False,
                "uses_slumbot_training_data": False,
                "uses_solver_labels": False,
                "uses_alphanlholdem_training_data": False,
            },
            path,
        )
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train-states", type=int, default=64)
    parser.add_argument("--n-eval-states", type=int, default=32)
    parser.add_argument("--rollouts-per-action", type=int, default=16)
    parser.add_argument("--max-steps-per-rollout", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_training_gate(
        n_train_states=args.n_train_states,
        n_eval_states=args.n_eval_states,
        rollouts_per_action=args.rollouts_per_action,
        max_steps_per_rollout=args.max_steps_per_rollout,
        initial_chips=args.initial_chips,
        hidden_dim=args.hidden_dim,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        target_temperature=args.target_temperature,
        seed=args.seed,
        device=args.device,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
