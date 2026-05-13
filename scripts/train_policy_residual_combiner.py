#!/usr/bin/env python3
"""Train a small learned combiner for low-budget solver residual correction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.belief_value_probe import save_metrics
from poker_ai.research.evaluation import load_value_network_checkpoint
from poker_ai.research.resolver_benchmark import (
    _policy_decision,
    _solver_decision,
    load_cases_json,
)
from play_slumbot import parse_action


COMBINER_INPUT_DIM = N_FEATURES + 3 * N_ACTIONS


@dataclass(frozen=True)
class CombinerDataset:
    features: np.ndarray
    legal_masks: np.ndarray
    target_probs: np.ndarray
    low_strategies: np.ndarray
    policy_strategies: np.ndarray
    labels: list[str]


class PolicyResidualCombiner(nn.Module):
    """Predict a high-budget resolver policy from public features and priors."""

    def __init__(self, input_dim: int = COMBINER_INPUT_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_combiner_features(
    features: np.ndarray,
    legal_masks: np.ndarray,
    low_strategies: np.ndarray,
    policy_strategies: np.ndarray,
) -> np.ndarray:
    parts = [
        np.asarray(features, dtype=np.float32),
        np.asarray(legal_masks, dtype=np.float32),
        np.asarray(low_strategies, dtype=np.float32),
        np.asarray(policy_strategies, dtype=np.float32),
    ]
    n_rows = parts[0].shape[0]
    expected = [(n_rows, N_FEATURES), (n_rows, N_ACTIONS), (n_rows, N_ACTIONS), (n_rows, N_ACTIONS)]
    for idx, (part, shape) in enumerate(zip(parts, expected, strict=True)):
        if part.shape != shape:
            raise ValueError(f"combiner feature part {idx} shape {part.shape} != {shape}")
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def masked_softmax_np(logits: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    legal = np.asarray(legal_masks, dtype=np.float64) > 0
    masked = np.where(legal, logits, -1e9)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    probs = np.exp(shifted) * legal
    totals = probs.sum(axis=1, keepdims=True)
    uniform = legal / np.maximum(legal.sum(axis=1, keepdims=True), 1)
    return np.where(totals > 0, probs / totals, uniform).astype(np.float32)


def policy_metrics(pred: np.ndarray, target: np.ndarray, legal_masks: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64) * (legal_masks > 0)
    target = np.asarray(target, dtype=np.float64) * (legal_masks > 0)
    pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-12)
    target = target / np.maximum(target.sum(axis=1, keepdims=True), 1e-12)
    l1 = np.abs(pred - target).sum(axis=1)
    kl = (target * (np.log(np.clip(target, 1e-12, 1.0)) - np.log(np.clip(pred, 1e-12, 1.0)))).sum(axis=1)
    pred_top = np.argmax(pred, axis=1)
    target_top = np.argmax(target, axis=1)
    return {
        "mean_l1": round(float(l1.mean()), 8),
        "mean_kl": round(float(kl.mean()), 8),
        "top_action_agreement": round(float(np.mean(pred_top == target_top)), 8),
        "top_allin_rate": round(float(np.mean(pred_top == 8)), 8),
        "mean_allin_prob": round(float(pred[:, 8].mean()), 8),
    }


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_combiner_dataset(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    targets_npz: str | Path,
    device: str | torch.device = "auto",
    low_iterations: int = 5,
    solver_backend: str = "cpu",
) -> CombinerDataset:
    resolved_device = _resolve_device(str(device)) if not isinstance(device, torch.device) else device
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    cases = load_cases_json(cases_json)
    targets = PolicyTargetBuffer.from_npz(targets_npz)
    if targets.size != len(cases):
        raise ValueError(f"targets size {targets.size} != cases size {len(cases)}")

    features = []
    legal_masks = []
    target_probs = []
    low_strategies = []
    policy_strategies = []
    labels = []
    for idx, case in enumerate(cases):
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            continue
        low = _solver_decision(
            case,
            parsed,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
        )
        policy = _policy_decision(
            loaded.value_net,
            resolved_device,
            case,
            parsed,
            strategy_source="policy_head",
        )
        if low is None:
            continue
        features.append(targets.features[idx])
        legal_masks.append(targets.legal_masks[idx])
        target_probs.append(targets.target_probs[idx])
        low_strategies.append(low.strategy.astype(np.float32))
        policy_strategies.append(policy.strategy.astype(np.float32))
        labels.append(case.label)

    if not features:
        raise RuntimeError("no combiner training rows were generated")
    return CombinerDataset(
        features=np.asarray(features, dtype=np.float32).reshape(-1, N_FEATURES),
        legal_masks=np.asarray(legal_masks, dtype=np.float32).reshape(-1, N_ACTIONS),
        target_probs=np.asarray(target_probs, dtype=np.float32).reshape(-1, N_ACTIONS),
        low_strategies=np.asarray(low_strategies, dtype=np.float32).reshape(-1, N_ACTIONS),
        policy_strategies=np.asarray(policy_strategies, dtype=np.float32).reshape(-1, N_ACTIONS),
        labels=labels,
    )


def _masked_cross_entropy(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(legal_masks <= 0, -1e4)
    return -(target_probs * torch.log_softmax(masked_logits, dim=1)).sum(dim=1).mean()


def _predict_combiner(
    model: PolicyResidualCombiner,
    dataset: CombinerDataset,
    device: torch.device,
) -> np.ndarray:
    x = make_combiner_features(
        dataset.features,
        dataset.legal_masks,
        dataset.low_strategies,
        dataset.policy_strategies,
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    return masked_softmax_np(logits, dataset.legal_masks)


def train_policy_residual_combiner(
    *,
    checkpoint: str | Path,
    train_cases_json: str | Path,
    train_targets_npz: str | Path,
    holdout_cases_json: str | Path,
    holdout_targets_npz: str | Path,
    device: str = "auto",
    low_iterations: int = 5,
    solver_backend: str = "cpu",
    hidden_dim: int = 64,
    n_steps: int = 800,
    lr: float = 1e-3,
    max_combiner_allin_rate: float = 0.05,
    output: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train = build_combiner_dataset(
        checkpoint=checkpoint,
        cases_json=train_cases_json,
        targets_npz=train_targets_npz,
        device=resolved_device,
        low_iterations=low_iterations,
        solver_backend=solver_backend,
    )
    holdout = build_combiner_dataset(
        checkpoint=checkpoint,
        cases_json=holdout_cases_json,
        targets_npz=holdout_targets_npz,
        device=resolved_device,
        low_iterations=low_iterations,
        solver_backend=solver_backend,
    )

    model = PolicyResidualCombiner(hidden_dim=int(hidden_dim)).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_train = torch.from_numpy(
        make_combiner_features(
            train.features,
            train.legal_masks,
            train.low_strategies,
            train.policy_strategies,
        )
    ).to(resolved_device)
    mask_train = torch.from_numpy(train.legal_masks).to(resolved_device)
    target_train = torch.from_numpy(train.target_probs).to(resolved_device)

    losses: list[float] = []
    model.train()
    for _ in range(int(n_steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = _masked_cross_entropy(logits, mask_train, target_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    train_pred = _predict_combiner(model, train, resolved_device)
    holdout_pred = _predict_combiner(model, holdout, resolved_device)
    train_metrics = policy_metrics(train_pred, train.target_probs, train.legal_masks)
    holdout_metrics = policy_metrics(holdout_pred, holdout.target_probs, holdout.legal_masks)
    low_metrics = policy_metrics(holdout.low_strategies, holdout.target_probs, holdout.legal_masks)
    policy_head_metrics = policy_metrics(
        holdout.policy_strategies,
        holdout.target_probs,
        holdout.legal_masks,
    )
    passed = (
        holdout_metrics["mean_l1"] < min(low_metrics["mean_l1"], policy_head_metrics["mean_l1"])
        and holdout_metrics["top_allin_rate"] <= float(max_combiner_allin_rate)
    )

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": COMBINER_INPUT_DIM,
                "hidden_dim": int(hidden_dim),
                "n_actions": N_ACTIONS,
                "checkpoint": str(checkpoint),
                "low_iterations": int(low_iterations),
            },
            output_path,
        )

    return {
        "mode": "policy_residual_combiner",
        "passed": bool(passed),
        "checkpoint": str(checkpoint),
        "device": str(resolved_device),
        "solver_backend": solver_backend,
        "low_iterations": int(low_iterations),
        "hidden_dim": int(hidden_dim),
        "n_steps": int(n_steps),
        "lr": float(lr),
        "n_train": int(len(train.labels)),
        "n_holdout": int(len(holdout.labels)),
        "train_loss_initial": round(float(losses[0]), 8) if losses else 0.0,
        "train_loss_final": round(float(losses[-1]), 8) if losses else 0.0,
        "train_combiner": train_metrics,
        "holdout_low_solver": low_metrics,
        "holdout_policy_head": policy_head_metrics,
        "holdout_combiner": holdout_metrics,
        "max_combiner_allin_rate": float(max_combiner_allin_rate),
        "output": str(output) if output is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a learned residual combiner for low-budget solver policy correction."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cases-json", required=True)
    parser.add_argument("--train-targets-npz", required=True)
    parser.add_argument("--holdout-cases-json", required=True)
    parser.add_argument("--holdout-targets-npz", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-combiner-allin-rate", type=float, default=0.05)
    parser.add_argument("--output")
    parser.add_argument("--metrics-output")
    args = parser.parse_args(argv)
    metrics = train_policy_residual_combiner(
        checkpoint=args.checkpoint,
        train_cases_json=args.train_cases_json,
        train_targets_npz=args.train_targets_npz,
        holdout_cases_json=args.holdout_cases_json,
        holdout_targets_npz=args.holdout_targets_npz,
        device=args.device,
        low_iterations=args.low_iterations,
        solver_backend=args.solver_backend,
        hidden_dim=args.hidden_dim,
        n_steps=args.n_steps,
        lr=args.lr,
        max_combiner_allin_rate=args.max_combiner_allin_rate,
        output=args.output,
    )
    if args.metrics_output:
        save_metrics(metrics, args.metrics_output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
