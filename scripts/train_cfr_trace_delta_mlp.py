#!/usr/bin/env python3
"""Train a nonlinear diagnostic model for CFR trace update deltas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_cfr_trace_predictor import _load_payload  # noqa: E402
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from train_cfr_trace_policy_residual import (  # noqa: E402
    _feature_row,
    _mean,
    _normalize_prediction,
    _rate,
    _record_map,
    _standardize,
    _top_match,
)


class TraceDeltaMLP(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def _common_labels(*maps: dict[str, dict[str, Any]]) -> list[str]:
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    labels = sorted(common)
    if not labels:
        raise ValueError("no common labels across requested trace iterations")
    return labels


def _advantage_context_row(record: dict[str, Any], action_dim: int) -> np.ndarray:
    """Return bounded per-action counterfactual-advantage context features."""
    raw = np.asarray(record.get("counterfactual_advantage", []), dtype=np.float64).reshape(-1)
    policy = np.asarray(record.get("advantage_policy", []), dtype=np.float64).reshape(-1)
    legal_mask = np.zeros(action_dim, dtype=np.float64)
    for action in record.get("legal_actions", ()):
        action = int(action)
        if 0 <= action < action_dim:
            legal_mask[action] = 1.0
    if raw.shape[0] != action_dim:
        raw = np.zeros(action_dim, dtype=np.float64)
    if policy.shape[0] != action_dim:
        policy = np.zeros(action_dim, dtype=np.float64)
    centered = np.zeros(action_dim, dtype=np.float64)
    legal = legal_mask > 0
    if legal.any():
        legal_raw = raw[legal]
        centered_legal = legal_raw - float(np.mean(legal_raw))
        scale = max(float(np.max(np.abs(centered_legal))), 1e-8)
        centered[legal] = centered_legal / scale
    legal_policy = np.where(legal_mask > 0, policy, 0.0)
    total = float(legal_policy.sum())
    if total > 1e-12:
        legal_policy = legal_policy / total
    max_advantage = float(np.max(raw[legal])) if legal.any() else 0.0
    min_advantage = float(np.min(raw[legal])) if legal.any() else 0.0
    spread = max_advantage - min_advantage
    policy_entropy = 0.0
    positive_policy = legal_policy[legal_policy > 1e-12]
    if positive_policy.size:
        policy_entropy = float(-(positive_policy * np.log(positive_policy)).sum())
    policy_margin = 0.0
    if positive_policy.size >= 2:
        top_two = np.sort(positive_policy)[-2:]
        policy_margin = float(top_two[-1] - top_two[-2])
    elif positive_policy.size == 1:
        policy_margin = float(positive_policy[0])
    return np.concatenate(
        [
            centered,
            legal_policy,
            np.asarray(
                [max_advantage, min_advantage, spread, policy_entropy, policy_margin],
                dtype=np.float64,
            ),
        ]
    )


def _build_dataset(
    low_by_label: dict[str, dict[str, Any]],
    target_by_label: dict[str, dict[str, Any]],
    labels: list[str],
    *,
    include_advantage_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_dim = len(low_by_label[labels[0]]["strategy_policy"])
    features: list[np.ndarray] = []
    low_policies: list[np.ndarray] = []
    target_deltas: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    for label in labels:
        low_record = low_by_label[label]
        low_policy = np.asarray(low_record["strategy_policy"], dtype=np.float64)
        target_policy = np.asarray(target_by_label[label]["strategy_policy"], dtype=np.float64)
        legal_mask = np.zeros(action_dim, dtype=np.float64)
        for action in low_record.get("legal_actions", ()):
            action = int(action)
            if 0 <= action < action_dim:
                legal_mask[action] = 1.0
        if float(legal_mask.sum()) <= 0.0:
            raise ValueError(f"record {label} has no legal actions")
        context = np.asarray(low_record.get("public_belief_features", ()), dtype=np.float64).reshape(-1)
        parts = [_feature_row(low_record, action_dim), context]
        if include_advantage_features:
            parts.append(_advantage_context_row(low_record, action_dim))
        features.append(np.concatenate(parts))
        low_policies.append(low_policy)
        target_deltas.append(target_policy - low_policy)
        legal_masks.append(legal_mask)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(low_policies, dtype=np.float64),
        np.asarray(target_deltas, dtype=np.float64),
        np.asarray(legal_masks, dtype=np.float64),
    )


def _train_mlp(
    x_train: np.ndarray,
    y_delta: np.ndarray,
    legal_mask: np.ndarray,
    *,
    hidden_dim: int,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[TraceDeltaMLP, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device_name = _device_name(device)
    model = TraceDeltaMLP(
        input_dim=x_train.shape[1],
        action_dim=y_delta.shape[1],
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
    ).to(device_name)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device_name)
    y = torch.as_tensor(y_delta, dtype=torch.float32, device=device_name)
    mask = torch.as_tensor(legal_mask, dtype=torch.float32, device=device_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    n = int(x.shape[0])
    batch_size = max(1, min(int(batch_size), n))
    started = time.perf_counter()
    final_loss = 0.0
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=device_name)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x[idx])
            batch_mask = mask[idx]
            denom = batch_mask.sum().clamp_min(1.0)
            loss = (((pred - y[idx]) * batch_mask) ** 2).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    seconds = time.perf_counter() - started
    return model, {
        "device": device_name,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "train_seconds": round(float(seconds), 6),
        "final_loss": round(float(final_loss), 10),
    }


def _predict_delta(model: TraceDeltaMLP, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        pred = model(tensor).detach().cpu().numpy().astype(np.float64)
    return pred


def _evaluate(
    labels: list[str],
    low_by_label: dict[str, dict[str, Any]],
    target_by_label: dict[str, dict[str, Any]],
    uniform_by_label: dict[str, dict[str, Any]],
    reference_by_label: dict[str, dict[str, Any]],
    pred_delta: np.ndarray,
) -> dict[str, Any]:
    pred_l1: list[float] = []
    pred_to_target_l1: list[float] = []
    low_l1: list[float] = []
    low_to_target_l1: list[float] = []
    uniform_l1: list[float] = []
    pred_match: list[bool] = []
    low_match: list[bool] = []
    uniform_match: list[bool] = []
    records: list[dict[str, Any]] = []
    for row_idx, label in enumerate(labels):
        low_record = low_by_label[label]
        low_policy = np.asarray(low_record["strategy_policy"], dtype=np.float64)
        target = np.asarray(target_by_label[label]["strategy_policy"], dtype=np.float64)
        reference = np.asarray(reference_by_label[label]["strategy_policy"], dtype=np.float64)
        uniform_policy = np.asarray(uniform_by_label[label]["strategy_policy"], dtype=np.float64)
        pred_policy = _normalize_prediction(
            low_policy + pred_delta[row_idx],
            [int(action) for action in low_record.get("legal_actions", ())],
            fallback=low_policy,
        )
        pred_value = float(np.abs(pred_policy - reference).sum())
        pred_to_target_value = float(np.abs(pred_policy - target).sum())
        low_value = float(np.abs(low_policy - reference).sum())
        low_to_target_value = float(np.abs(low_policy - target).sum())
        uniform_value = float(np.abs(uniform_policy - reference).sum())
        pred_l1.append(pred_value)
        pred_to_target_l1.append(pred_to_target_value)
        low_l1.append(low_value)
        low_to_target_l1.append(low_to_target_value)
        uniform_l1.append(uniform_value)
        pred_match.append(_top_match(pred_policy, reference))
        low_match.append(_top_match(low_policy, reference))
        uniform_match.append(_top_match(uniform_policy, reference))
        records.append(
            {
                "label": label,
                "pred_l1_to_reference": round(pred_value, 8),
                "pred_l1_to_target": round(pred_to_target_value, 8),
                "low_l1_to_reference": round(low_value, 8),
                "low_l1_to_target": round(low_to_target_value, 8),
                "uniform_l1_to_reference": round(uniform_value, 8),
                "pred_top_matches_reference": bool(pred_match[-1]),
                "low_top_matches_reference": bool(low_match[-1]),
                "uniform_top_matches_reference": bool(uniform_match[-1]),
                "pred_policy": pred_policy.round(8).tolist(),
            }
        )
    mean_pred = _mean(pred_l1)
    mean_pred_to_target = _mean(pred_to_target_l1)
    mean_low = _mean(low_l1)
    mean_low_to_target = _mean(low_to_target_l1)
    mean_uniform = _mean(uniform_l1)
    target_fit_passed = mean_pred_to_target < mean_low_to_target
    decision_passed = mean_pred < mean_low and mean_pred < mean_uniform
    return {
        "target_fit_passed": bool(target_fit_passed),
        "decision_passed": bool(decision_passed),
        "passed": bool(target_fit_passed and decision_passed),
        "mean_pred_l1_to_reference": mean_pred,
        "mean_pred_l1_to_target": mean_pred_to_target,
        "mean_low_l1_to_reference": mean_low,
        "mean_low_l1_to_target": mean_low_to_target,
        "mean_uniform_l1_to_reference": mean_uniform,
        "pred_top_match_rate": _rate(pred_match),
        "low_top_match_rate": _rate(low_match),
        "uniform_top_match_rate": _rate(uniform_match),
        "records": records,
    }


def fit_trace_delta_mlp_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    low_trace_iteration: int = 5,
    target_trace_iteration: int = 10,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 20260661,
    device: str = "auto",
    include_advantage_features: bool = False,
) -> dict[str, Any]:
    train_low = _record_map(train_payload, low_trace_iteration)
    train_target = _record_map(train_payload, target_trace_iteration)
    train_labels = _common_labels(train_low, train_target)
    x_train, _low_train, y_train_delta, train_mask = _build_dataset(
        train_low,
        train_target,
        train_labels,
        include_advantage_features=include_advantage_features,
    )
    x_train, feature_mean, feature_std = _standardize(x_train)

    model, train_metrics = _train_mlp(
        x_train,
        y_train_delta,
        train_mask,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    holdout_low = _record_map(holdout_payload, low_trace_iteration)
    holdout_target = _record_map(holdout_payload, target_trace_iteration)
    holdout_uniform = _record_map(holdout_payload, uniform_trace_iteration)
    holdout_reference = _record_map(holdout_payload, reference_trace_iteration)
    holdout_labels = _common_labels(
        holdout_low,
        holdout_target,
        holdout_uniform,
        holdout_reference,
    )
    x_holdout, _low_holdout, _holdout_delta, _holdout_mask = _build_dataset(
        holdout_low,
        holdout_target,
        holdout_labels,
        include_advantage_features=include_advantage_features,
    )
    x_holdout, _, _ = _standardize(x_holdout, mean=feature_mean, std=feature_std)
    pred_delta = _predict_delta(model, x_holdout, train_metrics["device"])
    metrics = _evaluate(
        holdout_labels,
        holdout_low,
        holdout_target,
        holdout_uniform,
        holdout_reference,
        pred_delta,
    )
    metrics.update(
        {
            "mode": "cfr_trace_delta_mlp",
            "promotion": False,
            "low_trace_iteration": int(low_trace_iteration),
            "target_trace_iteration": int(target_trace_iteration),
            "uniform_trace_iteration": int(uniform_trace_iteration),
            "reference_trace_iteration": int(reference_trace_iteration),
            "n_train": int(len(train_labels)),
            "n_holdout": int(len(holdout_labels)),
            "include_advantage_features": bool(include_advantage_features),
            "model": {
                **train_metrics,
                "feature_mean": feature_mean.round(10).tolist(),
                "feature_std": feature_std.round(10).tolist(),
            },
        }
    )
    return metrics


def fit_trace_delta_mlp(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    low_trace_iteration: int = 5,
    target_trace_iteration: int = 10,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 20260661,
    device: str = "auto",
    include_advantage_features: bool = False,
) -> dict[str, Any]:
    metrics = fit_trace_delta_mlp_from_payloads(
        _load_payload(train_trace_json),
        _load_payload(holdout_trace_json),
        low_trace_iteration=low_trace_iteration,
        target_trace_iteration=target_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
        include_advantage_features=include_advantage_features,
    )
    metrics["train_trace_json"] = str(train_trace_json)
    metrics["holdout_trace_json"] = str(holdout_trace_json)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-json", required=True)
    parser.add_argument("--holdout-trace-json", required=True)
    parser.add_argument("--low-trace-iteration", type=int, default=5)
    parser.add_argument("--target-trace-iteration", type=int, default=10)
    parser.add_argument("--uniform-trace-iteration", type=int, default=10)
    parser.add_argument("--reference-trace-iteration", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260661)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--include-advantage-features",
        action="store_true",
        help="Append normalized counterfactual-advantage and advantage-policy trace fields.",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_trace_delta_mlp(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        low_trace_iteration=args.low_trace_iteration,
        target_trace_iteration=args.target_trace_iteration,
        uniform_trace_iteration=args.uniform_trace_iteration,
        reference_trace_iteration=args.reference_trace_iteration,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        include_advantage_features=args.include_advantage_features,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
