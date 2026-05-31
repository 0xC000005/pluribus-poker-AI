#!/usr/bin/env python3
"""Train a closed-loop policy-update diagnostic from CFR trace states."""

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
from train_cfr_trace_policy_residual import _mean, _rate, _top_match  # noqa: E402
from train_cfr_trace_sequence_model import _records_by_label_iteration  # noqa: E402


class ClosedLoopUpdateOperator(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def _policy(record: dict[str, Any], key: str, action_dim: int) -> np.ndarray:
    value = np.asarray(record.get(key, []), dtype=np.float64).reshape(-1)
    if value.shape[0] != action_dim:
        return np.zeros(action_dim, dtype=np.float64)
    return value


def _legal_mask(record: dict[str, Any], action_dim: int) -> np.ndarray:
    mask = np.zeros(action_dim, dtype=np.float64)
    for action in record.get("legal_actions", ()):
        action = int(action)
        if 0 <= action < action_dim:
            mask[action] = 1.0
    return mask


def _normalize_policy(policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(policy, dtype=np.float64), 0.0) * legal_mask
    total = float(clipped.sum())
    if total <= 1e-12:
        return legal_mask / max(float(legal_mask.sum()), 1.0)
    return clipped / total


def _feature_row(
    record: dict[str, Any],
    *,
    current_policy: np.ndarray,
    action_dim: int,
    iteration: int,
    start_iteration: int,
    rollout_end_iteration: int,
) -> np.ndarray:
    public = np.asarray(record.get("public_belief_features", ()), dtype=np.float64).reshape(-1)
    advantage = np.asarray(record.get("counterfactual_advantage", ()), dtype=np.float64).reshape(-1)
    if advantage.shape[0] != action_dim:
        advantage = np.zeros(action_dim, dtype=np.float64)
    scalars = np.asarray(
        [
            float(iteration - start_iteration) / max(float(rollout_end_iteration - start_iteration), 1.0),
            float(rollout_end_iteration - iteration) / max(float(rollout_end_iteration - start_iteration), 1.0),
            np.log1p(max(float(record.get("regret_mass", 0.0)), 0.0)),
            np.log1p(max(float(record.get("strategy_mass", 0.0)), 0.0)),
            float(record.get("hero_reach_mass", 0.0)),
            float(record.get("villain_reach_mass", 0.0)),
            float(record.get("street", 0.0)) / 3.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            np.asarray(current_policy, dtype=np.float64).reshape(-1),
            _policy(record, "regret_policy", action_dim),
            _legal_mask(record, action_dim),
            advantage,
            scalars,
            public,
        ]
    )


def _common_labels(
    payload: dict[str, Any],
    *,
    start_iteration: int,
    rollout_end_iteration: int,
    uniform_iteration: int,
    reference_iteration: int,
) -> list[str]:
    by_label = _records_by_label_iteration(payload)
    required = set(range(int(start_iteration), int(rollout_end_iteration) + 1))
    required.update({int(uniform_iteration), int(reference_iteration)})
    labels = [label for label, rows in by_label.items() if required.issubset(rows)]
    if not labels:
        raise ValueError("no labels contain the requested trace interval")
    return sorted(labels)


def _build_dataset(
    payload: dict[str, Any],
    labels: list[str],
    *,
    start_iteration: int,
    rollout_end_iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_label = _records_by_label_iteration(payload)
    action_dim = len(by_label[labels[0]][int(start_iteration)]["strategy_policy"])
    features: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for label in labels:
        rows = by_label[label]
        for iteration in range(int(start_iteration), int(rollout_end_iteration)):
            current = rows[iteration]
            current_policy = _policy(current, "strategy_policy", action_dim)
            next_policy = _policy(rows[iteration + 1], "strategy_policy", action_dim)
            legal_mask = _legal_mask(current, action_dim)
            features.append(
                _feature_row(
                    current,
                    current_policy=current_policy,
                    action_dim=action_dim,
                    iteration=iteration,
                    start_iteration=start_iteration,
                    rollout_end_iteration=rollout_end_iteration,
                )
            )
            deltas.append(next_policy - current_policy)
            masks.append(legal_mask)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(deltas, dtype=np.float64),
        np.asarray(masks, dtype=np.float64),
    )


def _standardize(
    values: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = values.mean(axis=0, keepdims=True)
    if std is None:
        std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / std, mean.reshape(-1), std.reshape(-1)


def _train_model(
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
) -> tuple[ClosedLoopUpdateOperator, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device_name = _device_name(device)
    model = ClosedLoopUpdateOperator(
        input_dim=int(x_train.shape[1]),
        action_dim=int(y_delta.shape[1]),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
    ).to(device_name)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device_name)
    y = torch.as_tensor(y_delta, dtype=torch.float32, device=device_name)
    mask = torch.as_tensor(legal_mask, dtype=torch.float32, device=device_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    n = int(x.shape[0])
    batch_size = max(1, min(int(batch_size), n))
    started = time.perf_counter()
    final_loss = 0.0
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=device_name)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x[idx])
            denom = mask[idx].sum().clamp_min(1.0)
            loss = (((pred - y[idx]) * mask[idx]) ** 2).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    return model, {
        "device": device_name,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "train_seconds": round(float(time.perf_counter() - started), 6),
        "final_loss": round(float(final_loss), 10),
    }


def _predict_delta(model: ClosedLoopUpdateOperator, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model(tensor).detach().cpu().numpy().astype(np.float64)


def _rollout_label(
    by_iteration: dict[int, dict[str, Any]],
    *,
    model: ClosedLoopUpdateOperator,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    start_iteration: int,
    rollout_end_iteration: int,
    device: str,
    max_step_l1: float,
) -> np.ndarray:
    action_dim = len(by_iteration[int(start_iteration)]["strategy_policy"])
    policy = _policy(by_iteration[int(start_iteration)], "strategy_policy", action_dim)
    for iteration in range(int(start_iteration), int(rollout_end_iteration)):
        record = by_iteration[iteration]
        legal_mask = _legal_mask(record, action_dim)
        features = _feature_row(
            record,
            current_policy=policy,
            action_dim=action_dim,
            iteration=iteration,
            start_iteration=start_iteration,
            rollout_end_iteration=rollout_end_iteration,
        )
        features, _, _ = _standardize(features.reshape(1, -1), mean=feature_mean, std=feature_std)
        delta = _predict_delta(model, features, device)[0]
        step_l1 = float(np.abs(delta * legal_mask).sum())
        if max_step_l1 > 0.0 and step_l1 > max_step_l1:
            delta = delta * (float(max_step_l1) / max(step_l1, 1e-12))
        policy = _normalize_policy(policy + delta, legal_mask)
    return policy


def _evaluate(
    payload: dict[str, Any],
    labels: list[str],
    *,
    model: ClosedLoopUpdateOperator,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    start_iteration: int,
    rollout_end_iteration: int,
    uniform_iteration: int,
    reference_iteration: int,
    device: str,
    max_step_l1: float,
) -> dict[str, Any]:
    by_label = _records_by_label_iteration(payload)
    closed_l1: list[float] = []
    target_l1: list[float] = []
    low_l1: list[float] = []
    low_to_target_l1: list[float] = []
    uniform_l1: list[float] = []
    closed_match: list[bool] = []
    uniform_match: list[bool] = []
    records: list[dict[str, Any]] = []
    for label in labels:
        rows = by_label[label]
        action_dim = len(rows[int(start_iteration)]["strategy_policy"])
        closed = _rollout_label(
            rows,
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            start_iteration=start_iteration,
            rollout_end_iteration=rollout_end_iteration,
            device=device,
            max_step_l1=max_step_l1,
        )
        low = _policy(rows[int(start_iteration)], "strategy_policy", action_dim)
        target = _policy(rows[int(rollout_end_iteration)], "strategy_policy", action_dim)
        uniform = _policy(rows[int(uniform_iteration)], "strategy_policy", action_dim)
        reference = _policy(rows[int(reference_iteration)], "strategy_policy", action_dim)
        closed_value = float(np.abs(closed - reference).sum())
        target_value = float(np.abs(closed - target).sum())
        low_value = float(np.abs(low - reference).sum())
        uniform_value = float(np.abs(uniform - reference).sum())
        closed_l1.append(closed_value)
        target_l1.append(target_value)
        low_l1.append(low_value)
        low_to_target_l1.append(float(np.abs(low - target).sum()))
        uniform_l1.append(uniform_value)
        closed_match.append(_top_match(closed, reference))
        uniform_match.append(_top_match(uniform, reference))
        records.append(
            {
                "label": label,
                "closed_loop_l1_to_reference": round(closed_value, 8),
                "closed_loop_l1_to_rollout_target": round(target_value, 8),
                "low_l1_to_reference": round(low_value, 8),
                "uniform_l1_to_reference": round(uniform_value, 8),
                "closed_loop_top_matches_reference": bool(closed_match[-1]),
                "uniform_top_matches_reference": bool(uniform_match[-1]),
                "closed_loop_policy": closed.round(8).tolist(),
            }
        )
    mean_closed = _mean(closed_l1)
    mean_low = _mean(low_l1)
    mean_uniform = _mean(uniform_l1)
    return {
        "target_fit_passed": bool(_mean(target_l1) < _mean(low_to_target_l1)),
        "decision_passed": bool(
            mean_closed < mean_low
            and mean_closed <= mean_uniform + 1e-8
            and _rate(closed_match) >= _rate(uniform_match)
        ),
        "mean_closed_loop_l1_to_reference": mean_closed,
        "mean_closed_loop_l1_to_rollout_target": _mean(target_l1),
        "mean_low_l1_to_reference": mean_low,
        "mean_low_l1_to_rollout_target": _mean(low_to_target_l1),
        "mean_uniform_l1_to_reference": mean_uniform,
        "closed_loop_top_match_rate": _rate(closed_match),
        "uniform_top_match_rate": _rate(uniform_match),
        "records": records,
    }


def fit_closed_loop_update_operator_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    start_iteration: int = 5,
    rollout_end_iteration: int = 24,
    uniform_iteration: int = 10,
    reference_iteration: int = 24,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 20260521,
    device: str = "auto",
    trust_region_quantile: float = 0.95,
) -> dict[str, Any]:
    if int(rollout_end_iteration) <= int(start_iteration):
        raise ValueError("rollout_end_iteration must be greater than start_iteration")
    train_labels = _common_labels(
        train_payload,
        start_iteration=start_iteration,
        rollout_end_iteration=rollout_end_iteration,
        uniform_iteration=uniform_iteration,
        reference_iteration=reference_iteration,
    )
    x_train, y_train, mask_train = _build_dataset(
        train_payload,
        train_labels,
        start_iteration=start_iteration,
        rollout_end_iteration=rollout_end_iteration,
    )
    step_l1 = np.abs(y_train * mask_train).sum(axis=1)
    max_step_l1 = float(np.quantile(step_l1, float(trust_region_quantile))) if len(step_l1) else 0.0
    x_train, feature_mean, feature_std = _standardize(x_train)
    model, train_metrics = _train_model(
        x_train,
        y_train,
        mask_train,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    holdout_labels = _common_labels(
        holdout_payload,
        start_iteration=start_iteration,
        rollout_end_iteration=rollout_end_iteration,
        uniform_iteration=uniform_iteration,
        reference_iteration=reference_iteration,
    )
    holdout = _evaluate(
        holdout_payload,
        holdout_labels,
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        start_iteration=start_iteration,
        rollout_end_iteration=rollout_end_iteration,
        uniform_iteration=uniform_iteration,
        reference_iteration=reference_iteration,
        device=train_metrics["device"],
        max_step_l1=max_step_l1,
    )
    return {
        "mode": "cfr_trace_closed_loop_update_operator",
        "promotion": False,
        "passed": bool(holdout["target_fit_passed"] and holdout["decision_passed"]),
        "target_fit_passed": bool(holdout["target_fit_passed"]),
        "decision_passed": bool(holdout["decision_passed"]),
        "start_iteration": int(start_iteration),
        "rollout_end_iteration": int(rollout_end_iteration),
        "uniform_iteration": int(uniform_iteration),
        "reference_iteration": int(reference_iteration),
        "n_train_roots": int(len(train_labels)),
        "n_train_steps": int(x_train.shape[0]),
        "n_holdout_roots": int(len(holdout_labels)),
        "holdout": holdout,
        "model": {
            **train_metrics,
            "trust_region_quantile": float(trust_region_quantile),
            "max_step_l1": round(float(max_step_l1), 8),
            "feature_mean": feature_mean.round(10).tolist(),
            "feature_std": feature_std.round(10).tolist(),
        },
    }


def fit_closed_loop_update_operator(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    start_iteration: int = 5,
    rollout_end_iteration: int = 24,
    uniform_iteration: int = 10,
    reference_iteration: int = 24,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 20260521,
    device: str = "auto",
    trust_region_quantile: float = 0.95,
) -> dict[str, Any]:
    metrics = fit_closed_loop_update_operator_from_payloads(
        _load_payload(train_trace_json),
        _load_payload(holdout_trace_json),
        start_iteration=start_iteration,
        rollout_end_iteration=rollout_end_iteration,
        uniform_iteration=uniform_iteration,
        reference_iteration=reference_iteration,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
        trust_region_quantile=trust_region_quantile,
    )
    metrics["train_trace_json"] = str(train_trace_json)
    metrics["holdout_trace_json"] = str(holdout_trace_json)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-json", required=True)
    parser.add_argument("--holdout-trace-json", required=True)
    parser.add_argument("--start-iteration", type=int, default=5)
    parser.add_argument("--rollout-end-iteration", type=int, default=24)
    parser.add_argument("--uniform-iteration", type=int, default=10)
    parser.add_argument("--reference-iteration", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-region-quantile", type=float, default=0.95)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_closed_loop_update_operator(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        start_iteration=args.start_iteration,
        rollout_end_iteration=args.rollout_end_iteration,
        uniform_iteration=args.uniform_iteration,
        reference_iteration=args.reference_iteration,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        trust_region_quantile=args.trust_region_quantile,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
