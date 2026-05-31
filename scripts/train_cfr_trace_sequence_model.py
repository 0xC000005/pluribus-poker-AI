#!/usr/bin/env python3
"""Train a recurrent diagnostic model from CFR trace trajectories."""

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
from train_cfr_trace_policy_residual import _mean, _normalize_prediction, _rate, _top_match  # noqa: E402


class TraceSequenceModel(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _sequence, hidden = self.gru(x)
        return self.head(hidden[-1])


def _device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def _records_by_label_iteration(payload: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    by_label: dict[str, dict[int, dict[str, Any]]] = {}
    for record in payload.get("records", []):
        if "label" not in record or "iteration" not in record:
            continue
        by_label.setdefault(str(record["label"]), {})[int(record["iteration"])] = dict(record)
    return by_label


def _common_labels(
    payload: dict[str, Any],
    *,
    low_trace_iteration: int,
    uniform_trace_iteration: int,
    target_trace_iteration: int,
    reference_trace_iteration: int,
) -> list[str]:
    by_label = _records_by_label_iteration(payload)
    required = set(range(0, int(low_trace_iteration) + 1))
    required.update(
        {
            int(uniform_trace_iteration),
            int(target_trace_iteration),
            int(reference_trace_iteration),
        }
    )
    labels = [label for label, rows in by_label.items() if required.issubset(rows)]
    if not labels:
        raise ValueError("no labels contain all requested trace iterations")
    return sorted(labels)


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


def _step_features(record: dict[str, Any], *, action_dim: int, low_trace_iteration: int) -> np.ndarray:
    iteration = int(record.get("iteration", 0))
    public = np.asarray(record.get("public_belief_features", ()), dtype=np.float64).reshape(-1)
    scalars = np.asarray(
        [
            float(iteration) / max(float(low_trace_iteration), 1.0),
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
            _policy(record, "strategy_policy", action_dim),
            _policy(record, "regret_policy", action_dim),
            _legal_mask(record, action_dim),
            scalars,
            public,
        ]
    )


def _l1(policy_a: np.ndarray, policy_b: np.ndarray) -> float:
    return float(np.abs(policy_a - policy_b).sum())


def _build_dataset(
    payload: dict[str, Any],
    *,
    low_trace_iteration: int,
    uniform_trace_iteration: int,
    target_trace_iteration: int,
    reference_trace_iteration: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    labels = _common_labels(
        payload,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        target_trace_iteration=target_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
    )
    by_label = _records_by_label_iteration(payload)
    action_dim = len(by_label[labels[0]][int(low_trace_iteration)]["strategy_policy"])
    sequences: list[np.ndarray] = []
    target_deltas: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    records: dict[str, dict[str, Any]] = {}
    for label in labels:
        rows = by_label[label]
        sequence = [
            _step_features(rows[iteration], action_dim=action_dim, low_trace_iteration=low_trace_iteration)
            for iteration in range(0, int(low_trace_iteration) + 1)
        ]
        low = rows[int(low_trace_iteration)]
        target = rows[int(target_trace_iteration)]
        uniform = rows[int(uniform_trace_iteration)]
        reference = rows[int(reference_trace_iteration)]
        low_policy = _policy(low, "strategy_policy", action_dim)
        target_policy = _policy(target, "strategy_policy", action_dim)
        mask = _legal_mask(low, action_dim)
        if float(mask.sum()) <= 0.0:
            raise ValueError(f"record {label} has no legal actions")
        sequences.append(np.asarray(sequence, dtype=np.float64))
        target_deltas.append(target_policy - low_policy)
        legal_masks.append(mask)
        records[label] = {
            "low": low,
            "target": target,
            "uniform": uniform,
            "reference": reference,
        }
    return (
        labels,
        np.asarray(sequences, dtype=np.float64),
        np.asarray(target_deltas, dtype=np.float64),
        np.asarray(legal_masks, dtype=np.float64),
        records,
    )


def _standardize_sequence(
    values: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = values.reshape(-1, values.shape[-1]).mean(axis=0, keepdims=True)
    if std is None:
        std = values.reshape(-1, values.shape[-1]).std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return normalized, mean.reshape(-1), std.reshape(-1)


def _train_model(
    x_train: np.ndarray,
    y_delta: np.ndarray,
    legal_mask: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[TraceSequenceModel, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device_name = _device_name(device)
    model = TraceSequenceModel(
        input_dim=int(x_train.shape[-1]),
        action_dim=int(y_delta.shape[-1]),
        hidden_dim=int(hidden_dim),
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
            batch_mask = mask[idx]
            denom = batch_mask.sum().clamp_min(1.0)
            loss = (((pred - y[idx]) * batch_mask) ** 2).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    return model, {
        "device": device_name,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "train_seconds": round(float(time.perf_counter() - started), 6),
        "final_loss": round(float(final_loss), 10),
    }


def _predict_delta(model: TraceSequenceModel, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model(tensor).detach().cpu().numpy().astype(np.float64)


def _evaluate(
    labels: list[str],
    records: dict[str, dict[str, Any]],
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
    rows: list[dict[str, Any]] = []
    for row_idx, label in enumerate(labels):
        row = records[label]
        action_dim = len(row["low"]["strategy_policy"])
        low_policy = _policy(row["low"], "strategy_policy", action_dim)
        target_policy = _policy(row["target"], "strategy_policy", action_dim)
        uniform_policy = _policy(row["uniform"], "strategy_policy", action_dim)
        reference_policy = _policy(row["reference"], "strategy_policy", action_dim)
        legal_actions = [int(action) for action in row["low"].get("legal_actions", ())]
        pred_policy = _normalize_prediction(
            low_policy + pred_delta[row_idx],
            legal_actions,
            fallback=low_policy,
        )
        pred_value = _l1(pred_policy, reference_policy)
        pred_target_value = _l1(pred_policy, target_policy)
        low_value = _l1(low_policy, reference_policy)
        low_target_value = _l1(low_policy, target_policy)
        uniform_value = _l1(uniform_policy, reference_policy)
        pred_l1.append(pred_value)
        pred_to_target_l1.append(pred_target_value)
        low_l1.append(low_value)
        low_to_target_l1.append(low_target_value)
        uniform_l1.append(uniform_value)
        pred_match.append(_top_match(pred_policy, reference_policy))
        low_match.append(_top_match(low_policy, reference_policy))
        uniform_match.append(_top_match(uniform_policy, reference_policy))
        rows.append(
            {
                "label": label,
                "pred_l1_to_reference": round(float(pred_value), 8),
                "pred_l1_to_target": round(float(pred_target_value), 8),
                "low_l1_to_reference": round(float(low_value), 8),
                "low_l1_to_target": round(float(low_target_value), 8),
                "uniform_l1_to_reference": round(float(uniform_value), 8),
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
    pred_top = _rate(pred_match)
    uniform_top = _rate(uniform_match)
    target_fit_passed = mean_pred_to_target < mean_low_to_target
    decision_passed = mean_pred < mean_low and mean_pred < mean_uniform and pred_top >= uniform_top
    return {
        "target_fit_passed": bool(target_fit_passed),
        "decision_passed": bool(decision_passed),
        "passed": bool(target_fit_passed and decision_passed),
        "mean_pred_l1_to_reference": mean_pred,
        "mean_pred_l1_to_target": mean_pred_to_target,
        "mean_low_l1_to_reference": mean_low,
        "mean_low_l1_to_target": mean_low_to_target,
        "mean_uniform_l1_to_reference": mean_uniform,
        "pred_top_match_rate": pred_top,
        "low_top_match_rate": _rate(low_match),
        "uniform_top_match_rate": uniform_top,
        "records": rows,
    }


def fit_trace_sequence_model_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    target_trace_iteration: int = 24,
    reference_trace_iteration: int = 24,
    hidden_dim: int = 64,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 20260521,
    device: str = "auto",
) -> dict[str, Any]:
    train_labels, x_train, y_train_delta, train_mask, _train_records = _build_dataset(
        train_payload,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        target_trace_iteration=target_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
    )
    holdout_labels, x_holdout, _holdout_delta, _holdout_mask, holdout_records = _build_dataset(
        holdout_payload,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        target_trace_iteration=target_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
    )
    x_train, feature_mean, feature_std = _standardize_sequence(x_train)
    x_holdout, _, _ = _standardize_sequence(x_holdout, mean=feature_mean, std=feature_std)
    model, train_metrics = _train_model(
        x_train,
        y_train_delta,
        train_mask,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    pred_delta = _predict_delta(model, x_holdout, train_metrics["device"])
    metrics = _evaluate(holdout_labels, holdout_records, pred_delta)
    metrics.update(
        {
            "mode": "cfr_trace_sequence_model",
            "promotion": False,
            "low_trace_iteration": int(low_trace_iteration),
            "uniform_trace_iteration": int(uniform_trace_iteration),
            "target_trace_iteration": int(target_trace_iteration),
            "reference_trace_iteration": int(reference_trace_iteration),
            "n_train": int(len(train_labels)),
            "n_holdout": int(len(holdout_labels)),
            "sequence_length": int(low_trace_iteration) + 1,
            "input_dim": int(x_train.shape[-1]),
            "model": {
                **train_metrics,
                "feature_mean": feature_mean.round(10).tolist(),
                "feature_std": feature_std.round(10).tolist(),
            },
        }
    )
    return metrics


def fit_trace_sequence_model(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    target_trace_iteration: int = 24,
    reference_trace_iteration: int = 24,
    hidden_dim: int = 64,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 20260521,
    device: str = "auto",
) -> dict[str, Any]:
    metrics = fit_trace_sequence_model_from_payloads(
        _load_payload(train_trace_json),
        _load_payload(holdout_trace_json),
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        target_trace_iteration=target_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    metrics["train_trace_json"] = str(train_trace_json)
    metrics["holdout_trace_json"] = str(holdout_trace_json)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-json", required=True)
    parser.add_argument("--holdout-trace-json", required=True)
    parser.add_argument("--low-trace-iteration", type=int, default=5)
    parser.add_argument("--uniform-trace-iteration", type=int, default=10)
    parser.add_argument("--target-trace-iteration", type=int, default=24)
    parser.add_argument("--reference-trace-iteration", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_trace_sequence_model(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        low_trace_iteration=args.low_trace_iteration,
        uniform_trace_iteration=args.uniform_trace_iteration,
        target_trace_iteration=args.target_trace_iteration,
        reference_trace_iteration=args.reference_trace_iteration,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
