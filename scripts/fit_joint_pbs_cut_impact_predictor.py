#!/usr/bin/env python3
"""Fit a structural predictor for successor-cut root policy impact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from poker_ai.research.belief_value_probe import save_metrics


NUMERIC_FIELDS = (
    "actor_to_act",
    "bet_count",
    "client_pos",
    "cut_pos",
    "hero_first",
    "hero_stack",
    "legal_action_count",
    "pot",
    "root_bet_count",
    "villain_stack",
)
CATEGORICAL_FIELDS = ("action_shape", "root_action_shape")
_RIDGE = 1e-3


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for record in payload.get("records", []):
        if "action_l1_drift" not in record:
            continue
        rows.append(dict(record))
    if not rows:
        raise ValueError(f"no cut-impact records with action_l1_drift in {path}")
    return rows


def _numeric_value(row: dict[str, Any], field: str) -> float:
    value = row.get(field, 0.0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value or 0.0)


def _category_vocab(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        field: sorted({str(row.get(field, "missing")) for row in rows})
        for field in CATEGORICAL_FIELDS
    }


def _design_matrix(
    rows: list[dict[str, Any]],
    *,
    vocab: dict[str, list[str]],
    numeric_mean: np.ndarray | None = None,
    numeric_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric = np.asarray(
        [[_numeric_value(row, field) for field in NUMERIC_FIELDS] for row in rows],
        dtype=np.float64,
    )
    if numeric_mean is None:
        numeric_mean = numeric.mean(axis=0)
    if numeric_std is None:
        numeric_std = numeric.std(axis=0)
    numeric_std = np.where(numeric_std > 1e-8, numeric_std, 1.0)
    parts = [
        np.ones((len(rows), 1), dtype=np.float64),
        (numeric - numeric_mean) / numeric_std,
    ]
    for field in CATEGORICAL_FIELDS:
        values = vocab[field]
        encoded = np.zeros((len(rows), len(values)), dtype=np.float64)
        index = {value: idx for idx, value in enumerate(values)}
        for row_idx, row in enumerate(rows):
            value_idx = index.get(str(row.get(field, "missing")))
            if value_idx is not None:
                encoded[row_idx, value_idx] = 1.0
        parts.append(encoded)
    return np.concatenate(parts, axis=1), numeric_mean, numeric_std


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * _RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _top_quantile_recall(pred: np.ndarray, target: np.ndarray, *, quantile: float = 0.8) -> float:
    pred_cut = float(np.quantile(pred, quantile))
    target_cut = float(np.quantile(target, quantile))
    target_high = target >= target_cut
    if not bool(target_high.any()):
        return 0.0
    return float(np.mean(pred[target_high] >= pred_cut))


def _split_metrics(pred_log: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.maximum(np.exp(pred_log) - 1e-6, 0.0)
    return {
        "actual_mean": round(float(np.mean(target)), 8),
        "predicted_mean": round(float(np.mean(pred)), 8),
        "pearson": round(_pearson(pred, target), 8),
        "top_quintile_recall": round(_top_quantile_recall(pred, target), 8),
    }


def fit_cut_impact_predictor_from_rows(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    vocab = _category_vocab(train_rows)
    x_train, numeric_mean, numeric_std = _design_matrix(train_rows, vocab=vocab)
    x_holdout, _, _ = _design_matrix(
        holdout_rows,
        vocab=vocab,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
    )
    y_train = np.asarray([float(row["action_l1_drift"]) for row in train_rows], dtype=np.float64)
    y_holdout = np.asarray([float(row["action_l1_drift"]) for row in holdout_rows], dtype=np.float64)
    weights = _fit_ridge(x_train, np.log(y_train + 1e-6))
    train_pred = x_train @ weights
    holdout_pred = x_holdout @ weights
    train_pred_drift = np.maximum(np.exp(train_pred) - 1e-6, 0.0)
    holdout_pred_drift = np.maximum(np.exp(holdout_pred) - 1e-6, 0.0)
    threshold = float(np.quantile(train_pred_drift, 0.5))
    selected = holdout_pred_drift <= threshold
    rejected = ~selected
    selected_mean = float(np.mean(y_holdout[selected])) if bool(selected.any()) else None
    rejected_mean = float(np.mean(y_holdout[rejected])) if bool(rejected.any()) else None
    holdout_mean = float(np.mean(y_holdout))
    passed = (
        selected_mean is not None
        and rejected_mean is not None
        and selected_mean < holdout_mean
        and selected_mean < rejected_mean
        and _pearson(holdout_pred_drift, y_holdout) > 0.0
    )
    return {
        "mode": "joint_pbs_cut_impact_predictor",
        "passed": bool(passed),
        "promotion": False,
        "n_train": int(len(train_rows)),
        "n_holdout": int(len(holdout_rows)),
        "numeric_fields": list(NUMERIC_FIELDS),
        "categorical_fields": list(CATEGORICAL_FIELDS),
        "train": _split_metrics(train_pred, y_train),
        "holdout": _split_metrics(holdout_pred, y_holdout),
        "abstention": {
            "rule": "use_learned_value_when_predicted_action_l1_at_or_below_train_median",
            "predicted_action_l1_cut": round(threshold, 8),
            "selected_count": int(np.sum(selected)),
            "rejected_count": int(np.sum(rejected)),
            "selected_mean_action_l1": (
                round(selected_mean, 8) if selected_mean is not None else None
            ),
            "rejected_mean_action_l1": (
                round(rejected_mean, 8) if rejected_mean is not None else None
            ),
            "holdout_mean_action_l1": round(holdout_mean, 8),
        },
        "model": {
            "ridge": float(_RIDGE),
            "numeric_fields": list(NUMERIC_FIELDS),
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "category_vocab": vocab,
            "numeric_mean": numeric_mean.round(10).tolist(),
            "numeric_std": numeric_std.round(10).tolist(),
            "weights": weights.round(10).tolist(),
            "abstention_predicted_action_l1_cut": round(threshold, 8),
        },
        "worst_predicted_holdout": [
            {
                "label": str(holdout_rows[index].get("label", "")),
                "action_shape": str(holdout_rows[index].get("action_shape", "")),
                "actual_action_l1": round(float(y_holdout[index]), 8),
                "predicted_action_l1": round(float(holdout_pred_drift[index]), 8),
            }
            for index in np.argsort(holdout_pred_drift)[-10:][::-1]
        ],
    }


def fit_cut_impact_predictor(
    *,
    train_impact_json: str | Path,
    holdout_impact_json: str | Path,
) -> dict[str, Any]:
    train_rows = _load_records(train_impact_json)
    holdout_rows = _load_records(holdout_impact_json)
    metrics = fit_cut_impact_predictor_from_rows(train_rows, holdout_rows)
    metrics.update(
        {
            "train_impact_json": str(train_impact_json),
            "holdout_impact_json": str(holdout_impact_json),
        }
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit a structural predictor for one-cut root action drift."
    )
    parser.add_argument("--train-impact-json", required=True)
    parser.add_argument("--holdout-impact-json", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_cut_impact_predictor(
        train_impact_json=args.train_impact_json,
        holdout_impact_json=args.holdout_impact_json,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
