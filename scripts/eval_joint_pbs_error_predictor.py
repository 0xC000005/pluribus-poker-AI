#!/usr/bin/env python3
"""Learn a metadata risk model for joint-PBS value prediction errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from analyze_joint_pbs_value_errors import analyze_joint_pbs_value_errors  # noqa: E402


NUMERIC_FIELDS = (
    "actor_to_act",
    "bet_count",
    "client_pos",
    "cut_pos",
    "hero_reach_normalized_entropy",
    "hero_reach_top1_mass",
    "hero_reach_top10_mass",
    "legal_action_count",
    "villain_reach_normalized_entropy",
    "villain_reach_top1_mass",
    "villain_reach_top10_mass",
)
STRUCTURAL_NUMERIC_FIELDS = (
    "actor_to_act",
    "bet_count",
    "client_pos",
    "cut_pos",
    "legal_action_count",
)
CATEGORICAL_FIELDS = ("action_shape",)
_RIDGE = 1e-3


def _metadata_by_label(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(record["label"]): dict(record)
        for record in payload.get("cut_records", [])
        if record.get("label") is not None
    }


def _join_error_metadata(
    error_metrics: dict[str, Any],
    metadata_json: str | Path,
) -> list[dict[str, Any]]:
    metadata = _metadata_by_label(metadata_json)
    rows = []
    for record in error_metrics.get("state_errors", []):
        label = str(record["label"])
        if label not in metadata:
            continue
        rows.append({**metadata[label], **record})
    return rows


def _category_vocab(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        field: sorted({str(row.get(field, "missing")) for row in rows})
        for field in CATEGORICAL_FIELDS
    }


def _design_matrix(
    rows: list[dict[str, Any]],
    *,
    numeric_fields: tuple[str, ...],
    vocab: dict[str, list[str]],
    numeric_mean: np.ndarray | None = None,
    numeric_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric = np.asarray(
        [
            [float(row.get(field, 0.0) or 0.0) for field in numeric_fields]
            for row in rows
        ],
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
    if len(pred) == 0:
        return 0.0
    pred_cut = float(np.quantile(pred, quantile))
    target_cut = float(np.quantile(target, quantile))
    target_high = target >= target_cut
    if not bool(target_high.any()):
        return 0.0
    return float(np.mean(pred[target_high] >= pred_cut))


def _metrics(pred_log: np.ndarray, target_mae: np.ndarray) -> dict[str, float]:
    pred_mae = np.exp(pred_log) - 1e-6
    return {
        "pearson_mae": round(_pearson(pred_mae, target_mae), 8),
        "top_quintile_recall": round(_top_quantile_recall(pred_mae, target_mae), 8),
        "predicted_mae_mean": round(float(np.mean(pred_mae)), 8),
        "actual_mae_mean": round(float(np.mean(target_mae)), 8),
    }


def _fit_error_predictor_with_fields(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    numeric_fields: tuple[str, ...],
) -> dict[str, Any]:
    vocab = _category_vocab(train_rows)
    x_train, numeric_mean, numeric_std = _design_matrix(
        train_rows,
        numeric_fields=numeric_fields,
        vocab=vocab,
    )
    x_holdout, _, _ = _design_matrix(
        holdout_rows,
        numeric_fields=numeric_fields,
        vocab=vocab,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
    )
    y_train = np.log(
        np.asarray([float(row["mae"]) for row in train_rows], dtype=np.float64) + 1e-6
    )
    y_holdout_mae = np.asarray(
        [float(row["mae"]) for row in holdout_rows],
        dtype=np.float64,
    )
    weights = _fit_ridge(x_train, y_train)
    train_pred = x_train @ weights
    holdout_pred = x_holdout @ weights
    train_pred_mae = np.exp(train_pred) - 1e-6
    return {
        "passed": bool(np.isfinite(holdout_pred).all()),
        "n_train": len(train_rows),
        "n_holdout": len(holdout_rows),
        "numeric_fields": list(numeric_fields),
        "categorical_fields": list(CATEGORICAL_FIELDS),
        "category_vocab": vocab,
        "train": _metrics(
            train_pred,
            np.asarray([float(row["mae"]) for row in train_rows], dtype=np.float64),
        ),
        "holdout": _metrics(holdout_pred, y_holdout_mae),
        "model": {
            "ridge": float(_RIDGE),
            "numeric_fields": list(numeric_fields),
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "category_vocab": vocab,
            "numeric_mean": numeric_mean.round(10).tolist(),
            "numeric_std": numeric_std.round(10).tolist(),
            "weights": weights.round(10).tolist(),
            "abstention_rule": "use_learned_value_when_predicted_mae_at_or_below_train_median",
            "abstention_predicted_mae_cut": round(float(np.quantile(train_pred_mae, 0.5)), 8),
            "train_predicted_mae_quantiles": {
                "p50": round(float(np.quantile(train_pred_mae, 0.5)), 8),
                "p80": round(float(np.quantile(train_pred_mae, 0.8)), 8),
                "p90": round(float(np.quantile(train_pred_mae, 0.9)), 8),
            },
        },
        "worst_predicted_holdout": [
            {
                "label": str(holdout_rows[index]["label"]),
                "actual_mae": round(float(y_holdout_mae[index]), 8),
                "predicted_mae": round(float(np.exp(holdout_pred[index]) - 1e-6), 8),
            }
            for index in np.argsort(holdout_pred)[-10:][::-1]
        ],
    }


def fit_error_predictor_from_rows(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    full = _fit_error_predictor_with_fields(
        train_rows,
        holdout_rows,
        numeric_fields=NUMERIC_FIELDS,
    )
    structural = _fit_error_predictor_with_fields(
        train_rows,
        holdout_rows,
        numeric_fields=STRUCTURAL_NUMERIC_FIELDS,
    )
    return {
        "mode": "joint_pbs_error_predictor",
        **full,
        "passed": bool(full["passed"] and structural["passed"]),
        "structural_only": structural,
    }


def evaluate_joint_pbs_error_predictor(
    *,
    checkpoint: str | Path,
    train_joint_npz: str | Path,
    holdout_joint_npz: str | Path,
    train_metadata_json: str | Path,
    holdout_metadata_json: str | Path,
    device: str = "auto",
    batch_size: int = 8192,
) -> dict[str, Any]:
    train_errors = analyze_joint_pbs_value_errors(
        checkpoint=checkpoint,
        joint_npz=train_joint_npz,
        metadata_json=train_metadata_json,
        device=device,
        batch_size=batch_size,
    )
    holdout_errors = analyze_joint_pbs_value_errors(
        checkpoint=checkpoint,
        joint_npz=holdout_joint_npz,
        metadata_json=holdout_metadata_json,
        device=device,
        batch_size=batch_size,
    )
    train_rows = _join_error_metadata(train_errors, train_metadata_json)
    holdout_rows = _join_error_metadata(holdout_errors, holdout_metadata_json)
    metrics = fit_error_predictor_from_rows(train_rows, holdout_rows)
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "train_joint_npz": str(train_joint_npz),
            "holdout_joint_npz": str(holdout_joint_npz),
            "train_metadata_json": str(train_metadata_json),
            "holdout_metadata_json": str(holdout_metadata_json),
            "device": str(device),
            "batch_size": int(batch_size),
        }
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit a learned metadata predictor for joint-PBS cut-value error."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-joint", required=True)
    parser.add_argument("--holdout-joint", required=True)
    parser.add_argument("--train-metadata-json", required=True)
    parser.add_argument("--holdout-metadata-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = evaluate_joint_pbs_error_predictor(
        checkpoint=args.checkpoint,
        train_joint_npz=args.train_joint,
        holdout_joint_npz=args.holdout_joint,
        train_metadata_json=args.train_metadata_json,
        holdout_metadata_json=args.holdout_metadata_json,
        device=args.device,
        batch_size=args.batch_size,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
