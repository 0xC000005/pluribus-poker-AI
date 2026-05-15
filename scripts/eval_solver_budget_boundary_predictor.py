#!/usr/bin/env python3
"""Evaluate a cheap predictor for profile-drift budget escalation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


RIDGE_ALPHA = 1e-2


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _case_index(label: str) -> int:
    match = re.search(r"-(\d+)-", str(label))
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", str(label))
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse case index from label: {label}")


def _profile_l1(record: dict[str, Any], reference_profile: str, candidate_profile: str) -> float:
    reference = np.asarray(record["profiles"][reference_profile]["strategy"], dtype=np.float64)
    candidate = np.asarray(record["profiles"][candidate_profile]["strategy"], dtype=np.float64)
    return float(np.abs(candidate - reference).sum())


def _frontier_by_label(frontier_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["label"]): record
        for record in frontier_metrics.get("records", [])
        if record.get("passed")
    }


def _feature_by_label(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels and features must have the same number of rows")
    return {
        str(label): features[idx]
        for idx, label in enumerate(labels)
    }


def _rows(
    frontier_metrics: dict[str, Any],
    profile_metrics: dict[str, Any],
    features: np.ndarray,
    feature_labels: np.ndarray,
    *,
    reference_profile: str,
    candidate_profile: str,
    escalation_budget: int,
) -> list[dict[str, Any]]:
    frontier = _frontier_by_label(frontier_metrics)
    feature_lookup = _feature_by_label(features, feature_labels)
    rows: list[dict[str, Any]] = []
    for record in profile_metrics.get("records", []):
        if not record.get("passed"):
            continue
        label = str(record["label"])
        if label not in frontier or label not in feature_lookup:
            continue
        live_iterations = str(int(record["profiles"][reference_profile]["iterations"]))
        frontier_record = frontier[label]
        live_budget = frontier_record["budgets"][live_iterations]
        escalation = frontier_record["budgets"][str(int(escalation_budget))]
        rows.append(
            {
                "label": label,
                "case_index": _case_index(label),
                "features": feature_lookup[label],
                "profile_l1": _profile_l1(record, reference_profile, candidate_profile),
                "live_l1": float(live_budget["l1_to_reference"]),
                "live_kl": float(live_budget["kl_to_reference"]),
                "live_latency_ms": float(live_budget["latency_ms"]),
                "escalation_l1": float(escalation["l1_to_reference"]),
                "escalation_kl": float(escalation["kl_to_reference"]),
                "escalation_latency_ms": float(escalation["latency_ms"]),
            }
        )
    return rows


def _standardize(train_x: np.ndarray, holdout_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (holdout_x - mean) / std


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    x_aug = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ train_y)


def _predict_ridge(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return x_aug @ weights


def _select_threshold(rows: list[dict[str, Any]], score_key: str, select_train_top_k: int) -> float:
    if not rows:
        raise ValueError("train split is empty")
    if select_train_top_k <= 0:
        raise ValueError("select_train_top_k must be positive")
    if select_train_top_k > len(rows):
        raise ValueError("select_train_top_k exceeds train split size")
    ranked = sorted(rows, key=lambda row: row[score_key], reverse=True)
    return float(ranked[int(select_train_top_k) - 1][score_key])


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    pred_std = float(np.std(pred))
    target_std = float(np.std(target))
    if pred_std < 1e-9 or target_std < 1e-9:
        return 0.0
    return round(float(np.corrcoef(pred, target)[0, 1]), 8)


def _selective_values(
    holdout_rows: list[dict[str, Any]],
    selected_labels: set[str],
    value_key: str,
    escalation_key: str,
) -> list[float]:
    return [
        row[escalation_key] if row["label"] in selected_labels else row[value_key]
        for row in holdout_rows
    ]


def evaluate_boundary_predictor(
    frontier_metrics: dict[str, Any],
    profile_metrics: dict[str, Any],
    features: np.ndarray,
    feature_labels: np.ndarray,
    *,
    train_start_index: int = 128,
    train_limit: int = 64,
    holdout_start_index: int = 192,
    holdout_limit: int = 64,
    select_train_top_k: int = 8,
    escalation_budget: int = 350,
    reference_profile: str = "live",
    candidate_profile: str = "fast-live",
) -> dict[str, Any]:
    rows = _rows(
        frontier_metrics,
        profile_metrics,
        features,
        feature_labels,
        reference_profile=reference_profile,
        candidate_profile=candidate_profile,
        escalation_budget=escalation_budget,
    )
    train_end = int(train_start_index) + int(train_limit)
    holdout_end = int(holdout_start_index) + int(holdout_limit)
    train_rows = [
        row for row in rows
        if int(train_start_index) <= row["case_index"] < train_end
    ]
    holdout_rows = [
        row for row in rows
        if int(holdout_start_index) <= row["case_index"] < holdout_end
    ]
    if not train_rows or not holdout_rows:
        raise ValueError("train and holdout splits must both be non-empty")

    train_x = np.stack([row["features"] for row in train_rows], axis=0)
    holdout_x = np.stack([row["features"] for row in holdout_rows], axis=0)
    train_y = np.asarray([row["profile_l1"] for row in train_rows], dtype=np.float64)
    holdout_y = np.asarray([row["profile_l1"] for row in holdout_rows], dtype=np.float64)
    train_x_std, holdout_x_std = _standardize(train_x, holdout_x)
    weights = _fit_ridge(train_x_std, train_y)
    train_pred = _predict_ridge(weights, train_x_std)
    holdout_pred = _predict_ridge(weights, holdout_x_std)

    for row, score in zip(train_rows, train_pred, strict=True):
        row["predicted_profile_l1"] = float(score)
    for row, score in zip(holdout_rows, holdout_pred, strict=True):
        row["predicted_profile_l1"] = float(score)

    predicted_threshold = _select_threshold(train_rows, "predicted_profile_l1", int(select_train_top_k))
    oracle_threshold = _select_threshold(train_rows, "profile_l1", int(select_train_top_k))
    selected = [
        row for row in holdout_rows
        if row["predicted_profile_l1"] >= predicted_threshold
    ]
    oracle_selected = [
        row for row in holdout_rows
        if row["profile_l1"] >= oracle_threshold
    ]
    selected_labels = {row["label"] for row in selected}
    oracle_selected_labels = {row["label"] for row in oracle_selected}
    overlap = selected_labels.intersection(oracle_selected_labels)

    live_l1 = _mean([row["live_l1"] for row in holdout_rows])
    predicted_l1 = _mean(
        _selective_values(holdout_rows, selected_labels, "live_l1", "escalation_l1")
    )
    live_kl = _mean([row["live_kl"] for row in holdout_rows])
    predicted_kl = _mean(
        _selective_values(holdout_rows, selected_labels, "live_kl", "escalation_kl")
    )
    live_latency = _mean([row["live_latency_ms"] for row in holdout_rows])
    predicted_latency = _mean(
        _selective_values(
            holdout_rows,
            selected_labels,
            "live_latency_ms",
            "escalation_latency_ms",
        )
    )
    holdout_mae = _mean(list(np.abs(holdout_pred - holdout_y)))
    baseline_mae = _mean(list(np.abs(float(np.mean(train_y)) - holdout_y)))

    return {
        "mode": "solver_budget_boundary_predictor",
        "reference_profile": reference_profile,
        "candidate_profile": candidate_profile,
        "escalation_budget": int(escalation_budget),
        "train_start_index": int(train_start_index),
        "train_limit": int(train_limit),
        "holdout_start_index": int(holdout_start_index),
        "holdout_limit": int(holdout_limit),
        "select_train_top_k": int(select_train_top_k),
        "ridge_alpha": float(RIDGE_ALPHA),
        "feature_dim": int(train_x.shape[1]),
        "n_train": int(len(train_rows)),
        "n_holdout": int(len(holdout_rows)),
        "predicted_threshold": round(float(predicted_threshold), 8),
        "oracle_threshold_profile_l1": round(float(oracle_threshold), 8),
        "holdout_selected": int(len(selected)),
        "oracle_holdout_selected": int(len(oracle_selected)),
        "selected_labels": sorted(selected_labels),
        "oracle_selected_labels": sorted(oracle_selected_labels),
        "top_k_recall": round(
            float(len(overlap) / max(len(oracle_selected_labels), 1)),
            8,
        ),
        "top_k_precision": round(
            float(len(overlap) / max(len(selected_labels), 1)),
            8,
        ),
        "holdout_profile_l1_mae": holdout_mae,
        "holdout_profile_l1_mean_baseline_mae": baseline_mae,
        "holdout_profile_l1_pearson": _pearson(holdout_pred, holdout_y),
        "live_mean_l1": live_l1,
        "predicted_selective_mean_l1": predicted_l1,
        "oracle_selective_mean_l1": _mean(
            _selective_values(holdout_rows, oracle_selected_labels, "live_l1", "escalation_l1")
        ),
        "uniform_escalation_mean_l1": _mean([row["escalation_l1"] for row in holdout_rows]),
        "live_mean_kl": live_kl,
        "predicted_selective_mean_kl": predicted_kl,
        "oracle_selective_mean_kl": _mean(
            _selective_values(holdout_rows, oracle_selected_labels, "live_kl", "escalation_kl")
        ),
        "uniform_escalation_mean_kl": _mean([row["escalation_kl"] for row in holdout_rows]),
        "live_mean_latency_ms": live_latency,
        "predicted_selective_mean_latency_ms": predicted_latency,
        "oracle_selective_mean_latency_ms": _mean(
            _selective_values(
                holdout_rows,
                oracle_selected_labels,
                "live_latency_ms",
                "escalation_latency_ms",
            )
        ),
        "uniform_escalation_mean_latency_ms": _mean(
            [row["escalation_latency_ms"] for row in holdout_rows]
        ),
        "predicted_latency_ratio_to_live": round(
            float(predicted_latency / max(live_latency, 1e-9)),
            8,
        ),
        "predicted_l1_improvement": round(float(live_l1 - predicted_l1), 8),
        "predicted_kl_improvement": round(float(live_kl - predicted_kl), 8),
        "passed": bool(
            len(train_rows) == int(train_limit)
            and len(holdout_rows) == int(holdout_limit)
            and len(selected) > 0
            and holdout_mae <= baseline_mae
            and predicted_l1 <= live_l1
            and predicted_kl <= live_kl
        ),
        "promotion": False,
    }


def _load_feature_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["features"].astype(np.float32, copy=False), data["labels"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate cheap feature prediction of profile-L1 budget boundaries."
    )
    parser.add_argument("--frontier-json", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--train-start-index", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--holdout-start-index", type=int, default=192)
    parser.add_argument("--holdout-limit", type=int, default=64)
    parser.add_argument("--select-train-top-k", type=int, default=8)
    parser.add_argument("--escalation-budget", type=int, default=350)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    frontier_metrics = json.loads(Path(args.frontier_json).read_text(encoding="utf-8"))
    profile_metrics = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    features, feature_labels = _load_feature_cache(Path(args.cfv_cache))
    metrics = evaluate_boundary_predictor(
        frontier_metrics,
        profile_metrics,
        features,
        feature_labels,
        train_start_index=args.train_start_index,
        train_limit=args.train_limit,
        holdout_start_index=args.holdout_start_index,
        holdout_limit=args.holdout_limit,
        select_train_top_k=args.select_train_top_k,
        escalation_budget=args.escalation_budget,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
