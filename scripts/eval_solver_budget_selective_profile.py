#!/usr/bin/env python3
"""Evaluate selective fast-live-to-live solver budget escalation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from poker_ai.research.solver_budget_policy import (
    score_selective_policy,
    solver_native_vector,
)

RIDGE_ALPHA = 1e-2


def _case_index(label: str) -> int:
    match = re.search(r"-(\d+)-", str(label))
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", str(label))
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse case index from label: {label}")


def _feature_by_label(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels and features must have the same number of rows")
    return {str(label): features[idx] for idx, label in enumerate(labels)}


def build_solver_native_features(
    profile_metrics: dict[str, Any],
    *,
    candidate_profile: str = "fast-live",
) -> tuple[np.ndarray, np.ndarray]:
    """Build deployment-available features from the cheap solver's own output."""

    records = [record for record in profile_metrics.get("records", []) if record.get("passed")]
    if not records:
        raise ValueError("profile_metrics has no passed records")
    width = max(
        len(record["profiles"][candidate_profile]["strategy"])
        for record in records
        if candidate_profile in record.get("profiles", {})
    )
    features: list[np.ndarray] = []
    labels: list[str] = []
    for record in records:
        profiles = record.get("profiles", {})
        if candidate_profile not in profiles:
            continue
        candidate = profiles[candidate_profile]
        labels.append(str(record["label"]))
        features.append(
            solver_native_vector(
                candidate["strategy"],
                action=int(candidate["action"]),
                width=width,
            )
        )
    if not features:
        raise ValueError(f"profile_metrics has no records for candidate profile {candidate_profile}")
    return np.stack(features, axis=0).astype(np.float32), np.asarray(labels)


def _combine_features(
    left_features: np.ndarray,
    left_labels: np.ndarray,
    right_features: np.ndarray,
    right_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    right_lookup = _feature_by_label(right_features, right_labels)
    combined_features: list[np.ndarray] = []
    combined_labels: list[str] = []
    for idx, label in enumerate(np.asarray(left_labels)):
        label_str = str(label)
        if label_str not in right_lookup:
            continue
        combined_labels.append(label_str)
        combined_features.append(np.concatenate([left_features[idx], right_lookup[label_str]], axis=0))
    if not combined_features:
        raise ValueError("feature sources do not share labels")
    return np.stack(combined_features, axis=0).astype(np.float32), np.asarray(combined_labels)


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _standardize(train_x: np.ndarray, holdout_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = _standardization_stats(train_x)
    return (train_x - mean) / std, (holdout_x - mean) / std


def _standardization_stats(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    x_aug = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ train_y)


def _predict_ridge(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return x_aug @ weights


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    pred_std = float(np.std(pred))
    target_std = float(np.std(target))
    if pred_std < 1e-9 or target_std < 1e-9:
        return 0.0
    return round(float(np.corrcoef(pred, target)[0, 1]), 8)


def _select_threshold(rows: list[dict[str, Any]], score_key: str, select_train_top_k: int) -> float:
    if not rows:
        raise ValueError("train split is empty")
    if select_train_top_k <= 0:
        raise ValueError("select_train_top_k must be positive")
    if select_train_top_k > len(rows):
        raise ValueError("select_train_top_k exceeds train split size")
    ranked = sorted(rows, key=lambda row: row[score_key], reverse=True)
    return float(ranked[int(select_train_top_k) - 1][score_key])


def _rows(
    profile_metrics: dict[str, Any],
    features: np.ndarray,
    feature_labels: np.ndarray,
    *,
    reference_profile: str,
    candidate_profile: str,
) -> list[dict[str, Any]]:
    feature_lookup = _feature_by_label(features, feature_labels)
    rows: list[dict[str, Any]] = []
    for record in profile_metrics.get("records", []):
        if not record.get("passed"):
            continue
        label = str(record["label"])
        if label not in feature_lookup:
            continue
        reference = record["profiles"][reference_profile]
        candidate = record["profiles"][candidate_profile]
        reference_strategy = np.asarray(reference["strategy"], dtype=np.float64)
        candidate_strategy = np.asarray(candidate["strategy"], dtype=np.float64)
        profile_l1 = float(np.abs(candidate_strategy - reference_strategy).sum())
        action_disagreement = float(int(candidate["action"]) != int(reference["action"]))
        rows.append(
            {
                "label": label,
                "case_index": _case_index(label),
                "features": feature_lookup[label],
                "profile_l1": profile_l1,
                "action_disagreement": action_disagreement,
                "reference_action": int(reference["action"]),
                "candidate_action": int(candidate["action"]),
                "reference_latency_ms": float(reference["latency_ms"]),
                "candidate_latency_ms": float(candidate["latency_ms"]),
            }
        )
    return rows


def _target_key(target_name: str) -> str:
    if target_name == "profile-l1":
        return "profile_l1"
    if target_name == "action-disagreement":
        return "action_disagreement"
    raise ValueError(f"unsupported target_name: {target_name}")


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    start_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    end = int(start_index) + int(limit)
    return [row for row in rows if int(start_index) <= row["case_index"] < end]


def fit_selective_policy(
    profile_metrics: dict[str, Any],
    features: np.ndarray,
    feature_labels: np.ndarray,
    *,
    train_start_index: int = 128,
    train_limit: int = 64,
    select_train_top_k: int = 8,
    target_name: str = "profile-l1",
    reference_profile: str = "live",
    candidate_profile: str = "fast-live",
    feature_source: str = "unknown",
) -> dict[str, Any]:
    rows = _rows(
        profile_metrics,
        features,
        feature_labels,
        reference_profile=reference_profile,
        candidate_profile=candidate_profile,
    )
    train_rows = _split_rows(rows, start_index=train_start_index, limit=train_limit)
    if len(train_rows) != int(train_limit):
        raise ValueError("train split is incomplete")

    target_key = _target_key(target_name)
    train_x = np.stack([row["features"] for row in train_rows], axis=0)
    train_y = np.asarray([row[target_key] for row in train_rows], dtype=np.float64)
    mean, std = _standardization_stats(train_x)
    train_x_std = (train_x - mean) / std
    weights = _fit_ridge(train_x_std, train_y)
    train_pred = _predict_ridge(weights, train_x_std)
    for row, score in zip(train_rows, train_pred, strict=True):
        row["predicted_target"] = float(score)
    threshold = _select_threshold(train_rows, "predicted_target", select_train_top_k)
    return {
        "schema_version": 1,
        "policy_type": "solver_budget_selective_profile",
        "feature_source": str(feature_source),
        "target_name": str(target_name),
        "reference_profile": str(reference_profile),
        "candidate_profile": str(candidate_profile),
        "train_start_index": int(train_start_index),
        "train_limit": int(train_limit),
        "select_train_top_k": int(select_train_top_k),
        "ridge_alpha": float(RIDGE_ALPHA),
        "feature_dim": int(train_x.shape[1]),
        "feature_mean": mean.reshape(-1).astype(float).tolist(),
        "feature_std": std.reshape(-1).astype(float).tolist(),
        "ridge_weights": weights.astype(float).tolist(),
        "score_threshold": round(float(threshold), 8),
        "train_target_mean": round(float(np.mean(train_y)), 8),
    }


def _selective_metrics(rows: list[dict[str, Any]], selected_labels: set[str]) -> dict[str, float]:
    if not rows:
        return {
            "mean_l1_to_live": 0.0,
            "action_agreement": 0.0,
            "mean_latency_ms": 0.0,
        }
    l1_values = [
        0.0 if row["label"] in selected_labels else row["profile_l1"]
        for row in rows
    ]
    action_agreements = [
        1.0
        if row["label"] in selected_labels or row["action_disagreement"] == 0.0
        else 0.0
        for row in rows
    ]
    latencies = [
        row["reference_latency_ms"]
        if row["label"] in selected_labels
        else row["candidate_latency_ms"]
        for row in rows
    ]
    return {
        "mean_l1_to_live": _mean(l1_values),
        "action_agreement": round(float(np.mean(action_agreements)), 8),
        "mean_latency_ms": _mean(latencies),
    }


def evaluate_selective_profile(
    profile_metrics: dict[str, Any],
    features: np.ndarray,
    feature_labels: np.ndarray,
    *,
    train_start_index: int = 128,
    train_limit: int = 64,
    holdout_start_index: int = 192,
    holdout_limit: int = 64,
    select_train_top_k: int = 8,
    target_name: str = "profile-l1",
    reference_profile: str = "live",
    candidate_profile: str = "fast-live",
) -> dict[str, Any]:
    rows = _rows(
        profile_metrics,
        features,
        feature_labels,
        reference_profile=reference_profile,
        candidate_profile=candidate_profile,
    )
    train_rows = _split_rows(rows, start_index=train_start_index, limit=train_limit)
    holdout_rows = _split_rows(rows, start_index=holdout_start_index, limit=holdout_limit)
    if not train_rows or not holdout_rows:
        raise ValueError("train and holdout splits must both be non-empty")

    target_key = _target_key(target_name)
    train_x = np.stack([row["features"] for row in train_rows], axis=0)
    holdout_x = np.stack([row["features"] for row in holdout_rows], axis=0)
    train_y = np.asarray([row[target_key] for row in train_rows], dtype=np.float64)
    holdout_y = np.asarray([row[target_key] for row in holdout_rows], dtype=np.float64)
    train_x_std, holdout_x_std = _standardize(train_x, holdout_x)
    weights = _fit_ridge(train_x_std, train_y)
    train_pred = _predict_ridge(weights, train_x_std)
    holdout_pred = _predict_ridge(weights, holdout_x_std)

    for row, score in zip(train_rows, train_pred, strict=True):
        row["predicted_target"] = float(score)
    for row, score in zip(holdout_rows, holdout_pred, strict=True):
        row["predicted_target"] = float(score)

    predicted_threshold = _select_threshold(train_rows, "predicted_target", select_train_top_k)
    oracle_threshold = _select_threshold(train_rows, target_key, select_train_top_k)
    predicted_selected = {
        row["label"] for row in holdout_rows if row["predicted_target"] >= predicted_threshold
    }
    oracle_selected = {
        row["label"] for row in holdout_rows if row[target_key] >= oracle_threshold
    }
    disagreement_selected = {
        row["label"] for row in holdout_rows if row["action_disagreement"] > 0.0
    }

    fast_metrics = _selective_metrics(holdout_rows, set())
    predicted_metrics = _selective_metrics(holdout_rows, predicted_selected)
    oracle_metrics = _selective_metrics(holdout_rows, oracle_selected)
    disagreement_metrics = _selective_metrics(holdout_rows, disagreement_selected)
    live_latency = _mean([row["reference_latency_ms"] for row in holdout_rows])
    target_mae = _mean(list(np.abs(holdout_pred - holdout_y)))
    baseline_mae = _mean(list(np.abs(float(np.mean(train_y)) - holdout_y)))
    overlap = predicted_selected.intersection(oracle_selected)

    return {
        "mode": "solver_budget_selective_profile",
        "reference_profile": reference_profile,
        "candidate_profile": candidate_profile,
        "target_name": target_name,
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
        "oracle_threshold": round(float(oracle_threshold), 8),
        "holdout_selected": int(len(predicted_selected)),
        "oracle_holdout_selected": int(len(oracle_selected)),
        "disagreement_holdout_selected": int(len(disagreement_selected)),
        "selected_labels": sorted(predicted_selected),
        "oracle_selected_labels": sorted(oracle_selected),
        "disagreement_labels": sorted(disagreement_selected),
        "top_k_recall": round(float(len(overlap) / max(len(oracle_selected), 1)), 8),
        "top_k_precision": round(float(len(overlap) / max(len(predicted_selected), 1)), 8),
        "target_mae": target_mae,
        "target_mean_baseline_mae": baseline_mae,
        "target_pearson": _pearson(holdout_pred, holdout_y),
        "fast_mean_l1_to_live": fast_metrics["mean_l1_to_live"],
        "predicted_selective_mean_l1_to_live": predicted_metrics["mean_l1_to_live"],
        "oracle_selective_mean_l1_to_live": oracle_metrics["mean_l1_to_live"],
        "disagreement_oracle_mean_l1_to_live": disagreement_metrics["mean_l1_to_live"],
        "fast_action_agreement": fast_metrics["action_agreement"],
        "predicted_selective_action_agreement": predicted_metrics["action_agreement"],
        "oracle_selective_action_agreement": oracle_metrics["action_agreement"],
        "disagreement_oracle_action_agreement": disagreement_metrics["action_agreement"],
        "fast_mean_latency_ms": fast_metrics["mean_latency_ms"],
        "predicted_selective_mean_latency_ms": predicted_metrics["mean_latency_ms"],
        "oracle_selective_mean_latency_ms": oracle_metrics["mean_latency_ms"],
        "disagreement_oracle_mean_latency_ms": disagreement_metrics["mean_latency_ms"],
        "live_mean_latency_ms": live_latency,
        "predicted_latency_ratio_to_fast": round(
            float(predicted_metrics["mean_latency_ms"] / max(fast_metrics["mean_latency_ms"], 1e-9)),
            8,
        ),
        "predicted_latency_ratio_to_live": round(
            float(predicted_metrics["mean_latency_ms"] / max(live_latency, 1e-9)),
            8,
        ),
        "predicted_l1_improvement": round(
            float(fast_metrics["mean_l1_to_live"] - predicted_metrics["mean_l1_to_live"]),
            8,
        ),
        "predicted_action_agreement_improvement": round(
            float(predicted_metrics["action_agreement"] - fast_metrics["action_agreement"]),
            8,
        ),
        "passed": bool(
            len(train_rows) == int(train_limit)
            and len(holdout_rows) == int(holdout_limit)
            and len(predicted_selected) > 0
            and target_mae <= baseline_mae
            and predicted_metrics["mean_l1_to_live"] <= fast_metrics["mean_l1_to_live"]
            and predicted_metrics["action_agreement"] >= fast_metrics["action_agreement"]
            and predicted_metrics["mean_latency_ms"] < live_latency
        ),
        "promotion": False,
    }


def _load_feature_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["features"].astype(np.float32, copy=False), data["labels"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate selective fast-live to live solver escalation."
    )
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--cfv-cache")
    parser.add_argument(
        "--feature-source",
        choices=("cfv-cache", "solver-native", "combined"),
        default="cfv-cache",
        help="Features used by the boundary predictor. solver-native uses only the cheap solver output.",
    )
    parser.add_argument("--target", choices=("profile-l1", "action-disagreement"), default="profile-l1")
    parser.add_argument("--train-start-index", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--holdout-start-index", type=int, default=192)
    parser.add_argument("--holdout-limit", type=int, default=64)
    parser.add_argument("--select-train-top-k", type=int, default=8)
    parser.add_argument("--policy-output", help="Optional JSON path for the fitted selector policy.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    profile_metrics = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    if args.feature_source == "solver-native":
        features, labels = build_solver_native_features(profile_metrics)
    elif args.feature_source == "combined":
        if not args.cfv_cache:
            raise SystemExit("--cfv-cache is required for --feature-source combined")
        cfv_features, cfv_labels = _load_feature_cache(Path(args.cfv_cache))
        solver_features, solver_labels = build_solver_native_features(profile_metrics)
        features, labels = _combine_features(cfv_features, cfv_labels, solver_features, solver_labels)
    else:
        if not args.cfv_cache:
            raise SystemExit("--cfv-cache is required for --feature-source cfv-cache")
        features, labels = _load_feature_cache(Path(args.cfv_cache))
    metrics = evaluate_selective_profile(
        profile_metrics,
        features,
        labels,
        train_start_index=args.train_start_index,
        train_limit=args.train_limit,
        holdout_start_index=args.holdout_start_index,
        holdout_limit=args.holdout_limit,
        select_train_top_k=args.select_train_top_k,
        target_name=args.target,
    )
    metrics["profile_json"] = str(args.profile_json)
    metrics["cfv_cache"] = str(args.cfv_cache) if args.cfv_cache else None
    metrics["feature_source"] = str(args.feature_source)
    if args.policy_output:
        policy = fit_selective_policy(
            profile_metrics,
            features,
            labels,
            train_start_index=args.train_start_index,
            train_limit=args.train_limit,
            select_train_top_k=args.select_train_top_k,
            target_name=args.target,
            feature_source=args.feature_source,
        )
        policy["profile_json"] = str(args.profile_json)
        policy["cfv_cache"] = str(args.cfv_cache) if args.cfv_cache else None
        policy_output = Path(args.policy_output)
        policy_output.parent.mkdir(parents=True, exist_ok=True)
        policy_output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metrics["policy_output"] = str(policy_output)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
