#!/usr/bin/env python3
"""Evaluate a learned selector over exact public-belief CFR budget records."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from poker_ai.research.solver_budget_policy import solver_native_vector  # noqa: E402


RIDGE_ALPHA = 1e-3
STRATEGY_FEATURE_WIDTH = 9


def _case_index(label: str) -> int:
    match = re.search(r"-(\d+)-", str(label))
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", str(label))
    if match:
        return int(match.group(1))
    raise ValueError(f"cannot parse case index from label: {label}")


def _budget(record: dict[str, Any], budget: int) -> dict[str, Any]:
    budgets = record.get("budgets", {})
    key = str(int(budget))
    if key not in budgets:
        raise ValueError(f"record {record.get('label')} is missing budget {key}")
    return dict(budgets[key])


def _rows(
    frontier_metrics: dict[str, Any],
    *,
    low_budget: int,
    uniform_budget: int,
    high_budget: int,
) -> list[dict[str, Any]]:
    records = [record for record in frontier_metrics.get("records", []) if record.get("passed")]
    if not records:
        raise ValueError("no passed frontier records")
    rows: list[dict[str, Any]] = []
    for record in records:
        label = str(record["label"])
        low = _budget(record, low_budget)
        uniform = _budget(record, uniform_budget)
        high = _budget(record, high_budget)
        rows.append(
            {
                "label": label,
                "case_index": _case_index(label),
                "low": low,
                "uniform": uniform,
                "high": high,
                "target_improvement": float(low["l1_to_reference"])
                - float(high["l1_to_reference"]),
                "features": _low_budget_features(low),
                "feature_source": _low_budget_feature_source(low),
            }
        )
    return rows


def _low_budget_features(low_budget_record: dict[str, Any]) -> np.ndarray:
    if "strategy" in low_budget_record:
        strategy = low_budget_record["strategy"]
        return solver_native_vector(
            strategy,
            action=int(low_budget_record["action"]),
            width=STRATEGY_FEATURE_WIDTH,
        )
    action = int(low_budget_record["action"])
    latency_ms = float(low_budget_record.get("latency_ms", 0.0))
    allin_prob = float(low_budget_record.get("allin_prob", 0.0))
    return np.asarray(
        [
            float(action) / 8.0,
            allin_prob,
            1.0 if bool(low_budget_record.get("allin_selected", False)) else 0.0,
            math.log1p(max(latency_ms, 0.0)) / 8.0,
        ],
        dtype=np.float64,
    )


def _low_budget_feature_source(low_budget_record: dict[str, Any]) -> str:
    if "strategy" in low_budget_record:
        return "low_budget_strategy_vector"
    return "low_budget_public_belief_solver_summary"


def _low_budget_feature_names(feature_source: str) -> list[str]:
    if feature_source == "low_budget_strategy_vector":
        return (
            [f"strategy_p{idx}" for idx in range(STRATEGY_FEATURE_WIDTH)]
            + [
                "strategy_top",
                "strategy_second",
                "strategy_margin",
                "strategy_entropy",
                "strategy_normalized_entropy",
                "strategy_concentration",
                "strategy_legal_fraction",
                "action_scaled",
            ]
        )
    return ["action_scaled", "allin_prob", "allin_selected", "log_latency"]


def _split_rows(rows: list[dict[str, Any]], *, start_index: int, limit: int) -> list[dict[str, Any]]:
    end = int(start_index) + int(limit)
    out = [row for row in rows if int(start_index) <= int(row["case_index"]) < end]
    if len(out) != int(limit):
        raise ValueError(
            f"split {start_index}:{end} expected {limit} rows but found {len(out)}"
        )
    return sorted(out, key=lambda row: int(row["case_index"]))


def _standardization_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)


def _predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return x_aug @ weights


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _rate(values: list[bool]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _select_threshold(scores: np.ndarray, select_train_top_k: int) -> float:
    if select_train_top_k <= 0:
        raise ValueError("select_train_top_k must be positive")
    if select_train_top_k > len(scores):
        raise ValueError("select_train_top_k exceeds train split size")
    ordered = np.sort(scores)
    return float(ordered[-int(select_train_top_k)])


def _select_top_fraction_labels(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    select_train_top_k: int,
    train_limit: int,
) -> set[str]:
    fraction = float(select_train_top_k) / float(max(int(train_limit), 1))
    count = max(1, int(math.ceil(fraction * len(rows))))
    ranked = sorted(
        zip(rows, scores, strict=True),
        key=lambda item: (float(item[1]), -int(item[0]["case_index"])),
        reverse=True,
    )
    return {str(row["label"]) for row, _score in ranked[:count]}


def evaluate_search_budget_selector(
    frontier_metrics: dict[str, Any],
    *,
    train_start_index: int = 128,
    train_limit: int = 64,
    holdout_start_index: int = 192,
    holdout_limit: int = 64,
    select_train_top_k: int = 8,
    low_budget: int = 5,
    uniform_budget: int = 10,
    high_budget: int = 25,
) -> dict[str, Any]:
    rows = _rows(
        frontier_metrics,
        low_budget=low_budget,
        uniform_budget=uniform_budget,
        high_budget=high_budget,
    )
    train_rows = _split_rows(rows, start_index=train_start_index, limit=train_limit)
    holdout_rows = _split_rows(rows, start_index=holdout_start_index, limit=holdout_limit)
    feature_sources = {str(row["feature_source"]) for row in [*train_rows, *holdout_rows]}
    if len(feature_sources) != 1:
        raise ValueError(f"mixed low-budget feature sources are not supported: {sorted(feature_sources)}")
    feature_source = next(iter(feature_sources))
    feature_names = _low_budget_feature_names(feature_source)
    train_x = np.stack([row["features"] for row in train_rows], axis=0)
    holdout_x = np.stack([row["features"] for row in holdout_rows], axis=0)
    train_y = np.asarray([float(row["target_improvement"]) for row in train_rows])
    mean, std = _standardization_stats(train_x)
    train_x_std = (train_x - mean) / std
    holdout_x_std = (holdout_x - mean) / std
    weights = _fit_ridge(train_x_std, train_y)
    train_scores = _predict(weights, train_x_std)
    holdout_scores = _predict(weights, holdout_x_std)
    threshold = _select_threshold(train_scores, select_train_top_k)
    selected_labels = _select_top_fraction_labels(
        holdout_rows,
        holdout_scores,
        select_train_top_k=select_train_top_k,
        train_limit=train_limit,
    )

    records: list[dict[str, Any]] = []
    illegal_masses: list[float] = []
    for row, score in zip(holdout_rows, holdout_scores, strict=True):
        selected = str(row["label"]) in selected_labels
        chosen = row["high"] if selected else row["low"]
        low = row["low"]
        uniform = row["uniform"]
        high = row["high"]
        illegal_masses.extend(
            [
                float(low.get("illegal_mass", 0.0)),
                float(uniform.get("illegal_mass", 0.0)),
                float(high.get("illegal_mass", 0.0)),
            ]
        )
        records.append(
            {
                "label": row["label"],
                "score": round(float(score), 8),
                "selected_for_high_budget": selected,
                "low_l1": round(float(low["l1_to_reference"]), 8),
                "uniform_l1": round(float(uniform["l1_to_reference"]), 8),
                "high_l1": round(float(high["l1_to_reference"]), 8),
                "selective_l1": round(float(chosen["l1_to_reference"]), 8),
                "low_latency_ms": round(float(low["latency_ms"]), 8),
                "uniform_latency_ms": round(float(uniform["latency_ms"]), 8),
                "high_latency_ms": round(float(high["latency_ms"]), 8),
                "selective_latency_ms": round(float(chosen["latency_ms"]), 8),
                "uniform_matches_high_action": int(uniform["action"]) == int(high["action"]),
                "selective_matches_high_action": int(chosen["action"]) == int(high["action"]),
            }
        )

    low_l1 = _mean([row["low_l1"] for row in records])
    uniform_l1 = _mean([row["uniform_l1"] for row in records])
    selective_l1 = _mean([row["selective_l1"] for row in records])
    low_latency = _mean([row["low_latency_ms"] for row in records])
    uniform_latency = _mean([row["uniform_latency_ms"] for row in records])
    selective_latency = _mean([row["selective_latency_ms"] for row in records])
    uniform_agreement = _rate([row["uniform_matches_high_action"] for row in records])
    selective_agreement = _rate([row["selective_matches_high_action"] for row in records])
    max_illegal = max(illegal_masses) if illegal_masses else 0.0
    return {
        "mode": "public_belief_search_budget_selector",
        "passed": bool(
            len(records) == int(holdout_limit)
            and max_illegal <= 1e-6
            and selective_l1 < uniform_l1
            and selective_agreement >= uniform_agreement
            and selective_latency <= uniform_latency
        ),
        "promotion": False,
        "low_budget": int(low_budget),
        "uniform_budget": int(uniform_budget),
        "high_budget": int(high_budget),
        "train_start_index": int(train_start_index),
        "train_limit": int(train_limit),
        "holdout_start_index": int(holdout_start_index),
        "holdout_limit": int(holdout_limit),
        "select_train_top_k": int(select_train_top_k),
        "selection_rule": "top_predicted_fraction_from_train_split",
        "holdout_selected": int(len(selected_labels)),
        "selected_labels": [row["label"] for row in records if row["selected_for_high_budget"]],
        "low_mean_l1": low_l1,
        "uniform_mean_l1": uniform_l1,
        "selective_mean_l1": selective_l1,
        "low_mean_latency_ms": low_latency,
        "uniform_mean_latency_ms": uniform_latency,
        "selective_mean_latency_ms": selective_latency,
        "uniform_action_agreement_to_high": uniform_agreement,
        "selective_action_agreement_to_high": selective_agreement,
        "max_illegal_mass": round(float(max_illegal), 8),
        "policy": {
            "policy_type": "public_belief_search_budget_selector",
            "feature_source": feature_source,
            "feature_dim": int(train_x.shape[1]),
            "feature_names": feature_names,
            "ridge_alpha": float(RIDGE_ALPHA),
            "feature_mean": mean.reshape(-1).astype(float).tolist(),
            "feature_std": std.reshape(-1).astype(float).tolist(),
            "ridge_weights": weights.astype(float).tolist(),
            "score_threshold": round(float(threshold), 8),
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-json", required=True)
    parser.add_argument("--train-start-index", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--holdout-start-index", type=int, default=192)
    parser.add_argument("--holdout-limit", type=int, default=64)
    parser.add_argument("--select-train-top-k", type=int, default=8)
    parser.add_argument("--low-budget", type=int, default=5)
    parser.add_argument("--uniform-budget", type=int, default=10)
    parser.add_argument("--high-budget", type=int, default=25)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    frontier_metrics = json.loads(Path(args.frontier_json).read_text(encoding="utf-8"))
    metrics = evaluate_search_budget_selector(
        frontier_metrics,
        train_start_index=args.train_start_index,
        train_limit=args.train_limit,
        holdout_start_index=args.holdout_start_index,
        holdout_limit=args.holdout_limit,
        select_train_top_k=args.select_train_top_k,
        low_budget=args.low_budget,
        uniform_budget=args.uniform_budget,
        high_budget=args.high_budget,
    )
    metrics["frontier_json"] = str(args.frontier_json)
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
