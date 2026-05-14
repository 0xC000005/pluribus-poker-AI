#!/usr/bin/env python3
"""Fit a simple diagnostic predictor for CFR trace convergence error."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from poker_ai.research.belief_value_probe import save_metrics


FEATURE_NAMES = (
    "bias",
    "regret_entropy",
    "strategy_entropy",
    "regret_margin",
    "strategy_margin",
    "regret_strategy_l1",
    "log_regret_mass",
    "log_strategy_mass",
    "hero_reach_mass",
    "villain_reach_mass",
    "legal_action_count",
    "street",
)
DEFAULT_ITERATIONS = (1, 2, 3, 5, 10, 15, 20)
_RIDGE = 1e-3


def _load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _policy_entropy(policy: np.ndarray) -> float:
    probs = policy[policy > 1e-12]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _policy_margin(policy: np.ndarray, legal_actions: Iterable[int]) -> float:
    legal_values = [float(policy[int(action)]) for action in legal_actions]
    if len(legal_values) <= 1:
        return 1.0
    ordered = sorted(legal_values)
    return float(ordered[-1] - ordered[-2])


def _trace_feature_row(record: dict[str, Any]) -> list[float]:
    regret = np.asarray(record["regret_policy"], dtype=np.float64)
    strategy = np.asarray(record["strategy_policy"], dtype=np.float64)
    legal_actions = tuple(int(action) for action in record.get("legal_actions", ()))
    return [
        1.0,
        _policy_entropy(regret),
        _policy_entropy(strategy),
        _policy_margin(regret, legal_actions),
        _policy_margin(strategy, legal_actions),
        float(np.abs(regret - strategy).sum()),
        math.log1p(float(record.get("regret_mass", 0.0))),
        math.log1p(float(record.get("strategy_mass", 0.0))),
        float(record.get("hero_reach_mass", 0.0)),
        float(record.get("villain_reach_mass", 0.0)),
        float(len(legal_actions)),
        float(record.get("street", 0)),
    ]


def _records_for_iteration(payload: dict[str, Any], iteration: int) -> list[dict[str, Any]]:
    rows = [
        dict(record)
        for record in payload.get("records", [])
        if int(record.get("iteration", -1)) == int(iteration)
        and "l1_to_final_strategy" in record
    ]
    if not rows:
        raise ValueError(f"no trace records for iteration {iteration}")
    return rows


def _design(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([_trace_feature_row(row) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["l1_to_final_strategy"]) for row in rows], dtype=np.float64)
    top_match = np.asarray([1.0 if row.get("top_matches_final") else 0.0 for row in rows])
    return x, y, top_match


def _standardize(
    x: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = x[:, 1:].mean(axis=0)
    if std is None:
        std = x[:, 1:].std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    out = x.copy()
    out[:, 1:] = (x[:, 1:] - mean) / std
    return out, mean, std


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * _RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _top_quintile_recall(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) == 0:
        return 0.0
    k = max(1, int(math.ceil(0.2 * len(pred))))
    pred_high = set(np.argsort(pred)[-k:].tolist())
    target_high = set(np.argsort(target)[-k:].tolist())
    return float(len(pred_high & target_high) / k)


def _iteration_metrics(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    x_train, y_train, _train_match = _design(train_rows)
    x_holdout, y_holdout, holdout_match = _design(holdout_rows)
    x_train, mean, std = _standardize(x_train)
    x_holdout, _, _ = _standardize(x_holdout, mean=mean, std=std)
    weights = _fit_ridge(x_train, y_train)
    pred = x_holdout @ weights
    train_mean = float(np.mean(y_train))
    constant = np.full_like(y_holdout, train_mean)
    k = max(1, int(math.ceil(0.2 * len(y_holdout))))
    predicted_high_idx = np.argsort(pred)[-k:]
    predicted_low_mask = np.ones(len(y_holdout), dtype=bool)
    predicted_low_mask[predicted_high_idx] = False
    predicted_high_mean = float(np.mean(y_holdout[predicted_high_idx]))
    predicted_low_mean = (
        float(np.mean(y_holdout[predicted_low_mask]))
        if bool(predicted_low_mask.any())
        else None
    )
    ridge_mae = float(np.mean(np.abs(pred - y_holdout)))
    train_mean_mae = float(np.mean(np.abs(constant - y_holdout)))
    pearson = _pearson(pred, y_holdout)
    top_recall = _top_quintile_recall(pred, y_holdout)
    signal_found = (
        ridge_mae < train_mean_mae
        and pearson > 0.0
        and predicted_low_mean is not None
        and predicted_high_mean > predicted_low_mean
    )
    return {
        "signal_found": bool(signal_found),
        "n_train": int(len(train_rows)),
        "n_holdout": int(len(holdout_rows)),
        "holdout_l1_mean": round(float(np.mean(y_holdout)), 8),
        "holdout_top_match_rate": round(float(np.mean(holdout_match)), 8),
        "ridge_mae": round(ridge_mae, 8),
        "train_mean_mae": round(train_mean_mae, 8),
        "mae_improvement": round(train_mean_mae - ridge_mae, 8),
        "pearson": round(pearson, 8),
        "top_quintile_recall": round(top_recall, 8),
        "predicted_high_count": int(k),
        "predicted_high_mean_l1": round(predicted_high_mean, 8),
        "predicted_low_mean_l1": (
            round(predicted_low_mean, 8) if predicted_low_mean is not None else None
        ),
        "model": {
            "ridge": float(_RIDGE),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": mean.round(10).tolist(),
            "feature_std": std.round(10).tolist(),
            "weights": weights.round(10).tolist(),
        },
    }


def fit_trace_predictor_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    iterations: Iterable[int] = DEFAULT_ITERATIONS,
) -> dict[str, Any]:
    per_iteration: dict[str, Any] = {}
    for iteration in iterations:
        train_rows = _records_for_iteration(train_payload, int(iteration))
        holdout_rows = _records_for_iteration(holdout_payload, int(iteration))
        per_iteration[str(int(iteration))] = _iteration_metrics(train_rows, holdout_rows)
    passing = [
        (int(iteration), metrics)
        for iteration, metrics in per_iteration.items()
        if metrics["signal_found"]
    ]
    if passing:
        best_iteration, best_metrics = max(
            passing,
            key=lambda item: (
                float(item[1]["mae_improvement"]),
                float(item[1]["pearson"]),
            ),
        )
    else:
        best_iteration, best_metrics = max(
            ((int(iteration), metrics) for iteration, metrics in per_iteration.items()),
            key=lambda item: float(item[1]["mae_improvement"]),
        )
    return {
        "mode": "cfr_trace_predictor_diagnostic",
        "passed": bool(passing),
        "promotion": False,
        "best_iteration": int(best_iteration),
        "best_iteration_metrics": best_metrics,
        "iterations": per_iteration,
    }


def fit_trace_predictor(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    iterations: Iterable[int] = DEFAULT_ITERATIONS,
) -> dict[str, Any]:
    train_payload = _load_payload(train_trace_json)
    holdout_payload = _load_payload(holdout_trace_json)
    metrics = fit_trace_predictor_from_payloads(
        train_payload,
        holdout_payload,
        iterations=iterations,
    )
    metrics["train_trace_json"] = str(train_trace_json)
    metrics["holdout_trace_json"] = str(holdout_trace_json)
    return metrics


def _parse_iterations(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-json", required=True)
    parser.add_argument("--holdout-trace-json", required=True)
    parser.add_argument("--iterations", default=",".join(str(i) for i in DEFAULT_ITERATIONS))
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_trace_predictor(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        iterations=_parse_iterations(args.iterations),
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
