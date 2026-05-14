#!/usr/bin/env python3
"""Fit a linear trace-state policy residual toward the final CFR strategy."""

from __future__ import annotations

import argparse
import json
import math
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

from analyze_cfr_trace_predictor import (  # noqa: E402
    _load_payload,
    _policy_entropy,
    _policy_margin,
    _records_for_iteration,
)
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402


_RIDGE = 1e-3


def _record_map(payload: dict[str, Any], iteration: int) -> dict[str, dict[str, Any]]:
    return {str(record["label"]): dict(record) for record in _records_for_iteration(payload, iteration)}


def _feature_row(record: dict[str, Any], action_dim: int) -> np.ndarray:
    regret = np.asarray(record["regret_policy"], dtype=np.float64)
    strategy = np.asarray(record["strategy_policy"], dtype=np.float64)
    legal = tuple(int(action) for action in record.get("legal_actions", ()))
    legal_mask = np.zeros(action_dim, dtype=np.float64)
    for action in legal:
        if 0 <= action < action_dim:
            legal_mask[action] = 1.0
    return np.concatenate(
        [
            np.ones(1, dtype=np.float64),
            regret,
            strategy,
            legal_mask,
            np.asarray(
                [
                    _policy_entropy(regret),
                    _policy_entropy(strategy),
                    _policy_margin(regret, legal),
                    _policy_margin(strategy, legal),
                    float(np.abs(regret - strategy).sum()),
                    math.log1p(float(record.get("regret_mass", 0.0))),
                    math.log1p(float(record.get("strategy_mass", 0.0))),
                    float(record.get("hero_reach_mass", 0.0)),
                    float(record.get("villain_reach_mass", 0.0)),
                    float(len(legal)),
                    float(record.get("street", 0)),
                ],
                dtype=np.float64,
            ),
        ]
    )


def _build_xy(
    low_by_label: dict[str, dict[str, Any]],
    final_by_label: dict[str, dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = sorted(set(low_by_label) & set(final_by_label))
    if not labels:
        raise ValueError("no common labels between low and final records")
    action_dim = len(low_by_label[labels[0]]["strategy_policy"])
    x = np.asarray([_feature_row(low_by_label[label], action_dim) for label in labels])
    y = np.asarray(
        [final_by_label[label]["strategy_policy"] for label in labels],
        dtype=np.float64,
    )
    return labels, x, y


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


def _normalize_prediction(pred: np.ndarray, legal_actions: list[int], fallback: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.float64)
    legal = [int(action) for action in legal_actions if 0 <= int(action) < len(pred)]
    if not legal:
        raise ValueError("legal_actions must contain at least one valid action")
    clipped = np.maximum(pred[legal], 0.0)
    total = float(clipped.sum())
    if total <= 1e-12:
        fallback_legal = np.maximum(fallback[legal], 0.0)
        total = float(fallback_legal.sum())
        if total <= 1e-12:
            out[legal] = 1.0 / float(len(legal))
        else:
            out[legal] = fallback_legal / total
    else:
        out[legal] = clipped / total
    return out


def _top_match(policy: np.ndarray, reference: np.ndarray) -> bool:
    return bool(int(np.argmax(policy)) == int(np.argmax(reference)))


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _rate(values: list[bool]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def fit_trace_policy_residual_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
) -> dict[str, Any]:
    train_low = _record_map(train_payload, low_trace_iteration)
    train_final = _record_map(train_payload, reference_trace_iteration)
    _train_labels, x_train, y_train = _build_xy(train_low, train_final)
    x_train, feature_mean, feature_std = _standardize(x_train)
    weights = _fit_ridge(x_train, y_train)

    holdout_low = _record_map(holdout_payload, low_trace_iteration)
    holdout_uniform = _record_map(holdout_payload, uniform_trace_iteration)
    holdout_final = _record_map(holdout_payload, reference_trace_iteration)
    labels, x_holdout, y_holdout = _build_xy(holdout_low, holdout_final)
    x_holdout, _, _ = _standardize(x_holdout, mean=feature_mean, std=feature_std)
    raw_pred = x_holdout @ weights

    pred_l1: list[float] = []
    low_l1: list[float] = []
    uniform_l1: list[float] = []
    pred_match: list[bool] = []
    low_match: list[bool] = []
    uniform_match: list[bool] = []
    records: list[dict[str, Any]] = []
    for row_idx, label in enumerate(labels):
        low_record = holdout_low[label]
        uniform_record = holdout_uniform[label]
        reference = np.asarray(holdout_final[label]["strategy_policy"], dtype=np.float64)
        low_policy = np.asarray(low_record["strategy_policy"], dtype=np.float64)
        uniform_policy = np.asarray(uniform_record["strategy_policy"], dtype=np.float64)
        pred_policy = _normalize_prediction(
            raw_pred[row_idx],
            [int(action) for action in low_record.get("legal_actions", ())],
            fallback=low_policy,
        )
        pred_value = float(np.abs(pred_policy - reference).sum())
        low_value = float(np.abs(low_policy - reference).sum())
        uniform_value = float(np.abs(uniform_policy - reference).sum())
        pred_l1.append(pred_value)
        low_l1.append(low_value)
        uniform_l1.append(uniform_value)
        pred_match.append(_top_match(pred_policy, reference))
        low_match.append(_top_match(low_policy, reference))
        uniform_match.append(_top_match(uniform_policy, reference))
        records.append(
            {
                "label": label,
                "pred_l1_to_reference": round(pred_value, 8),
                "low_l1_to_reference": round(low_value, 8),
                "uniform_l1_to_reference": round(uniform_value, 8),
                "pred_top_matches_reference": bool(pred_match[-1]),
                "low_top_matches_reference": bool(low_match[-1]),
                "uniform_top_matches_reference": bool(uniform_match[-1]),
                "pred_policy": pred_policy.round(8).tolist(),
            }
        )
    mean_pred = _mean(pred_l1)
    mean_low = _mean(low_l1)
    mean_uniform = _mean(uniform_l1)
    passed = mean_pred < mean_low and mean_pred < mean_uniform
    return {
        "mode": "cfr_trace_policy_residual",
        "passed": bool(passed),
        "promotion": False,
        "low_trace_iteration": int(low_trace_iteration),
        "uniform_trace_iteration": int(uniform_trace_iteration),
        "reference_trace_iteration": int(reference_trace_iteration),
        "n_train": int(len(_train_labels)),
        "n_holdout": int(len(labels)),
        "mean_pred_l1_to_reference": mean_pred,
        "mean_low_l1_to_reference": mean_low,
        "mean_uniform_l1_to_reference": mean_uniform,
        "pred_top_match_rate": _rate(pred_match),
        "low_top_match_rate": _rate(low_match),
        "uniform_top_match_rate": _rate(uniform_match),
        "model": {
            "ridge": float(_RIDGE),
            "feature_mean": feature_mean.round(10).tolist(),
            "feature_std": feature_std.round(10).tolist(),
            "weights_shape": list(weights.shape),
        },
        "records": records,
    }


def fit_trace_policy_residual(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
) -> dict[str, Any]:
    metrics = fit_trace_policy_residual_from_payloads(
        _load_payload(train_trace_json),
        _load_payload(holdout_trace_json),
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
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
    parser.add_argument("--reference-trace-iteration", type=int, default=24)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_trace_policy_residual(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        low_trace_iteration=args.low_trace_iteration,
        uniform_trace_iteration=args.uniform_trace_iteration,
        reference_trace_iteration=args.reference_trace_iteration,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
