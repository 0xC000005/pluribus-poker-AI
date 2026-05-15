#!/usr/bin/env python3
"""Fit a diagnostic predictor from CFR trace trajectories."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_cfr_trace_predictor import (  # noqa: E402
    _fit_ridge,
    _load_payload,
    _pearson,
    _policy_entropy,
    _policy_margin,
    _standardize,
    _top_quintile_recall,
    _trace_feature_row,
)
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402


SEQUENCE_FEATURE_NAMES = (
    "strategy_delta_l1_mean",
    "strategy_delta_l1_max",
    "strategy_delta_l1_last",
    "regret_delta_l1_mean",
    "regret_delta_l1_max",
    "regret_delta_l1_last",
    "strategy_entropy_change",
    "regret_entropy_change",
    "strategy_margin_change",
    "regret_margin_change",
    "log_strategy_mass_change",
    "log_regret_mass_change",
)


def _records_by_label_iteration(payload: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    by_label: dict[str, dict[int, dict[str, Any]]] = {}
    for record in payload.get("records", []):
        if "label" not in record or "iteration" not in record:
            continue
        label = str(record["label"])
        iteration = int(record["iteration"])
        by_label.setdefault(label, {})[iteration] = dict(record)
    return by_label


def _l1(policy_a: Iterable[float], policy_b: Iterable[float]) -> float:
    return float(np.abs(np.asarray(policy_a, dtype=np.float64) - np.asarray(policy_b, dtype=np.float64)).sum())


def _top_match(policy_a: Iterable[float], policy_b: Iterable[float]) -> bool:
    a = np.asarray(policy_a, dtype=np.float64)
    b = np.asarray(policy_b, dtype=np.float64)
    return int(np.argmax(a)) == int(np.argmax(b))


def _delta_stats(records: list[dict[str, Any]], key: str) -> tuple[float, float, float]:
    deltas: list[float] = []
    for previous, current in zip(records, records[1:]):
        deltas.append(_l1(previous[key], current[key]))
    if not deltas:
        return 0.0, 0.0, 0.0
    values = np.asarray(deltas, dtype=np.float64)
    return float(values.mean()), float(values.max()), float(values[-1])


def _sequence_feature_row(records: list[dict[str, Any]]) -> list[float]:
    first = records[0]
    last = records[-1]
    legal_actions = tuple(int(action) for action in last.get("legal_actions", ()))
    strategy_delta = _delta_stats(records, "strategy_policy")
    regret_delta = _delta_stats(records, "regret_policy")
    first_strategy = np.asarray(first["strategy_policy"], dtype=np.float64)
    last_strategy = np.asarray(last["strategy_policy"], dtype=np.float64)
    first_regret = np.asarray(first["regret_policy"], dtype=np.float64)
    last_regret = np.asarray(last["regret_policy"], dtype=np.float64)
    return [
        *strategy_delta,
        *regret_delta,
        _policy_entropy(last_strategy) - _policy_entropy(first_strategy),
        _policy_entropy(last_regret) - _policy_entropy(first_regret),
        _policy_margin(last_strategy, legal_actions) - _policy_margin(first_strategy, legal_actions),
        _policy_margin(last_regret, legal_actions) - _policy_margin(first_regret, legal_actions),
        math.log1p(float(last.get("strategy_mass", 0.0))) - math.log1p(float(first.get("strategy_mass", 0.0))),
        math.log1p(float(last.get("regret_mass", 0.0))) - math.log1p(float(first.get("regret_mass", 0.0))),
    ]


def _common_labels(
    payload: dict[str, Any],
    *,
    low_trace_iteration: int,
    uniform_trace_iteration: int,
    reference_trace_iteration: int,
) -> list[str]:
    by_label = _records_by_label_iteration(payload)
    required = set(range(0, int(low_trace_iteration) + 1))
    required.add(int(uniform_trace_iteration))
    required.add(int(reference_trace_iteration))
    labels = [
        label
        for label, by_iteration in by_label.items()
        if required.issubset(set(by_iteration))
    ]
    if not labels:
        raise ValueError("no labels contain all requested trace iterations")
    return sorted(labels)


def _design(
    payload: dict[str, Any],
    *,
    low_trace_iteration: int,
    uniform_trace_iteration: int,
    reference_trace_iteration: int,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    by_label = _records_by_label_iteration(payload)
    labels = _common_labels(
        payload,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
    )
    features: list[list[float]] = []
    targets: list[float] = []
    records: dict[str, dict[str, Any]] = {}
    for label in labels:
        by_iteration = by_label[label]
        sequence = [by_iteration[iteration] for iteration in range(0, int(low_trace_iteration) + 1)]
        low = by_iteration[int(low_trace_iteration)]
        uniform = by_iteration[int(uniform_trace_iteration)]
        reference = by_iteration[int(reference_trace_iteration)]
        low_l1 = _l1(low["strategy_policy"], reference["strategy_policy"])
        uniform_l1 = _l1(uniform["strategy_policy"], reference["strategy_policy"])
        features.append([*_trace_feature_row(low), *_sequence_feature_row(sequence)])
        targets.append(low_l1 - uniform_l1)
        records[label] = {
            "low": low,
            "uniform": uniform,
            "reference": reference,
            "low_l1": low_l1,
            "uniform_l1": uniform_l1,
        }
    return labels, np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64), records


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(float(np.mean(values)), 8) if values else 0.0


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return round(float(np.mean(values)), 8) if values else 0.0


def _selected_indices(predictions: np.ndarray, selection_fraction: float) -> set[int]:
    if not 0.0 < float(selection_fraction) <= 1.0:
        raise ValueError("selection_fraction must be in (0, 1]")
    count = max(1, int(math.ceil(float(selection_fraction) * len(predictions))))
    return set(np.argsort(predictions)[-count:].tolist())


def fit_trace_sequence_predictor_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    selection_fraction: float = 0.2,
) -> dict[str, Any]:
    train_labels, x_train, y_train, _train_records = _design(
        train_payload,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
    )
    holdout_labels, x_holdout, y_holdout, holdout_records = _design(
        holdout_payload,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
    )
    x_train, mean, std = _standardize(x_train)
    x_holdout, _, _ = _standardize(x_holdout, mean=mean, std=std)
    weights = _fit_ridge(x_train, y_train)
    predictions = x_holdout @ weights
    train_mean = float(np.mean(y_train))
    constant = np.full_like(y_holdout, train_mean)
    ridge_mae = float(np.mean(np.abs(predictions - y_holdout)))
    train_mean_mae = float(np.mean(np.abs(constant - y_holdout)))
    pearson = _pearson(predictions, y_holdout)
    top_recall = _top_quintile_recall(predictions, y_holdout)
    selected = _selected_indices(predictions, selection_fraction)

    adaptive_l1: list[float] = []
    low_l1: list[float] = []
    uniform_l1: list[float] = []
    adaptive_match: list[bool] = []
    uniform_match: list[bool] = []
    selected_improvement: list[float] = []
    rejected_improvement: list[float] = []
    row_metrics: list[dict[str, Any]] = []
    for index, label in enumerate(holdout_labels):
        row = holdout_records[label]
        was_selected = index in selected
        chosen = row["uniform"] if was_selected else row["low"]
        chosen_l1 = row["uniform_l1"] if was_selected else row["low_l1"]
        improvement = float(y_holdout[index])
        if was_selected:
            selected_improvement.append(improvement)
        else:
            rejected_improvement.append(improvement)
        adaptive_l1.append(float(chosen_l1))
        low_l1.append(float(row["low_l1"]))
        uniform_l1.append(float(row["uniform_l1"]))
        adaptive_match.append(_top_match(chosen["strategy_policy"], row["reference"]["strategy_policy"]))
        uniform_match.append(_top_match(row["uniform"]["strategy_policy"], row["reference"]["strategy_policy"]))
        row_metrics.append(
            {
                "label": label,
                "predicted_improvement": round(float(predictions[index]), 8),
                "target_improvement": round(improvement, 8),
                "selected_for_continuation": bool(was_selected),
                "low_l1_to_reference": round(float(row["low_l1"]), 8),
                "uniform_l1_to_reference": round(float(row["uniform_l1"]), 8),
                "adaptive_l1_to_reference": round(float(chosen_l1), 8),
            }
        )

    selected_mean = _mean(selected_improvement)
    rejected_mean = _mean(rejected_improvement)
    prediction_signal_found = (
        ridge_mae < train_mean_mae
        and pearson > 0.0
        and selected_improvement
        and rejected_improvement
        and selected_mean > rejected_mean
    )
    decision_passed = (
        _mean(adaptive_l1) < _mean(uniform_l1)
        and _rate(adaptive_match) >= _rate(uniform_match)
    )
    return {
        "mode": "cfr_trace_sequence_predictor_diagnostic",
        "passed": bool(prediction_signal_found and decision_passed),
        "promotion": False,
        "prediction_signal_found": bool(prediction_signal_found),
        "decision_passed": bool(decision_passed),
        "low_trace_iteration": int(low_trace_iteration),
        "uniform_trace_iteration": int(uniform_trace_iteration),
        "reference_trace_iteration": int(reference_trace_iteration),
        "selection_fraction": float(selection_fraction),
        "n_train": int(len(train_labels)),
        "n_holdout": int(len(holdout_labels)),
        "selected_count": int(len(selected)),
        "ridge_mae": round(ridge_mae, 8),
        "train_mean_mae": round(train_mean_mae, 8),
        "pearson": round(pearson, 8),
        "top_quintile_recall": round(top_recall, 8),
        "selected_mean_target_improvement": selected_mean,
        "rejected_mean_target_improvement": rejected_mean,
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_uniform_l1_to_reference": _mean(uniform_l1),
        "mean_adaptive_l1_to_reference": _mean(adaptive_l1),
        "uniform_top_match_rate": _rate(uniform_match),
        "adaptive_top_match_rate": _rate(adaptive_match),
        "model": {
            "ridge": 1e-3,
            "feature_names": [
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
                *SEQUENCE_FEATURE_NAMES,
            ],
            "feature_mean": mean.round(10).tolist(),
            "feature_std": std.round(10).tolist(),
            "weights": weights.round(10).tolist(),
        },
        "records": row_metrics,
    }


def fit_trace_sequence_predictor(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    selection_fraction: float = 0.2,
) -> dict[str, Any]:
    train_payload = _load_payload(train_trace_json)
    holdout_payload = _load_payload(holdout_trace_json)
    metrics = fit_trace_sequence_predictor_from_payloads(
        train_payload,
        holdout_payload,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        selection_fraction=selection_fraction,
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
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_trace_sequence_predictor(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        low_trace_iteration=args.low_trace_iteration,
        uniform_trace_iteration=args.uniform_trace_iteration,
        reference_trace_iteration=args.reference_trace_iteration,
        selection_fraction=args.selection_fraction,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
