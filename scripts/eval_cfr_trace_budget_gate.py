#!/usr/bin/env python3
"""Evaluate a selective CFR iteration budget from early trace predictions."""

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
    _design,
    _fit_ridge,
    _load_payload,
    _records_for_iteration,
    _standardize,
)
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402


def _record_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["label"]): dict(record) for record in records}


def _max_iteration(payload: dict[str, Any]) -> int:
    iterations = [int(record.get("iteration", -1)) for record in payload.get("records", [])]
    if not iterations:
        raise ValueError("trace payload has no records")
    return max(iterations)


def _fit_low_iteration_predictions(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    low_trace_iteration: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    train_rows = _records_for_iteration(train_payload, low_trace_iteration)
    holdout_rows = _records_for_iteration(holdout_payload, low_trace_iteration)
    x_train, y_train, _ = _design(train_rows)
    x_holdout, _y_holdout, _ = _design(holdout_rows)
    x_train, mean, std = _standardize(x_train)
    x_holdout, _, _ = _standardize(x_holdout, mean=mean, std=std)
    weights = _fit_ridge(x_train, y_train)
    return holdout_rows, x_holdout @ weights


def _select_predicted_high(predictions: np.ndarray, selection_fraction: float) -> set[str]:
    if not 0.0 < float(selection_fraction) <= 1.0:
        raise ValueError("selection_fraction must be in (0, 1]")
    count = max(1, int(math.ceil(float(selection_fraction) * len(predictions))))
    return {str(index) for index in np.argsort(predictions)[-count:].tolist()}


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _rate(values: list[bool]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def eval_trace_budget_gate_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int | None = None,
    reference_trace_iteration: int | None = None,
    selection_fraction: float = 0.2,
) -> dict[str, Any]:
    if reference_trace_iteration is None:
        reference_trace_iteration = _max_iteration(holdout_payload)
    holdout_low_rows, predictions = _fit_low_iteration_predictions(
        train_payload,
        holdout_payload,
        low_trace_iteration=low_trace_iteration,
    )
    selected_positions = _select_predicted_high(predictions, selection_fraction)
    selected_labels = {
        str(holdout_low_rows[int(position)]["label"])
        for position in selected_positions
    }
    selected_count = len(selected_labels)
    n_rows = len(holdout_low_rows)
    adaptive_mean_iterations = (
        ((n_rows - selected_count) * (low_trace_iteration + 1))
        + (selected_count * (reference_trace_iteration + 1))
    ) / float(n_rows)
    if uniform_trace_iteration is None:
        uniform_trace_iteration = min(
            reference_trace_iteration,
            max(low_trace_iteration, int(math.ceil(adaptive_mean_iterations)) - 1),
        )
    low_by_label = _record_map(holdout_low_rows)
    uniform_by_label = _record_map(
        _records_for_iteration(holdout_payload, int(uniform_trace_iteration))
    )
    reference_by_label = _record_map(
        _records_for_iteration(holdout_payload, int(reference_trace_iteration))
    )
    labels = sorted(set(low_by_label) & set(uniform_by_label) & set(reference_by_label))
    if not labels:
        raise ValueError("no common labels across low, uniform, and reference records")
    low_l1: list[float] = []
    uniform_l1: list[float] = []
    adaptive_l1: list[float] = []
    low_match: list[bool] = []
    uniform_match: list[bool] = []
    adaptive_match: list[bool] = []
    records: list[dict[str, Any]] = []
    for label in labels:
        low_record = low_by_label[label]
        uniform_record = uniform_by_label[label]
        reference_record = reference_by_label[label]
        selected = label in selected_labels
        adaptive_record = reference_record if selected else low_record
        low_value = float(low_record["l1_to_final_strategy"])
        uniform_value = float(uniform_record["l1_to_final_strategy"])
        adaptive_value = float(adaptive_record["l1_to_final_strategy"])
        low_l1.append(low_value)
        uniform_l1.append(uniform_value)
        adaptive_l1.append(adaptive_value)
        low_match.append(bool(low_record.get("top_matches_final")))
        uniform_match.append(bool(uniform_record.get("top_matches_final")))
        adaptive_match.append(bool(adaptive_record.get("top_matches_final")))
        records.append(
            {
                "label": label,
                "selected_for_extra_iterations": bool(selected),
                "low_l1_to_reference": round(low_value, 8),
                "uniform_l1_to_reference": round(uniform_value, 8),
                "adaptive_l1_to_reference": round(adaptive_value, 8),
                "low_top_matches_reference": bool(low_record.get("top_matches_final")),
                "uniform_top_matches_reference": bool(uniform_record.get("top_matches_final")),
                "adaptive_top_matches_reference": bool(adaptive_record.get("top_matches_final")),
            }
        )
    adaptive_mean = _mean(adaptive_l1)
    uniform_mean = _mean(uniform_l1)
    passed = adaptive_mean < uniform_mean and _rate(adaptive_match) >= _rate(uniform_match)
    return {
        "mode": "cfr_trace_selective_budget_gate",
        "passed": bool(passed),
        "promotion": False,
        "low_trace_iteration": int(low_trace_iteration),
        "uniform_trace_iteration": int(uniform_trace_iteration),
        "reference_trace_iteration": int(reference_trace_iteration),
        "selection_fraction": float(selection_fraction),
        "n_evaluated": int(len(labels)),
        "selected_count": int(selected_count),
        "low_mean_trace_iterations": float(low_trace_iteration + 1),
        "uniform_mean_trace_iterations": float(uniform_trace_iteration + 1),
        "adaptive_mean_trace_iterations": round(float(adaptive_mean_iterations), 8),
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_uniform_l1_to_reference": uniform_mean,
        "mean_adaptive_l1_to_reference": adaptive_mean,
        "low_top_match_rate": _rate(low_match),
        "uniform_top_match_rate": _rate(uniform_match),
        "adaptive_top_match_rate": _rate(adaptive_match),
        "selection_rule": "top_predicted_l1_to_final_from_low_iteration_trace",
        "records": records,
    }


def eval_trace_budget_gate(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int | None = None,
    reference_trace_iteration: int | None = None,
    selection_fraction: float = 0.2,
) -> dict[str, Any]:
    train_payload = _load_payload(train_trace_json)
    holdout_payload = _load_payload(holdout_trace_json)
    metrics = eval_trace_budget_gate_from_payloads(
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
    parser.add_argument("--uniform-trace-iteration", type=int)
    parser.add_argument("--reference-trace-iteration", type=int)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_trace_budget_gate(
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
