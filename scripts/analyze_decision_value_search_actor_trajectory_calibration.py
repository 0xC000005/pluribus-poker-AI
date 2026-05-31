#!/usr/bin/env python3
"""Test whether search-actor decision traces predict whole-hand transfer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402


STREETS = ("preflop", "flop", "turn", "river")


def _pearson_or_none(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _sign_agreement(xs: list[float], ys: list[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys, strict=True)
        if abs(float(x)) > 1e-12 and abs(float(y)) > 1e-12
    ]
    if not pairs:
        return None
    return float(sum(np.sign(x) == np.sign(y) for x, y in pairs) / len(pairs))


def _mae(xs: list[float], ys: list[float]) -> float | None:
    if not xs:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    return float(np.mean(np.abs(x - y)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pair_features(paired_deltas: list[float], records: list[dict[str, Any]]) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_idx = int(record.get("duplicate_pair", -1))
        if 0 <= pair_idx < len(paired_deltas):
            grouped[pair_idx].append(record)

    pair_indices = sorted(grouped)
    rows: list[list[float]] = []
    targets: list[float] = []
    feature_names = [
        "local_sum_all",
        "local_sum_disagreement",
        "n_decisions",
        "n_disagreements",
        "positive_local_sum",
        "negative_local_sum",
        "mean_abs_local",
        "max_abs_local",
        "mean_pot_total",
        "max_pot_total",
        "mean_to_call",
        *[f"{street}_local_sum" for street in STREETS],
        *[f"{street}_n_disagreements" for street in STREETS],
    ]

    for pair_idx in pair_indices:
        pair_records = grouped[pair_idx]
        local_values = [
            _safe_float(record.get("local_search_minus_deployed_value"))
            for record in pair_records
        ]
        disagreements = [
            record
            for record in pair_records
            if record.get("deployed_action") != record.get("search_action")
        ]
        disagreement_values = [
            _safe_float(record.get("local_search_minus_deployed_value"))
            for record in disagreements
        ]
        pots = [_safe_float(record.get("pot_total")) for record in pair_records]
        to_calls = [_safe_float(record.get("to_call")) for record in pair_records]
        row = [
            float(np.sum(local_values)) if local_values else 0.0,
            float(np.sum(disagreement_values)) if disagreement_values else 0.0,
            float(len(pair_records)),
            float(len(disagreements)),
            float(np.sum([max(0.0, value) for value in local_values])),
            float(np.sum([min(0.0, value) for value in local_values])),
            float(np.mean(np.abs(local_values))) if local_values else 0.0,
            float(np.max(np.abs(local_values))) if local_values else 0.0,
            float(np.mean(pots)) if pots else 0.0,
            float(np.max(pots)) if pots else 0.0,
            float(np.mean(to_calls)) if to_calls else 0.0,
        ]
        for street in STREETS:
            street_values = [
                _safe_float(record.get("local_search_minus_deployed_value"))
                for record in pair_records
                if record.get("street") == street
            ]
            row.append(float(np.sum(street_values)) if street_values else 0.0)
        for street in STREETS:
            row.append(
                float(
                    sum(
                        record.get("street") == street
                        and record.get("deployed_action") != record.get("search_action")
                        for record in pair_records
                    )
                )
            )
        rows.append(row)
        targets.append(float(paired_deltas[pair_idx]))

    return pair_indices, np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), feature_names


def _fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    alpha: float,
) -> list[float]:
    if len(y_train) == 0 or x_eval.shape[0] == 0:
        return []
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma <= 1e-8] = 1.0
    x_train_z = (x_train - mu) / sigma
    x_eval_z = (x_eval - mu) / sigma
    design = np.concatenate(
        [np.ones((x_train_z.shape[0], 1), dtype=np.float64), x_train_z],
        axis=1,
    )
    eval_design = np.concatenate(
        [np.ones((x_eval_z.shape[0], 1), dtype=np.float64), x_eval_z],
        axis=1,
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
    return [float(value) for value in eval_design @ weights]


def _summarize_predictions(predictions: list[float], targets: list[float]) -> dict[str, Any]:
    return {
        "holdout_pearson": _pearson_or_none(predictions, targets),
        "holdout_sign_agreement": _sign_agreement(predictions, targets),
        "holdout_mae": _mae(predictions, targets),
    }


def analyze_trajectory_calibration(
    payload: dict[str, Any],
    *,
    train_fraction: float = 0.5,
    ridge_alpha: float = 1.0,
    min_holdout_pairs: int = 4,
    min_sign_agreement: float = 0.5,
) -> dict[str, Any]:
    paired_deltas = [float(value) for value in payload.get("paired_deltas", [])]
    records = list(payload.get("decision_records") or [])
    pair_indices, features, targets, feature_names = _pair_features(paired_deltas, records)
    n_pairs = int(len(pair_indices))
    n_train = int(np.floor(n_pairs * float(train_fraction)))
    n_train = max(1, min(n_train, max(1, n_pairs - 1))) if n_pairs >= 2 else n_pairs
    train_idx = list(range(n_train))
    holdout_idx = list(range(n_train, n_pairs))

    raw_local = features[:, 0].tolist() if n_pairs else []
    holdout_targets = targets[holdout_idx].tolist() if holdout_idx else []
    holdout_raw = [raw_local[idx] for idx in holdout_idx]
    train_mean = float(np.mean(targets[train_idx])) if train_idx else 0.0
    mean_predictions = [train_mean for _ in holdout_targets]
    calibrated_predictions = _fit_ridge_predict(
        features[train_idx],
        targets[train_idx],
        features[holdout_idx],
        alpha=ridge_alpha,
    ) if holdout_idx else []

    raw_summary = _summarize_predictions(holdout_raw, holdout_targets)
    mean_summary = _summarize_predictions(mean_predictions, holdout_targets)
    calibrated_summary = _summarize_predictions(calibrated_predictions, holdout_targets)

    blockers: list[str] = []
    if len(holdout_idx) < int(min_holdout_pairs):
        blockers.append("insufficient_holdout_pairs")
    raw_pearson = raw_summary["holdout_pearson"]
    calibrated_pearson = calibrated_summary["holdout_pearson"]
    if calibrated_pearson is None or calibrated_pearson <= 0.0:
        blockers.append("non_positive_holdout_pearson")
    if (
        raw_pearson is not None
        and calibrated_pearson is not None
        and calibrated_pearson <= raw_pearson
    ):
        blockers.append("does_not_improve_raw_local_sum")
    sign_agreement = calibrated_summary["holdout_sign_agreement"]
    if sign_agreement is None or sign_agreement < float(min_sign_agreement):
        blockers.append("weak_holdout_sign_agreement")
    if (
        calibrated_summary["holdout_mae"] is not None
        and mean_summary["holdout_mae"] is not None
        and calibrated_summary["holdout_mae"] >= mean_summary["holdout_mae"]
    ):
        blockers.append("does_not_beat_train_mean_mae")

    return {
        "mode": "decision_value_search_actor_trajectory_calibration",
        "passed": bool(paired_deltas and records),
        "promotable": False,
        "promotion_blockers": [
            "trajectory_calibration_only_not_strength_gate",
            "requires_actor_training_h2h_and_slumbot_confirmation",
        ],
        "trajectory_calibration_passed": not blockers,
        "calibration_blockers": blockers,
        "input_mode": payload.get("mode"),
        "n_pairs": n_pairs,
        "n_train_pairs": int(len(train_idx)),
        "n_holdout_pairs": int(len(holdout_idx)),
        "train_pair_indices": [int(pair_indices[idx]) for idx in train_idx],
        "holdout_pair_indices": [int(pair_indices[idx]) for idx in holdout_idx],
        "feature_names": feature_names,
        "raw_local_sum_baseline": raw_summary,
        "train_mean_baseline": mean_summary,
        "trajectory_calibration": calibrated_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-holdout-pairs", type=int, default=4)
    parser.add_argument("--min-sign-agreement", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.input_json).read_text())
    metrics = analyze_trajectory_calibration(
        payload,
        train_fraction=args.train_fraction,
        ridge_alpha=args.ridge_alpha,
        min_holdout_pairs=args.min_holdout_pairs,
        min_sign_agreement=args.min_sign_agreement,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
