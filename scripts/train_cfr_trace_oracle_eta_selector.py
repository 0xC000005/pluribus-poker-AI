#!/usr/bin/env python3
"""Fit a decision-focused selector for CFR trace mirror-update step sizes."""

from __future__ import annotations

import argparse
import json
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

from analyze_cfr_trace_predictor import _load_payload, _trace_feature_row  # noqa: E402
from analyze_cfr_trace_sequence_predictor import (  # noqa: E402
    _records_by_label_iteration,
    _sequence_feature_row,
)
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from train_cfr_trace_delta_mlp import _advantage_context_row  # noqa: E402
from train_cfr_trace_mirror_update_operator import mirror_update_policy  # noqa: E402
from train_cfr_trace_policy_residual import (  # noqa: E402
    _feature_row,
    _fit_ridge,
    _mean,
    _rate,
    _record_map,
    _standardize,
    _top_match,
)


def _l1(policy_a: Iterable[float], policy_b: Iterable[float]) -> float:
    return float(np.abs(np.asarray(policy_a, dtype=np.float64) - np.asarray(policy_b, dtype=np.float64)).sum())


def _common_labels(*maps: dict[str, dict[str, Any]]) -> list[str]:
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    labels = sorted(common)
    if not labels:
        raise ValueError("no common labels across requested trace iterations")
    return labels


def _labels_with_sequence(
    payload: dict[str, Any],
    labels: list[str],
    *,
    low_trace_iteration: int,
) -> list[str]:
    by_label = _records_by_label_iteration(payload)
    required = set(range(0, int(low_trace_iteration) + 1))
    out = [
        label
        for label in labels
        if label in by_label and required.issubset(set(by_label[label]))
    ]
    if not out:
        raise ValueError("no labels contain the requested early trace sequence")
    return out


def _oracle_rows(
    payload: dict[str, Any],
    *,
    etas: list[float],
    low_trace_iteration: int,
    uniform_trace_iteration: int,
    reference_trace_iteration: int,
    min_base_prob: float,
) -> tuple[list[str], dict[str, dict[str, Any]], np.ndarray]:
    low_by_label = _record_map(payload, low_trace_iteration)
    uniform_by_label = _record_map(payload, uniform_trace_iteration)
    reference_by_label = _record_map(payload, reference_trace_iteration)
    labels = _common_labels(low_by_label, uniform_by_label, reference_by_label)
    rows: dict[str, dict[str, Any]] = {}
    losses: list[list[float]] = []
    for label in labels:
        low_record = low_by_label[label]
        reference = np.asarray(reference_by_label[label]["strategy_policy"], dtype=np.float64)
        low_policy = np.asarray(low_record["strategy_policy"], dtype=np.float64)
        uniform_policy = np.asarray(uniform_by_label[label]["strategy_policy"], dtype=np.float64)
        eta_losses: list[float] = []
        eta_matches: list[bool] = []
        eta_policies: list[list[float]] = []
        for eta in etas:
            update_policy = mirror_update_policy(low_record, eta=float(eta), min_base_prob=min_base_prob)
            eta_losses.append(_l1(update_policy, reference))
            eta_matches.append(_top_match(update_policy, reference))
            eta_policies.append(update_policy.round(8).tolist())
        best_idx = int(np.argmin(np.asarray(eta_losses, dtype=np.float64)))
        rows[label] = {
            "label": label,
            "oracle_eta": float(etas[best_idx]),
            "oracle_eta_index": best_idx,
            "oracle_l1_to_reference": float(eta_losses[best_idx]),
            "oracle_top_matches_reference": bool(eta_matches[best_idx]),
            "low_l1_to_reference": _l1(low_policy, reference),
            "uniform_l1_to_reference": _l1(uniform_policy, reference),
            "low_top_matches_reference": _top_match(low_policy, reference),
            "uniform_top_matches_reference": _top_match(uniform_policy, reference),
            "eta_losses": [float(value) for value in eta_losses],
            "eta_top_matches_reference": [bool(value) for value in eta_matches],
            "eta_policies": eta_policies,
        }
        losses.append(eta_losses)
    return labels, rows, np.asarray(losses, dtype=np.float64)


def oracle_eta_targets(
    payload: dict[str, Any],
    *,
    etas: list[float],
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    min_base_prob: float = 1e-4,
) -> dict[str, Any]:
    if not etas:
        raise ValueError("etas must contain at least one value")
    etas = [float(eta) for eta in etas]
    labels, rows, _losses = _oracle_rows(
        payload,
        etas=etas,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        min_base_prob=min_base_prob,
    )
    oracle_l1 = [float(rows[label]["oracle_l1_to_reference"]) for label in labels]
    low_l1 = [float(rows[label]["low_l1_to_reference"]) for label in labels]
    uniform_l1 = [float(rows[label]["uniform_l1_to_reference"]) for label in labels]
    oracle_match = [bool(rows[label]["oracle_top_matches_reference"]) for label in labels]
    uniform_match = [bool(rows[label]["uniform_top_matches_reference"]) for label in labels]
    return {
        "labels": labels,
        "etas": etas,
        "n_eval": int(len(labels)),
        "oracle_eta_by_label": {label: float(rows[label]["oracle_eta"]) for label in labels},
        "mean_oracle_l1_to_reference": _mean(oracle_l1),
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_uniform_l1_to_reference": _mean(uniform_l1),
        "oracle_top_match_rate": _rate(oracle_match),
        "uniform_top_match_rate": _rate(uniform_match),
        "oracle_beats_uniform_count": int(sum(o < u for o, u in zip(oracle_l1, uniform_l1))),
        "oracle_passed": bool(
            _mean(oracle_l1) < _mean(low_l1)
            and _mean(oracle_l1) < _mean(uniform_l1)
            and _rate(oracle_match) >= _rate(uniform_match)
        ),
        "records": [rows[label] for label in labels],
    }


def _feature_matrix(
    payload: dict[str, Any],
    labels: list[str],
    *,
    low_trace_iteration: int,
) -> np.ndarray:
    by_label = _records_by_label_iteration(payload)
    labels = _labels_with_sequence(payload, labels, low_trace_iteration=low_trace_iteration)
    action_dim = len(by_label[labels[0]][int(low_trace_iteration)]["strategy_policy"])
    features: list[np.ndarray] = []
    for label in labels:
        by_iteration = by_label[label]
        sequence = [by_iteration[iteration] for iteration in range(0, int(low_trace_iteration) + 1)]
        low = by_iteration[int(low_trace_iteration)]
        public_context = np.asarray(low.get("public_belief_features", ()), dtype=np.float64).reshape(-1)
        features.append(
            np.concatenate(
                [
                    np.asarray(_trace_feature_row(low), dtype=np.float64),
                    np.asarray(_sequence_feature_row(sequence), dtype=np.float64),
                    _feature_row(low, action_dim),
                    public_context,
                    _advantage_context_row(low, action_dim),
                ]
            )
        )
    return np.asarray(features, dtype=np.float64)


def _select_and_evaluate(
    labels: list[str],
    rows: dict[str, dict[str, Any]],
    predicted_losses: np.ndarray,
    etas: list[float],
) -> dict[str, Any]:
    selected_l1: list[float] = []
    oracle_l1: list[float] = []
    low_l1: list[float] = []
    uniform_l1: list[float] = []
    selected_match: list[bool] = []
    oracle_match: list[bool] = []
    uniform_match: list[bool] = []
    selected_oracle_index_match: list[bool] = []
    records: list[dict[str, Any]] = []
    for row_idx, label in enumerate(labels):
        row = rows[label]
        selected_idx = int(np.argmin(predicted_losses[row_idx]))
        selected_l1_value = float(row["eta_losses"][selected_idx])
        selected_match_value = bool(row["eta_top_matches_reference"][selected_idx])
        selected_l1.append(selected_l1_value)
        oracle_l1.append(float(row["oracle_l1_to_reference"]))
        low_l1.append(float(row["low_l1_to_reference"]))
        uniform_l1.append(float(row["uniform_l1_to_reference"]))
        selected_match.append(selected_match_value)
        oracle_match.append(bool(row["oracle_top_matches_reference"]))
        uniform_match.append(bool(row["uniform_top_matches_reference"]))
        selected_oracle_index_match.append(selected_idx == int(row["oracle_eta_index"]))
        records.append(
            {
                "label": label,
                "selected_eta": float(etas[selected_idx]),
                "oracle_eta": float(row["oracle_eta"]),
                "selected_l1_to_reference": round(selected_l1_value, 8),
                "oracle_l1_to_reference": round(float(row["oracle_l1_to_reference"]), 8),
                "low_l1_to_reference": round(float(row["low_l1_to_reference"]), 8),
                "uniform_l1_to_reference": round(float(row["uniform_l1_to_reference"]), 8),
                "selected_top_matches_reference": selected_match_value,
                "oracle_top_matches_reference": bool(row["oracle_top_matches_reference"]),
                "uniform_top_matches_reference": bool(row["uniform_top_matches_reference"]),
                "predicted_eta_losses": predicted_losses[row_idx].round(8).tolist(),
                "true_eta_losses": [round(float(value), 8) for value in row["eta_losses"]],
            }
        )
    mean_selected = _mean(selected_l1)
    mean_low = _mean(low_l1)
    mean_uniform = _mean(uniform_l1)
    return {
        "decision_passed": bool(
            mean_selected < mean_low
            and mean_selected < mean_uniform
            and _rate(selected_match) >= _rate(uniform_match)
        ),
        "mean_selected_l1_to_reference": mean_selected,
        "mean_oracle_l1_to_reference": _mean(oracle_l1),
        "mean_low_l1_to_reference": mean_low,
        "mean_uniform_l1_to_reference": mean_uniform,
        "selected_top_match_rate": _rate(selected_match),
        "oracle_top_match_rate": _rate(oracle_match),
        "uniform_top_match_rate": _rate(uniform_match),
        "selected_oracle_eta_match_rate": _rate(selected_oracle_index_match),
        "records": records,
    }


def fit_oracle_eta_selector_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    etas: list[float],
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    min_base_prob: float = 1e-4,
) -> dict[str, Any]:
    if not etas:
        raise ValueError("etas must contain at least one value")
    etas = [float(eta) for eta in etas]
    raw_train_labels, train_rows, train_losses = _oracle_rows(
        train_payload,
        etas=etas,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        min_base_prob=min_base_prob,
    )
    raw_train_index = {label: idx for idx, label in enumerate(raw_train_labels)}
    train_labels = _labels_with_sequence(
        train_payload,
        raw_train_labels,
        low_trace_iteration=low_trace_iteration,
    )
    x_train = _feature_matrix(train_payload, train_labels, low_trace_iteration=low_trace_iteration)
    y_train = np.asarray([train_losses[raw_train_index[label]] for label in train_labels], dtype=np.float64)
    x_train, feature_mean, feature_std = _standardize(x_train)
    weights = _fit_ridge(x_train, y_train)
    train_pred = x_train @ weights
    train_eval = _select_and_evaluate(train_labels, train_rows, train_pred, etas)

    holdout_labels, holdout_rows, _holdout_losses = _oracle_rows(
        holdout_payload,
        etas=etas,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        min_base_prob=min_base_prob,
    )
    holdout_labels = _labels_with_sequence(
        holdout_payload,
        holdout_labels,
        low_trace_iteration=low_trace_iteration,
    )
    x_holdout = _feature_matrix(holdout_payload, holdout_labels, low_trace_iteration=low_trace_iteration)
    x_holdout, _, _ = _standardize(x_holdout, mean=feature_mean, std=feature_std)
    holdout_pred = x_holdout @ weights
    holdout_eval = _select_and_evaluate(holdout_labels, holdout_rows, holdout_pred, etas)
    target_mae = float(np.mean(np.abs(holdout_pred - np.asarray([holdout_rows[label]["eta_losses"] for label in holdout_labels]))))
    return {
        "mode": "cfr_trace_oracle_eta_selector",
        "promotion": False,
        "passed": bool(holdout_eval["decision_passed"]),
        "low_trace_iteration": int(low_trace_iteration),
        "uniform_trace_iteration": int(uniform_trace_iteration),
        "reference_trace_iteration": int(reference_trace_iteration),
        "min_base_prob": float(min_base_prob),
        "etas": etas,
        "n_train": int(len(train_labels)),
        "n_holdout": int(len(holdout_labels)),
        "train": train_eval,
        "holdout": holdout_eval,
        "model": {
            "type": "ridge_per_eta_loss_selector",
            "ridge": 1e-3,
            "holdout_loss_mae": round(target_mae, 8),
            "feature_mean": feature_mean.round(10).tolist(),
            "feature_std": feature_std.round(10).tolist(),
            "weights_shape": list(weights.shape),
        },
    }


def fit_oracle_eta_selector(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    etas: list[float],
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    min_base_prob: float = 1e-4,
) -> dict[str, Any]:
    metrics = fit_oracle_eta_selector_from_payloads(
        _load_payload(train_trace_json),
        _load_payload(holdout_trace_json),
        etas=etas,
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        min_base_prob=min_base_prob,
    )
    metrics["train_trace_json"] = str(train_trace_json)
    metrics["holdout_trace_json"] = str(holdout_trace_json)
    return metrics


def _parse_etas(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one eta is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-json", required=True)
    parser.add_argument("--holdout-trace-json", required=True)
    parser.add_argument("--etas", type=_parse_etas, default=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--low-trace-iteration", type=int, default=5)
    parser.add_argument("--uniform-trace-iteration", type=int, default=10)
    parser.add_argument("--reference-trace-iteration", type=int, default=24)
    parser.add_argument("--min-base-prob", type=float, default=1e-4)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_oracle_eta_selector(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        etas=args.etas,
        low_trace_iteration=args.low_trace_iteration,
        uniform_trace_iteration=args.uniform_trace_iteration,
        reference_trace_iteration=args.reference_trace_iteration,
        min_base_prob=args.min_base_prob,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
