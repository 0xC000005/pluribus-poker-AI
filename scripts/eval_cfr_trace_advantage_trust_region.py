#!/usr/bin/env python3
"""Evaluate calibrated trust-region updates from CFR trace advantages."""

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

from analyze_cfr_trace_predictor import _load_payload  # noqa: E402
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from train_cfr_trace_policy_residual import _mean, _rate, _record_map, _top_match  # noqa: E402


def _common_labels(*maps: dict[str, dict[str, Any]]) -> list[str]:
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    labels = sorted(common)
    if not labels:
        raise ValueError("no common labels across requested trace iterations")
    return labels


def _legal_mask(record: dict[str, Any], action_dim: int) -> np.ndarray:
    mask = np.zeros(action_dim, dtype=np.float64)
    for action in record.get("legal_actions", ()):
        action = int(action)
        if 0 <= action < action_dim:
            mask[action] = 1.0
    if float(mask.sum()) <= 0.0:
        raise ValueError(f"record {record.get('label', '<unknown>')} has no legal actions")
    return mask


def _normalize_policy(policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    masked = np.asarray(policy, dtype=np.float64) * legal_mask
    total = float(masked.sum())
    if total <= 1e-12:
        return legal_mask / float(legal_mask.sum())
    return masked / total


def _center_scale_advantage(
    advantage: np.ndarray,
    legal_mask: np.ndarray,
    base_policy: np.ndarray,
) -> np.ndarray:
    adv = np.asarray(advantage, dtype=np.float64).reshape(-1)
    if adv.shape != legal_mask.shape:
        adv = np.zeros_like(legal_mask)
    legal = legal_mask > 0
    centered = np.zeros_like(legal_mask)
    expected = float(np.sum(base_policy[legal] * adv[legal]))
    centered[legal] = adv[legal] - expected
    scale = max(float(np.max(np.abs(centered[legal]))), 1e-8)
    centered[legal] /= scale
    return centered


def _trust_region_policy(record: dict[str, Any], *, eta: float, min_base_prob: float) -> np.ndarray:
    low_policy = np.asarray(record["strategy_policy"], dtype=np.float64).reshape(-1)
    action_dim = int(low_policy.shape[0])
    legal_mask = _legal_mask(record, action_dim)
    legal_uniform = legal_mask / float(legal_mask.sum())
    base_policy = _normalize_policy(
        np.maximum(low_policy, float(min_base_prob) * legal_uniform),
        legal_mask,
    )
    advantage = _center_scale_advantage(
        np.asarray(record.get("counterfactual_advantage", []), dtype=np.float64),
        legal_mask,
        base_policy,
    )
    logits = np.log(np.clip(base_policy, 1e-12, 1.0)) + float(eta) * advantage
    logits = np.where(legal_mask > 0, logits, -1e9)
    logits = logits - float(np.max(logits[legal_mask > 0]))
    probs = np.exp(logits) * legal_mask
    return probs / max(float(probs.sum()), 1e-12)


def _evaluate_eta(
    payload: dict[str, Any],
    *,
    eta: float,
    low_trace_iteration: int,
    uniform_trace_iteration: int,
    reference_trace_iteration: int,
    min_base_prob: float,
) -> dict[str, Any]:
    low_by_label = _record_map(payload, low_trace_iteration)
    uniform_by_label = _record_map(payload, uniform_trace_iteration)
    reference_by_label = _record_map(payload, reference_trace_iteration)
    labels = _common_labels(low_by_label, uniform_by_label, reference_by_label)

    update_l1: list[float] = []
    low_l1: list[float] = []
    uniform_l1: list[float] = []
    update_match: list[bool] = []
    low_match: list[bool] = []
    uniform_match: list[bool] = []
    records: list[dict[str, Any]] = []
    for label in labels:
        low_record = low_by_label[label]
        update_policy = _trust_region_policy(
            low_record,
            eta=eta,
            min_base_prob=min_base_prob,
        )
        low_policy = np.asarray(low_record["strategy_policy"], dtype=np.float64)
        uniform_policy = np.asarray(uniform_by_label[label]["strategy_policy"], dtype=np.float64)
        reference = np.asarray(reference_by_label[label]["strategy_policy"], dtype=np.float64)
        update_value = float(np.abs(update_policy - reference).sum())
        low_value = float(np.abs(low_policy - reference).sum())
        uniform_value = float(np.abs(uniform_policy - reference).sum())
        update_l1.append(update_value)
        low_l1.append(low_value)
        uniform_l1.append(uniform_value)
        update_match.append(_top_match(update_policy, reference))
        low_match.append(_top_match(low_policy, reference))
        uniform_match.append(_top_match(uniform_policy, reference))
        records.append(
            {
                "label": label,
                "update_l1_to_reference": round(update_value, 8),
                "low_l1_to_reference": round(low_value, 8),
                "uniform_l1_to_reference": round(uniform_value, 8),
                "update_top_matches_reference": bool(update_match[-1]),
                "low_top_matches_reference": bool(low_match[-1]),
                "uniform_top_matches_reference": bool(uniform_match[-1]),
                "update_policy": update_policy.round(8).tolist(),
            }
        )
    mean_update = _mean(update_l1)
    mean_low = _mean(low_l1)
    mean_uniform = _mean(uniform_l1)
    return {
        "eta": float(eta),
        "n_eval": int(len(labels)),
        "mean_update_l1_to_reference": mean_update,
        "mean_low_l1_to_reference": mean_low,
        "mean_uniform_l1_to_reference": mean_uniform,
        "update_top_match_rate": _rate(update_match),
        "low_top_match_rate": _rate(low_match),
        "uniform_top_match_rate": _rate(uniform_match),
        "passed": bool(
            mean_update < mean_low
            and mean_update < mean_uniform
            and _rate(update_match) >= _rate(uniform_match)
        ),
        "records": records,
    }


def evaluate_advantage_trust_region(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    etas: list[float],
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    min_base_prob: float = 1e-4,
) -> dict[str, Any]:
    if not etas:
        raise ValueError("etas must contain at least one value")
    train_payload = _load_payload(train_trace_json)
    holdout_payload = _load_payload(holdout_trace_json)
    train_results = [
        _evaluate_eta(
            train_payload,
            eta=eta,
            low_trace_iteration=low_trace_iteration,
            uniform_trace_iteration=uniform_trace_iteration,
            reference_trace_iteration=reference_trace_iteration,
            min_base_prob=min_base_prob,
        )
        for eta in etas
    ]
    selected = min(
        train_results,
        key=lambda result: (
            result["mean_update_l1_to_reference"],
            -result["update_top_match_rate"],
            abs(result["eta"]),
        ),
    )
    holdout = _evaluate_eta(
        holdout_payload,
        eta=float(selected["eta"]),
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        min_base_prob=min_base_prob,
    )
    return {
        "mode": "cfr_trace_advantage_trust_region",
        "promotion": False,
        "passed": bool(holdout["passed"] and float(selected["eta"]) != 0.0),
        "train_trace_json": str(train_trace_json),
        "holdout_trace_json": str(holdout_trace_json),
        "low_trace_iteration": int(low_trace_iteration),
        "uniform_trace_iteration": int(uniform_trace_iteration),
        "reference_trace_iteration": int(reference_trace_iteration),
        "min_base_prob": float(min_base_prob),
        "etas": [float(eta) for eta in etas],
        "selected_eta": float(selected["eta"]),
        "train_selected": selected,
        "train_grid": [
            {key: value for key, value in result.items() if key != "records"}
            for result in train_results
        ],
        "holdout": holdout,
    }


def _parse_etas(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one eta is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-json", required=True)
    parser.add_argument("--holdout-trace-json", required=True)
    parser.add_argument(
        "--etas",
        type=_parse_etas,
        default=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument("--low-trace-iteration", type=int, default=5)
    parser.add_argument("--uniform-trace-iteration", type=int, default=10)
    parser.add_argument("--reference-trace-iteration", type=int, default=24)
    parser.add_argument("--min-base-prob", type=float, default=1e-4)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = evaluate_advantage_trust_region(
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
