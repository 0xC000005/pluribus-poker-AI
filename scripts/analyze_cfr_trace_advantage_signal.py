#!/usr/bin/env python3
"""Evaluate whether CFR trace counterfactual advantages align with final actions."""

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


def _policy(record: dict[str, Any], field: str) -> np.ndarray:
    if field not in record:
        raise ValueError(f"trace record {record.get('label', '<unknown>')} missing {field}")
    return np.asarray(record[field], dtype=np.float64).reshape(-1)


def _l1(policy: np.ndarray, reference: np.ndarray) -> float:
    if policy.shape != reference.shape:
        raise ValueError("policy shapes do not match")
    return float(np.abs(policy - reference).sum())


def analyze_trace_advantage_signal_from_payload(
    payload: dict[str, Any],
    *,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
) -> dict[str, Any]:
    """Compare low-iteration advantage policy to low/uniform trace baselines."""
    low_by_label = _record_map(payload, low_trace_iteration)
    uniform_by_label = _record_map(payload, uniform_trace_iteration)
    reference_by_label = _record_map(payload, reference_trace_iteration)
    labels = _common_labels(low_by_label, uniform_by_label, reference_by_label)

    low_l1: list[float] = []
    regret_l1: list[float] = []
    advantage_l1: list[float] = []
    uniform_l1: list[float] = []
    low_match: list[bool] = []
    regret_match: list[bool] = []
    advantage_match: list[bool] = []
    uniform_match: list[bool] = []
    records: list[dict[str, Any]] = []

    for label in labels:
        low_record = low_by_label[label]
        reference = _policy(reference_by_label[label], "strategy_policy")
        low_policy = _policy(low_record, "strategy_policy")
        regret_policy = _policy(low_record, "regret_policy")
        advantage_policy = _policy(low_record, "advantage_policy")
        uniform_policy = _policy(uniform_by_label[label], "strategy_policy")

        low_value = _l1(low_policy, reference)
        regret_value = _l1(regret_policy, reference)
        advantage_value = _l1(advantage_policy, reference)
        uniform_value = _l1(uniform_policy, reference)
        low_l1.append(low_value)
        regret_l1.append(regret_value)
        advantage_l1.append(advantage_value)
        uniform_l1.append(uniform_value)
        low_match.append(_top_match(low_policy, reference))
        regret_match.append(_top_match(regret_policy, reference))
        advantage_match.append(_top_match(advantage_policy, reference))
        uniform_match.append(_top_match(uniform_policy, reference))
        records.append(
            {
                "label": label,
                "low_l1_to_reference": round(low_value, 8),
                "regret_l1_to_reference": round(regret_value, 8),
                "advantage_l1_to_reference": round(advantage_value, 8),
                "uniform_l1_to_reference": round(uniform_value, 8),
                "low_top_matches_reference": bool(low_match[-1]),
                "regret_top_matches_reference": bool(regret_match[-1]),
                "advantage_top_matches_reference": bool(advantage_match[-1]),
                "uniform_top_matches_reference": bool(uniform_match[-1]),
            }
        )

    mean_advantage = _mean(advantage_l1)
    mean_low = _mean(low_l1)
    mean_uniform = _mean(uniform_l1)
    advantage_top_rate = _rate(advantage_match)
    uniform_top_rate = _rate(uniform_match)
    passed = bool(
        mean_advantage < mean_low
        and mean_advantage < mean_uniform
        and advantage_top_rate >= uniform_top_rate
    )
    return {
        "mode": "cfr_trace_advantage_signal",
        "passed": passed,
        "promotion": False,
        "low_trace_iteration": int(low_trace_iteration),
        "uniform_trace_iteration": int(uniform_trace_iteration),
        "reference_trace_iteration": int(reference_trace_iteration),
        "n_eval": int(len(labels)),
        "mean_low_l1_to_reference": mean_low,
        "mean_regret_l1_to_reference": _mean(regret_l1),
        "mean_advantage_l1_to_reference": mean_advantage,
        "mean_uniform_l1_to_reference": mean_uniform,
        "low_top_match_rate": _rate(low_match),
        "regret_top_match_rate": _rate(regret_match),
        "advantage_top_match_rate": advantage_top_rate,
        "uniform_top_match_rate": uniform_top_rate,
        "records": records,
    }


def analyze_trace_advantage_signal(
    *,
    trace_json: str | Path,
    low_trace_iteration: int = 5,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
) -> dict[str, Any]:
    metrics = analyze_trace_advantage_signal_from_payload(
        _load_payload(trace_json),
        low_trace_iteration=low_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
    )
    metrics["trace_json"] = str(trace_json)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--low-trace-iteration", type=int, default=5)
    parser.add_argument("--uniform-trace-iteration", type=int, default=10)
    parser.add_argument("--reference-trace-iteration", type=int, default=24)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = analyze_trace_advantage_signal(
        trace_json=args.trace_json,
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
