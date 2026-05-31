#!/usr/bin/env python3
"""Analyze whether search-actor local value gains predict H2H transfer."""

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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize_records(
    *,
    paired_deltas: list[float],
    records: list[dict[str, Any]],
    disagreement_only: bool,
) -> dict[str, Any]:
    pair_local: dict[int, float] = defaultdict(float)
    n_decisions = 0
    for record in records:
        if disagreement_only and record.get("deployed_action") == record.get("search_action"):
            continue
        value = record.get("local_search_minus_deployed_value")
        if value is None:
            continue
        pair_idx = int(record["duplicate_pair"])
        if pair_idx < 0 or pair_idx >= len(paired_deltas):
            continue
        pair_local[pair_idx] += float(value)
        n_decisions += 1

    matched_indices = sorted(pair_local)
    local_sums = [float(pair_local[idx]) for idx in matched_indices]
    h2h = [float(paired_deltas[idx]) for idx in matched_indices]
    return {
        "n_decisions": int(n_decisions),
        "n_pairs": int(len(matched_indices)),
        "mean_pair_local_delta_sum": _mean(local_sums),
        "mean_pair_h2h_delta": _mean(h2h),
        "pair_local_delta_sum_to_h2h_delta_pearson": _pearson_or_none(
            local_sums,
            h2h,
        ),
        "sign_agreement_fraction": _sign_agreement(local_sums, h2h),
        "positive_local_negative_h2h_pairs": int(
            sum(local > 0.0 and delta < 0.0 for local, delta in zip(local_sums, h2h, strict=True))
        ),
        "negative_local_positive_h2h_pairs": int(
            sum(local < 0.0 and delta > 0.0 for local, delta in zip(local_sums, h2h, strict=True))
        ),
    }


def analyze_search_actor_transfer(payload: dict[str, Any]) -> dict[str, Any]:
    paired_deltas = [float(value) for value in payload.get("paired_deltas", [])]
    records = list(payload.get("decision_records") or [])
    all_summary = _summarize_records(
        paired_deltas=paired_deltas,
        records=records,
        disagreement_only=False,
    )
    disagreement_summary = _summarize_records(
        paired_deltas=paired_deltas,
        records=records,
        disagreement_only=True,
    )
    return {
        "mode": "decision_value_search_actor_transfer_analysis",
        "passed": bool(paired_deltas and records),
        "promotable": False,
        "promotion_blockers": [
            "attribution_only_not_strength_gate",
            "requires_positive_h2h_confidence_and_slumbot_confirmation",
        ],
        "input_mode": payload.get("mode"),
        "n_duplicate_pairs": int(len(paired_deltas)),
        "n_pairs_with_decision_records": int(all_summary["n_pairs"]),
        "h2h_avg_chips_per_hand": payload.get("avg_chips_per_hand"),
        "h2h_lower95_chips_per_hand": payload.get("paired_delta_lower95_chips_per_hand"),
        "all_decisions": all_summary,
        "disagreement_decisions": disagreement_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(Path(args.input_json).read_text())
    metrics = analyze_search_actor_transfer(payload)
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
