"""Summarize duplicate-swapped H2H league artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_record(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "candidate_checkpoint",
        "baseline_checkpoint",
        "mean_candidate_payoff",
        "lower95_candidate_payoff",
    ):
        if key not in payload:
            raise ValueError(f"{path} missing required H2H field: {key}")
    return payload


def summarize_h2h_league(record_paths: list[str], *, min_lower95: float = 0.0) -> dict[str, Any]:
    records = [_load_record(path) for path in record_paths]
    if not records:
        return {
            "algorithm": "poker_h2h_league_summary",
            "passed": False,
            "candidate_checkpoint": None,
            "n_records": 0,
            "errors": ["no_records"],
            "records": [],
        }

    candidate_checkpoints = {str(record["candidate_checkpoint"]) for record in records}
    if len(candidate_checkpoints) != 1:
        raise ValueError("all H2H records must use the same candidate_checkpoint")

    baselines = [str(record["baseline_checkpoint"]) for record in records]
    lower95_values = [float(record["lower95_candidate_payoff"]) for record in records]
    mean_values = [float(record["mean_candidate_payoff"]) for record in records]
    failed_baselines = [
        baseline
        for baseline, lower95 in zip(baselines, lower95_values, strict=True)
        if lower95 <= float(min_lower95)
    ]
    passed = not failed_baselines
    return {
        "algorithm": "poker_h2h_league_summary",
        "passed": bool(passed),
        "candidate_checkpoint": next(iter(candidate_checkpoints)),
        "n_records": len(records),
        "min_lower95_threshold": float(min_lower95),
        "worst_lower95_candidate_payoff": min(lower95_values),
        "mean_lower95_candidate_payoff": sum(lower95_values) / float(len(lower95_values)),
        "mean_candidate_payoff": sum(mean_values) / float(len(mean_values)),
        "failed_baselines": failed_baselines,
        "baselines": baselines,
        "records": records,
        "errors": [],
    }
