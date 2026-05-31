"""Audit XDO-lite information-state response targets."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import numpy as np


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 0.0


def _target_entropy(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy)


def _action_value_margin(record: dict[str, Any], oracle_action: str) -> float:
    values = record.get("action_values", {})
    if not isinstance(values, dict) or oracle_action not in values:
        return 0.0
    oracle_value = _finite_float(values.get(oracle_action))
    alternatives = [
        _finite_float(value)
        for action, value in values.items()
        if str(action) != str(oracle_action)
    ]
    if not alternatives:
        return 0.0
    return float(oracle_value - max(alternatives))


def _summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [
        _finite_float(
            record.get(
                "oracle_gap",
                _finite_float(record.get("oracle_action_value"))
                - _finite_float(record.get("parent_action_value")),
            )
        )
        for record in records
    ]
    margins = [
        _action_value_margin(record, str(record.get("oracle_action", "unknown")))
        for record in records
    ]
    return {
        "n_targets": int(len(records)),
        "mean_oracle_gap": _mean(gaps),
        "mean_action_value_margin": _mean(margins),
    }


def audit_xdo_target_records(
    records: Sequence[dict[str, Any]],
    *,
    min_targets: int = 16,
    max_top_action_fraction: float = 0.75,
    min_mean_gap: float = 0.0,
    min_holdout_mean_gap: float = 0.0,
) -> dict[str, Any]:
    """Validate whether exact/search targets are healthy enough to train from."""
    rows = [dict(record) for record in records]
    counts = Counter(str(record.get("oracle_action", "unknown")) for record in rows)
    n_targets = len(rows)
    max_fraction = (
        max(counts.values()) / float(n_targets)
        if n_targets > 0 and counts
        else 0.0
    )
    train_rows = [row for row in rows if int(row.get("root_idx", 0)) % 2 == 0]
    holdout_rows = [row for row in rows if int(row.get("root_idx", 0)) % 2 == 1]
    all_summary = _summarize_split(rows)
    train_summary = _summarize_split(train_rows)
    holdout_summary = _summarize_split(holdout_rows)
    n_truncated = int(sum(int(row.get("n_truncated_rollouts", 0)) for row in rows))

    blockers: list[str] = []
    if n_targets < int(min_targets):
        blockers.append("insufficient_targets")
    if max_fraction > float(max_top_action_fraction):
        blockers.append("target_action_collapse")
    if all_summary["mean_oracle_gap"] < float(min_mean_gap):
        blockers.append("mean_gap_below_threshold")
    if holdout_summary["n_targets"] <= 0:
        blockers.append("missing_holdout_targets")
    elif holdout_summary["mean_oracle_gap"] < float(min_holdout_mean_gap):
        blockers.append("holdout_gap_below_threshold")
    if n_truncated > 0:
        blockers.append("truncated_rollouts")

    return {
        "algorithm": "xdo_lite_information_state_target_audit",
        "passed": bool(not blockers),
        "promotion": False,
        "promotable": False,
        "blockers": blockers,
        "n_targets": int(n_targets),
        "min_targets": int(min_targets),
        "target_action_counts": dict(sorted(counts.items())),
        "max_target_top_action_fraction": float(max_fraction),
        "max_allowed_top_action_fraction": float(max_top_action_fraction),
        "target_action_entropy": _target_entropy(counts),
        "mean_oracle_gap": float(all_summary["mean_oracle_gap"]),
        "mean_action_value_margin": float(all_summary["mean_action_value_margin"]),
        "min_mean_gap": float(min_mean_gap),
        "min_holdout_mean_gap": float(min_holdout_mean_gap),
        "n_truncated_rollouts": int(n_truncated),
        "train": train_summary,
        "holdout": holdout_summary,
        "promotion_blockers": [
            "target_audit_only",
            "requires_neural_consumer_parent_population_h2h",
            "slumbot_heldout_confirmation_required",
        ],
    }
