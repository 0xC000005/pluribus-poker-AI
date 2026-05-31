"""Decision-impact guard for collapsed deployed policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_action_collapse(
    root_rollout_metrics: dict[str, Any],
    *,
    max_top_action_fraction: float = 0.75,
    min_distinct_actions: int = 2,
) -> dict[str, Any]:
    """Return a gate result for selected-action concentration."""
    counts = {
        str(action): int(count)
        for action, count in (root_rollout_metrics.get("selected_action_counts") or {}).items()
        if int(count) > 0
    }
    total = int(sum(counts.values()))
    failures: list[str] = []
    if total <= 0:
        failures.append("selected_action_counts is empty")
        top_action = ""
        top_count = 0
        top_fraction = 0.0
    else:
        top_action, top_count = max(counts.items(), key=lambda item: item[1])
        top_fraction = float(top_count) / float(total)
        if top_fraction > float(max_top_action_fraction):
            failures.append(
                "top_action_fraction "
                f"{top_fraction:.6f} > {float(max_top_action_fraction):.6f}"
            )
    distinct_actions = int(len(counts))
    if distinct_actions < int(min_distinct_actions):
        failures.append(
            f"distinct_actions {distinct_actions} < {int(min_distinct_actions)}"
        )
    return {
        "passed": not failures,
        "mode": "policy_action_collapse_gate",
        "failures": failures,
        "selected_action_counts": counts,
        "n_selected_actions": total,
        "distinct_actions": distinct_actions,
        "top_action": top_action,
        "top_action_count": int(top_count),
        "top_action_fraction": float(top_fraction),
        "max_top_action_fraction": float(max_top_action_fraction),
        "min_distinct_actions": int(min_distinct_actions),
    }


def evaluate_action_collapse_file(
    input_json: str | Path,
    *,
    max_top_action_fraction: float = 0.75,
    min_distinct_actions: int = 2,
) -> dict[str, Any]:
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    metrics = evaluate_action_collapse(
        payload,
        max_top_action_fraction=max_top_action_fraction,
        min_distinct_actions=min_distinct_actions,
    )
    metrics["input_json"] = str(input_json)
    return metrics
