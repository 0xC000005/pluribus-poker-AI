#!/usr/bin/env python3
"""Evaluate an exported selective solver-budget policy on fixed profile records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from poker_ai.research.solver_budget_policy import (
    load_selective_policy,
    score_selective_policy,
    solver_native_vector,
)


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _strategy(profile: dict[str, Any]) -> np.ndarray:
    return np.asarray(profile["strategy"], dtype=np.float64)


def _l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).sum())


def _policy_score(policy: dict[str, Any], candidate: dict[str, Any]) -> float:
    width = int(policy["feature_dim"]) - 8
    feature = solver_native_vector(
        candidate["strategy"],
        action=int(candidate["action"]),
        width=width,
    )
    return score_selective_policy(policy, feature)


def evaluate_budget_policy_integration(
    profile_metrics: dict[str, Any],
    policy: dict[str, Any],
    *,
    reference_profile: str = "live",
    candidate_profile: str = "fast-live",
    min_evaluated: int = 1,
) -> dict[str, Any]:
    threshold = float(policy["score_threshold"])
    rows: list[dict[str, Any]] = []
    illegal_masses: list[float] = []
    for record in profile_metrics.get("records", []):
        profiles = record.get("profiles", {})
        if not record.get("passed") or reference_profile not in profiles or candidate_profile not in profiles:
            continue
        reference = profiles[reference_profile]
        candidate = profiles[candidate_profile]
        reference_strategy = _strategy(reference)
        candidate_strategy = _strategy(candidate)
        score = _policy_score(policy, candidate)
        escalated = bool(score >= threshold)
        final = reference if escalated else candidate
        final_strategy = reference_strategy if escalated else candidate_strategy
        candidate_latency = float(candidate["latency_ms"])
        reference_latency = float(reference["latency_ms"])
        rows.append(
            {
                "label": str(record.get("label", "")),
                "score": round(float(score), 8),
                "escalated": escalated,
                "fast_l1": _l1(candidate_strategy, reference_strategy),
                "selective_l1": _l1(final_strategy, reference_strategy),
                "fast_action_match": int(candidate["action"]) == int(reference["action"]),
                "selective_action_match": int(final["action"]) == int(reference["action"]),
                "fast_latency_ms": candidate_latency,
                "live_latency_ms": reference_latency,
                "selective_online_latency_ms": (
                    candidate_latency + reference_latency if escalated else candidate_latency
                ),
            }
        )
        illegal_masses.extend([
            float(candidate.get("illegal_mass", 0.0)),
            float(reference.get("illegal_mass", 0.0)),
        ])

    fast_l1 = _mean([row["fast_l1"] for row in rows])
    selective_l1 = _mean([row["selective_l1"] for row in rows])
    fast_action_agreement = _mean([float(row["fast_action_match"]) for row in rows])
    selective_action_agreement = _mean([float(row["selective_action_match"]) for row in rows])
    fast_latency = _mean([row["fast_latency_ms"] for row in rows])
    live_latency = _mean([row["live_latency_ms"] for row in rows])
    selective_latency = _mean([row["selective_online_latency_ms"] for row in rows])
    max_illegal_mass = max(illegal_masses) if illegal_masses else 0.0

    return {
        "mode": "solver_budget_policy_integration",
        "reference_profile": reference_profile,
        "candidate_profile": candidate_profile,
        "policy_type": policy.get("policy_type"),
        "feature_source": policy.get("feature_source"),
        "score_threshold": threshold,
        "n_evaluated": int(len(rows)),
        "min_evaluated": int(min_evaluated),
        "n_escalated": int(sum(1 for row in rows if row["escalated"])),
        "escalation_rate": _mean([float(row["escalated"]) for row in rows]),
        "fast_mean_l1_to_live": fast_l1,
        "selective_mean_l1_to_live": selective_l1,
        "fast_action_agreement": fast_action_agreement,
        "selective_action_agreement": selective_action_agreement,
        "fast_mean_latency_ms": fast_latency,
        "selective_online_mean_latency_ms": selective_latency,
        "live_mean_latency_ms": live_latency,
        "selective_latency_ratio_to_fast": round(float(selective_latency / max(fast_latency, 1e-9)), 8),
        "selective_latency_ratio_to_live": round(float(selective_latency / max(live_latency, 1e-9)), 8),
        "max_illegal_mass": round(float(max_illegal_mass), 8),
        "escalated_labels": [row["label"] for row in rows if row["escalated"]],
        "passed": bool(
            len(rows) >= int(min_evaluated)
            and max_illegal_mass <= 1e-6
            and selective_l1 <= fast_l1
            and selective_action_agreement >= fast_action_agreement
            and selective_latency < live_latency
        ),
        "promotion": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a selective solver-budget policy with true online latency accounting."
    )
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    profile_metrics = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    policy = load_selective_policy(args.policy_json)
    metrics = evaluate_budget_policy_integration(
        profile_metrics,
        policy,
        min_evaluated=args.min_evaluated,
    )
    metrics["profile_json"] = str(args.profile_json)
    metrics["policy_json"] = str(args.policy_json)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
