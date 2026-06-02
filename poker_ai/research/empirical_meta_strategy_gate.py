"""Promotion gate for solved empirical-game checkpoint meta-strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_support(
    empirical_game_json: str | Path,
    *,
    strategy_key: str,
    min_support_weight: float,
) -> tuple[list[str], list[float], set[str]]:
    payload = _load_json(empirical_game_json)
    policies = [str(policy) for policy in payload.get("policies", [])]
    meta_strategy = payload.get("meta_strategy", {})
    if meta_strategy.get("solved") is not True:
        raise ValueError("empirical meta-strategy must be solved before gate evaluation")
    weights = meta_strategy.get(strategy_key)
    if not isinstance(weights, list):
        raise ValueError(f"meta_strategy missing list field: {strategy_key}")
    if len(weights) != len(policies):
        raise ValueError("meta_strategy weights must match empirical-game policy count")
    normalized_weights = [float(weight) for weight in weights]
    support = {
        policy
        for policy, weight in zip(policies, normalized_weights, strict=True)
        if float(weight) > float(min_support_weight)
    }
    return policies, normalized_weights, support


def _records_by_baseline(record_paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in record_paths:
        payload = _load_json(path)
        baseline = str(payload.get("baseline_checkpoint", ""))
        if not baseline:
            raise ValueError(f"{path} missing baseline_checkpoint")
        records[baseline] = payload
    return records


def evaluate_empirical_meta_strategy_h2h_gate(
    empirical_game_json: str | Path,
    h2h_records: Sequence[str | Path],
    *,
    strategy_key: str = "row_strategy",
    min_support_weight: float = 1e-12,
    max_support_abs_mean: float = 0.01,
    max_support_lower95_loss: float = 0.01,
    min_off_support_lower95: float = 0.0,
) -> dict[str, Any]:
    """Check that a solved meta-strategy ties support and beats off-support policies."""
    policies, weights, support = _load_support(
        empirical_game_json,
        strategy_key=strategy_key,
        min_support_weight=min_support_weight,
    )
    records = _records_by_baseline(h2h_records)
    support_results: list[dict[str, Any]] = []
    off_support_results: list[dict[str, Any]] = []
    blockers: list[str] = []

    for index, policy in enumerate(policies):
        record = records.get(policy)
        if record is None:
            blockers.append("missing_h2h_record")
            result = {
                "policy_index": index,
                "policy": policy,
                "weight": float(weights[index]),
                "missing": True,
            }
        else:
            mean = float(record["mean_candidate_payoff"])
            lower95 = float(record["lower95_candidate_payoff"])
            upper95 = float(record["upper95_candidate_payoff"])
            result = {
                "policy_index": index,
                "policy": policy,
                "weight": float(weights[index]),
                "mean_candidate_payoff": mean,
                "lower95_candidate_payoff": lower95,
                "upper95_candidate_payoff": upper95,
                "n_games": int(record.get("n_games", 0) or 0),
            }

        if policy in support:
            support_results.append(result)
            if result.get("missing"):
                continue
            mean = float(result["mean_candidate_payoff"])
            lower95 = float(result["lower95_candidate_payoff"])
            upper95 = float(result["upper95_candidate_payoff"])
            if abs(mean) > float(max_support_abs_mean):
                blockers.append("support_policy_mean_not_near_zero")
            if lower95 > 0.0 or upper95 < 0.0:
                blockers.append("support_policy_ci_excludes_zero")
            if lower95 < -float(max_support_lower95_loss):
                blockers.append("support_policy_lower95_too_negative")
        else:
            off_support_results.append(result)
            if result.get("missing"):
                continue
            if float(result["lower95_candidate_payoff"]) < float(min_off_support_lower95):
                blockers.append("off_support_policy_not_beaten")

    unique_blockers = sorted(set(blockers))
    return {
        "algorithm": "empirical_meta_strategy_h2h_gate",
        "empirical_game_json": str(empirical_game_json),
        "strategy_key": str(strategy_key),
        "support_policy_count": int(len(support_results)),
        "off_support_policy_count": int(len(off_support_results)),
        "support_results": support_results,
        "off_support_results": off_support_results,
        "min_support_weight": float(min_support_weight),
        "max_support_abs_mean": float(max_support_abs_mean),
        "max_support_lower95_loss": float(max_support_lower95_loss),
        "min_off_support_lower95": float(min_off_support_lower95),
        "promotion": False,
        "uses_slumbot_training_data": False,
        "promotion_blockers": unique_blockers,
        "passed": not unique_blockers,
    }
