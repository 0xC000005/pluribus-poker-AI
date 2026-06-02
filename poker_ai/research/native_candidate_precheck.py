"""Native-only local precheck for full-deck poker candidates.

This gate deliberately does not replace the dual-surface pre-Slumbot promotion
gate. It answers a narrower question: whether a native 9-action candidate has
enough local H2H and empirical-game evidence to justify gathering missing
external evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lower95(payload: dict[str, Any]) -> float | None:
    value = payload.get("lower95_candidate_payoff")
    if value is None:
        return None
    return float(value)


def _bool_field(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return bool(default)
    return bool(payload[key])


def _check_h2h(
    payload: dict[str, Any],
    *,
    candidate_checkpoint: str,
    min_lower95: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    lower95 = _lower95(payload)
    if payload.get("algorithm") != "mixed_native_policy_h2h":
        blockers.append("native_h2h_wrong_algorithm")
    if payload.get("environment") != "poker_ai:full_deck_hu_nlhe":
        blockers.append("native_h2h_wrong_environment")
    if str(payload.get("candidate_checkpoint", "")) != str(candidate_checkpoint):
        blockers.append("native_h2h_candidate_checkpoint_mismatch")
    if not _bool_field(payload, "trained_environment_native", False):
        blockers.append("native_h2h_not_environment_native")
    if _bool_field(payload, "native_action_projection", True):
        blockers.append("native_h2h_uses_projection")
    if _bool_field(payload, "uses_slumbot_training_data", False) or _bool_field(
        payload,
        "uses_slumbot_data",
        False,
    ):
        blockers.append("native_h2h_uses_slumbot_data")
    if _bool_field(payload, "uses_alphanlholdem_training_data", False):
        blockers.append("native_h2h_uses_alphanlholdem_training_data")
    if lower95 is None:
        blockers.append("native_h2h_missing_lower95")
    elif lower95 < float(min_lower95):
        blockers.append("native_h2h_lower95_below_threshold")
    return (
        {
            "passed": not blockers,
            "baseline_checkpoint": payload.get("baseline_checkpoint"),
            "candidate_kind": payload.get("candidate_kind"),
            "baseline_kind": payload.get("baseline_kind"),
            "n_games": payload.get("n_games"),
            "lower95": lower95,
            "mean": payload.get("mean_candidate_payoff"),
            "min_lower95": float(min_lower95),
        },
        blockers,
    )


def _check_empirical_game(
    payload: dict[str, Any],
    *,
    candidate_checkpoint: str,
    min_candidate_support: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if payload.get("algorithm") != "native_empirical_payoff_matrix":
        blockers.append("empirical_game_wrong_algorithm")
    coverage = float(payload.get("off_diagonal_coverage", 0.0) or 0.0)
    if coverage < 1.0:
        blockers.append("empirical_game_incomplete_matrix")
    meta = payload.get("meta_strategy", {})
    if not bool(meta.get("solved", False)):
        blockers.append("empirical_game_meta_strategy_not_solved")
    policies = [str(policy) for policy in payload.get("policies", [])]
    support: float | None = None
    if str(candidate_checkpoint) not in policies:
        blockers.append("empirical_game_candidate_missing")
    elif bool(meta.get("solved", False)):
        index = policies.index(str(candidate_checkpoint))
        row_strategy = list(meta.get("row_strategy", []))
        if index >= len(row_strategy):
            blockers.append("empirical_game_candidate_support_missing")
        else:
            support = float(row_strategy[index])
            if support < float(min_candidate_support):
                blockers.append("empirical_game_candidate_support_below_threshold")
    return (
        {
            "passed": not blockers,
            "off_diagonal_coverage": coverage,
            "meta_strategy_solved": bool(meta.get("solved", False)),
            "candidate_support_probability": support,
            "min_candidate_support": float(min_candidate_support),
        },
        blockers,
    )


def evaluate_native_candidate_local_precheck(
    *,
    candidate_checkpoint: str | Path,
    native_h2h_jsons: Sequence[str | Path],
    empirical_game_json: str | Path,
    min_lower95: float = 0.0,
    min_candidate_support: float = 1.0e-9,
) -> dict[str, Any]:
    """Evaluate local native evidence for a candidate checkpoint.

    The result may pass while still reporting external blockers. Passing means
    local native evidence is coherent; it does not authorize Slumbot confidence.
    """
    candidate = str(candidate_checkpoint)
    h2h_paths = [Path(path) for path in native_h2h_jsons]
    if not h2h_paths:
        raise ValueError("at least one native H2H JSON is required")

    h2h_checks = []
    local_blockers: list[str] = []
    candidate_kinds: set[str] = set()
    for path in h2h_paths:
        payload = _load_json(path)
        check, blockers = _check_h2h(
            payload,
            candidate_checkpoint=candidate,
            min_lower95=float(min_lower95),
        )
        check["path"] = str(path)
        h2h_checks.append(check)
        local_blockers.extend(blockers)
        kind = check.get("candidate_kind")
        if kind:
            candidate_kinds.add(str(kind))

    empirical_payload = _load_json(empirical_game_json)
    empirical_check, empirical_blockers = _check_empirical_game(
        empirical_payload,
        candidate_checkpoint=candidate,
        min_candidate_support=float(min_candidate_support),
    )
    empirical_check["path"] = str(empirical_game_json)
    local_blockers.extend(empirical_blockers)

    local_passed = not local_blockers
    external_blockers = [
        "rlcard_reference_evidence_missing",
        "slumbot_confidence_not_authorized_by_native_only_precheck",
        "held_out_slumbot_smoke_missing",
    ]
    slumbot_supported_kinds = {"deep-cfr", "tianshou-rainbow"}
    unsupported_kinds = sorted(kind for kind in candidate_kinds if kind not in slumbot_supported_kinds)
    if unsupported_kinds:
        external_blockers.append("slumbot_adapter_missing_for_candidate_kind")

    return {
        "algorithm": "native_candidate_local_precheck",
        "role": "native_only_pre_slumbot_local_gate",
        "passed": bool(local_passed),
        "local_precheck_passed": bool(local_passed),
        "promotion": False,
        "slumbot_confidence_eligible": False,
        "held_out_slumbot_next_allowed": False,
        "formal_dual_surface_gate_required": True,
        "candidate_checkpoint": candidate,
        "candidate_kinds": sorted(candidate_kinds),
        "slumbot_supported_candidate_kinds": sorted(slumbot_supported_kinds),
        "min_lower95": float(min_lower95),
        "min_candidate_support": float(min_candidate_support),
        "checks": {
            "native_h2h": h2h_checks,
            "empirical_game": empirical_check,
        },
        "local_blockers": local_blockers,
        "external_promotion_blockers": external_blockers,
        "promotion_blockers": [*local_blockers, *external_blockers],
        "uses_slumbot_training_data": False,
        "uses_alphanlholdem_training_data": False,
        "next_required_evidence": [
            "same-philosophy RLCard-native AlphaHoldem reference evidence or explicit workflow decision that native Slumbot can proceed without it",
            "held-out Slumbot smoke/confidence evaluation after formal gate approval",
        ],
    }
