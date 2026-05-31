"""Pre-Slumbot promotion gate for AlphaHoldem-style poker candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lower95(payload: dict[str, Any]) -> float | None:
    for key in (
        "lower95_candidate_payoff",
        "paired_delta_lower95_chips_per_hand",
        "paired_delta_lower95_chips_per_hand_across_seeds",
        "best_worst_lower95_chips_per_hand",
    ):
        if key in payload and payload[key] is not None:
            return float(payload[key])
    return None


def _bool_field(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return bool(default)
    return bool(payload[key])


def _check_rlcard_reference(payload: dict[str, Any], *, min_lower95: float) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    lower95 = _lower95(payload)
    allowed_reference_baselines = {
        "alphanlholdem",
        "alphanlholdem-reference",
        "alphanlholdem-reproduction",
    }
    if payload.get("environment") != "rlcard:no-limit-holdem":
        blockers.append("rlcard_reference_wrong_environment")
    if payload.get("candidate_kind") in {None, "random", "alphanlholdem"}:
        blockers.append("rlcard_reference_missing_project_candidate")
    if payload.get("baseline_kind") not in allowed_reference_baselines:
        blockers.append("rlcard_reference_not_alphanlholdem_baseline")
    if not _bool_field(payload, "trained_environment_native", False):
        blockers.append("rlcard_reference_not_environment_native")
    if _bool_field(payload, "native_action_projection", True):
        blockers.append("rlcard_reference_uses_native_projection")
    if not _bool_field(payload, "reference_integrity_passed", False):
        blockers.append("rlcard_reference_integrity_not_passed")
    if not _bool_field(payload, "reference_checkout_ignored", False):
        blockers.append("rlcard_reference_checkout_not_ignored")
    if payload.get("tracked_external_reference_files"):
        blockers.append("rlcard_reference_has_tracked_external_files")
    if _bool_field(payload, "native_slumbot_evidence", False):
        blockers.append("rlcard_reference_claims_native_slumbot_evidence")
    if _bool_field(payload, "uses_slumbot_data", False):
        blockers.append("rlcard_reference_uses_slumbot_data")
    if _bool_field(payload, "uses_alphanlholdem_training_data", False):
        blockers.append("rlcard_reference_uses_alphanlholdem_training_data")
    if lower95 is None:
        blockers.append("rlcard_reference_missing_lower95")
    elif lower95 < float(min_lower95):
        blockers.append("rlcard_reference_lower95_below_threshold")
    return (
        {
            "passed": not blockers,
            "environment": payload.get("environment"),
            "candidate_kind": payload.get("candidate_kind"),
            "baseline_kind": payload.get("baseline_kind"),
            "candidate_weights": payload.get("candidate_weights"),
            "reference_integrity_passed": _bool_field(payload, "reference_integrity_passed", False),
            "reference_checkout_ignored": _bool_field(payload, "reference_checkout_ignored", False),
            "tracked_external_reference_files": list(payload.get("tracked_external_reference_files", [])),
            "lower95": lower95,
            "min_lower95": float(min_lower95),
        },
        blockers,
    )


def _check_native_h2h(payload: dict[str, Any], *, min_lower95: float) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    lower95 = _lower95(payload)
    if payload.get("environment") != "poker_ai:full_deck_hu_nlhe":
        blockers.append("native_h2h_wrong_environment")
    if not payload.get("candidate_checkpoint"):
        blockers.append("native_h2h_missing_candidate_checkpoint")
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
            "environment": payload.get("environment"),
            "candidate_checkpoint": payload.get("candidate_checkpoint"),
            "baseline_checkpoint": payload.get("baseline_checkpoint"),
            "lower95": lower95,
            "min_lower95": float(min_lower95),
        },
        blockers,
    )


def _check_empirical_game(
    payload: dict[str, Any] | None,
    *,
    native_candidate_checkpoint: str | None,
    require_empirical: bool,
    min_candidate_support: float,
) -> tuple[dict[str, Any], list[str]]:
    if payload is None:
        if require_empirical:
            return (
                {
                    "passed": False,
                    "required": True,
                    "candidate_support_probability": None,
                },
                ["empirical_game_missing"],
            )
        return (
            {
                "passed": True,
                "required": False,
                "candidate_support_probability": None,
            },
            [],
        )

    blockers: list[str] = []
    if payload.get("algorithm") != "native_empirical_payoff_matrix":
        blockers.append("empirical_game_wrong_algorithm")
    coverage = float(payload.get("off_diagonal_coverage", 0.0) or 0.0)
    if coverage < 1.0:
        blockers.append("empirical_game_incomplete_matrix")
    meta = payload.get("meta_strategy", {})
    if not bool(meta.get("solved", False)):
        blockers.append("empirical_game_meta_strategy_not_solved")
    candidate_support: float | None = None
    if native_candidate_checkpoint is not None:
        policies = [str(policy) for policy in payload.get("policies", [])]
        if str(native_candidate_checkpoint) not in policies:
            blockers.append("empirical_game_candidate_missing")
        elif bool(meta.get("solved", False)):
            index = policies.index(str(native_candidate_checkpoint))
            row_strategy = list(meta.get("row_strategy", []))
            if index >= len(row_strategy):
                blockers.append("empirical_game_candidate_support_missing")
            else:
                candidate_support = float(row_strategy[index])
                if candidate_support < float(min_candidate_support):
                    blockers.append("empirical_game_candidate_support_below_threshold")
    return (
        {
            "passed": not blockers,
            "required": bool(require_empirical),
            "off_diagonal_coverage": coverage,
            "meta_strategy_solved": bool(meta.get("solved", False)),
            "candidate_support_probability": candidate_support,
            "min_candidate_support": float(min_candidate_support),
        },
        blockers,
    )


def evaluate_candidate_promotion_evidence(
    *,
    rlcard_reference_json: str | Path,
    native_h2h_json: str | Path,
    empirical_game_json: str | Path | None = None,
    native_candidate_checkpoint: str | None = None,
    min_lower95: float = 0.0,
    require_empirical: bool = True,
    min_candidate_support: float = 1.0e-9,
) -> dict[str, Any]:
    """Evaluate whether a candidate has enough evidence for Slumbot confidence."""
    rlcard_payload = _load_json(rlcard_reference_json)
    native_payload = _load_json(native_h2h_json)
    empirical_payload = None if empirical_game_json is None else _load_json(empirical_game_json)

    rlcard_check, rlcard_blockers = _check_rlcard_reference(
        rlcard_payload,
        min_lower95=float(min_lower95),
    )
    native_check, native_blockers = _check_native_h2h(
        native_payload,
        min_lower95=float(min_lower95),
    )
    if native_candidate_checkpoint is None:
        native_candidate_checkpoint = native_check.get("candidate_checkpoint")
    empirical_check, empirical_blockers = _check_empirical_game(
        empirical_payload,
        native_candidate_checkpoint=native_candidate_checkpoint,
        require_empirical=bool(require_empirical),
        min_candidate_support=float(min_candidate_support),
    )
    blockers = [*rlcard_blockers, *native_blockers, *empirical_blockers]
    passed = not blockers
    return {
        "algorithm": "poker_candidate_promotion_gate",
        "role": "pre_slumbot_dual_surface_promotion_gate",
        "passed": bool(passed),
        "slumbot_confidence_eligible": bool(passed),
        "promotion": bool(passed),
        "min_lower95": float(min_lower95),
        "require_empirical": bool(require_empirical),
        "native_candidate_checkpoint": native_candidate_checkpoint,
        "checks": {
            "rlcard_reference": rlcard_check,
            "native_h2h": native_check,
            "empirical_game": empirical_check,
        },
        "promotion_blockers": blockers,
        "uses_slumbot_training_data": False,
        "uses_alphanlholdem_training_data": False,
        "held_out_slumbot_next_allowed": bool(passed),
    }
