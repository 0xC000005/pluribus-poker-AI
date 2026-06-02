"""Select native poker population expansions from empirical-game evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from poker_ai.research.empirical_game import (
    build_empirical_payoff_matrix,
    solve_zero_sum_meta_strategy,
)


@dataclass(frozen=True)
class ExpansionCandidate:
    checkpoint: str
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or self.checkpoint


def _load_record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_confirmation(
    records: Sequence[dict[str, Any]],
    *,
    candidate: str,
    incumbent: str,
) -> dict[str, Any]:
    direct = [
        record
        for record in records
        if str(record.get("candidate_checkpoint")) == candidate
        and str(record.get("baseline_checkpoint")) == incumbent
    ]
    if not direct:
        return {
            "available": False,
            "blocker": "missing_direct_candidate_vs_incumbent_h2h",
        }
    record = max(direct, key=lambda item: int(item.get("n_games", 0) or 0))
    return {
        "available": True,
        "path": record.get("_path"),
        "mean_candidate_payoff": float(record.get("mean_candidate_payoff", 0.0)),
        "lower95_candidate_payoff": float(record.get("lower95_candidate_payoff", float("nan"))),
        "upper95_candidate_payoff": float(record.get("upper95_candidate_payoff", float("nan"))),
        "n_games": int(record.get("n_games", 0) or 0),
    }


def _candidate_support_probability(
    *,
    candidate: str,
    policies: Sequence[str],
    meta_strategy: dict[str, Any],
) -> float:
    if not meta_strategy.get("solved"):
        return 0.0
    try:
        index = list(policies).index(candidate)
    except ValueError:
        return 0.0
    row_strategy = meta_strategy.get("row_strategy") or []
    if index >= len(row_strategy):
        return 0.0
    return float(row_strategy[index])


def _pure_strategy_fallback(payoff_matrix: Sequence[Sequence[float | None]]) -> dict[str, Any]:
    matrix = np.asarray(payoff_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        return {"solved": False, "reason": "non_finite_or_non_square_matrix"}
    row_security = matrix.min(axis=1)
    col_exposure = matrix.max(axis=0)
    for row_i, security in enumerate(row_security):
        if float(security) < -1.0e-12:
            continue
        if float(col_exposure[row_i]) > 1.0e-12:
            continue
        strategy = np.zeros(matrix.shape[0], dtype=float)
        strategy[row_i] = 1.0
        return {
            "solved": True,
            "solver": "pure_strategy_fallback",
            "row_strategy": strategy.tolist(),
            "column_strategy": strategy.tolist(),
            "value": 0.0,
        }
    return {
        "solved": False,
        "solver": "pure_strategy_fallback",
        "reason": "no_pure_zero_sum_equilibrium_found",
    }


def _solve_meta_strategy(
    payoff_matrix: Sequence[Sequence[float | None]],
    *,
    nashpy_module: Any | None,
) -> dict[str, Any]:
    solution = solve_zero_sum_meta_strategy(
        payoff_matrix,
        nashpy_module=nashpy_module,
    )
    if solution.get("solved"):
        return solution
    fallback = _pure_strategy_fallback(payoff_matrix)
    if fallback.get("solved"):
        fallback["primary_solver_error"] = solution
        return fallback
    return solution


def evaluate_expansion_quality_selector(
    *,
    support_policies: Sequence[str],
    candidate_checkpoints: Sequence[str],
    h2h_records: Sequence[str | Path],
    incumbent_checkpoint: str,
    min_candidate_support: float = 1.0e-9,
    min_lower95: float = 0.0,
    nashpy_module: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed slate of candidate population insertions.

    A candidate is eligible only if its expanded payoff matrix is complete, the
    restricted-game solution assigns it support, and its direct H2H confirmation
    against the incumbent has a positive-enough lower bound.
    """

    support = [str(policy) for policy in support_policies]
    candidates = [str(candidate) for candidate in candidate_checkpoints]
    incumbent = str(incumbent_checkpoint)
    record_paths = [Path(path) for path in h2h_records]
    records = []
    for path in record_paths:
        record = _load_record(path)
        record["_path"] = str(path)
        records.append(record)

    candidate_results: list[dict[str, Any]] = []
    for candidate in candidates:
        policies = list(dict.fromkeys([*support, candidate]))
        matrix = build_empirical_payoff_matrix(record_paths, policies=policies)
        meta_strategy = (
            _solve_meta_strategy(
                matrix["payoff_matrix"],
                nashpy_module=nashpy_module,
            )
            if matrix["recommendation"] == "solve_meta_strategy"
            else {"solved": False, "reason": "incomplete_matrix"}
        )
        support_probability = _candidate_support_probability(
            candidate=candidate,
            policies=policies,
            meta_strategy=meta_strategy,
        )
        confirmation = _candidate_confirmation(
            records,
            candidate=candidate,
            incumbent=incumbent,
        )
        blockers: list[str] = []
        if matrix["off_diagonal_coverage"] < 1.0:
            blockers.append("incomplete_expanded_empirical_game")
        if support_probability < float(min_candidate_support):
            blockers.append("candidate_not_in_empirical_game_support")
        if not confirmation.get("available"):
            blockers.append(str(confirmation.get("blocker")))
        elif float(confirmation["lower95_candidate_payoff"]) < float(min_lower95):
            blockers.append("candidate_confirmation_lower95_below_threshold")

        candidate_results.append(
            {
                "candidate_checkpoint": candidate,
                "eligible": not blockers,
                "blockers": blockers,
                "support_probability": support_probability,
                "confirmation": confirmation,
                "empirical_game": {
                    "policies": policies,
                    "off_diagonal_coverage": matrix["off_diagonal_coverage"],
                    "covered_pairs": matrix["covered_pairs"],
                    "total_pairs": matrix["total_pairs"],
                    "missing_pairs": matrix["missing_pairs"],
                    "meta_strategy": meta_strategy,
                },
            }
        )

    eligible = [item for item in candidate_results if item["eligible"]]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                float(item["support_probability"]),
                float(item["confirmation"]["lower95_candidate_payoff"]),
                float(item["confirmation"]["mean_candidate_payoff"]),
            ),
        )

    return {
        "algorithm": "native_expansion_quality_selector",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "support_policies": support,
        "candidate_checkpoints": candidates,
        "incumbent_checkpoint": incumbent,
        "min_candidate_support": float(min_candidate_support),
        "min_lower95": float(min_lower95),
        "n_records": len(record_paths),
        "selection_metric": "empirical_support_then_incumbent_lower95",
        "passed": selected is not None,
        "selected_candidate": None if selected is None else selected["candidate_checkpoint"],
        "candidate_results": candidate_results,
        "promotion": False,
        "uses_slumbot_training_data": False,
        "trained_environment_native": True,
        "native_action_projection": False,
    }
