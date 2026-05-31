"""Empirical payoff-matrix helpers for native poker policy populations."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _load_h2h_record(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("candidate_checkpoint", "baseline_checkpoint", "mean_candidate_payoff"):
        if key not in payload:
            raise ValueError(f"{path} missing required H2H field: {key}")
    return payload


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    total_weight = sum(weight for _value, weight in values)
    if total_weight <= 0:
        return sum(value for value, _weight in values) / float(len(values))
    return sum(value * weight for value, weight in values) / total_weight


def build_empirical_payoff_matrix(
    record_paths: Sequence[str | Path],
    *,
    policies: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build an antisymmetric empirical game from duplicate-swapped H2H records."""
    records = [_load_h2h_record(path) for path in record_paths]
    if policies is None:
        policy_names = sorted(
            {
                str(record["candidate_checkpoint"])
                for record in records
            }
            | {
                str(record["baseline_checkpoint"])
                for record in records
            }
        )
    else:
        policy_names = [str(policy) for policy in policies]
    policy_set = set(policy_names)
    directed_values: dict[tuple[str, str], list[tuple[float, float]]] = {}

    for record in records:
        candidate = str(record["candidate_checkpoint"])
        baseline = str(record["baseline_checkpoint"])
        if candidate == baseline or candidate not in policy_set or baseline not in policy_set:
            continue
        payoff = float(record["mean_candidate_payoff"])
        weight = float(record.get("n_games", 1) or 1)
        directed_values.setdefault((candidate, baseline), []).append((payoff, weight))
        directed_values.setdefault((baseline, candidate), []).append((-payoff, weight))

    matrix: list[list[float | None]] = []
    missing_pairs: list[list[str]] = []
    covered_pairs = 0
    total_pairs = max(0, len(policy_names) * (len(policy_names) - 1) // 2)
    for row_i, row_policy in enumerate(policy_names):
        row: list[float | None] = []
        for col_i, col_policy in enumerate(policy_names):
            if row_i == col_i:
                row.append(0.0)
                continue
            value = _weighted_mean(directed_values.get((row_policy, col_policy), []))
            row.append(value)
            if row_i < col_i:
                if value is None:
                    missing_pairs.append([row_policy, col_policy])
                else:
                    covered_pairs += 1
        matrix.append(row)

    coverage = (covered_pairs / float(total_pairs)) if total_pairs else 1.0
    recommendation = "solve_meta_strategy" if coverage == 1.0 else "fill_pairwise_payoff_matrix"
    return {
        "algorithm": "native_empirical_payoff_matrix",
        "policies": policy_names,
        "n_policies": len(policy_names),
        "n_records": len(records),
        "payoff_matrix": matrix,
        "off_diagonal_coverage": coverage,
        "covered_pairs": covered_pairs,
        "total_pairs": total_pairs,
        "missing_pairs": missing_pairs,
        "recommendation": recommendation,
        "promotion": False,
    }


def _finite_matrix(payoff_matrix: Sequence[Sequence[float | None]]) -> np.ndarray:
    matrix = np.asarray(payoff_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("payoff_matrix must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("payoff_matrix contains missing or non-finite entries")
    return matrix


def solve_zero_sum_meta_strategy(
    payoff_matrix: Sequence[Sequence[float | None]],
    *,
    nashpy_module: Any | None = None,
) -> dict[str, Any]:
    """Solve a complete two-player zero-sum empirical game with nashpy."""
    matrix = _finite_matrix(payoff_matrix)
    try:
        nashpy = nashpy_module or importlib.import_module("nashpy")
        game = nashpy.Game(matrix)
        row_strategy, column_strategy = next(game.support_enumeration())
    except Exception as exc:
        return {
            "solved": False,
            "solver": "nashpy.support_enumeration",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    row = np.asarray(row_strategy, dtype=float)
    col = np.asarray(column_strategy, dtype=float)
    row = row / row.sum() if row.sum() else np.ones_like(row) / len(row)
    col = col / col.sum() if col.sum() else np.ones_like(col) / len(col)
    value = float(row @ matrix @ col)
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return {
        "solved": True,
        "solver": "nashpy.support_enumeration",
        "row_strategy": row.tolist(),
        "column_strategy": col.tolist(),
        "value": value,
    }
