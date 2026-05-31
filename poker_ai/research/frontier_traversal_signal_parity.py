"""Distributional parity helpers for frontier-indexed traversal diagnostics."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def summarize_signal_samples(
    features: np.ndarray,
    regrets: np.ndarray,
) -> dict[str, object]:
    """Summarize one traversal signal sample set."""
    count = int(regrets.shape[0])
    if count == 0:
        feature_mean = np.zeros(features.shape[1], dtype=np.float32)
        regret_mean = np.zeros(regrets.shape[1], dtype=np.float32)
    else:
        feature_mean = features[:count].mean(axis=0)
        regret_mean = regrets[:count].mean(axis=0)
    return {
        "count": count,
        "feature_mean": feature_mean.astype(float).tolist(),
        "regret_mean": regret_mean.astype(float).tolist(),
    }


def _mean_count(group: Sequence[dict[str, object]]) -> float:
    if not group:
        return 0.0
    return float(np.mean([float(item["count"]) for item in group]))


def _mean_regret(group: Sequence[dict[str, object]]) -> np.ndarray:
    if not group:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(
        [item["regret_mean"] for item in group],
        dtype=np.float32,
    ).mean(axis=0)


def _within_regret_l1(group: Sequence[dict[str, object]]) -> float:
    if len(group) < 2:
        return 0.0
    distances = []
    for left, right in combinations(group, 2):
        left_regret = np.asarray(left["regret_mean"], dtype=np.float32)
        right_regret = np.asarray(right["regret_mean"], dtype=np.float32)
        distances.append(float(np.abs(left_regret - right_regret).sum()))
    return float(np.mean(distances)) if distances else 0.0


def compare_signal_groups(
    baseline: Sequence[dict[str, object]],
    frontier: Sequence[dict[str, object]],
    *,
    max_count_rel_gap: float = 0.05,
    max_regret_l1_ratio: float = 2.0,
) -> dict[str, object]:
    """Compare frontier-vs-baseline divergence against simple thresholds."""
    baseline_count = _mean_count(baseline)
    frontier_count = _mean_count(frontier)
    denom_count = max(abs(baseline_count), 1.0)
    count_rel_gap = abs(frontier_count - baseline_count) / denom_count

    baseline_regret = _mean_regret(baseline)
    frontier_regret = _mean_regret(frontier)
    regret_l1 = float(np.abs(frontier_regret - baseline_regret).sum())
    within_l1 = max(
        1e-8,
        _within_regret_l1(baseline),
        _within_regret_l1(frontier),
    )
    regret_l1_ratio = regret_l1 / within_l1

    failures: list[str] = []
    if count_rel_gap > max_count_rel_gap:
        failures.append(
            f"count_rel_gap {count_rel_gap:.6f} > {max_count_rel_gap:.6f}"
        )
    if regret_l1_ratio > max_regret_l1_ratio:
        failures.append(
            f"regret_l1_ratio {regret_l1_ratio:.6f} > {max_regret_l1_ratio:.6f}"
        )
    return {
        "mode": "frontier_traversal_signal_parity",
        "baseline_count_mean": baseline_count,
        "frontier_count_mean": frontier_count,
        "count_rel_gap": count_rel_gap,
        "baseline_within_regret_l1": _within_regret_l1(baseline),
        "frontier_within_regret_l1": _within_regret_l1(frontier),
        "between_regret_l1": regret_l1,
        "regret_l1_ratio": regret_l1_ratio,
        "gate_thresholds": {
            "max_count_rel_gap": max_count_rel_gap,
            "max_regret_l1_ratio": max_regret_l1_ratio,
        },
        "gate_failures": failures,
        "passed": not failures,
    }
