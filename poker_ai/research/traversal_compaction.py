"""Analyze whether GPU traversal should use active-frontier compaction."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any


def _as_float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    if value is None:
        return default
    return float(value)


def _as_int(record: dict[str, Any], key: str, default: int = 0) -> int:
    value = record.get(key, default)
    if value is None:
        return default
    return int(value)


def load_traversal_profiles(
    path: str | Path,
    *,
    include_warmup: bool = False,
) -> list[dict[str, Any]]:
    """Load measured traversal profile rows from a benchmark JSON artifact."""

    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    profiles = list(artifact.get("profiles", []))
    if include_warmup:
        profiles = list(artifact.get("warmup_profiles", [])) + profiles
    if not profiles and "traversal_mean_allocated_to_live_ratio" in artifact:
        profiles = [artifact]
    return [dict(profile) for profile in profiles]


def analyze_traversal_compaction_potential(
    profiles: list[dict[str, Any]],
    *,
    min_mean_allocated_to_live_ratio: float = 2.0,
    max_overflow_chunk_fraction: float = 0.0,
    max_pool_exhausted_per_traversal: float = 0.0,
) -> dict[str, Any]:
    """Return a gate report for enumeration-preserving frontier compaction.

    The analyzer does not claim a wall-clock speedup. It asks whether the
    accepted traversal uses many more allocated slots than live nonterminal
    slots, which is the narrow signal needed before implementing prefix-sum
    active-frontier packing in the CUDA traversal.
    """

    if not profiles:
        return {
            "mode": "traversal_compaction_potential",
            "profile_count": 0,
            "passed": False,
            "promotion": False,
            "failures": ["no traversal profiles found"],
            "next_action": "run scripts/benchmark_gpu_deep_cfr.py with traversal pool telemetry enabled",
        }

    mean_allocated_to_live = fmean(
        _as_float(profile, "traversal_mean_allocated_to_live_ratio")
        for profile in profiles
    )
    max_allocated_to_live = max(
        _as_float(profile, "traversal_max_allocated_to_live_ratio")
        for profile in profiles
    )
    mean_slots_per_traversal = fmean(
        _as_float(profile, "traversal_mean_slots_per_traversal")
        for profile in profiles
    )
    mean_live_slots_per_traversal = fmean(
        _as_float(profile, "traversal_mean_max_nonterminal_slots_per_traversal")
        for profile in profiles
    )
    max_live_slots_per_traversal = max(
        _as_float(profile, "traversal_max_nonterminal_slots_per_traversal")
        for profile in profiles
    )
    max_overflow = max(
        _as_float(profile, "traversal_overflow_chunk_fraction")
        for profile in profiles
    )
    max_pool_exhausted = max(
        _as_float(profile, "traversal_pool_exhausted_per_traversal")
        for profile in profiles
    )
    max_pool_exhausted_nodes = max(
        _as_int(profile, "traversal_pool_exhausted_nodes")
        for profile in profiles
    )
    mean_traverse_seconds = fmean(
        _as_float(profile, "traverse_seconds")
        for profile in profiles
    )
    mean_traversals_per_second = fmean(
        _as_float(profile, "traversals_per_second")
        for profile in profiles
    )

    fidelity_clean = (
        max_overflow <= max_overflow_chunk_fraction
        and max_pool_exhausted <= max_pool_exhausted_per_traversal
        and max_pool_exhausted_nodes == 0
    )
    waste_large = mean_allocated_to_live >= min_mean_allocated_to_live_ratio
    estimated_slot_reduction = (
        max(0.0, 1.0 - (1.0 / mean_allocated_to_live))
        if mean_allocated_to_live > 0.0
        else 0.0
    )
    failures: list[str] = []
    if not fidelity_clean:
        failures.append(
            "accepted traversal is not fidelity-clean "
            f"(overflow_fraction={max_overflow:.6f}, "
            f"pool_exhausted_per_traversal={max_pool_exhausted:.6f}, "
            f"pool_exhausted_nodes={max_pool_exhausted_nodes})"
        )
    if not waste_large:
        failures.append(
            "allocated/live slot ratio is too small "
            f"({mean_allocated_to_live:.6f} < {min_mean_allocated_to_live_ratio:.6f})"
        )

    if fidelity_clean and waste_large:
        next_action = (
            "implement an opt-in active-frontier compaction prototype that keeps "
            "traverser-action enumeration unchanged"
        )
    elif not fidelity_clean and waste_large:
        next_action = (
            "keep the larger safe slot budget for training, then prototype "
            "compaction as a fidelity and memory-pressure fix"
        )
    else:
        next_action = (
            "do not prioritize compaction; look for batching, network, or "
            "cache-bound bottlenecks instead"
        )

    return {
        "mode": "traversal_compaction_potential",
        "profile_count": len(profiles),
        "mean_allocated_to_live_ratio": round(mean_allocated_to_live, 6),
        "max_allocated_to_live_ratio": round(max_allocated_to_live, 6),
        "mean_slots_per_traversal": round(mean_slots_per_traversal, 6),
        "mean_live_slots_per_traversal": round(mean_live_slots_per_traversal, 6),
        "max_live_slots_per_traversal": round(max_live_slots_per_traversal, 6),
        "estimated_slot_reduction_fraction": round(estimated_slot_reduction, 6),
        "max_overflow_chunk_fraction": round(max_overflow, 6),
        "max_pool_exhausted_per_traversal": round(max_pool_exhausted, 6),
        "max_pool_exhausted_nodes": int(max_pool_exhausted_nodes),
        "mean_traverse_seconds": round(mean_traverse_seconds, 6),
        "mean_traversals_per_second": round(mean_traversals_per_second, 6),
        "thresholds": {
            "min_mean_allocated_to_live_ratio": float(min_mean_allocated_to_live_ratio),
            "max_overflow_chunk_fraction": float(max_overflow_chunk_fraction),
            "max_pool_exhausted_per_traversal": float(max_pool_exhausted_per_traversal),
        },
        "fidelity_clean": fidelity_clean,
        "compaction_promising": waste_large,
        "passed": fidelity_clean and waste_large,
        "promotion": False,
        "failures": failures,
        "next_action": next_action,
    }
