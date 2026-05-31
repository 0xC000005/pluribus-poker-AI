"""Summaries for CFR solver cold/warm latency diagnostics."""

from __future__ import annotations

from statistics import mean
from typing import Any


def _round(value: float) -> float:
    return round(float(value), 8)


def summarize_warmup_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-root cold and warm CFR solve latency records."""
    evaluated = [record for record in records if record.get("passed")]
    cold = [float(record["cold_solve_ms"]) for record in evaluated]
    warm_means = [float(record["warm_solve_mean_ms"]) for record in evaluated]
    construction = [float(record.get("construction_ms", 0.0)) for record in evaluated]
    speedups = [
        float(record["cold_solve_ms"]) / max(float(record["warm_solve_mean_ms"]), 1e-9)
        for record in evaluated
    ]
    return {
        "mode": "cfr_solver_warmup_profile",
        "passed": bool(evaluated) and len(evaluated) == len(records),
        "n_records": int(len(records)),
        "n_evaluated": int(len(evaluated)),
        "mean_cold_solve_ms": _round(mean(cold)) if cold else 0.0,
        "mean_warm_solve_ms": _round(mean(warm_means)) if warm_means else 0.0,
        "mean_construction_ms": _round(mean(construction)) if construction else 0.0,
        "mean_cold_to_warm_speedup": _round(mean(speedups)) if speedups else 0.0,
        "max_cold_to_warm_speedup": _round(max(speedups)) if speedups else 0.0,
        "records": records,
    }
