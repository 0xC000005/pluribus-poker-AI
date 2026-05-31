"""Join CFR frontier latency metrics with matrix footprint shape metrics."""

from __future__ import annotations

from statistics import mean
from typing import Any


def _round(value: float) -> float:
    return round(float(value), 8)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    numerator = sum(x * y for x, y in zip(x_centered, y_centered, strict=True))
    x_var = sum(x * x for x in x_centered)
    y_var = sum(y * y for y in y_centered)
    denom = (x_var * y_var) ** 0.5
    if denom <= 0.0:
        return None
    return _round(numerator / denom)


def _solver_state_mib(record: dict[str, Any]) -> float:
    memory_mib = record.get("memory_mib", {})
    if "solver_state_total" in memory_mib:
        return float(memory_mib["solver_state_total"])
    memory_bytes = record.get("memory_bytes", {})
    if "solver_state_total" in memory_bytes:
        return float(memory_bytes["solver_state_total"]) / (1024.0 * 1024.0)
    return 0.0


def _select_budget(frontier: dict[str, Any], budget: str | int | None) -> str:
    if budget is not None:
        return str(budget)
    best = frontier.get("best_l1_budget")
    if best is not None:
        return str(best)
    for record in frontier.get("records", []):
        budget_keys = record.get("budgets", {}).keys()
        numeric = sorted(int(key) for key in budget_keys)
        if numeric:
            return str(numeric[-1])
    raise ValueError("frontier records do not contain budget metrics")


def analyze_frontier_latency_shape(
    frontier: dict[str, Any],
    footprint: dict[str, Any],
    *,
    budget: str | int | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Summarize how exact-CFR latency relates to solver tree shape."""
    selected_budget = _select_budget(frontier, budget)
    footprints = {
        str(record.get("label")): record
        for record in footprint.get("records", [])
        if record.get("passed", True)
    }
    joined: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    for record in frontier.get("records", []):
        if not record.get("passed"):
            continue
        label = str(record.get("label"))
        budget_metrics = record.get("budgets", {}).get(selected_budget)
        shape = footprints.get(label)
        if budget_metrics is None or shape is None:
            missing_labels.append(label)
            continue
        n_nodes = int(shape.get("n_nodes", 0))
        n_edges = int(shape.get("n_edges", 0))
        solver_state_mib = _solver_state_mib(shape)
        latency_ms = float(budget_metrics["latency_ms"])
        joined.append(
            {
                "label": label,
                "latency_ms": _round(latency_ms),
                "l1_to_reference": _round(float(budget_metrics.get("l1_to_reference", 0.0))),
                "kl_to_reference": _round(float(budget_metrics.get("kl_to_reference", 0.0))),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "max_depth": int(shape.get("max_depth", 0)),
                "solver_state_mib": _round(solver_state_mib),
                "latency_ms_per_1k_nodes": _round(
                    latency_ms / max(float(n_nodes), 1.0) * 1000.0
                ),
                "latency_ms_per_mib": _round(latency_ms / max(solver_state_mib, 1e-9)),
            }
        )

    latencies = [float(row["latency_ms"]) for row in joined]
    nodes = [float(row["n_nodes"]) for row in joined]
    edges = [float(row["n_edges"]) for row in joined]
    state_mib = [float(row["solver_state_mib"]) for row in joined]
    l1s = [float(row["l1_to_reference"]) for row in joined]
    top = sorted(joined, key=lambda row: row["latency_ms"], reverse=True)[: int(top_k)]
    return {
        "mode": "cfr_frontier_latency_shape",
        "passed": bool(joined),
        "budget": selected_budget,
        "n_joined": int(len(joined)),
        "n_missing": int(len(missing_labels)),
        "missing_labels": missing_labels,
        "mean_latency_ms": _round(mean(latencies)) if latencies else 0.0,
        "max_latency_ms": _round(max(latencies)) if latencies else 0.0,
        "mean_l1_to_reference": _round(mean(l1s)) if l1s else 0.0,
        "mean_latency_ms_per_1k_nodes": _round(
            mean([float(row["latency_ms_per_1k_nodes"]) for row in joined])
        )
        if joined
        else 0.0,
        "mean_latency_ms_per_mib": _round(
            mean([float(row["latency_ms_per_mib"]) for row in joined])
        )
        if joined
        else 0.0,
        "correlations": {
            "latency_vs_n_nodes": _pearson(latencies, nodes),
            "latency_vs_n_edges": _pearson(latencies, edges),
            "latency_vs_solver_state_mib": _pearson(latencies, state_mib),
            "latency_vs_l1_to_reference": _pearson(latencies, l1s),
        },
        "top_roots_by_latency": top,
        "records": joined,
    }
