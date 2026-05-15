"""Slumbot smoke-evaluation helpers for autoresearch."""

from __future__ import annotations

import re


_FINAL_RE = re.compile(r"FINAL:\s+(?P<hands>\d+)\s+hands\s+\|\s+(?P<total>[+-]?\d+)\s+chips")
_AVG_RE = re.compile(
    r"Avg:\s+(?P<avg>[+-]?\d+)\s+\+/-\s+(?P<ci95>\d+)\s+chips/hand"
)
_RATE_RE = re.compile(r"Rate:\s+(?P<mbb>[+-]?\d+)\s+mbb/hand")
_WIN_RE = re.compile(r"Win rate:\s+(?P<win>[0-9.]+)%")
_DECISIONS_RE = re.compile(
    r"Decisions:\s+total=(?P<total>\d+)\s+policy=(?P<policy>\d+)\s+"
    r"solver=(?P<solver>\d+)\s+fallback=(?P<fallback>\d+)\s+"
    r"parse_errors=(?P<parse_errors>\d+)(?:\s+api_errors=(?P<api_errors>\d+))?"
)
_ACTION_MIX_RE = re.compile(r"Action mix:\s+(?P<mix>.+)")
_INCREMENTS_RE = re.compile(r"Increments:\s+(?P<mix>.+)")
_STREET_MIX_RE = re.compile(r"Street mix:\s+(?P<mix>.+)")
_STREET_ITEM_RE = re.compile(
    r"(?P<street>preflop|flop|turn|river)\("
    r"total=(?P<total>\d+)\s+all-in=(?P<allin>\d+)\s+solver=(?P<solver>\d+)\)"
)
_MAPPING_DRIFT_RE = re.compile(
    r"Mapping drift:\s+n=(?P<n>\d+)\s+mean=(?P<mean>[0-9.]+)\s+max=(?P<max>[0-9.]+)"
)
_SOLVER_PERF_RE = re.compile(
    r"Solver perf:\s+n=(?P<n>\d+)\s+mean_ms=(?P<mean_ms>[0-9.]+)\s+"
    r"max_ms=(?P<max_ms>[0-9.]+)\s+cache_hits=(?P<cache_hits>\d+)\s+"
    r"mean_hands=(?P<mean_hands>[0-9.]+)/(?P<mean_full_hands>[0-9.]+)\s+"
    r"prune_ratio=(?P<prune_ratio>[0-9.]+)"
)


def _parse_key_value_counts(text: str) -> dict[str, int]:
    counts = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.rsplit("=", 1)
        counts[key] = int(value)
    return counts


def _parse_street_mix(text: str) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    decisions = {}
    all_in = {}
    solver = {}
    for match in _STREET_ITEM_RE.finditer(text):
        street = match.group("street")
        decisions[street] = int(match.group("total"))
        all_in[street] = int(match.group("allin"))
        solver[street] = int(match.group("solver"))
    return decisions, all_in, solver


def parse_slumbot_summary(output: str) -> dict:
    final = _FINAL_RE.search(output)
    avg = _AVG_RE.search(output)
    rate = _RATE_RE.search(output)
    win = _WIN_RE.search(output)
    if not all([final, avg, rate, win]):
        return {
            "passed": False,
            "parse_error": "Could not parse Slumbot final summary.",
            "raw_output_tail": output[-2000:],
        }

    metrics = {
        "passed": True,
        "hands": int(final.group("hands")),
        "total_chips": int(final.group("total")),
        "avg_chips_per_hand": int(avg.group("avg")),
        "ci95_chips_per_hand": int(avg.group("ci95")),
        "mbb_per_hand": int(rate.group("mbb")),
        "win_rate": float(win.group("win")) / 100.0,
    }
    decisions = _DECISIONS_RE.search(output)
    if decisions:
        metrics.update(
            {
                "decision_total": int(decisions.group("total")),
                "decision_policy": int(decisions.group("policy")),
                "decision_solver": int(decisions.group("solver")),
                "decision_fallback": int(decisions.group("fallback")),
                "parse_errors": int(decisions.group("parse_errors")),
                "api_errors": int(decisions.group("api_errors") or 0),
            }
        )
    action_mix = _ACTION_MIX_RE.search(output)
    if action_mix:
        metrics["action_mix"] = _parse_key_value_counts(action_mix.group("mix"))
    increments = _INCREMENTS_RE.search(output)
    if increments:
        metrics["increment_mix"] = _parse_key_value_counts(increments.group("mix"))
    street_mix = _STREET_MIX_RE.search(output)
    if street_mix:
        street_decisions, street_all_in, street_solver = _parse_street_mix(
            street_mix.group("mix")
        )
        if street_decisions:
            metrics["street_decisions"] = street_decisions
            metrics["street_all_in"] = street_all_in
            metrics["street_solver"] = street_solver
    mapping_drift = _MAPPING_DRIFT_RE.search(output)
    if mapping_drift:
        metrics.update(
            {
                "mapping_drift_n": int(mapping_drift.group("n")),
                "mapping_drift_mean": float(mapping_drift.group("mean")),
                "mapping_drift_max": float(mapping_drift.group("max")),
            }
        )
    solver_perf = _SOLVER_PERF_RE.search(output)
    if solver_perf:
        metrics.update(
            {
                "solver_latency_n": int(solver_perf.group("n")),
                "solver_latency_mean_ms": float(solver_perf.group("mean_ms")),
                "solver_latency_max_ms": float(solver_perf.group("max_ms")),
                "solver_cache_hits": int(solver_perf.group("cache_hits")),
                "solver_mean_hands": float(solver_perf.group("mean_hands")),
                "solver_mean_full_hands": float(solver_perf.group("mean_full_hands")),
                "solver_mean_prune_ratio": float(solver_perf.group("prune_ratio")),
            }
        )
    return metrics


def add_runtime_metrics(metrics: dict, *, elapsed_seconds: float) -> dict:
    enriched = dict(metrics)
    elapsed = round(float(elapsed_seconds), 3)
    enriched["elapsed_seconds"] = elapsed
    hands = enriched.get("hands")
    if hands:
        enriched["seconds_per_hand"] = round(elapsed / float(hands), 3)
    return enriched
