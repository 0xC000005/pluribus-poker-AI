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
_MAPPING_DRIFT_RE = re.compile(
    r"Mapping drift:\s+n=(?P<n>\d+)\s+mean=(?P<mean>[0-9.]+)\s+max=(?P<max>[0-9.]+)"
)


def _parse_key_value_counts(text: str) -> dict[str, int]:
    counts = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.rsplit("=", 1)
        counts[key] = int(value)
    return counts


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
    mapping_drift = _MAPPING_DRIFT_RE.search(output)
    if mapping_drift:
        metrics.update(
            {
                "mapping_drift_n": int(mapping_drift.group("n")),
                "mapping_drift_mean": float(mapping_drift.group("mean")),
                "mapping_drift_max": float(mapping_drift.group("max")),
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
