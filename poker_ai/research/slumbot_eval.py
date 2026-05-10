"""Slumbot smoke-evaluation helpers for autoresearch."""

from __future__ import annotations

import re


_FINAL_RE = re.compile(r"FINAL:\s+(?P<hands>\d+)\s+hands\s+\|\s+(?P<total>[+-]?\d+)\s+chips")
_AVG_RE = re.compile(
    r"Avg:\s+(?P<avg>[+-]?\d+)\s+\+/-\s+(?P<ci95>\d+)\s+chips/hand"
)
_RATE_RE = re.compile(r"Rate:\s+(?P<mbb>[+-]?\d+)\s+mbb/hand")
_WIN_RE = re.compile(r"Win rate:\s+(?P<win>[0-9.]+)%")


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

    return {
        "passed": True,
        "hands": int(final.group("hands")),
        "total_chips": int(final.group("total")),
        "avg_chips_per_hand": int(avg.group("avg")),
        "ci95_chips_per_hand": int(avg.group("ci95")),
        "mbb_per_hand": int(rate.group("mbb")),
        "win_rate": float(win.group("win")) / 100.0,
    }


def add_runtime_metrics(metrics: dict, *, elapsed_seconds: float) -> dict:
    enriched = dict(metrics)
    elapsed = round(float(elapsed_seconds), 3)
    enriched["elapsed_seconds"] = elapsed
    hands = enriched.get("hands")
    if hands:
        enriched["seconds_per_hand"] = round(elapsed / float(hands), 3)
    return enriched
