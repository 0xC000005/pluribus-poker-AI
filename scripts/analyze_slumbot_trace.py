#!/usr/bin/env python
"""Analyze Slumbot JSONL traces for large-pot calibration failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_ai.research.slumbot_trace_analysis import analyze_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="Path to a Slumbot JSONL trace.")
    parser.add_argument("--output-json", help="Optional output JSON path.")
    parser.add_argument("--risk-bet-threshold", type=int, default=5000)
    parser.add_argument("--risk-call-threshold", type=int, default=5000)
    parser.add_argument("--low-margin-threshold", type=float, default=0.01)
    args = parser.parse_args()

    summary = analyze_trace(
        args.trace,
        risk_bet_threshold=args.risk_bet_threshold,
        risk_call_threshold=args.risk_call_threshold,
        low_margin_threshold=args.low_margin_threshold,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
