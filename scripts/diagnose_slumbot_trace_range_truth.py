#!/usr/bin/env python3
"""Score revealed Slumbot hands under trace-conditioned opponent ranges."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.slumbot_trace_range_truth import diagnose_trace_true_range  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure revealed Slumbot hand likelihood under trace-conditioned ranges."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace", required=True, help="Input Slumbot JSONL trace.")
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    metrics = diagnose_trace_true_range(
        args.checkpoint,
        args.trace,
        strategy_source=args.strategy_source,
        device=args.device,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
