#!/usr/bin/env python3
"""Analyze empirical opponent-response baselines for Slumbot trace actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.slumbot_opponent_response_baseline import (  # noqa: E402
    analyze_opponent_response_baseline,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare model Slumbot action likelihood with empirical LOO priors."
    )
    parser.add_argument(
        "--action-likelihood",
        required=True,
        help="Output JSON from diagnose_slumbot_trace_action_likelihood.py.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Symmetric legal-action smoothing for empirical priors.",
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    metrics = analyze_opponent_response_baseline(
        args.action_likelihood,
        alpha=args.alpha,
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
