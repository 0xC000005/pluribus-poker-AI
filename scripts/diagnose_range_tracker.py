#!/usr/bin/env python3
"""Run learned range-likelihood diagnostics for resolver cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.range_diagnostics import diagnose_range_likelihood  # noqa: E402
from poker_ai.research.resolver_benchmark import (  # noqa: E402
    default_benchmark_cases,
    load_cases_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure hand-conditional action likelihoods used by RangeTracker."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases-json")
    parser.add_argument("--strategy-source", choices=("regret", "policy-head", "average-policy"), default="regret")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    cases = load_cases_json(args.cases_json) if args.cases_json else default_benchmark_cases()
    metrics = diagnose_range_likelihood(
        args.checkpoint,
        cases,
        strategy_source=args.strategy_source,
        device=args.device,
        max_cases=args.max_cases,
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
