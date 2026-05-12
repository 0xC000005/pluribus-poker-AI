#!/usr/bin/env python3
"""Evaluate checkpoint policy fit to search-generated targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate learned policy probabilities against search targets."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--strategy-source",
        choices=("policy-head", "regret", "average-policy"),
        default="policy-head",
    )
    args = parser.parse_args(argv)

    from poker_ai.research.search_target_eval import evaluate_search_targets

    metrics = evaluate_search_targets(
        args.checkpoint,
        args.targets,
        device=args.device,
        strategy_source=args.strategy_source,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
