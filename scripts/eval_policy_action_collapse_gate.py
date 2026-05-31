#!/usr/bin/env python3
"""Fail a root-decision eval when selected actions collapse to one action."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.action_collapse_gate import evaluate_action_collapse_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--max-top-action-fraction", type=float, default=0.75)
    parser.add_argument("--min-distinct-actions", type=int, default=2)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = evaluate_action_collapse_file(
        args.input_json,
        max_top_action_fraction=args.max_top_action_fraction,
        min_distinct_actions=args.min_distinct_actions,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
