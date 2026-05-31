#!/usr/bin/env python3
"""Build an empirical payoff matrix from native duplicate-swapped H2H records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.empirical_game import (  # noqa: E402
    build_empirical_payoff_matrix,
    solve_zero_sum_meta_strategy,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", help="Duplicate-swapped H2H JSON artifacts.")
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help="Policy/checkpoint to include. May be supplied multiple times to expose missing pairs.",
    )
    parser.add_argument(
        "--solve-meta-strategy",
        action="store_true",
        help="Use nashpy to solve a complete zero-sum empirical matrix.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    summary = build_empirical_payoff_matrix(args.records, policies=args.policies)
    if args.solve_meta_strategy:
        if summary["recommendation"] != "solve_meta_strategy":
            summary["meta_strategy"] = {
                "status": "skipped_incomplete_matrix",
                "missing_pairs": summary["missing_pairs"],
            }
        else:
            summary["meta_strategy"] = solve_zero_sum_meta_strategy(summary["payoff_matrix"])
    else:
        summary["meta_strategy"] = {"status": "not_requested"}

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
