#!/usr/bin/env python
"""Analyze GPU Deep CFR traversal telemetry for compaction potential."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.traversal_compaction import (  # noqa: E402
    analyze_traversal_compaction_potential,
    load_traversal_profiles,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GPU Deep CFR traversal pool telemetry into a gate for "
            "enumeration-preserving active-frontier compaction."
        )
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--min-mean-allocated-to-live-ratio", type=float, default=2.0)
    parser.add_argument("--max-overflow-chunk-fraction", type=float, default=0.0)
    parser.add_argument("--max-pool-exhausted-per-traversal", type=float, default=0.0)
    args = parser.parse_args(argv)

    profiles = load_traversal_profiles(
        args.input_json,
        include_warmup=args.include_warmup,
    )
    report = analyze_traversal_compaction_potential(
        profiles,
        min_mean_allocated_to_live_ratio=args.min_mean_allocated_to_live_ratio,
        max_overflow_chunk_fraction=args.max_overflow_chunk_fraction,
        max_pool_exhausted_per_traversal=args.max_pool_exhausted_per_traversal,
    )
    report["input_json"] = args.input_json
    report["include_warmup"] = bool(args.include_warmup)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
