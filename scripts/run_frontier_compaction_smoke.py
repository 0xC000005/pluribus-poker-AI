#!/usr/bin/env python
"""Run the research-only active-frontier compaction smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.frontier_compaction_smoke import (  # noqa: E402
    run_frontier_index_interop_smoke,
    run_frontier_compaction_smoke,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test prefix-sum/scatter active-frontier compaction before "
            "changing the CUDA traversal kernel."
        )
    )
    parser.add_argument("--n-rows", type=int, default=1_000_000)
    parser.add_argument(
        "--mode",
        choices=("row-processing", "frontier-index-interop"),
        default="row-processing",
    )
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--live-density", type=float, default=0.13)
    parser.add_argument("--work-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-slot-reduction-fraction", type=float, default=0.5)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    if args.mode == "frontier-index-interop":
        report = run_frontier_index_interop_smoke(
            n_rows=args.n_rows,
            live_density=args.live_density,
            repeats=args.repeats,
            seed=args.seed,
            device=args.device,
            min_slot_reduction_fraction=args.min_slot_reduction_fraction,
        )
    else:
        report = run_frontier_compaction_smoke(
            n_rows=args.n_rows,
            feature_dim=args.feature_dim,
            live_density=args.live_density,
            work_repeats=args.work_repeats,
            repeats=args.repeats,
            seed=args.seed,
            device=args.device,
            min_slot_reduction_fraction=args.min_slot_reduction_fraction,
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
