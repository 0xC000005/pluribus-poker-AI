#!/usr/bin/env python3
"""Create deterministic stratified train/holdout search-target splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_input(text: str):
    from poker_ai.research.search_target_split import SearchTargetSplitInput

    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "--input must be TARGETS_NPZ:CASES_JSON or TARGETS_NPZ:CASES_JSON:CFV_CACHE_NPZ"
        )
    return SearchTargetSplitInput(
        targets_npz=Path(parts[0]),
        cases_json=Path(parts[1]),
        cfv_cache_npz=Path(parts[2]) if len(parts) == 3 else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine search-target artifacts and write stratified train/holdout "
            "splits. With CFV caches, strata are street plus searched-CFV mean bins."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=_parse_input,
        help="TARGETS_NPZ:CASES_JSON[:CFV_CACHE_NPZ]. Repeat for multiple artifacts.",
    )
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--holdout-targets", required=True)
    parser.add_argument("--holdout-cases", required=True)
    parser.add_argument("--train-cfv-cache")
    parser.add_argument("--holdout-cfv-cache")
    parser.add_argument("--metadata-json")
    parser.add_argument("--train-size", required=True, type=int)
    parser.add_argument("--holdout-size", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfv-bins", type=int, default=4)
    args = parser.parse_args(argv)

    from poker_ai.research.search_target_split import (
        SearchTargetSplitOutput,
        split_search_targets_stratified,
    )

    metadata = split_search_targets_stratified(
        args.input,
        SearchTargetSplitOutput(
            train_targets_npz=Path(args.train_targets),
            train_cases_json=Path(args.train_cases),
            holdout_targets_npz=Path(args.holdout_targets),
            holdout_cases_json=Path(args.holdout_cases),
            train_cfv_cache_npz=Path(args.train_cfv_cache) if args.train_cfv_cache else None,
            holdout_cfv_cache_npz=Path(args.holdout_cfv_cache) if args.holdout_cfv_cache else None,
            metadata_json=Path(args.metadata_json) if args.metadata_json else None,
        ),
        train_size=args.train_size,
        holdout_size=args.holdout_size,
        seed=args.seed,
        cfv_bins=args.cfv_bins,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
