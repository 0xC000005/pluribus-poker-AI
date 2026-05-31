#!/usr/bin/env python3
"""Build a neural policy-target dataset from XDO-lite exact-oracle rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.xdo_target_dataset import build_xdo_policy_target_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--output", required=True, help="Output .npz policy-target artifact.")
    parser.add_argument("--output-json")
    parser.add_argument("--min-targets", type=int, default=16)
    parser.add_argument("--max-top-action-fraction", type=float, default=0.75)
    parser.add_argument("--min-mean-gap", type=float, default=0.0)
    parser.add_argument("--min-holdout-mean-gap", type=float, default=0.0)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)

    metrics = build_xdo_policy_target_dataset(
        args.records_json,
        args.output,
        output_json=args.output_json,
        min_targets=int(args.min_targets),
        max_top_action_fraction=float(args.max_top_action_fraction),
        min_mean_gap=float(args.min_mean_gap),
        min_holdout_mean_gap=float(args.min_holdout_mean_gap),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.require_pass and not bool(metrics.get("passed", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
