#!/usr/bin/env python3
"""Reweight search policy targets by incumbent disagreement."""

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
        description="Write search targets with weights proportional to incumbent disagreement."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--strategy-source",
        choices=("policy-head", "regret", "average-policy"),
        default="regret",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    from poker_ai.research.search_target_reweight import (
        reweight_targets_by_checkpoint_disagreement,
    )

    metrics = reweight_targets_by_checkpoint_disagreement(
        checkpoint=args.checkpoint,
        targets=args.targets,
        output=args.output,
        strategy_source=args.strategy_source,
        device=args.device,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
