#!/usr/bin/env python3
"""Evaluate a duplicate-swapped league for native policy checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.native_policy_league import (  # noqa: E402
    evaluate_native_policy_league,
    load_native_policy_league_records,
    summarize_native_policy_league,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument(
        "--record-json",
        action="append",
        default=[],
        help="Existing native_policy_h2h JSON artifact to summarize without rerunning evaluation.",
    )
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--strategy-source", choices=("auto", "actor", "average"), default="auto")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--require-best-positive-lower95",
        action="store_true",
        help="Exit nonzero unless the selected checkpoint has positive worst lower95.",
    )
    args = parser.parse_args(argv)

    if args.record_json:
        metrics = summarize_native_policy_league(
            load_native_policy_league_records(args.record_json)
        )
        metrics.update(
            {
                "record_json_paths": [str(path) for path in args.record_json],
                "mode": "summarize_existing_records",
            }
        )
    else:
        if not args.candidate or not args.baseline:
            parser.error("--candidate and --baseline are required unless --record-json is supplied")
        metrics = evaluate_native_policy_league(
            args.candidate,
            args.baseline,
            n_games=args.n_games,
            device=args.device,
            seed=args.seed,
            strategy_source=args.strategy_source,
        )
        metrics["mode"] = "run_evaluation"
    if args.require_best_positive_lower95 and float(metrics.get("best_worst_lower95", -1.0)) <= 0.0:
        metrics["passed"] = False
        metrics.setdefault("errors", []).append("best_checkpoint_requires_positive_worst_lower95")
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
