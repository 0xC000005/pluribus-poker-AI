#!/usr/bin/env python3
"""Build average-policy calibration targets from learned self-play decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate masked policy-head calibration targets."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-targets", type=int, default=2048)
    parser.add_argument(
        "--hand-sweep",
        action="store_true",
        help="Build targets by sweeping private hands at turn/river public states.",
    )
    parser.add_argument("--cases-json")
    parser.add_argument("--sampled-cases", type=int, default=0)
    parser.add_argument("--hands-per-case", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy-source", choices=("regret", "policy-head"), default="regret")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-hands", type=int)
    args = parser.parse_args(argv)

    from poker_ai.research.policy_calibration import (
        save_policy_calibration_targets,
        save_public_state_hand_sweep_targets,
    )
    from poker_ai.research.resolver_benchmark import load_cases_json
    from poker_ai.research.search_targets import sample_resolver_cases

    if args.hand_sweep:
        if args.cases_json:
            cases = load_cases_json(args.cases_json)
        elif args.sampled_cases > 0:
            cases = sample_resolver_cases(args.sampled_cases, seed=args.seed)
        else:
            parser.error("--hand-sweep requires --cases-json or --sampled-cases")
        metadata = save_public_state_hand_sweep_targets(
            args.output,
            cases,
            checkpoint=args.checkpoint,
            seed=args.seed,
            strategy_source=args.strategy_source,
            device=args.device,
            hands_per_case=args.hands_per_case,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    metadata = save_policy_calibration_targets(
        args.output,
        checkpoint=args.checkpoint,
        n_targets=args.n_targets,
        seed=args.seed,
        strategy_source=args.strategy_source,
        device=args.device,
        max_hands=args.max_hands,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
