#!/usr/bin/env python3
"""Run the restricted early-action value diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.restricted_action_value import (  # noqa: E402
    RestrictedActionValueConfig,
    evaluate_restricted_action_values,
)


def build_config(argv: list[str] | None = None) -> RestrictedActionValueConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate legal first actions with a showdown-abstraction diagnostic."
    )
    parser.add_argument("--n-roots", type=int, default=64)
    parser.add_argument("--n-equity-samples", type=int, default=512)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--br-player", type=int, choices=(0,), default=0)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-positive-controls",
        action="store_true",
        help="Skip premium/trash positive controls.",
    )
    args = parser.parse_args(argv)
    return RestrictedActionValueConfig(
        n_roots=args.n_roots,
        n_equity_samples=args.n_equity_samples,
        initial_chips=args.initial_chips,
        br_player=args.br_player,
        seed=args.seed,
        include_positive_controls=not args.no_positive_controls,
        checkpoint=args.checkpoint,
        strategy_source=args.strategy_source,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = build_config(argv)
    metrics = evaluate_restricted_action_values(cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
