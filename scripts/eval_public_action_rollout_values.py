#!/usr/bin/env python3
"""Run public-information early action rollout diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.public_action_rollout_value import (  # noqa: E402
    PublicActionRolloutConfig,
    evaluate_public_action_rollout_values,
)


def build_config(argv: list[str] | None = None) -> tuple[PublicActionRolloutConfig, str | None]:
    parser = argparse.ArgumentParser(
        description="Evaluate first actions with hidden-card-safe public rollout samples."
    )
    parser.add_argument("--n-roots", type=int, default=16)
    parser.add_argument("--n-worlds", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    parser.add_argument(
        "--continuation-checkpoint",
        action="append",
        default=[],
        help="Optional fixed checkpoint used only for rollout continuation.",
    )
    parser.add_argument(
        "--continuation-strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        help="Strategy source for --continuation-checkpoint; defaults to --strategy-source.",
    )
    parser.add_argument("--greedy-continuation", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-positive-controls", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    return (
        PublicActionRolloutConfig(
            n_roots=args.n_roots,
            n_worlds=args.n_worlds,
            initial_chips=args.initial_chips,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            max_steps_per_hand=args.max_steps_per_hand,
            seed=args.seed,
            checkpoint=args.checkpoint,
            strategy_source=args.strategy_source,
            continuation_checkpoints=tuple(args.continuation_checkpoint),
            continuation_strategy_source=args.continuation_strategy_source,
            greedy_continuation=args.greedy_continuation,
            device=args.device,
            include_positive_controls=not args.no_positive_controls,
        ),
        args.output_json,
    )


def main(argv: list[str] | None = None) -> int:
    cfg, output_json = build_config(argv)
    metrics = evaluate_public_action_rollout_values(cfg)
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
