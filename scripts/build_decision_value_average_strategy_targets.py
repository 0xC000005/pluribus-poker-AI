#!/usr/bin/env python3
"""Build decision-value targets for GPU Deep CFR average-policy training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.decision_value_actor import (  # noqa: E402
    OnPolicyDecisionValueTargetConfig,
    build_on_policy_average_strategy_target_artifact,
    build_on_policy_behavior_average_strategy_target_artifact,
    build_on_policy_trajectory_return_average_strategy_target_artifact,
)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output .npz target artifact.")
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="average-policy",
    )
    parser.add_argument("--continuation-checkpoint", action="append", default=[])
    parser.add_argument(
        "--continuation-strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
    )
    parser.add_argument("--greedy-continuation", action="store_true")
    parser.add_argument("--state-policy-greedy", action="store_true")
    parser.add_argument(
        "--behavior-policy",
        choices=("base", "search-improved"),
        default="search-improved",
    )
    parser.add_argument(
        "--target-mode",
        choices=("search-target", "behavior", "trajectory-return-behavior"),
        default="search-target",
        help=(
            "search-target trains on the soft search-improved target; behavior "
            "trains on the actual action used while collecting the trajectory; "
            "trajectory-return-behavior additionally weights those behavior "
            "actions by realized terminal return."
        ),
    )
    parser.add_argument(
        "--search-improved-behavior-sample",
        action="store_true",
        help="Sample from the search-improved behavior target instead of taking argmax.",
    )
    parser.add_argument("--n-states", type=int, default=64)
    parser.add_argument("--n-worlds", type=int, default=32)
    parser.add_argument("--target-streets", default="0,1,2,3")
    parser.add_argument("--require-streets", default="")
    parser.add_argument("--min-rows-per-required-street", type=int, default=1)
    parser.add_argument("--max-hands", type=int, default=256)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--return-temperature", type=float, default=1.0)
    parser.add_argument("--max-return-weight", type=float, default=20.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recommended-average-strategy-weight", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = OnPolicyDecisionValueTargetConfig(
        checkpoint=args.checkpoint,
        strategy_source=args.strategy_source,
        continuation_checkpoints=tuple(args.continuation_checkpoint),
        continuation_strategy_source=args.continuation_strategy_source,
        greedy_continuation=args.greedy_continuation,
        state_policy_greedy=args.state_policy_greedy,
        behavior_policy=args.behavior_policy,
        search_improved_behavior_greedy=not args.search_improved_behavior_sample,
        n_states=args.n_states,
        n_worlds=args.n_worlds,
        target_streets=_parse_int_csv(args.target_streets),
        required_streets=_parse_int_csv(args.require_streets),
        min_rows_per_required_street=args.min_rows_per_required_street,
        initial_chips=args.initial_chips,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        max_steps_per_hand=args.max_steps_per_hand,
        max_hands=args.max_hands,
        seed=args.seed,
        eta=args.eta,
        device=args.device,
    )
    if args.target_mode == "behavior":
        metrics = build_on_policy_behavior_average_strategy_target_artifact(
            cfg,
            args.output,
            output_json=args.output_json or None,
            recommended_average_strategy_weight=args.recommended_average_strategy_weight,
        )
    elif args.target_mode == "trajectory-return-behavior":
        metrics = build_on_policy_trajectory_return_average_strategy_target_artifact(
            cfg,
            args.output,
            output_json=args.output_json or None,
            recommended_average_strategy_weight=args.recommended_average_strategy_weight,
            return_temperature=args.return_temperature,
            max_return_weight=args.max_return_weight,
        )
    else:
        metrics = build_on_policy_average_strategy_target_artifact(
            cfg,
            args.output,
            output_json=args.output_json or None,
            recommended_average_strategy_weight=args.recommended_average_strategy_weight,
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
