#!/usr/bin/env python3
"""Train an average-policy actor from search/public-rollout action values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from poker_ai.research.decision_value_actor import (  # noqa: E402
    DecisionValueTargetConfig,
    OnPolicyDecisionValueTargetConfig,
    run_decision_value_actor_pilot,
    run_on_policy_decision_value_actor_pilot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument(
        "--target-source",
        choices=("public-root", "on-policy"),
        default="public-root",
        help="Where to collect decision-value targets.",
    )
    parser.add_argument(
        "--actor-target",
        choices=("average-policy", "policy-head"),
        default="average-policy",
        help="Which deployed actor surface to update.",
    )
    parser.add_argument(
        "--actor-update",
        choices=("mirror-ce", "kl-q"),
        default="mirror-ce",
        help="Actor update objective for on-policy targets.",
    )
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
    parser.add_argument("--n-roots", type=int, default=32)
    parser.add_argument("--n-states", type=int)
    parser.add_argument("--n-worlds", type=int, default=32)
    parser.add_argument("--target-streets", default="0,1,2,3")
    parser.add_argument(
        "--require-streets",
        default="",
        help="Comma-separated streets that must appear in on-policy target coverage.",
    )
    parser.add_argument("--min-rows-per-required-street", type=int, default=1)
    parser.add_argument("--max-hands", type=int, default=256)
    parser.add_argument("--state-policy-greedy", action="store_true")
    parser.add_argument(
        "--behavior-policy",
        choices=("base", "search-improved"),
        default="base",
        help="Behavior policy used while collecting on-policy targets.",
    )
    parser.add_argument(
        "--search-improved-behavior-sample",
        action="store_true",
        help="Sample from the search-improved behavior target instead of taking argmax.",
    )
    parser.add_argument("--eval-roots", type=int, default=0)
    parser.add_argument("--eval-worlds", type=int)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    if args.target_source == "public-root":
        train_cfg = DecisionValueTargetConfig(
            checkpoint=args.checkpoint,
            strategy_source=args.strategy_source,
            continuation_checkpoints=tuple(args.continuation_checkpoint),
            continuation_strategy_source=args.continuation_strategy_source,
            greedy_continuation=args.greedy_continuation,
            n_roots=args.n_roots,
            n_worlds=args.n_worlds,
            initial_chips=args.initial_chips,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            max_steps_per_hand=args.max_steps_per_hand,
            seed=args.seed,
            eta=args.eta,
            device=args.device,
        )
        metrics = run_decision_value_actor_pilot(
            train_cfg=train_cfg,
            output_checkpoint=args.output_checkpoint,
            actor_target=args.actor_target,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_roots=args.eval_roots,
            eval_worlds=args.eval_worlds,
            eval_seed=args.eval_seed,
        )
    else:
        streets = tuple(
            int(item.strip())
            for item in args.target_streets.split(",")
            if item.strip()
        )
        required_streets = tuple(
            int(item.strip())
            for item in args.require_streets.split(",")
            if item.strip()
        )
        train_cfg = OnPolicyDecisionValueTargetConfig(
            checkpoint=args.checkpoint,
            strategy_source=args.strategy_source,
            continuation_checkpoints=tuple(args.continuation_checkpoint),
            continuation_strategy_source=args.continuation_strategy_source,
            greedy_continuation=args.greedy_continuation,
            state_policy_greedy=args.state_policy_greedy,
            behavior_policy=args.behavior_policy,
            search_improved_behavior_greedy=not args.search_improved_behavior_sample,
            n_states=int(args.n_states or args.n_roots),
            n_worlds=args.n_worlds,
            target_streets=streets,
            required_streets=required_streets,
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
        metrics = run_on_policy_decision_value_actor_pilot(
            train_cfg=train_cfg,
            output_checkpoint=args.output_checkpoint,
            actor_target=args.actor_target,
            actor_update=args.actor_update,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            kl_beta=args.kl_beta,
            eval_roots=args.eval_roots,
            eval_worlds=args.eval_worlds,
            eval_seed=args.eval_seed,
        )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
