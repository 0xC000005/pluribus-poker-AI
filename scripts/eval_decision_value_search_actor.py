#!/usr/bin/env python3
"""Evaluate the search-improved root actor without training a policy head."""

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
    collect_decision_value_policy_targets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--n-worlds", type=int, default=32)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json")
    return parser


def _oracle_match_rate(rows: list[dict[str, object]], *, selected_key: str) -> float:
    if not rows:
        return 0.0
    return float(
        sum(1 for row in rows if row.get(selected_key) == row.get("oracle_action"))
        / len(rows)
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = DecisionValueTargetConfig(
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
    _targets, target_metadata = collect_decision_value_policy_targets(cfg)
    rows = list(target_metadata.get("rows") or [])
    metrics = {
        "mode": "decision_value_search_actor_root_gate",
        "passed": bool(int(target_metadata.get("n_truncated_rollouts", 0)) == 0),
        "promotable": False,
        "promotion_blockers": [
            "diagnostic_only_not_exploitability_estimator",
            "requires_action_collapse_gate_before_h2h",
        ],
        "checkpoint": args.checkpoint,
        "strategy_source": args.strategy_source,
        "n_roots": int(args.n_roots),
        "n_worlds": int(args.n_worlds),
        "seed": int(args.seed),
        "eta": float(args.eta),
        "selected_action_counts": target_metadata["target_selected_action_counts"],
        "deployed_metrics": {
            "mean_selected_action_value": target_metadata["mean_selected_action_value"],
            "mean_oracle_gap": target_metadata["mean_oracle_action_value"]
            - target_metadata["mean_selected_action_value"],
            "selected_action_counts": target_metadata["selected_action_counts"],
            "oracle_match_rate": _oracle_match_rate(rows, selected_key="selected_action"),
        },
        "search_actor_metrics": {
            "mean_selected_action_value": target_metadata[
                "mean_target_selected_action_value"
            ],
            "mean_oracle_gap": target_metadata["mean_target_selected_oracle_gap"],
            "selected_action_counts": target_metadata["target_selected_action_counts"],
            "oracle_match_rate": _oracle_match_rate(
                rows,
                selected_key="target_selected_action",
            ),
        },
        "search_minus_deployed": {
            "mean_selected_action_value": target_metadata[
                "mean_target_selected_action_value"
            ]
            - target_metadata["mean_selected_action_value"],
            "mean_oracle_gap": target_metadata["mean_target_selected_oracle_gap"]
            - (
                target_metadata["mean_oracle_action_value"]
                - target_metadata["mean_selected_action_value"]
            ),
            "oracle_match_rate": _oracle_match_rate(
                rows,
                selected_key="target_selected_action",
            )
            - _oracle_match_rate(rows, selected_key="selected_action"),
        },
        "target_metadata": target_metadata,
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
