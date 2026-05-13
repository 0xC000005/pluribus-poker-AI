#!/usr/bin/env python3
"""Run a public-belief input probe for search value generalization."""

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
        description="Compare feature-only and range-belief probes on searched values."
    )
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--holdout-targets", required=True)
    parser.add_argument("--holdout-cases", required=True)
    parser.add_argument("--range-checkpoint", required=True)
    parser.add_argument(
        "--range-strategy-source",
        choices=("regret", "policy-head", "average-policy"),
        default="regret",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    parser.add_argument("--range-prune-threshold", type=float, default=1e-4)
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-value-cache",
        help="Optional .npz cache for computed train EV labels and belief inputs.",
    )
    parser.add_argument(
        "--holdout-value-cache",
        help="Optional .npz cache for computed holdout EV labels and belief inputs.",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    from poker_ai.research.belief_value_probe import (
        run_public_belief_value_probe,
        save_metrics,
    )

    metrics = run_public_belief_value_probe(
        train_targets_npz=args.train_targets,
        train_cases_json=args.train_cases,
        holdout_targets_npz=args.holdout_targets,
        holdout_cases_json=args.holdout_cases,
        range_checkpoint=args.range_checkpoint,
        range_strategy_source=args.range_strategy_source,
        device=args.device,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        range_prune_threshold=args.range_prune_threshold,
        value_scale=args.value_scale,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        train_value_cache=args.train_value_cache,
        holdout_value_cache=args.holdout_value_cache,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
