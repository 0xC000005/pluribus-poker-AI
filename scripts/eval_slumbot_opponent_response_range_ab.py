#!/usr/bin/env python3
"""Run the offline Slumbot opponent-response range-update A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_ai.research.slumbot_opponent_response_range_ab import (
    evaluate_opponent_response_range_ab,
    write_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B learned opponent-response range updates on held-out Slumbot traces."
    )
    parser.add_argument("--checkpoint", required=True, help="Blueprint checkpoint.")
    parser.add_argument("--trace", required=True, help="Slumbot trace JSONL.")
    parser.add_argument(
        "--action-likelihood",
        required=True,
        help="Action-likelihood artifact used to train the response probe.",
    )
    parser.add_argument("--output", required=True, help="Output metrics JSON.")
    parser.add_argument("--strategy-source", default="regret")
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    metrics = evaluate_opponent_response_range_ab(
        args.checkpoint,
        args.trace,
        args.action_likelihood,
        strategy_source=args.strategy_source,
        holdout_fraction=args.holdout_fraction,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
    )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
