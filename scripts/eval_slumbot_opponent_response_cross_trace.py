#!/usr/bin/env python3
"""Train a Slumbot response probe on one trace and evaluate on another."""

from __future__ import annotations

import argparse
import json

from poker_ai.research.slumbot_opponent_response_cross_trace import (
    train_and_eval_cross_trace_response_probe,
    write_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Slumbot opponent-response action likelihood across traces."
    )
    parser.add_argument("--train-action-likelihood", required=True)
    parser.add_argument("--eval-action-likelihood", required=True)
    parser.add_argument("--output", required=True)
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

    metrics = train_and_eval_cross_trace_response_probe(
        args.train_action_likelihood,
        args.eval_action_likelihood,
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
