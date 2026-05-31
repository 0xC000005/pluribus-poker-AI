#!/usr/bin/env python3
"""Evaluate a latent diagnostic selector for response-conditioned resolving."""

from __future__ import annotations

import argparse
import json

from poker_ai.research.slumbot_response_latent_conditioner import (
    evaluate_latent_response_conditioner,
    write_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a latent response selector on one EV replay artifact and evaluate another."
    )
    parser.add_argument("--train-ev-gate", required=True)
    parser.add_argument("--eval-ev-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    metrics = evaluate_latent_response_conditioner(
        args.train_ev_gate,
        args.eval_ev_gate,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        device=args.device,
        seed=args.seed,
    )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
