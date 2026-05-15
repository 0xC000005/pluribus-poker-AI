#!/usr/bin/env python3
"""Train a diagnostic Slumbot opponent-response likelihood probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.slumbot_opponent_response_probe import (  # noqa: E402
    train_opponent_response_probe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a held-out diagnostic opponent-response probe."
    )
    parser.add_argument(
        "--action-likelihood",
        required=True,
        help="Output JSON from diagnose_slumbot_trace_action_likelihood.py.",
    )
    parser.add_argument("--output")
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args(argv)

    metrics = train_opponent_response_probe(
        args.action_likelihood,
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
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
