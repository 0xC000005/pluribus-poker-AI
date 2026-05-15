#!/usr/bin/env python3
"""Train the restricted action-value capacity probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.restricted_value_probe import (  # noqa: E402
    RestrictedValueProbeConfig,
    train_restricted_value_probe,
)


def build_config(argv: list[str] | None = None) -> RestrictedValueProbeConfig:
    parser = argparse.ArgumentParser(
        description="Train a diagnostic probe on restricted root action-value targets."
    )
    parser.add_argument("--train-roots", type=int, default=512)
    parser.add_argument("--holdout-roots", type=int, default=128)
    parser.add_argument("--n-equity-samples", type=int, default=512)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-checkpoint")
    args = parser.parse_args(argv)
    return RestrictedValueProbeConfig(
        train_roots=args.train_roots,
        holdout_roots=args.holdout_roots,
        n_equity_samples=args.n_equity_samples,
        initial_chips=args.initial_chips,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        output_checkpoint=args.output_checkpoint,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = build_config(argv)
    metrics = train_restricted_value_probe(cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
