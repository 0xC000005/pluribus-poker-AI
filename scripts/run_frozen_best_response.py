#!/usr/bin/env python3
"""Run a bounded best-response diagnostic against a frozen checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.frozen_best_response import (  # noqa: E402
    FrozenBestResponseConfig,
    run_frozen_best_response,
)


def build_config(argv: list[str] | None = None) -> FrozenBestResponseConfig:
    parser = argparse.ArgumentParser(
        description="Train a small approximate BR against a frozen checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--br-player", type=int, choices=(0, 1), default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-buffer-size-to-learn", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--q-discount", type=float, default=0.99)
    parser.add_argument("--q-target-sync-interval", type=int, default=200)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    args = parser.parse_args(argv)
    return FrozenBestResponseConfig(
        checkpoint=args.checkpoint,
        train_episodes=args.train_episodes,
        eval_games=args.eval_games,
        br_player=args.br_player,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        min_buffer_size_to_learn=args.min_buffer_size_to_learn,
        epsilon=args.epsilon,
        lr=args.lr,
        q_discount=args.q_discount,
        q_target_sync_interval=args.q_target_sync_interval,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        seed=args.seed,
        device=args.device,
        strategy_source=args.strategy_source,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = build_config(argv)
    metrics = run_frozen_best_response(cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("promotion") is False else 1


if __name__ == "__main__":
    raise SystemExit(main())

