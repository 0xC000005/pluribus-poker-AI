#!/usr/bin/env python3
"""Run the native full-deck 9-action NFSP-style pilot.

This is an autoresearch plumbing and compute diagnostic. It uses the repo's
full-deck heads-up state/action contract, but it is not a promoted Slumbot
agent or replacement for Deep CFR/resolving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.native_nfsp import NativeNFSPConfig, run_native_nfsp_pilot  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-buffer-size-to-learn", type=int, default=32)
    parser.add_argument("--anticipatory-param", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.06)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json")
    return parser


def build_config(argv: list[str] | None = None) -> NativeNFSPConfig:
    args = _parser().parse_args(argv)
    return NativeNFSPConfig(
        train_episodes=args.train_episodes,
        eval_games=args.eval_games,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        min_buffer_size_to_learn=args.min_buffer_size_to_learn,
        anticipatory_param=args.anticipatory_param,
        epsilon=args.epsilon,
        lr=args.lr,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        seed=args.seed,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = build_config(argv)
    metrics = run_native_nfsp_pilot(cfg)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    args = _parser().parse_args(argv)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
