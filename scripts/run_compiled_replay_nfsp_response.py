#!/usr/bin/env python3
"""Train an NFSP-style average response from compiled native population replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.compiled_replay_nfsp_response import (  # noqa: E402
    run_compiled_replay_nfsp_response,
)


def _parse_policy(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError("policy must be KIND:CHECKPOINT")
    kind, checkpoint = spec.split(":", 1)
    kind = kind.strip()
    checkpoint = checkpoint.strip()
    if not kind or not checkpoint:
        raise argparse.ArgumentTypeError("policy must be KIND:CHECKPOINT")
    return kind, checkpoint


def _parse_meta_strategy(text: str | None) -> list[float] | None:
    if text is None or not text.strip():
        return None
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        type=_parse_policy,
        help="Population member as KIND:CHECKPOINT. May be supplied multiple times.",
    )
    parser.add_argument("--meta-strategy")
    parser.add_argument("--n-hands", type=int, default=4096)
    parser.add_argument("--collector-batch-size", type=int, default=256)
    parser.add_argument("--initial-chips", type=int, default=20000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--exploration-epsilon", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--q-train-steps", type=int, default=1000)
    parser.add_argument("--average-train-steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--response-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260850)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    metrics = run_compiled_replay_nfsp_response(
        policy_specs=list(args.policy),
        meta_strategy=_parse_meta_strategy(args.meta_strategy),
        n_hands=args.n_hands,
        collector_batch_size=args.collector_batch_size,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        exploration_epsilon=args.exploration_epsilon,
        hidden_dim=args.hidden_dim,
        q_train_steps=args.q_train_steps,
        average_train_steps=args.average_train_steps,
        train_batch_size=args.train_batch_size,
        lr=args.lr,
        response_temperature=args.response_temperature,
        seed=args.seed,
        device=args.device,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, default=str))
    return metrics


if __name__ == "__main__":
    main()
