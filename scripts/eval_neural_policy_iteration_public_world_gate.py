#!/usr/bin/env python3
"""Evaluate a neural policy-iteration checkpoint on held-out public-world roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.neural_policy_iteration import (  # noqa: E402
    NeuralPolicyIterationPublicWorldGateConfig,
    evaluate_neural_policy_iteration_public_world_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-roots", type=int, default=32)
    parser.add_argument("--n-worlds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument(
        "--continuation-policy",
        choices=("call", "uniform"),
        default="call",
    )
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics = evaluate_neural_policy_iteration_public_world_gate(
        NeuralPolicyIterationPublicWorldGateConfig(
            checkpoint_path=args.checkpoint,
            n_roots=args.n_roots,
            n_worlds=args.n_worlds,
            seed=args.seed,
            continuation_policy=args.continuation_policy,
            max_steps_per_hand=args.max_steps_per_hand,
            initial_chips=args.initial_chips,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            device=args.device,
        )
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
