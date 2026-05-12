#!/usr/bin/env python3
"""Evaluate an SD-CFR-style sampled checkpoint mixture against one baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Duplicate-swapped local H2H for a linearly weighted checkpoint mixture."
    )
    parser.add_argument(
        "--candidate-glob",
        action="append",
        default=[],
        help="Glob for candidate iteration checkpoints. Repeat to combine patterns.",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        action="append",
        default=[],
        help="Explicit candidate checkpoint path. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--n-games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--seeds")
    parser.add_argument("--initial-chips", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strategy-source", choices=("regret", "policy-head"), default="regret")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    from poker_ai.research.sd_cfr_mixture import (
        discover_checkpoint_paths,
        evaluate_checkpoint_mixture_across_seeds,
    )

    patterns = [*args.candidate_glob, *args.candidate_checkpoint]
    candidate_paths = discover_checkpoint_paths(patterns)
    seeds = (
        [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        if args.seeds
        else [args.seed]
    )
    metrics = evaluate_checkpoint_mixture_across_seeds(
        candidate_paths,
        args.baseline_checkpoint,
        _device(args.device),
        n_games=args.n_games,
        seeds=seeds,
        initial_chips=args.initial_chips,
        strategy_source=args.strategy_source,
    )
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
