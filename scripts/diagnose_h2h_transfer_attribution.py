#!/usr/bin/env python3
"""Attribute duplicate-swapped H2H transfer failures by action and street."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from poker_ai.research.h2h_transfer_attribution import (  # noqa: E402
    aggregate_attribution_runs,
    evaluate_h2h_transfer_attribution,
)


def _parse_seeds(seed: int, seeds: str | None) -> list[int]:
    if not seeds:
        return [int(seed)]
    return [int(item.strip()) for item in seeds.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--initial-chips", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strategy-source", default="regret")
    parser.add_argument("--candidate-strategy-source")
    parser.add_argument("--baseline-strategy-source")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    runs = [
        evaluate_h2h_transfer_attribution(
            candidate_checkpoint=args.candidate_checkpoint,
            baseline_checkpoint=args.baseline_checkpoint,
            n_games=args.n_games,
            initial_chips=args.initial_chips,
            seed=seed,
            device=args.device,
            strategy_source=args.strategy_source,
            candidate_strategy_source=args.candidate_strategy_source,
            baseline_strategy_source=args.baseline_strategy_source,
        )
        for seed in _parse_seeds(args.seed, args.seeds)
    ]
    metrics = aggregate_attribution_runs(runs)
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
