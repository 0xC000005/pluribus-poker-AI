#!/usr/bin/env python3
"""Run local model-vs-random evaluation for poker autoresearch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.evaluation import (  # noqa: E402
    aggregate_seed_runs,
    assert_strategy_source_supported,
    compare_checkpoint_metrics,
    evaluate_value_nets_head_to_head,
    evaluate_value_net_vs_random,
    load_value_network_checkpoint,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _evaluate_checkpoint(
    checkpoint: str,
    device: torch.device,
    *,
    n_games: int,
    seeds: list[int],
    n_players_override: int | None,
    initial_chips_override: int | None,
    strategy_source: str,
) -> dict:
    loaded = load_value_network_checkpoint(checkpoint, device)
    assert_strategy_source_supported(loaded, strategy_source)
    metadata = dict(loaded.metadata)
    n_players = n_players_override or int(metadata["n_players"])
    initial_chips = initial_chips_override or int(metadata["initial_chips"])
    runs = [
        evaluate_value_net_vs_random(
            loaded.value_net,
            device,
            n_games=n_games,
            n_players=n_players,
            initial_chips=initial_chips,
            seed=seed,
            checkpoint_metadata=metadata,
            strategy_source=strategy_source,
        )
        for seed in seeds
    ]
    return aggregate_seed_runs(runs)


def _evaluate_head_to_head(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    device: torch.device,
    *,
    n_games: int,
    seeds: list[int],
    initial_chips_override: int | None,
    strategy_source: str,
) -> dict:
    candidate = load_value_network_checkpoint(candidate_checkpoint, device)
    baseline = load_value_network_checkpoint(baseline_checkpoint, device)
    assert_strategy_source_supported(candidate, strategy_source)
    assert_strategy_source_supported(baseline, strategy_source)
    candidate_metadata = dict(candidate.metadata)
    baseline_metadata = dict(baseline.metadata)
    initial_chips = initial_chips_override or int(candidate_metadata["initial_chips"])
    runs = [
        evaluate_value_nets_head_to_head(
            candidate.value_net,
            baseline.value_net,
            device,
            n_games=n_games,
            initial_chips=initial_chips,
            seed=seed,
            candidate_metadata=candidate_metadata,
            baseline_metadata=baseline_metadata,
            strategy_source=strategy_source,
        )
        for seed in seeds
    ]
    return aggregate_seed_runs(runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against random play.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        help="Optional incumbent/baseline checkpoint for local comparison metrics.",
    )
    parser.add_argument(
        "--head-to-head",
        action="store_true",
        help="Evaluate candidate and baseline directly with duplicate swapped seats.",
    )
    parser.add_argument("--n-games", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument(
        "--seeds",
        help="Comma-separated seeds. Overrides --seed and aggregates runs.",
    )
    parser.add_argument("--n-players", type=int)
    parser.add_argument("--initial-chips", type=int)
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head"),
        default="regret",
        help="Use advantage regret matching or the trained average-strategy policy head.",
    )
    args = parser.parse_args(argv)

    device = _device(args.device)
    seeds = (
        [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        if args.seeds
        else [args.seed]
    )
    if args.head_to_head:
        if not args.baseline_checkpoint:
            parser.error("--head-to-head requires --baseline-checkpoint")
        metrics = _evaluate_head_to_head(
            args.checkpoint,
            args.baseline_checkpoint,
            device,
            n_games=args.n_games,
            seeds=seeds,
            initial_chips_override=args.initial_chips,
            strategy_source=args.strategy_source,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0 if metrics["passed"] else 1

    metrics = _evaluate_checkpoint(
        args.checkpoint,
        device,
        n_games=args.n_games,
        seeds=seeds,
        n_players_override=args.n_players,
        initial_chips_override=args.initial_chips,
        strategy_source=args.strategy_source,
    )
    if args.baseline_checkpoint:
        baseline = _evaluate_checkpoint(
            args.baseline_checkpoint,
            device,
            n_games=args.n_games,
            seeds=seeds,
            n_players_override=args.n_players,
            initial_chips_override=args.initial_chips,
            strategy_source=args.strategy_source,
        )
        metrics = compare_checkpoint_metrics(metrics, baseline)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
