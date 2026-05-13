#!/usr/bin/env python3
"""Duplicate-swapped round-robin evaluation for checkpoint history."""

from __future__ import annotations

import argparse
import itertools
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
    evaluate_value_nets_head_to_head,
    load_value_network_checkpoint,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_seeds(seed_text: str) -> list[int]:
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _pair_metrics(
    left,
    right,
    device: torch.device,
    *,
    n_games: int,
    seeds: list[int],
    initial_chips: int,
    strategy_source: str,
) -> dict:
    runs = [
        evaluate_value_nets_head_to_head(
            left.value_net,
            right.value_net,
            device,
            n_games=n_games,
            initial_chips=initial_chips,
            seed=seed,
            candidate_metadata=left.metadata,
            baseline_metadata=right.metadata,
            strategy_source=strategy_source,
        )
        for seed in seeds
    ]
    return aggregate_seed_runs(runs)


def evaluate_round_robin(
    checkpoints: list[str | Path],
    device: torch.device,
    *,
    n_games: int,
    seeds: list[int],
    initial_chips: int | None = None,
    strategy_source: str = "regret",
) -> dict:
    loaded = [load_value_network_checkpoint(path, device) for path in checkpoints]
    for checkpoint in loaded:
        assert_strategy_source_supported(checkpoint, strategy_source)
    if len(loaded) < 2:
        raise ValueError("round-robin requires at least two checkpoints")

    labels = [str(item.metadata["checkpoint"]) for item in loaded]
    resolved_initial_chips = int(
        initial_chips or loaded[0].metadata.get("initial_chips") or 20_000
    )
    mean_matrix = [[None for _ in loaded] for _ in loaded]
    lower95_matrix = [[None for _ in loaded] for _ in loaded]
    for i in range(len(loaded)):
        mean_matrix[i][i] = 0.0
        lower95_matrix[i][i] = 0.0

    pairs = []
    for i, j in itertools.combinations(range(len(loaded)), 2):
        metrics = _pair_metrics(
            loaded[i],
            loaded[j],
            device,
            n_games=n_games,
            seeds=seeds,
            initial_chips=resolved_initial_chips,
            strategy_source=strategy_source,
        )
        avg = float(metrics["avg_chips_per_hand"])
        ci95 = float(
            metrics.get("ci95_chips_per_hand_across_seeds")
            or metrics.get("ci95_chips_per_hand")
            or 0.0
        )
        lower = float(
            metrics.get("paired_delta_lower95_chips_per_hand_across_seeds")
            or metrics.get("paired_delta_lower95_chips_per_hand")
            or avg - ci95
        )
        mean_matrix[i][j] = avg
        mean_matrix[j][i] = -avg
        lower95_matrix[i][j] = lower
        lower95_matrix[j][i] = -avg - ci95
        pairs.append(
            {
                "left_index": i,
                "right_index": j,
                "left": labels[i],
                "right": labels[j],
                "avg_chips_per_hand": avg,
                "lower95_chips_per_hand": lower,
                "metrics": metrics,
            }
        )

    rankings = []
    for i, label in enumerate(labels):
        deltas = [
            float(mean_matrix[i][j])
            for j in range(len(labels))
            if i != j and mean_matrix[i][j] is not None
        ]
        lower_wins = sum(
            1
            for j in range(len(labels))
            if i != j and lower95_matrix[i][j] is not None and lower95_matrix[i][j] > 0
        )
        rankings.append(
            {
                "index": i,
                "checkpoint": label,
                "mean_delta_vs_field": float(sum(deltas) / max(len(deltas), 1)),
                "positive_lower95_wins": int(lower_wins),
            }
        )
    rankings.sort(
        key=lambda item: (
            item["positive_lower95_wins"],
            item["mean_delta_vs_field"],
        ),
        reverse=True,
    )
    return {
        "mode": "checkpoint_round_robin",
        "passed": all(bool(pair["metrics"].get("passed")) for pair in pairs),
        "promotable": False,
        "promotion_blockers": [
            "round_robin_is_model_selection_diagnostic_not_live_confidence",
            "local_head_to_head_requires_slumbot_confirmation",
        ],
        "strategy_source": strategy_source,
        "n_checkpoints": len(labels),
        "n_pairs": len(pairs),
        "n_games_per_pair_per_seed": int(n_games * 2),
        "seeds": [int(seed) for seed in seeds],
        "initial_chips": resolved_initial_chips,
        "labels": labels,
        "mean_delta_matrix": mean_matrix,
        "lower95_delta_matrix": lower95_matrix,
        "rankings": rankings,
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint history with duplicate-swapped H2H pairs."
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--n-games", type=int, default=300)
    parser.add_argument("--seeds", default="20260511,20260512,20260513")
    parser.add_argument("--initial-chips", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    metrics = evaluate_round_robin(
        args.checkpoint,
        _device(args.device),
        n_games=args.n_games,
        seeds=_parse_seeds(args.seeds),
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
