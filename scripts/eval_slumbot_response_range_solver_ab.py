#!/usr/bin/env python3
"""Run fixed-state resolver A/B with calibrated Slumbot response ranges."""

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
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.resolver_benchmark import load_cases_json  # noqa: E402
from poker_ai.research.slumbot_response_solver_ab import (  # noqa: E402
    evaluate_response_range_solver_ab,
    load_trace_bot_hands,
    train_response_model_from_action_likelihood,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare incumbent RangeTracker resolver inputs against calibrated "
            "Slumbot response-model villain ranges on fixed public states."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--action-likelihood", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--trace",
        help=(
            "Optional Slumbot trace JSONL. When supplied, score revealed bot "
            "hands under both ranges."
        ),
    )
    parser.add_argument("--strategy-source", default="regret")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument(
        "--solver-backend",
        choices=(
            "auto",
            "cpu",
            "cpu-levelsync",
            "torch-cuda",
            "torch-cpu",
            "torch-levelsync-cuda",
            "torch-levelsync-cpu",
            "segmented-cuda",
            "segmented-cpu",
        ),
        default="auto",
    )
    parser.add_argument("--range-prune-threshold", type=float, default=1e-4)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args(argv)

    device = _device(args.device)
    loaded = load_value_network_checkpoint(args.checkpoint, device)
    assert_strategy_source_supported(loaded, args.strategy_source)
    cases = load_cases_json(args.cases_json)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    response_model, mean, std, temperature, response_metrics = (
        train_response_model_from_action_likelihood(
            args.action_likelihood,
            holdout_fraction=args.holdout_fraction,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed,
        )
    )
    metrics = evaluate_response_range_solver_ab(
        loaded.value_net,
        response_model,
        mean,
        std,
        temperature,
        device,
        cases=cases,
        strategy_source=args.strategy_source,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        range_prune_threshold=args.range_prune_threshold,
        bot_hands_by_hand_index=(
            load_trace_bot_hands(args.trace) if args.trace else None
        ),
    )
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_iteration": loaded.metadata.get(
                "checkpoint_iteration", loaded.metadata.get("iteration")
            ),
            "cases_json": str(args.cases_json),
            "action_likelihood": str(args.action_likelihood),
            "trace": str(args.trace) if args.trace else None,
            "device": str(device),
            "holdout_fraction": float(args.holdout_fraction),
            "hidden_dim": int(args.hidden_dim),
            "n_layers": int(args.n_layers),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "temperature": float(temperature),
            "response_model_data": response_metrics,
        }
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
