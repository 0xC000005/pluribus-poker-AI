#!/usr/bin/env python3
"""Train a native 9-action R-NaD checkpoint from compiled self-play trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.native_rnad import run_compiled_native_rnad_learner  # noqa: E402


def _write_metrics(metrics: dict[str, Any], output_json: Path | None) -> dict[str, Any]:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(output_json)
        output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run_learner(
    *,
    train_iterations: int = 8,
    n_games: int = 512,
    collector_batch_size: int = 128,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    lr: float = 5e-5,
    target_network_avg: float = 0.001,
    seed: int = 20260602,
    device: str = "auto",
    checkpoint_in: str | Path | None = None,
    rnad_training_state_in: str | Path | None = None,
    rnad_training_state_out: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    learner_checkpoint_out: str | Path | None = None,
    parent_checkpoint_out: str | Path | None = None,
    opponent_checkpoints: Sequence[str | Path] | None = None,
    opponent_kinds: Sequence[str] | None = None,
    opponent_meta_strategy: Sequence[float] | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    metrics = run_compiled_native_rnad_learner(
        train_iterations=int(train_iterations),
        n_games=int(n_games),
        collector_batch_size=int(collector_batch_size),
        max_steps_per_game=int(max_steps_per_game),
        initial_chips=int(initial_chips),
        hidden_dim=int(hidden_dim),
        lr=float(lr),
        target_network_avg=float(target_network_avg),
        seed=int(seed),
        device=device,
        checkpoint_in=checkpoint_in,
        rnad_training_state_in=rnad_training_state_in,
        rnad_training_state_out=rnad_training_state_out,
        checkpoint_out=checkpoint_out,
        learner_checkpoint_out=learner_checkpoint_out,
        parent_checkpoint_out=parent_checkpoint_out,
        opponent_checkpoints=opponent_checkpoints,
        opponent_kinds=opponent_kinds,
        opponent_meta_strategy=opponent_meta_strategy,
    )
    return _write_metrics(metrics, Path(output_json) if output_json is not None else None)


def _parse_meta_strategy(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iterations", type=int, default=8)
    parser.add_argument("--n-games", type=int, default=512)
    parser.add_argument("--collector-batch-size", type=int, default=128)
    parser.add_argument("--max-steps-per-game", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--target-network-avg", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--rnad-training-state-in", type=Path)
    parser.add_argument("--rnad-training-state-out", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--learner-checkpoint-out", type=Path)
    parser.add_argument("--parent-checkpoint-out", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--opponent-kind", action="append", default=[])
    parser.add_argument("--opponent-meta-strategy")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_learner(
        train_iterations=args.train_iterations,
        n_games=args.n_games,
        collector_batch_size=args.collector_batch_size,
        max_steps_per_game=args.max_steps_per_game,
        initial_chips=args.initial_chips,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        target_network_avg=args.target_network_avg,
        seed=args.seed,
        device=args.device,
        checkpoint_in=args.checkpoint_in,
        rnad_training_state_in=args.rnad_training_state_in,
        rnad_training_state_out=args.rnad_training_state_out,
        checkpoint_out=args.checkpoint_out,
        learner_checkpoint_out=args.learner_checkpoint_out,
        parent_checkpoint_out=args.parent_checkpoint_out,
        opponent_checkpoints=args.opponent_checkpoint,
        opponent_kinds=args.opponent_kind,
        opponent_meta_strategy=_parse_meta_strategy(args.opponent_meta_strategy),
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
