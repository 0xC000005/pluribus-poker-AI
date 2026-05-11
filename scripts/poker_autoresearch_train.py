#!/usr/bin/env python3
"""Run a GPU Deep CFR training job and emit machine-readable metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoresearch GPU Deep CFR training gate.")
    parser.add_argument("--resume", default="", help="Optional checkpoint to resume.")
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--n-traversals", type=int, default=1000)
    parser.add_argument("--n-training-steps", type=int, default=1000)
    parser.add_argument("--buffer-capacity", type=int, default=2_000_000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--save-dir", default="models/autoresearch_gpu")
    parser.add_argument("--prefix", default="candidate")
    parser.add_argument("--eval-games", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from cuda_env import configure_numba_cuda_env

    configure_numba_cuda_env()
    import torch

    from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.resume:
        trainer = GPUDeepCFRTrainer.load(args.resume, device=device)
        trainer.n_traversals = args.n_traversals
        trainer.n_training_steps = args.n_training_steps
    else:
        trainer = GPUDeepCFRTrainer(
            n_players=2,
            initial_chips=20000,
            n_traversals=args.n_traversals,
            n_training_steps=args.n_training_steps,
            buffer_capacity=args.buffer_capacity,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            lr=0.001,
            device=device,
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    iteration_times: list[float] = []
    started = time.monotonic()
    for _ in range(args.n_iterations):
        iter_started = time.monotonic()
        trainer.run_iteration()
        iteration_times.append(time.monotonic() - iter_started)

    checkpoint = save_dir / f"{args.prefix}_final.pt"
    trainer.save(str(checkpoint))
    elapsed = time.monotonic() - started
    buffer_size = sum(len(buffer) for buffer in trainer.buffers)

    eval_chips = None
    if args.eval_games > 0:
        eval_chips = float(trainer.evaluate(n_games=args.eval_games))

    avg_iter_seconds = (
        sum(iteration_times) / len(iteration_times) if iteration_times else 0.0
    )
    metrics = {
        "passed": checkpoint.exists(),
        "mode": "autoresearch_gpu_deep_cfr_train",
        "checkpoint": str(checkpoint),
        "device": str(trainer.device),
        "n_iterations": int(args.n_iterations),
        "trainer_iteration": int(trainer.iteration),
        "n_traversals": int(args.n_traversals),
        "n_training_steps": int(args.n_training_steps),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "n_layers": int(args.n_layers),
        "buffer_capacity": int(args.buffer_capacity),
        "buffer_size": int(buffer_size),
        "elapsed_seconds": round(float(elapsed), 3),
        "avg_iter_seconds": round(float(avg_iter_seconds), 3),
        "iters_per_hour": round(3600.0 / avg_iter_seconds, 3) if avg_iter_seconds > 0 else 0.0,
        "traversals_per_second": round(
            float(args.n_iterations * args.n_traversals) / elapsed, 3
        ) if elapsed > 0 else 0.0,
        "eval_chips_per_game": eval_chips,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
