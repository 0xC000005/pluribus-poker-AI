#!/usr/bin/env python
"""Launch GPU Deep CFR training using CUDA traversal (Numba kernels).

This uses GPUDeepCFRTrainer in poker_ai.deep_cfr.cuda.gpu_trainer to move
the traversal (self-play) to GPU, which is the main bottleneck.

Example:
  python scripts/run_gpu_deep_cfr.py \
    --n-iterations 50 --n-traversals 400 \
    --hidden-dim 256 --n-training-steps 1500 \
    --batch-size 2048 --save-path ./models \
    --eval-every 5 --save-every 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cuda_env import configure_numba_cuda_env

configure_numba_cuda_env()
from numba import cuda
import logging as _logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-players", type=int, default=2)
    ap.add_argument("--n-iterations", type=int, default=50)
    ap.add_argument("--n-traversals", type=int, default=400)
    ap.add_argument("--buffer-size", type=int, default=2_000_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-training-steps", type=int, default=1500)
    ap.add_argument("--search-targets", type=str, default="")
    ap.add_argument("--search-target-weight", type=float, default=0.0)
    ap.add_argument("--search-target-batch-size", type=int, default=0)
    ap.add_argument("--average-strategy-weight", type=float, default=0.0)
    ap.add_argument("--average-strategy-memory-capacity", type=int, default=0)
    ap.add_argument("--average-strategy-batch-size", type=int, default=0)
    ap.add_argument("--traversal-slots-per-traversal", type=int, default=2000)
    ap.add_argument("--policy-slots-per-traversal", type=int, default=64)
    ap.add_argument("--save-path", type=str, default="./models")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=10)
    args = ap.parse_args()

    console = Console()

    # Silence verbose Numba CUDA driver logs.
    try:
        _logging.getLogger('numba.cuda.cudadrv.driver').setLevel(_logging.WARNING)
    except Exception:
        pass

    if not torch.cuda.is_available() or not cuda.is_available():
        console.print("[bold red]CUDA is not available[/bold red] — cannot run GPU traversal.")
        console.print("Check NVIDIA drivers and CUDA toolkit, then retry.")
        raise SystemExit(1)

    dev = torch.device("cuda")
    console.print("[bold]Device:[/bold] cuda")

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    search_target_buffer = (
        PolicyTargetBuffer.from_npz(args.search_targets)
        if args.search_targets
        else None
    )

    trainer = GPUDeepCFRTrainer(
        n_players=args.n_players,
        buffer_capacity=args.buffer_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        n_training_steps=args.n_training_steps,
        n_traversals=args.n_traversals,
        device=dev,
        policy_target_buffer=search_target_buffer,
        policy_target_weight=args.search_target_weight,
        policy_target_batch_size=args.search_target_batch_size or None,
        traversal_slots_per_traversal=args.traversal_slots_per_traversal,
        policy_slots_per_traversal=args.policy_slots_per_traversal,
        average_strategy_memory_capacity=args.average_strategy_memory_capacity or None,
        average_strategy_weight=args.average_strategy_weight,
        average_strategy_batch_size=args.average_strategy_batch_size or None,
    )

    console.print(
        f"[bold]Training (GPU traversal):[/bold] iters={args.n_iterations}, traversals/iter={args.n_traversals}"
    )

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training...", total=args.n_iterations)
        for i in range(1, args.n_iterations + 1):
            trainer.run_iteration()
            progress.update(task, advance=1)
            if args.eval_every > 0 and i % args.eval_every == 0:
                avg = trainer.evaluate(n_games=500)
                console.print(f"  [cyan]eval vs random: {avg:+.1f} chips/game[/cyan]")
            if args.save_every > 0 and i % args.save_every == 0:
                ckpt = save_dir / f"gpu_deep_cfr_iter_{trainer.iteration}.pt"
                trainer.save(str(ckpt))

    final = save_dir / "gpu_deep_cfr_final.pt"
    trainer.save(str(final))
    console.print(f"[bold green]Done.[/bold green] Saved {final}")


if __name__ == "__main__":
    main()
