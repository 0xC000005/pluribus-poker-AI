#!/usr/bin/env python
"""Launch Fast Deep CFR training without the Click CLI.

Usage example:
  python scripts/run_fast_deep_cfr.py \
    --n-iterations 300 --n-traversals 800 \
    --hidden-dim 256 --n-training-steps 1500 \
    --batch-size 2048 --save-path ./models \
    --eval-every 10 --save-every 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from poker_ai.deep_cfr.fast_trainer import FastDeepCFRTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-players", type=int, default=2)
    ap.add_argument("--n-iterations", type=int, default=200)
    ap.add_argument("--n-traversals", type=int, default=500)
    ap.add_argument("--buffer-size", type=int, default=2_000_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-training-steps", type=int, default=2000)
    ap.add_argument("--n-workers", type=int, default=0, help="0=auto-detect")
    ap.add_argument("--save-path", type=str, default="./models")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    console = Console()
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    # Avoid CPU thread oversubscription when using many workers.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    n_workers = None if args.n_workers == 0 else args.n_workers

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = FastDeepCFRTrainer(
        n_players=args.n_players,
        buffer_capacity=args.buffer_size,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        n_training_steps=args.n_training_steps,
        n_traversals=args.n_traversals,
        n_workers=n_workers,
        device=dev,
    )

    console.print(
        f"[bold]Training:[/bold] iters={args.n_iterations}, traversals/iter={args.n_traversals}, workers={trainer.n_workers}"
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
            console.print(
                f"  iter {trainer.iteration:4d} | buffers={[len(b) for b in trainer.buffers]}"
            )

            if args.eval_every > 0 and i % args.eval_every == 0:
                avg = trainer.evaluate(n_games=500)
                console.print(f"  [cyan]eval vs random: {avg:+.1f} chips/game[/cyan]")

            if args.save_every > 0 and i % args.save_every == 0:
                ckpt = save_dir / f"fast_deep_cfr_iter_{trainer.iteration}.pt"
                trainer.save(str(ckpt))

    final = save_dir / "fast_deep_cfr_final.pt"
    trainer.save(str(final))
    console.print(f"[bold green]Done.[/bold green] Saved {final}")


if __name__ == "__main__":
    main()
