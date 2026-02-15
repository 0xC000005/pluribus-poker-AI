"""Fast Deep CFR training with all 4 optimizations.

Combines:
1. FastPokerState (numpy arrays, ~0.9μs copy vs ~2ms deepcopy)
2. Coroutine-based batched inference (batch_size=N_traversals)
3. VectorizedPokerEnv for fast evaluation
4. Multi-process traversal across CPU cores

Usage:
    poker_ai train-fast-deep-cfr --n-iterations 200 --n-traversals 500
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.deep_cfr import train_value_network
from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.fast_traverse import batched_traverse, worker_fn
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.deep_cfr.vectorized_env import fast_evaluate_vs_random

logger = logging.getLogger("poker_ai.deep_cfr.fast_trainer")
console = Console()


class FastDeepCFRTrainer:
    """Optimized Deep CFR trainer.

    Parameters
    ----------
    n_players : int
    buffer_capacity : int
    hidden_dim : int
    batch_size : int
    lr : float
    n_training_steps : int
    n_traversals : int
    n_workers : int or None
        Number of CPU worker processes for traversal. Auto-detect if None.
    device : torch.device or None
    """

    def __init__(
        self,
        n_players: int = 2,
        buffer_capacity: int = 2_000_000,
        hidden_dim: int = 256,
        batch_size: int = 2048,
        lr: float = 0.001,
        n_training_steps: int = 2000,
        n_traversals: int = 500,
        n_workers: int | None = None,
        device: torch.device | None = None,
        initial_chips: int = 10000,
    ):
        self.n_players = n_players
        self.initial_chips = initial_chips
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.n_training_steps = n_training_steps
        self.n_traversals = n_traversals

        if n_workers is None:
            self.n_workers = min(os.cpu_count() or 4, 16)
        else:
            self.n_workers = n_workers

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.buffers: List[ReservoirBuffer] = [
            ReservoirBuffer(buffer_capacity) for _ in range(n_players)
        ]
        self.value_net = ValueNetwork(
            N_FEATURES, hidden_dim, N_ACTIONS
        ).to(self.device)
        self.iteration = 0

    def run_iteration(self):
        """Run one full CFR iteration with multi-process traversal."""
        self.iteration += 1
        self.value_net.eval()

        # Prepare model weights for workers (CPU tensors for pickling).
        state_dict_cpu = {
            k: v.cpu() for k, v in self.value_net.state_dict().items()
        }

        for player_i in range(self.n_players):
            if self.n_workers <= 1:
                # Single-process: run directly (simpler, avoids mp overhead).
                buffer = ReservoirBuffer(self.buffers[player_i].capacity)
                batched_traverse(
                    n_traversals=self.n_traversals,
                    traverser=player_i,
                    value_net=self.value_net,
                    buffer=buffer,
                    iteration=self.iteration,
                    device=self.device,
                    n_players=self.n_players,
                    initial_chips=self.initial_chips,
                )
                self.buffers[player_i].merge(
                    buffer.features, buffer.iterations,
                    buffer.advantages, buffer.size,
                )
            else:
                # Multi-process traversal.
                base = self.n_traversals // self.n_workers
                remainder = self.n_traversals % self.n_workers
                worker_counts = [
                    base + (1 if w < remainder else 0)
                    for w in range(self.n_workers)
                ]
                # Filter out zero-count workers.
                worker_counts = [c for c in worker_counts if c > 0]

                with ProcessPoolExecutor(max_workers=len(worker_counts)) as pool:
                    futures = [
                        pool.submit(
                            worker_fn,
                            w, count, player_i, state_dict_cpu,
                            self.iteration, self.n_players,
                            self.buffers[player_i].capacity,
                            self.hidden_dim,
                            self.initial_chips,
                        )
                        for w, count in enumerate(worker_counts)
                    ]
                    for f in futures:
                        features, iterations, advantages, size = f.result()
                        self.buffers[player_i].merge(
                            features, iterations, advantages, size,
                        )

        # Combine buffers and retrain.
        combined = self._combine_buffers()
        if len(combined) > 0:
            self.value_net = train_value_network(
                buffer=combined,
                hidden_dim=self.hidden_dim,
                n_epochs=self.n_training_steps,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
            )

    def _combine_buffers(self) -> ReservoirBuffer:
        total_size = sum(len(b) for b in self.buffers)
        combined = ReservoirBuffer(total_size)
        for buf in self.buffers:
            combined.merge(
                buf.features, buf.iterations, buf.advantages, buf.size,
            )
        return combined

    def evaluate(self, n_games: int = 500) -> float:
        """Evaluate agent vs random using fast vectorized env."""
        return fast_evaluate_vs_random(
            self.value_net, self.device, n_games, self.n_players,
            initial_chips=self.initial_chips,
        )

    def save(self, path: str):
        torch.save(
            {
                "value_net": self.value_net.state_dict(),
                "iteration": self.iteration,
                "n_players": self.n_players,
                "hidden_dim": self.hidden_dim,
                "initial_chips": self.initial_chips,
                "buffer_sizes": [len(b) for b in self.buffers],
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> FastDeepCFRTrainer:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        trainer = cls(
            n_players=checkpoint["n_players"],
            hidden_dim=checkpoint["hidden_dim"],
            initial_chips=checkpoint.get("initial_chips", 10000),
            device=device,
        )
        trainer.value_net.load_state_dict(checkpoint["value_net"])
        trainer.iteration = checkpoint["iteration"]
        logger.info(
            f"Loaded checkpoint from {path} (iteration {trainer.iteration})"
        )
        return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("train-fast-deep-cfr")
@click.option("--n-players", default=2, type=int, help="Number of players.")
@click.option("--n-iterations", default=200, type=int, help="CFR iterations.")
@click.option(
    "--n-traversals", default=500, type=int,
    help="Game tree traversals per player per iteration.",
)
@click.option("--buffer-size", default=2_000_000, type=int)
@click.option("--batch-size", default=2048, type=int)
@click.option("--lr", default=0.001, type=float)
@click.option("--hidden-dim", default=256, type=int)
@click.option("--n-training-steps", default=2000, type=int)
@click.option("--n-workers", default=0, type=int,
              help="CPU workers for traversal (0=auto-detect).")
@click.option("--save-path", default="./models", type=click.Path())
@click.option("--save-every", default=10, type=int)
@click.option("--eval-every", default=10, type=int)
@click.option("--resume", default="", type=str)
@click.option("--device", default="auto", type=str)
def train_fast_deep_cfr(
    n_players, n_iterations, n_traversals, buffer_size,
    batch_size, lr, hidden_dim, n_training_steps,
    n_workers, save_path, save_every, eval_every, resume, device,
):
    """Train a poker AI using Fast Deep CFR (optimized)."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    if n_workers == 0:
        n_workers = None  # auto-detect

    console.print(f"[bold]Device:[/bold] {dev}")

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        trainer = FastDeepCFRTrainer.load(resume, device=dev)
        console.print(
            f"[green]Resumed from iteration {trainer.iteration}[/green]"
        )
    else:
        trainer = FastDeepCFRTrainer(
            n_players=n_players,
            buffer_capacity=buffer_size,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            lr=lr,
            n_training_steps=n_training_steps,
            n_traversals=n_traversals,
            n_workers=n_workers,
            device=dev,
        )

    console.print(
        f"[bold]Training:[/bold] {n_iterations} iterations, "
        f"{n_players} players, {n_traversals} traversals/iter, "
        f"{trainer.n_workers} workers"
    )
    console.print(
        f"[bold]Network:[/bold] {hidden_dim}-dim hidden, "
        f"{sum(p.numel() for p in trainer.value_net.parameters()):,} params"
    )

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training...", total=n_iterations)
        for i in range(1, n_iterations + 1):
            t0 = time.time()
            trainer.run_iteration()
            elapsed = time.time() - t0

            buf_sizes = [len(b) for b in trainer.buffers]
            progress.update(task, advance=1)
            console.print(
                f"  iter {trainer.iteration:4d} | "
                f"time {elapsed:5.1f}s | "
                f"buffer {sum(buf_sizes):,} samples"
            )

            if eval_every > 0 and i % eval_every == 0:
                avg_payout = trainer.evaluate(n_games=500)
                console.print(
                    f"  [cyan]eval vs random: "
                    f"avg_payout={avg_payout:+.1f} chips/game[/cyan]"
                )

            if save_every > 0 and i % save_every == 0:
                ckpt = save_dir / f"fast_deep_cfr_iter_{trainer.iteration}.pt"
                trainer.save(str(ckpt))

    final = save_dir / "fast_deep_cfr_final.pt"
    trainer.save(str(final))
    console.print(f"[bold green]Training complete![/bold green] Saved to {final}")

    avg_payout = trainer.evaluate(n_games=1000)
    console.print(
        f"[bold]Final eval (1000 games): "
        f"avg_payout={avg_payout:+.1f} chips/game[/bold]"
    )
