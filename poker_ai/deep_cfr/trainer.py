"""Training orchestration for Deep CFR.

Manages the outer training loop with progress reporting, checkpointing,
and optional evaluation against a random opponent.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import click
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from poker_ai.deep_cfr.deep_cfr import DeepCFRTrainer, ACTION_TO_INDEX
from poker_ai.games.full_deck.state import new_game

logger = logging.getLogger("poker_ai.deep_cfr.trainer")
console = Console()


def evaluate_vs_random(
    trainer: DeepCFRTrainer, n_games: int = 500
) -> float:
    """Evaluate the trained agent against a random opponent.

    Player 0 uses the learned strategy, player 1 plays uniformly random.

    Returns
    -------
    avg_payout : float
        Average payout (chips) for the trained agent per game.
    """
    total_payout = 0.0
    for _ in range(n_games):
        state = new_game(trainer.n_players)
        while not state.is_terminal:
            if not state.current_player.is_active:
                state = state.apply_action(None)
                continue
            if state.player_i == 0:
                # Trained agent.
                strategy = trainer.get_strategy(state)
                legal_actions = [
                    a for a in state.legal_actions if a is not None
                ]
                probs = [strategy[ACTION_TO_INDEX[a]] for a in legal_actions]
                probs = np.array(probs, dtype=np.float64)
                probs /= probs.sum()
                action = np.random.choice(legal_actions, p=probs)
            else:
                # Random opponent.
                legal_actions = [
                    a for a in state.legal_actions if a is not None
                ]
                action = np.random.choice(legal_actions)
            state = state.apply_action(action)
        total_payout += state.payout[0]
    return total_payout / n_games


@click.command("train-deep-cfr")
@click.option("--n-players", default=2, type=int, help="Number of players.")
@click.option("--n-iterations", default=200, type=int, help="CFR iterations.")
@click.option(
    "--n-traversals", default=500, type=int,
    help="Game tree traversals per player per iteration.",
)
@click.option("--buffer-size", default=2_000_000, type=int, help="Reservoir buffer capacity.")
@click.option("--batch-size", default=2048, type=int, help="Training mini-batch size.")
@click.option("--lr", default=0.001, type=float, help="Learning rate.")
@click.option("--hidden-dim", default=256, type=int, help="Hidden layer size.")
@click.option(
    "--n-training-steps", default=2000, type=int,
    help="SGD steps per network retraining.",
)
@click.option(
    "--save-path", default="./models",
    type=click.Path(), help="Directory for checkpoints.",
)
@click.option(
    "--save-every", default=10, type=int,
    help="Save checkpoint every N iterations.",
)
@click.option(
    "--eval-every", default=10, type=int,
    help="Evaluate vs random every N iterations.",
)
@click.option(
    "--resume", default="", type=str,
    help="Path to checkpoint to resume from.",
)
@click.option(
    "--device", default="auto", type=str,
    help="Device: 'auto', 'cpu', or 'cuda'.",
)
def train_deep_cfr(
    n_players: int,
    n_iterations: int,
    n_traversals: int,
    buffer_size: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    n_training_steps: int,
    save_path: str,
    save_every: int,
    eval_every: int,
    resume: str,
    device: str,
):
    """Train a poker AI using Single Deep CFR."""
    # Resolve device.
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    console.print(f"[bold]Device:[/bold] {dev}")

    # Create save directory.
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create or resume trainer.
    if resume:
        trainer = DeepCFRTrainer.load(resume, device=dev)
        console.print(
            f"[green]Resumed from iteration {trainer.iteration}[/green]"
        )
    else:
        trainer = DeepCFRTrainer(
            n_players=n_players,
            buffer_capacity=buffer_size,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            lr=lr,
            n_training_steps=n_training_steps,
            n_traversals=n_traversals,
            device=dev,
        )
    console.print(
        f"[bold]Training:[/bold] {n_iterations} iterations, "
        f"{n_players} players, {n_traversals} traversals/iter"
    )
    console.print(
        f"[bold]Network:[/bold] {hidden_dim}-dim hidden, "
        f"{sum(p.numel() for p in trainer.value_net.parameters()):,} params"
    )
    console.print(
        f"[bold]Buffer:[/bold] {buffer_size:,} capacity, "
        f"batch_size={batch_size}"
    )

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Training...", total=n_iterations,
            completed=0,
        )
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

            # Periodic evaluation.
            if eval_every > 0 and i % eval_every == 0:
                avg_payout = evaluate_vs_random(trainer, n_games=500)
                console.print(
                    f"  [cyan]eval vs random: "
                    f"avg_payout={avg_payout:+.1f} chips/game[/cyan]"
                )

            # Periodic checkpoint.
            if save_every > 0 and i % save_every == 0:
                ckpt_path = save_dir / f"deep_cfr_iter_{trainer.iteration}.pt"
                trainer.save(str(ckpt_path))

    # Final save.
    final_path = save_dir / "deep_cfr_final.pt"
    trainer.save(str(final_path))
    console.print(f"[bold green]Training complete![/bold green] Saved to {final_path}")

    # Final evaluation.
    avg_payout = evaluate_vs_random(trainer, n_games=1000)
    console.print(
        f"[bold]Final eval vs random (1000 games): "
        f"avg_payout={avg_payout:+.1f} chips/game[/bold]"
    )
