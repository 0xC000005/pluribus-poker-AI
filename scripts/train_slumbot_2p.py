"""Train 2-player Deep CFR for Slumbot (200BB stacks).

Uses GPU trainer for fast wavefront traversal (~1s/iter vs ~15s on CPU).
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Suppress noisy numba CUDA driver logging before import.
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.cuda').setLevel(logging.WARNING)

import torch

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cuda_env import configure_numba_cuda_env

configure_numba_cuda_env()
from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--n-iterations', type=int, default=1000)
    parser.add_argument('--n-traversals', type=int, default=10000)
    parser.add_argument('--n-training-steps', type=int, default=4000)
    parser.add_argument('--buffer-capacity', type=int, default=10_000_000)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--average-strategy-weight', type=float, default=0.0)
    parser.add_argument('--policy-slots-per-traversal', type=int, default=64)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--prefix', type=str, default='slumbot_2p')
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=100)
    return parser


def main():
    args = build_parser().parse_args()

    print("=" * 60)
    print("2-Player Deep CFR Training — Slumbot (GPU Trainer)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume:
        trainer = GPUDeepCFRTrainer.load(args.resume, device=device)
        trainer.n_traversals = args.n_traversals
        trainer.n_training_steps = args.n_training_steps
        trainer.average_strategy_weight = args.average_strategy_weight
        trainer.policy_slots_per_traversal = args.policy_slots_per_traversal
        print(f"Resumed from iteration {trainer.iteration}")
    else:
        trainer = GPUDeepCFRTrainer(
            n_players=2,
            initial_chips=20000,       # 200BB to match Slumbot
            n_traversals=args.n_traversals,
            n_training_steps=args.n_training_steps,
            buffer_capacity=args.buffer_capacity,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            lr=0.001,
            device=device,
            average_strategy_weight=args.average_strategy_weight,
            policy_slots_per_traversal=args.policy_slots_per_traversal,
        )

    os.makedirs(args.save_dir, exist_ok=True)
    n_iterations = args.n_iterations
    eval_every = args.eval_every
    save_every = args.save_every

    print(f"Config: {n_iterations} iters, {trainer.n_traversals} trav, "
          f"{trainer.n_training_steps} steps, batch={trainer.batch_size}, "
          f"avg-strategy-weight={trainer.average_strategy_weight}, "
          f"buf={args.buffer_capacity//1_000_000}M, chips={trainer.initial_chips}, "
          f"device={trainer.device}")
    print()

    total_start = time.time()
    start_iter = trainer.iteration

    for i in range(n_iterations):
        iter_start = time.time()
        trainer.run_iteration()
        iter_time = time.time() - iter_start

        buf_size = sum(len(b) for b in trainer.buffers)
        print(f"Iter {trainer.iteration:3d}/{start_iter + n_iterations} | {iter_time:.1f}s | buffer: {buf_size:,}")

        if eval_every > 0 and trainer.iteration % eval_every == 0:
            payout = trainer.evaluate(n_games=1000)
            elapsed = time.time() - total_start
            print(f"  >>> Eval: {payout:+.0f} chips/game vs random | "
                  f"elapsed: {elapsed/60:.1f}min")

        if save_every > 0 and trainer.iteration % save_every == 0:
            path = os.path.join(args.save_dir, f"{args.prefix}_iter{trainer.iteration}.pt")
            trainer.save(path)
            print(f"  >>> Saved: {path}")

        sys.stdout.flush()

    final_path = os.path.join(args.save_dir, f"{args.prefix}_final.pt")
    trainer.save(final_path)
    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print(f"Done! {n_iterations} iters in {total_time/60:.1f}min")
    print(f"Model: {final_path}")

    payout = trainer.evaluate(n_games=5000)
    print(f"Final eval (5K games): {payout:+.0f} chips/game vs random")
    print("=" * 60)


if __name__ == "__main__":
    main()
