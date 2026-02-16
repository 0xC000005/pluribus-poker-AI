"""Train 2-player Deep CFR for Slumbot (200BB stacks).

Uses GPU trainer for fast wavefront traversal (~1s/iter vs ~15s on CPU).
"""
import argparse
import logging
import os
import sys
import time

# Suppress noisy numba CUDA driver logging before import.
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.cuda').setLevel(logging.WARNING)

import torch

sys.stdout.reconfigure(line_buffering=True)

# Must set LD_LIBRARY_PATH before numba import for CUDA nvvm.
nvvm = os.path.join(
    os.path.dirname(sys.executable), '..', 'lib', 'python3.13',
    'site-packages', 'nvidia', 'cuda_nvcc', 'nvvm', 'lib64',
)
if os.path.isdir(nvvm):
    os.environ.setdefault('LD_LIBRARY_PATH', '')
    if nvvm not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] = nvvm + ':' + os.environ['LD_LIBRARY_PATH']

from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--n-iterations', type=int, default=1000)
    parser.add_argument('--n-traversals', type=int, default=10000)
    parser.add_argument('--n-training-steps', type=int, default=4000)
    parser.add_argument('--buffer-capacity', type=int, default=10_000_000)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()

    print("=" * 60)
    print("2-Player Deep CFR Training — Slumbot (GPU Trainer)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume:
        trainer = GPUDeepCFRTrainer.load(args.resume, device=device)
        trainer.n_traversals = args.n_traversals
        trainer.n_training_steps = args.n_training_steps
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
        )

    os.makedirs("models", exist_ok=True)
    n_iterations = args.n_iterations
    eval_every = 50
    save_every = 100

    print(f"Config: {n_iterations} iters, {trainer.n_traversals} trav, "
          f"{trainer.n_training_steps} steps, batch={trainer.batch_size}, "
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

        if trainer.iteration % eval_every == 0:
            payout = trainer.evaluate(n_games=1000)
            elapsed = time.time() - total_start
            print(f"  >>> Eval: {payout:+.0f} chips/game vs random | "
                  f"elapsed: {elapsed/60:.1f}min")

        if trainer.iteration % save_every == 0:
            path = f"models/slumbot_2p_iter{trainer.iteration}.pt"
            trainer.save(path)
            print(f"  >>> Saved: {path}")

        sys.stdout.flush()

    trainer.save("models/slumbot_2p_final.pt")
    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print(f"Done! {n_iterations} iters in {total_time/60:.1f}min")
    print(f"Model: models/slumbot_2p_final.pt")

    payout = trainer.evaluate(n_games=5000)
    print(f"Final eval (5K games): {payout:+.0f} chips/game vs random")
    print("=" * 60)


if __name__ == "__main__":
    main()
