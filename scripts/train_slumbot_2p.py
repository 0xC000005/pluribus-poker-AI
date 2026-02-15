"""Train 2-player Deep CFR for Slumbot (200BB stacks)."""
import argparse
import os
import sys
import time

import torch

sys.stdout.reconfigure(line_buffering=True)

from poker_ai.deep_cfr.fast_trainer import FastDeepCFRTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--n-iterations', type=int, default=200)
    parser.add_argument('--n-traversals', type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("2-Player Deep CFR Training — Slumbot Config (200BB)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume:
        trainer = FastDeepCFRTrainer.load(args.resume, device=device)
        trainer.n_traversals = args.n_traversals
        trainer.n_training_steps = 300
        trainer.n_workers = 1
        print(f"Resumed from iteration {trainer.iteration}")
    else:
        trainer = FastDeepCFRTrainer(
            n_players=2,
            initial_chips=20000,       # 200BB to match Slumbot
            n_traversals=args.n_traversals,
            n_training_steps=300,
            buffer_capacity=2_000_000,
            hidden_dim=256,
            batch_size=2048,
            lr=0.001,
            n_workers=1,               # Single-process: batched GPU inference
            device=device,
        )

    os.makedirs("models", exist_ok=True)
    n_iterations = args.n_iterations
    eval_every = 25
    save_every = 50

    print(f"Config: {n_iterations} iters, {trainer.n_traversals} trav, "
          f"chips={trainer.initial_chips}, device={trainer.device}")
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
