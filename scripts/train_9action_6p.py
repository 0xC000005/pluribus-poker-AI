"""Train 6-player Deep CFR with 9-action space."""
import os
import time

from cuda_env import configure_numba_cuda_env

configure_numba_cuda_env()
from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

def main():
    print("=" * 60)
    print("6-Player Deep CFR Training — 9-Action Space")
    print("=" * 60)

    trainer = GPUDeepCFRTrainer(
        n_players=6,
        n_traversals=300,
        n_training_steps=1000,
        buffer_capacity=2_000_000,
        hidden_dim=256,
    )

    os.makedirs("models", exist_ok=True)
    n_iterations = 200
    eval_every = 10
    save_every = 50

    print(f"Config: {n_iterations} iters, {trainer.n_traversals} traversals, "
          f"{trainer.n_training_steps} train steps, hidden={256}")
    print(f"Eval every {eval_every} iters, save every {save_every} iters")
    print()

    total_start = time.time()

    for i in range(n_iterations):
        iter_start = time.time()
        trainer.run_iteration()
        iter_time = time.time() - iter_start

        buf_size = sum(len(b) for b in trainer.buffers)
        print(f"Iter {i+1:3d}/{n_iterations} | {iter_time:.1f}s | buffer: {buf_size:,}")

        if (i + 1) % eval_every == 0:
            payout = trainer.evaluate(n_games=1000)
            elapsed = time.time() - total_start
            print(f"  >>> Eval: {payout:+.0f} chips/game vs random | "
                  f"elapsed: {elapsed/60:.1f}min")

        if (i + 1) % save_every == 0:
            path = f"models/deep_cfr_9action_6p_iter{i+1}.pt"
            trainer.save(path)
            print(f"  >>> Saved checkpoint: {path}")

        sys.stdout.flush()

    # Final save
    trainer.save("models/deep_cfr_9action_6p_final.pt")
    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print(f"Training complete! {n_iterations} iterations in {total_time/60:.1f}min")
    print(f"Final model: models/deep_cfr_9action_6p_final.pt")

    # Final evaluation
    payout = trainer.evaluate(n_games=5000)
    print(f"Final eval (5000 games): {payout:+.0f} chips/game vs random")
    print("=" * 60)


if __name__ == "__main__":
    main()
