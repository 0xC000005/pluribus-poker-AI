"""Continue training 6-player Deep CFR from checkpoint."""
import os, sys, time

nvvm_path = os.path.join(
    os.path.dirname(sys.executable), "..", "lib", "python3.13",
    "site-packages", "nvidia", "cuda_nvcc", "nvvm", "lib64"
)
nvvm_path = os.path.normpath(nvvm_path)
if os.path.isdir(nvvm_path):
    os.environ["LD_LIBRARY_PATH"] = nvvm_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

def main():
    checkpoint_path = "models/deep_cfr_9action_6p_final.pt"
    print("=" * 60)
    print("Continuing 6-Player Deep CFR Training — 9-Action Space")
    print(f"Resuming from: {checkpoint_path}")
    print("=" * 60)

    trainer = GPUDeepCFRTrainer.load(checkpoint_path)
    start_iter = trainer.iteration
    target_iter = start_iter + 800  # Train 800 more → total 1000

    print(f"Resuming at iter {start_iter}, training to {target_iter}")
    print(f"Buffer sizes: {[len(b) for b in trainer.buffers]}")
    print()

    total_start = time.time()

    for i in range(start_iter, target_iter):
        iter_start = time.time()
        trainer.run_iteration()
        iter_time = time.time() - iter_start

        buf_size = sum(len(b) for b in trainer.buffers)
        print(f"Iter {trainer.iteration:4d}/{target_iter} | {iter_time:.1f}s | buffer: {buf_size:,}")

        if trainer.iteration % 25 == 0:
            payout = trainer.evaluate(n_games=1000)
            elapsed = time.time() - total_start
            print(f"  >>> Eval: {payout:+.0f} chips/game vs random | "
                  f"elapsed: {elapsed/60:.1f}min")

        if trainer.iteration % 100 == 0:
            path = f"models/deep_cfr_9action_6p_iter{trainer.iteration}.pt"
            trainer.save(path)
            print(f"  >>> Saved checkpoint: {path}")

        sys.stdout.flush()

    trainer.save("models/deep_cfr_9action_6p_final.pt")
    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print(f"Training complete! Iters {start_iter} -> {target_iter} in {total_time/60:.1f}min")
    print(f"Final model: models/deep_cfr_9action_6p_final.pt")

    payout = trainer.evaluate(n_games=5000)
    print(f"Final eval (5000 games): {payout:+.0f} chips/game vs random")
    print("=" * 60)


if __name__ == "__main__":
    main()
