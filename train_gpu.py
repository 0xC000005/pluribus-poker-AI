"""Train 6-player GPU Deep CFR agent."""
import warnings; warnings.filterwarnings("ignore")
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import time, os
from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

os.makedirs("models", exist_ok=True)

trainer = GPUDeepCFRTrainer(
    n_players=6,
    n_traversals=300,
    n_training_steps=500,
    buffer_capacity=2_000_000,
    hidden_dim=256,
)

print("Training 6-player Deep CFR (GPU)")
print("Config: 300 trav, 500 train steps, 256 hidden")
print("=" * 60)

t0 = time.perf_counter()
for i in range(1, 201):
    trainer.run_iteration()

    if i % 10 == 0:
        wall = time.perf_counter() - t0
        payout = trainer.evaluate(n_games=1000)
        samples = sum(len(b) for b in trainer.buffers)
        print(f"Iter {i:3d} | {wall:6.1f}s | {payout:+7.0f} chips/game | {samples:>8d} samples", flush=True)

    if i % 50 == 0:
        trainer.save(f"models/gpu_deep_cfr_6p_iter{i}.pt")

trainer.save("models/gpu_deep_cfr_6p_final.pt")
total = time.perf_counter() - t0
print(f"\nDone: 200 iters in {total:.1f}s ({total/60:.1f} min)")
