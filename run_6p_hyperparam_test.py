"""6-Player Hyperparameter sensitivity test for Fast Deep CFR.

Tests different configurations for 6-player poker to find the best
tradeoff between training speed and performance.
"""
import time
import torch
import gc

from poker_ai.deep_cfr.fast_trainer import FastDeepCFRTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ITERATIONS = 10
N_EVAL_GAMES = 1000

CONFIGS = {
    "A (small/fast)": {
        "hidden_dim": 128,
        "n_traversals": 100,
        "n_training_steps": 500,
        "buffer_capacity": 500_000,
    },
    "B (medium)": {
        "hidden_dim": 256,
        "n_traversals": 300,
        "n_training_steps": 1000,
        "buffer_capacity": 1_000_000,
    },
    "C (large)": {
        "hidden_dim": 256,
        "n_traversals": 500,
        "n_training_steps": 2000,
        "buffer_capacity": 2_000_000,
    },
    "D (wide net)": {
        "hidden_dim": 512,
        "n_traversals": 300,
        "n_training_steps": 1000,
        "buffer_capacity": 1_000_000,
    },
}


def run_config(name: str, params: dict) -> dict:
    """Train one config for N_ITERATIONS and evaluate."""
    print(f"\n{'='*70}")
    print(f"  CONFIG {name}  [6 PLAYERS]")
    print(f"  hidden_dim={params['hidden_dim']}, "
          f"n_traversals={params['n_traversals']}, "
          f"n_training_steps={params['n_training_steps']}, "
          f"buffer_capacity={params['buffer_capacity']:,}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}")

    trainer = FastDeepCFRTrainer(
        n_players=6,
        buffer_capacity=params["buffer_capacity"],
        hidden_dim=params["hidden_dim"],
        n_training_steps=params["n_training_steps"],
        n_traversals=params["n_traversals"],
        n_workers=1,
        device=DEVICE,
    )
    param_count = sum(p.numel() for p in trainer.value_net.parameters())
    print(f"  Network params: {param_count:,}")

    iter_times = []
    total_start = time.time()

    for i in range(1, N_ITERATIONS + 1):
        t0 = time.time()
        trainer.run_iteration()
        elapsed = time.time() - t0
        iter_times.append(elapsed)

        buf_sizes = [len(b) for b in trainer.buffers]
        total_buf = sum(buf_sizes)
        print(f"  iter {i}/{N_ITERATIONS} | time {elapsed:.1f}s | "
              f"buffer {total_buf:,} samples")

    total_time = time.time() - total_start
    final_buf = sum(len(b) for b in trainer.buffers)

    print(f"  Evaluating vs random ({N_EVAL_GAMES} games)...")
    eval_start = time.time()
    avg_payout = trainer.evaluate(n_games=N_EVAL_GAMES)
    eval_time = time.time() - eval_start
    print(f"  Avg payout vs random: {avg_payout:+.2f} chips/game "
          f"(eval took {eval_time:.1f}s)")

    avg_iter_time = sum(iter_times) / len(iter_times)
    est_200_iter_min = (avg_iter_time * 200) / 60

    result = {
        "name": name,
        "total_time": total_time,
        "avg_iter_time": avg_iter_time,
        "iter_times": iter_times,
        "final_buffer_size": final_buf,
        "avg_payout": avg_payout,
        "est_200_iter_min": est_200_iter_min,
        "eval_time": eval_time,
        "param_count": param_count,
    }

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_comparison(results: list[dict]):
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 100)
    print("  6-PLAYER HYPERPARAMETER SENSITIVITY TEST RESULTS (Fast Deep CFR)")
    print("=" * 100)
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Players: 6")
    print(f"  Iterations per config: {N_ITERATIONS}")
    print(f"  Eval games: {N_EVAL_GAMES}")
    print()

    header = (
        f"{'Config':<20} | {'Params':>8} | {'Total':>8} | {'Avg/Iter':>10} | "
        f"{'Buffer':>12} | {'Avg Payout':>11} | {'Est 200 iter':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        line = (
            f"{r['name']:<20} | {r['param_count']:>7,} | {r['total_time']:>6.0f}s | "
            f"{r['avg_iter_time']:>8.1f}s | "
            f"{r['final_buffer_size']:>12,} | {r['avg_payout']:>+10.2f} | "
            f"{r['est_200_iter_min']:>9.1f} min"
        )
        print(line)

    print()

    print("Per-iteration times (seconds):")
    print(f"{'Config':<20} | " + " | ".join(f"{'It '+str(i+1):>7}" for i in range(N_ITERATIONS)))
    print("-" * (22 + 10 * N_ITERATIONS))
    for r in results:
        times_str = " | ".join(f"{t:>7.1f}" for t in r["iter_times"])
        print(f"{r['name']:<20} | {times_str}")

    print()

    best_payout = max(results, key=lambda x: x["avg_payout"])
    fastest = min(results, key=lambda x: x["avg_iter_time"])

    print("ANALYSIS:")
    print(f"  Best performance:  {best_payout['name']} "
          f"(avg payout {best_payout['avg_payout']:+.2f})")
    print(f"  Fastest training:  {fastest['name']} "
          f"(avg {fastest['avg_iter_time']:.1f}s/iter)")

    print()
    print("Efficiency (performance per training minute for 200 iters):")
    for r in results:
        if r["est_200_iter_min"] > 0:
            eff = r["avg_payout"] / r["est_200_iter_min"]
            print(f"  {r['name']:<20}: {eff:+.4f} chips/game per min")

    print()
    print(f"  >>> RECOMMENDED for 6-player: {best_payout['name']} "
          f"({best_payout['avg_payout']:+.2f} payout, "
          f"~{best_payout['est_200_iter_min']:.0f} min for 200 iters)")
    print("=" * 100)


if __name__ == "__main__":
    print(f"Starting 6-PLAYER hyperparameter sensitivity test (Fast Deep CFR)")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = []
    for name, params in CONFIGS.items():
        result = run_config(name, params)
        results.append(result)

    print_comparison(results)
