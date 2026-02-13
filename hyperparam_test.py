"""Hyperparameter sensitivity test for Fast Deep CFR.

Runs three configurations sequentially (small, medium, large),
each for 10 iterations, then evaluates each vs random (1000 games).
Reports timing, buffer sizes, and performance comparison.

Uses FastDeepCFRTrainer with all 4 optimizations.
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
        "n_traversals": 200,
        "n_training_steps": 500,
        "buffer_capacity": 200_000,
    },
    "B (medium)": {
        "hidden_dim": 256,
        "n_traversals": 500,
        "n_training_steps": 1000,
        "buffer_capacity": 500_000,
    },
    "C (large/default)": {
        "hidden_dim": 256,
        "n_traversals": 1000,
        "n_training_steps": 2000,
        "buffer_capacity": 1_000_000,
    },
}


def run_config(name: str, params: dict) -> dict:
    """Train one config for N_ITERATIONS and evaluate."""
    print(f"\n{'='*60}")
    print(f"  CONFIG {name}")
    print(f"  hidden_dim={params['hidden_dim']}, "
          f"n_traversals={params['n_traversals']}, "
          f"n_training_steps={params['n_training_steps']}, "
          f"buffer_capacity={params['buffer_capacity']:,}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    trainer = FastDeepCFRTrainer(
        n_players=2,
        buffer_capacity=params["buffer_capacity"],
        hidden_dim=params["hidden_dim"],
        n_training_steps=params["n_training_steps"],
        n_traversals=params["n_traversals"],
        n_workers=1,  # single-process = GPU batched inference
        device=DEVICE,
    )

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

    # Evaluate
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
    }

    # Free memory
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_comparison(results: list[dict]):
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 95)
    print("  HYPERPARAMETER SENSITIVITY TEST RESULTS (Fast Deep CFR)")
    print("=" * 95)
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Iterations per config: {N_ITERATIONS}")
    print(f"  Eval games: {N_EVAL_GAMES}")
    print()

    # Header
    header = (
        f"{'Config':<20} | {'Total Time':>10} | {'Avg/Iter':>10} | "
        f"{'Buffer Size':>12} | {'Avg Payout':>11} | {'Est 200 iter':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        total_s = r["total_time"]
        avg_s = r["avg_iter_time"]
        buf = r["final_buffer_size"]
        payout = r["avg_payout"]
        est_m = r["est_200_iter_min"]

        line = (
            f"{r['name']:<20} | {total_s:>8.1f}s | {avg_s:>8.2f}s | "
            f"{buf:>12,} | {payout:>+10.2f} | {est_m:>9.1f} min"
        )
        print(line)

    print()

    # Per-iteration breakdown
    print("Per-iteration times (seconds):")
    print(f"{'Config':<20} | " + " | ".join(f"{'Iter '+str(i+1):>8}" for i in range(N_ITERATIONS)))
    print("-" * (22 + 11 * N_ITERATIONS))
    for r in results:
        times_str = " | ".join(f"{t:>8.2f}" for t in r["iter_times"])
        print(f"{r['name']:<20} | {times_str}")

    print()

    # Recommendation
    best_payout = max(results, key=lambda x: x["avg_payout"])
    fastest = min(results, key=lambda x: x["avg_iter_time"])

    print("RECOMMENDATION (RTX 3070 Ti, 8GB VRAM, 32GB RAM):")
    print(f"  Best performance:  {best_payout['name']} "
          f"(avg payout {best_payout['avg_payout']:+.2f})")
    print(f"  Fastest training:  {fastest['name']} "
          f"(avg {fastest['avg_iter_time']:.2f}s/iter)")

    # Compute efficiency score
    print()
    print("Efficiency (performance per training minute for 200 iters):")
    for r in results:
        if r["est_200_iter_min"] > 0:
            eff = r["avg_payout"] / r["est_200_iter_min"]
            print(f"  {r['name']:<20}: {eff:+.4f} chips/game per min")

    print()
    if best_payout["name"] == fastest["name"]:
        print(f"  >>> RECOMMENDED: Config {best_payout['name']} -- "
              f"best performance AND fastest training.")
    else:
        medium = next((r for r in results if "medium" in r["name"].lower()), None)
        if medium:
            large = next((r for r in results if "large" in r["name"].lower()), None)
            if large and medium["avg_payout"] >= large["avg_payout"] * 0.8:
                print(f"  >>> RECOMMENDED: Config {medium['name']} -- "
                      f"good balance of speed ({medium['avg_iter_time']:.2f}s/iter) "
                      f"and performance ({medium['avg_payout']:+.2f} payout). "
                      f"Full 200-iter run in ~{medium['est_200_iter_min']:.1f} min.")
            else:
                print(f"  >>> RECOMMENDED: Config {best_payout['name']} -- "
                      f"best performance ({best_payout['avg_payout']:+.2f} payout). "
                      f"Full 200-iter run in ~{best_payout['est_200_iter_min']:.1f} min.")
        else:
            print(f"  >>> RECOMMENDED: Config {best_payout['name']} -- "
                  f"best performance ({best_payout['avg_payout']:+.2f} payout).")
    print("=" * 95)


if __name__ == "__main__":
    print(f"Starting hyperparameter sensitivity test (Fast Deep CFR)")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = []
    for name, params in CONFIGS.items():
        result = run_config(name, params)
        results.append(result)

    print_comparison(results)
