"""6-Player Convergence experiment for Fast Deep CFR.

Trains for 30 iterations with 6 players, evaluating every 5 iterations
against random opponents. This is the Pluribus configuration — the whole
point of the project.
"""
import logging
import os
import signal
import sys
import time

import torch

from poker_ai.deep_cfr.fast_trainer import FastDeepCFRTrainer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ITERATIONS = 30
EVAL_EVERY = 5
EVAL_GAMES = 1000
SAVE_PATH = "/home/max/Documents/pluribus-poker-AI/models/6p_convergence_test.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINER_KWARGS = dict(
    n_players=6,
    buffer_capacity=1_000_000,
    hidden_dim=256,
    batch_size=2048,
    lr=0.001,
    n_training_steps=1000,
    n_traversals=300,
    n_workers=1,  # single-process = GPU batched inference
    device=DEVICE,
)

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("6p_convergence_test")

_trainer = None
_results = []
_total_start = None


def print_summary(results, total_elapsed, completed_iters, total_iters):
    """Print the summary results table."""
    print("\n" + "=" * 95)
    print("6-PLAYER CONVERGENCE TEST RESULTS (Fast Deep CFR)")
    print(f"Completed {completed_iters}/{total_iters} iterations in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Device: {DEVICE}")
    print(f"Players: 6")
    print("=" * 95)
    header = (
        f"{'Iter':>5} | {'Batch Time':>11} | {'Total Buf':>10} | "
        f"{'Avg Payout':>11} | {'Improving?'}"
    )
    print(header)
    print("-" * 75)
    for r in results:
        print(
            f"{r['iteration']:5d} | "
            f"{r['batch_time_s']:9.1f}s | "
            f"{r['buf_total']:10,} | "
            f"{r['avg_payout']:+10.2f} | "
            f"{r['improving']}"
        )
    print("=" * 95)

    if len(results) >= 2:
        first_payout = results[0]["avg_payout"]
        last_payout = results[-1]["avg_payout"]
        delta = last_payout - first_payout
        print(f"\nPayout change from iter {results[0]['iteration']} to {results[-1]['iteration']}: {delta:+.2f}")
        if last_payout > 0:
            print("CONCLUSION: Agent is WINNING vs random -- learning is occurring.")
        elif last_payout > first_payout:
            print("CONCLUSION: Agent is IMPROVING but not yet positive -- learning is occurring.")
        else:
            print("CONCLUSION: Agent may need more iterations or tuning.")
    print()


def handle_signal(signum, frame):
    total_elapsed = time.time() - _total_start if _total_start else 0
    completed = _trainer.iteration if _trainer else 0
    print(f"\n\nReceived signal {signum}, printing partial results...")
    if _trainer:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        _trainer.save(SAVE_PATH)
        print(f"Saved partial model to {SAVE_PATH}")
    print_summary(_results, total_elapsed, completed, N_ITERATIONS)
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def main():
    global _trainer, _results, _total_start

    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"6-PLAYER training: {N_ITERATIONS} iterations, eval every {EVAL_EVERY}")
    logger.info(f"Trainer config: {TRAINER_KWARGS}")

    trainer = FastDeepCFRTrainer(**TRAINER_KWARGS)
    _trainer = trainer
    param_count = sum(p.numel() for p in trainer.value_net.parameters())
    logger.info(f"Value network parameters: {param_count:,}")

    results = []
    _results = results

    batch_start = time.time()
    total_start = time.time()
    _total_start = total_start

    for iteration in range(1, N_ITERATIONS + 1):
        iter_start = time.time()
        trainer.run_iteration()
        iter_elapsed = time.time() - iter_start

        buf_sizes = [len(b) for b in trainer.buffers]
        total_buf = sum(buf_sizes)
        logger.info(
            f"Iter {iteration:3d}/{N_ITERATIONS} | "
            f"time {iter_elapsed:6.1f}s | "
            f"buffers total={total_buf:,}"
        )

        if iteration % EVAL_EVERY == 0:
            batch_elapsed = time.time() - batch_start

            eval_start = time.time()
            avg_payout = trainer.evaluate(n_games=EVAL_GAMES)
            eval_elapsed = time.time() - eval_start

            improving = ""
            if len(results) > 0:
                prev = results[-1]["avg_payout"]
                if avg_payout > prev:
                    improving = "YES (improved)"
                elif avg_payout == prev:
                    improving = "FLAT"
                else:
                    improving = "NO (regressed)"
            else:
                improving = "baseline"

            row = {
                "iteration": iteration,
                "batch_time_s": round(batch_elapsed, 1),
                "buf_total": total_buf,
                "avg_payout": round(avg_payout, 2),
                "improving": improving,
            }
            results.append(row)
            logger.info(
                f"  EVAL @ iter {iteration}: avg_payout={avg_payout:+.2f} "
                f"chips/game ({EVAL_GAMES} games, {eval_elapsed:.1f}s) | "
                f"{improving}"
            )
            batch_start = time.time()

    total_elapsed = time.time() - total_start

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    trainer.save(SAVE_PATH)
    logger.info(f"Model saved to {SAVE_PATH}")

    print_summary(results, total_elapsed, N_ITERATIONS, N_ITERATIONS)


if __name__ == "__main__":
    main()
