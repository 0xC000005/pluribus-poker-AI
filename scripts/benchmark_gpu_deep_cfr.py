#!/usr/bin/env python
"""Benchmark GPU Deep CFR iteration throughput with warmup separation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import statistics
import sys
import time
import warnings

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cuda_env import configure_numba_cuda_env  # noqa: E402

configure_numba_cuda_env()
from numba import cuda  # noqa: E402

from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer  # noqa: E402


def _mean(profiles: list[dict], key: str) -> float:
    if not profiles:
        return 0.0
    return float(statistics.fmean(float(profile[key]) for profile in profiles))


def _max(profiles: list[dict], key: str) -> float:
    if not profiles:
        return 0.0
    return max(float(profile[key]) for profile in profiles)


def evaluate_benchmark_gate_failures(
    result: dict,
    *,
    min_mean_traversals_per_second: float | None = None,
    max_pool_exhausted_per_traversal: float | None = None,
    max_overflow_chunk_fraction: float | None = None,
) -> list[str]:
    failures: list[str] = []
    if (
        min_mean_traversals_per_second is not None
        and result["mean_traversals_per_second"] < min_mean_traversals_per_second
    ):
        failures.append(
            f"mean_traversals_per_second {result['mean_traversals_per_second']:.6f} "
            f"< {min_mean_traversals_per_second:.6f}"
        )
    if (
        max_pool_exhausted_per_traversal is not None
        and result["max_pool_exhausted_per_traversal"]
        > max_pool_exhausted_per_traversal
    ):
        failures.append(
            "max_pool_exhausted_per_traversal "
            f"{result['max_pool_exhausted_per_traversal']:.6f} "
            f"> {max_pool_exhausted_per_traversal:.6f}"
        )
    if (
        max_overflow_chunk_fraction is not None
        and result["max_overflow_chunk_fraction"] > max_overflow_chunk_fraction
    ):
        failures.append(
            f"max_overflow_chunk_fraction {result['max_overflow_chunk_fraction']:.6f} "
            f"> {max_overflow_chunk_fraction:.6f}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run warmup-excluded GPU Deep CFR throughput benchmark."
    )
    parser.add_argument("--n-players", type=int, default=2)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-measure", type=int, default=3)
    parser.add_argument("--n-traversals", type=int, default=2000)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-training-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--initial-chips", type=int, default=10000)
    parser.add_argument("--traversal-pool-max-slots", type=int, default=1_000_000)
    parser.add_argument("--traversal-slots-per-traversal", type=int, default=2000)
    parser.add_argument("--policy-slots-per-traversal", type=int, default=64)
    parser.add_argument("--min-mean-traversals-per-second", type=float)
    parser.add_argument("--max-pool-exhausted-per-traversal", type=float)
    parser.add_argument("--max-overflow-chunk-fraction", type=float)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"Grid size .* will likely result in GPU under-utilization.*",
    )
    if not torch.cuda.is_available() or not cuda.is_available():
        raise SystemExit("CUDA is not available")

    trainer = GPUDeepCFRTrainer(
        n_players=args.n_players,
        buffer_capacity=args.buffer_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        n_training_steps=args.n_training_steps,
        n_traversals=args.n_traversals,
        initial_chips=args.initial_chips,
        traversal_pool_max_slots=args.traversal_pool_max_slots,
        traversal_slots_per_traversal=args.traversal_slots_per_traversal,
        policy_slots_per_traversal=args.policy_slots_per_traversal,
        device=torch.device("cuda"),
    )

    warmup_profiles: list[dict] = []
    measured_profiles: list[dict] = []
    started = time.perf_counter()
    for idx in range(int(args.n_warmup) + int(args.n_measure)):
        profile = trainer.run_iteration()
        if idx < int(args.n_warmup):
            warmup_profiles.append(profile)
        else:
            measured_profiles.append(profile)

    result = {
        "mode": "gpu_deep_cfr_benchmark",
        "warning": "Warmup iterations are excluded from aggregate throughput.",
        "device": torch.cuda.get_device_name(0),
        "n_players": int(args.n_players),
        "n_warmup": int(args.n_warmup),
        "n_measure": int(args.n_measure),
        "n_traversals": int(args.n_traversals),
        "hidden_dim": int(args.hidden_dim),
        "n_layers": int(args.n_layers),
        "batch_size": int(args.batch_size),
        "n_training_steps": int(args.n_training_steps),
        "traversal_pool_max_slots": int(args.traversal_pool_max_slots),
        "traversal_slots_per_traversal": int(args.traversal_slots_per_traversal),
        "policy_slots_per_traversal": int(args.policy_slots_per_traversal),
        "elapsed_seconds": round(float(time.perf_counter() - started), 6),
        "mean_iteration_seconds": round(_mean(measured_profiles, "iteration_seconds"), 6),
        "mean_traverse_seconds": round(_mean(measured_profiles, "traverse_seconds"), 6),
        "mean_train_seconds": round(_mean(measured_profiles, "train_seconds"), 6),
        "mean_traversals_per_second": round(
            _mean(measured_profiles, "traversals_per_second"),
            6,
        ),
        "mean_regret_samples_per_second": round(
            _mean(measured_profiles, "regret_samples_per_second"),
            6,
        ),
        "mean_train_samples_per_second": round(
            _mean(measured_profiles, "train_samples_per_second"),
            6,
        ),
        "max_pool_exhausted_per_traversal": round(
            _max(measured_profiles, "traversal_pool_exhausted_per_traversal"),
            6,
        ),
        "max_overflow_chunk_fraction": round(
            _max(measured_profiles, "traversal_overflow_chunk_fraction"),
            6,
        ),
        "warmup_profiles": warmup_profiles,
        "profiles": measured_profiles,
        "promotion": False,
    }
    failures = evaluate_benchmark_gate_failures(
        result,
        min_mean_traversals_per_second=args.min_mean_traversals_per_second,
        max_pool_exhausted_per_traversal=args.max_pool_exhausted_per_traversal,
        max_overflow_chunk_fraction=args.max_overflow_chunk_fraction,
    )
    result["gate_thresholds"] = {
        "min_mean_traversals_per_second": args.min_mean_traversals_per_second,
        "max_pool_exhausted_per_traversal": args.max_pool_exhausted_per_traversal,
        "max_overflow_chunk_fraction": args.max_overflow_chunk_fraction,
    }
    result["gate_failures"] = failures
    result["passed"] = not failures
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
