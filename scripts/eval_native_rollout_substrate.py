#!/usr/bin/env python3
"""Evaluate native rollout substrate parity and throughput."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.native_rollout_substrate import (  # noqa: E402
    benchmark_batched_fast_self_play_collection,
    collect_compiled_fast_self_play,
    evaluate_native_rollout_substrate,
    run_fast_state_policy_inference_throughput,
    run_batched_fast_policy_fit_smoke,
    run_batched_fast_policy_gradient_pilot,
)
from poker_ai.research.compiled_fast_rollout import run_compiled_fast_transition_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-parity-games", type=int, default=16)
    parser.add_argument("--parity-max-steps", type=int, default=64)
    parser.add_argument("--n-benchmark-games", type=int, default=256)
    parser.add_argument("--benchmark-max-steps", type=int, default=128)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260752)
    parser.add_argument("--min-speedup", type=float, default=5.0)
    parser.add_argument("--include-compiled-transition-benchmark", action="store_true")
    parser.add_argument("--compiled-benchmark-games", type=int, default=1024)
    parser.add_argument("--compiled-max-steps", type=int, default=16)
    parser.add_argument("--compiled-min-speedup", type=float, default=5.0)
    parser.add_argument("--include-policy-inference-benchmark", action="store_true")
    parser.add_argument("--policy-benchmark-games", type=int, default=256)
    parser.add_argument("--policy-batch-size", type=int, default=64)
    parser.add_argument("--policy-hidden-dim", type=int, default=128)
    parser.add_argument("--policy-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-batched-collector-benchmark", action="store_true")
    parser.add_argument("--include-compiled-collector-benchmark", action="store_true")
    parser.add_argument("--collector-games", type=int, default=256)
    parser.add_argument("--collector-batch-size", type=int, default=64)
    parser.add_argument("--collector-hidden-dim", type=int, default=128)
    parser.add_argument("--collector-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-batched-policy-fit-smoke", action="store_true")
    parser.add_argument("--policy-fit-games", type=int, default=256)
    parser.add_argument("--policy-fit-collector-batch-size", type=int, default=64)
    parser.add_argument("--policy-fit-train-batch-size", type=int, default=512)
    parser.add_argument("--policy-fit-hidden-dim", type=int, default=128)
    parser.add_argument("--policy-fit-train-steps", type=int, default=100)
    parser.add_argument("--policy-fit-lr", type=float, default=1e-3)
    parser.add_argument("--policy-fit-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-batched-policy-gradient-pilot", action="store_true")
    parser.add_argument("--pg-train-iterations", type=int, default=4)
    parser.add_argument("--pg-games-per-iteration", type=int, default=256)
    parser.add_argument("--pg-collector-batch-size", type=int, default=64)
    parser.add_argument("--pg-rollout-backend", choices=("fast-state", "compiled"), default="fast-state")
    parser.add_argument("--pg-train-batch-size", type=int, default=512)
    parser.add_argument("--pg-train-epochs", type=int, default=2)
    parser.add_argument("--pg-hidden-dim", type=int, default=128)
    parser.add_argument("--pg-lr", type=float, default=3e-4)
    parser.add_argument("--pg-entropy-weight", type=float, default=0.01)
    parser.add_argument("--pg-value-loss-weight", type=float, default=0.0)
    parser.add_argument("--pg-q-boost-lambda", type=float)
    parser.add_argument("--pg-q-loss-weight", type=float, default=1.0)
    parser.add_argument("--pg-ppo-clip-epsilon", type=float, default=0.0)
    parser.add_argument("--pg-gamma", type=float, default=1.0)
    parser.add_argument("--pg-centralized-q-critic", action="store_true")
    parser.add_argument("--pg-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pg-history-opponent-interval", type=int, default=0)
    parser.add_argument("--pg-history-opponent-capacity", type=int, default=8)
    parser.add_argument("--pg-checkpoint-out")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = evaluate_native_rollout_substrate(
        n_parity_games=args.n_parity_games,
        parity_max_steps=args.parity_max_steps,
        n_benchmark_games=args.n_benchmark_games,
        benchmark_max_steps=args.benchmark_max_steps,
        initial_chips=args.initial_chips,
        seed=args.seed,
        min_speedup=args.min_speedup,
    )
    if args.include_compiled_transition_benchmark:
        metrics["compiled_transition_benchmark"] = run_compiled_fast_transition_benchmark(
            n_games=args.compiled_benchmark_games,
            max_steps_per_game=args.compiled_max_steps,
            initial_chips=args.initial_chips,
            seed=args.seed + 5,
            min_speedup=args.compiled_min_speedup,
        )
    if args.include_policy_inference_benchmark:
        sequential = run_fast_state_policy_inference_throughput(
            mode="sequential",
            n_games=args.policy_benchmark_games,
            batch_size=args.policy_batch_size,
            max_steps_per_game=args.benchmark_max_steps,
            initial_chips=args.initial_chips,
            hidden_dim=args.policy_hidden_dim,
            device=args.policy_device,
            seed=args.seed + 10,
        )
        batched = run_fast_state_policy_inference_throughput(
            mode="batched",
            n_games=args.policy_benchmark_games,
            batch_size=args.policy_batch_size,
            max_steps_per_game=args.benchmark_max_steps,
            initial_chips=args.initial_chips,
            hidden_dim=args.policy_hidden_dim,
            device=args.policy_device,
            seed=args.seed + 10,
        )
        metrics["policy_inference_benchmark"] = {
            "sequential": sequential,
            "batched": batched,
            "speedup": float(
                batched["steps_per_second"] / max(float(sequential["steps_per_second"]), 1e-12)
            ),
        }
    if args.include_batched_collector_benchmark:
        metrics["batched_self_play_collection_benchmark"] = (
            benchmark_batched_fast_self_play_collection(
                n_games=args.collector_games,
                batch_size=args.collector_batch_size,
                max_steps_per_game=args.benchmark_max_steps,
                initial_chips=args.initial_chips,
                hidden_dim=args.collector_hidden_dim,
                device=args.collector_device,
                seed=args.seed + 20,
            )
        )
    if args.include_compiled_collector_benchmark:
        metrics["compiled_self_play_collection"] = collect_compiled_fast_self_play(
            n_games=args.collector_games,
            batch_size=args.collector_batch_size,
            max_steps_per_game=args.benchmark_max_steps,
            initial_chips=args.initial_chips,
            hidden_dim=args.collector_hidden_dim,
            device=args.collector_device,
            seed=args.seed + 25,
        )["metrics"]
    if args.include_batched_policy_fit_smoke:
        metrics["batched_policy_fit_smoke"] = run_batched_fast_policy_fit_smoke(
            n_games=args.policy_fit_games,
            collector_batch_size=args.policy_fit_collector_batch_size,
            train_batch_size=args.policy_fit_train_batch_size,
            max_steps_per_game=args.benchmark_max_steps,
            initial_chips=args.initial_chips,
            hidden_dim=args.policy_fit_hidden_dim,
            train_steps=args.policy_fit_train_steps,
            lr=args.policy_fit_lr,
            device=args.policy_fit_device,
            seed=args.seed + 30,
        )
    if args.include_batched_policy_gradient_pilot:
        metrics["batched_policy_gradient_pilot"] = run_batched_fast_policy_gradient_pilot(
            train_iterations=args.pg_train_iterations,
            games_per_iteration=args.pg_games_per_iteration,
            collector_batch_size=args.pg_collector_batch_size,
            rollout_backend=args.pg_rollout_backend,
            train_batch_size=args.pg_train_batch_size,
            train_epochs=args.pg_train_epochs,
            max_steps_per_game=args.benchmark_max_steps,
            initial_chips=args.initial_chips,
            hidden_dim=args.pg_hidden_dim,
            lr=args.pg_lr,
            entropy_weight=args.pg_entropy_weight,
            value_loss_weight=args.pg_value_loss_weight,
            q_boost_lambda=args.pg_q_boost_lambda,
            q_loss_weight=args.pg_q_loss_weight,
            ppo_clip_epsilon=args.pg_ppo_clip_epsilon,
            gamma=args.pg_gamma,
            centralized_q_critic=args.pg_centralized_q_critic,
            device=args.pg_device,
            seed=args.seed + 40,
            checkpoint_path=args.pg_checkpoint_out,
            history_opponent_interval=args.pg_history_opponent_interval,
            history_opponent_capacity=args.pg_history_opponent_capacity,
        )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
