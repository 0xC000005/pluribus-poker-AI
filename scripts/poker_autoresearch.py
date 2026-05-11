#!/usr/bin/env python3
"""CLI entrypoint for the poker autoresearch workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.autoresearch import (  # noqa: E402
    close_cycle,
    continuous,
    enqueue_candidate_comparison,
    enqueue_gpu_training,
    enqueue_methodology_review,
    enqueue_resolver_benchmark,
    enqueue_slumbot_smoke,
    enqueue_cycle,
    init_state,
    new_cycle,
    readiness_report,
    register_research_knob,
    run_gate,
    set_incumbent,
)


def _emit(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run safe poker autoresearch workflow gates and cycles."
    )
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Repository root containing autoresearch-session/ state.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create local workflow state.")
    init.add_argument("--force", action="store_true", help="Overwrite existing state files.")

    subparsers.add_parser("status", help="Print workflow readiness as JSON.")

    incumbent = subparsers.add_parser("set-incumbent", help="Record incumbent checkpoint.")
    incumbent.add_argument("--checkpoint", required=True)
    incumbent.add_argument("--reason", required=True)

    gate = subparsers.add_parser("gate", help="Run one configured workflow gate.")
    gate.add_argument("name", help="Gate name, for example tier0.")
    gate.add_argument("--run-dir", help="Optional directory for metrics.json.")

    new = subparsers.add_parser("new-cycle", help="Open a new research cycle.")
    new.add_argument("--hypothesis", required=True)
    new.add_argument("--cycle-type", default="experiment")
    new.add_argument("--failure-class", default="eval_invalid")
    new.add_argument("--gate", default="tier0")

    enqueue = subparsers.add_parser("enqueue", help="Add a queued research cycle.")
    enqueue.add_argument("--hypothesis", required=True)
    enqueue.add_argument("--cycle-type", default="experiment")
    enqueue.add_argument("--failure-class", default="eval_invalid")
    enqueue.add_argument("--gate", default="tier0")

    compare = subparsers.add_parser(
        "enqueue-compare",
        help="Create and queue a candidate-vs-incumbent local comparison gate.",
    )
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline")
    compare.add_argument("--n-games", type=int, default=500)
    compare.add_argument("--seeds", default="20260511,20260512,20260513")
    compare.add_argument("--device", default="auto")
    compare.add_argument("--timeout-seconds", type=int, default=2400)
    compare.add_argument(
        "--head-to-head",
        action="store_true",
        help="Queue duplicate-swapped model-vs-model evaluation instead of vs-random deltas.",
    )
    compare.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head"),
        default="regret",
        help="Use advantage regret matching or the trained average-strategy policy head.",
    )

    slumbot = subparsers.add_parser(
        "enqueue-slumbot",
        help="Create and queue a sparse live Slumbot smoke gate for one model.",
    )
    slumbot.add_argument("--model", required=True)
    slumbot.add_argument("--hands", type=int, default=5)
    slumbot.add_argument("--greedy", action="store_true")
    slumbot.add_argument("--no-allin", action="store_true")
    slumbot.add_argument("--no-solver", action="store_true")
    slumbot.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    slumbot.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head"),
        default="regret",
    )
    slumbot.add_argument("--timeout-seconds", type=int, default=600)

    resolver = subparsers.add_parser(
        "enqueue-resolver",
        help="Create and queue a fixed public-state resolver benchmark gate.",
    )
    resolver.add_argument("--model", required=True)
    resolver.add_argument("--solver-iterations", type=int, default=25)
    resolver.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    resolver.add_argument("--max-cases", type=int)
    resolver.add_argument("--device", default="auto")
    resolver.add_argument("--timeout-seconds", type=int, default=1200)

    review = subparsers.add_parser(
        "enqueue-review",
        help="Create and queue a methodology review gate with verifier and related-work artifacts.",
    )
    review.add_argument("--subject", required=True)
    review.add_argument("--trigger", required=True)
    review.add_argument("--claim", required=True)
    review.add_argument("--timeout-seconds", type=int, default=600)

    knob = subparsers.add_parser(
        "add-knob",
        help="Register one persistent research knob with a mechanism and removal criterion.",
    )
    knob.add_argument("--name", required=True)
    knob.add_argument("--default", required=True)
    knob.add_argument("--failure-class", required=True)
    knob.add_argument("--mechanism", required=True)
    knob.add_argument("--rationale", required=True)
    knob.add_argument("--removal-criterion", required=True)

    train = subparsers.add_parser(
        "enqueue-train",
        help="Create and queue a GPU Deep CFR candidate-training gate.",
    )
    train.add_argument("--n-iterations", type=int, default=10)
    train.add_argument("--n-traversals", type=int, default=1000)
    train.add_argument("--n-training-steps", type=int, default=1000)
    train.add_argument("--buffer-capacity", type=int, default=2_000_000)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--n-layers", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--traversal-pool-max-slots", type=int, default=1_000_000)
    train.add_argument("--traversal-slots-per-traversal", type=int, default=500)
    train.add_argument("--save-dir")
    train.add_argument("--prefix", default="candidate")
    train.add_argument("--save-every", type=int, default=0)
    train.add_argument("--resume")
    train.add_argument("--eval-games", type=int, default=0)
    train.add_argument(
        "--auto-compare",
        action="store_true",
        help="After training passes, queue incumbent head-to-head comparisons for emitted checkpoints.",
    )
    train.add_argument("--compare-n-games", type=int, default=500)
    train.add_argument("--compare-seeds", default="20260511,20260512,20260513")
    train.add_argument("--compare-device", default="auto")
    train.add_argument("--compare-timeout-seconds", type=int, default=2400)
    train.add_argument(
        "--compare-strategy-source",
        choices=("regret", "policy-head"),
        default="regret",
        help="Strategy source to use for auto-queued checkpoint comparisons.",
    )
    train.add_argument("--timeout-seconds", type=int, default=7200)

    close = subparsers.add_parser("close-cycle", help="Close the active research cycle.")
    close.add_argument("--run-id", required=True)
    close.add_argument("--outcome", required=True, choices=["passed", "failed", "blocked"])
    close.add_argument("--failure-class", required=True)
    close.add_argument("--metrics-path")
    close.add_argument("--summary", required=True)

    loop = subparsers.add_parser("continuous", help="Run queued cycles until stopped.")
    loop.add_argument("--max-cycles", type=int)
    loop.add_argument("--sleep-seconds", type=float, default=30.0)
    loop.add_argument(
        "--max-idle-checks",
        type=int,
        help="Dry-run helper: stop after this many empty-queue checks.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.command == "init":
        _emit(init_state(root, force=args.force))
        return 0

    if args.command == "status":
        report = readiness_report(root)
        _emit(report)
        return 0 if report["ready"] else 2

    if args.command == "set-incumbent":
        _emit(set_incumbent(root, args.checkpoint, reason=args.reason))
        return 0

    if args.command == "gate":
        result = run_gate(root, args.name, run_dir=args.run_dir)
        _emit(result)
        return 0 if result["passed"] else 1

    if args.command == "new-cycle":
        _emit(
            new_cycle(
                root,
                hypothesis=args.hypothesis,
                cycle_type=args.cycle_type,
                failure_class=args.failure_class,
                gate=args.gate,
            )
        )
        return 0

    if args.command == "enqueue":
        _emit(
            enqueue_cycle(
                root,
                hypothesis=args.hypothesis,
                cycle_type=args.cycle_type,
                failure_class=args.failure_class,
                gate=args.gate,
            )
        )
        return 0

    if args.command == "enqueue-compare":
        _emit(
            enqueue_candidate_comparison(
                root,
                args.candidate,
                baseline_checkpoint=args.baseline,
                n_games=args.n_games,
                seeds=args.seeds,
                device=args.device,
                timeout_seconds=args.timeout_seconds,
                head_to_head=args.head_to_head,
                strategy_source=args.strategy_source,
            )
        )
        return 0

    if args.command == "enqueue-slumbot":
        _emit(
            enqueue_slumbot_smoke(
                root,
                args.model,
                hands=args.hands,
                greedy=args.greedy,
                no_allin=args.no_allin,
                no_solver=args.no_solver,
                strategy_source=args.strategy_source,
                solver_backend=args.solver_backend,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-resolver":
        _emit(
            enqueue_resolver_benchmark(
                root,
                args.model,
                solver_iterations=args.solver_iterations,
                solver_backend=args.solver_backend,
                max_cases=args.max_cases,
                device=args.device,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-review":
        _emit(
            enqueue_methodology_review(
                root,
                subject=args.subject,
                trigger=args.trigger,
                claim=args.claim,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "add-knob":
        _emit(
            register_research_knob(
                root,
                name=args.name,
                default=args.default,
                failure_class=args.failure_class,
                mechanism=args.mechanism,
                rationale=args.rationale,
                removal_criterion=args.removal_criterion,
            )
        )
        return 0

    if args.command == "enqueue-train":
        _emit(
            enqueue_gpu_training(
                root,
                n_iterations=args.n_iterations,
                n_traversals=args.n_traversals,
                n_training_steps=args.n_training_steps,
                buffer_capacity=args.buffer_capacity,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                batch_size=args.batch_size,
                traversal_pool_max_slots=args.traversal_pool_max_slots,
                traversal_slots_per_traversal=args.traversal_slots_per_traversal,
                save_dir=args.save_dir,
                prefix=args.prefix,
                save_every=args.save_every,
                resume=args.resume,
                eval_games=args.eval_games,
                auto_compare=args.auto_compare,
                compare_n_games=args.compare_n_games,
                compare_seeds=args.compare_seeds,
                compare_device=args.compare_device,
                compare_timeout_seconds=args.compare_timeout_seconds,
                compare_strategy_source=args.compare_strategy_source,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "close-cycle":
        _emit(
            close_cycle(
                root,
                args.run_id,
                outcome=args.outcome,
                failure_class=args.failure_class,
                metrics_path=args.metrics_path,
                summary=args.summary,
            )
        )
        return 0

    if args.command == "continuous":
        result = continuous(
            root,
            max_cycles=args.max_cycles,
            sleep_seconds=args.sleep_seconds,
            max_idle_checks=args.max_idle_checks,
        )
        _emit(result)
        return 1 if result["stopped_reason"] in {"not_ready", "gate_failed"} else 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
