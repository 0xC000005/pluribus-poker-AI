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
    enqueue_slumbot_smoke,
    enqueue_cycle,
    init_state,
    new_cycle,
    readiness_report,
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

    slumbot = subparsers.add_parser(
        "enqueue-slumbot",
        help="Create and queue a sparse live Slumbot smoke gate for one model.",
    )
    slumbot.add_argument("--model", required=True)
    slumbot.add_argument("--hands", type=int, default=5)
    slumbot.add_argument("--greedy", action="store_true")
    slumbot.add_argument("--no-allin", action="store_true")
    slumbot.add_argument("--no-solver", action="store_true")
    slumbot.add_argument("--timeout-seconds", type=int, default=600)

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
