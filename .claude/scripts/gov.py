#!/usr/bin/env python3
"""Claude-native autoresearch governance dispatcher.

The thin CLI the skills wrap:
  python .claude/scripts/gov.py [--root DIR] <command> ...

Read-only: status, drift-status, synthesis-status, is-protected.
Mutating:  set-incumbent, new-cycle, enqueue, close-cycle, run-gate.
Every command emits a JSON document on stdout. The *decisions* (which cycle,
which gate, when) are the Claude-native driver's; this just performs the
deterministic, tested operations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checks  # noqa: E402
import session  # noqa: E402
import surfaces  # noqa: E402


def _emit(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gov.py", description="Claude-native autoresearch governance (JSON out)."
    )
    parser.add_argument("--root", default=".", help="Repo root (default: cwd).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Readiness report.")
    sub.add_parser("drift-status", help="Objective-drift guard status.")
    sub.add_parser("synthesis-status", help="Failure-synthesis due status.")

    breaker = sub.add_parser("circuit-check", help="Halt + trip STOP after N consecutive failed cycles.")
    breaker.add_argument("--max-failures", type=int, default=3)

    protected = sub.add_parser("is-protected", help="Is a path a protected eval surface?")
    protected.add_argument("path")

    incumbent = sub.add_parser("set-incumbent", help="Record the incumbent checkpoint.")
    incumbent.add_argument("--checkpoint", required=True)
    incumbent.add_argument("--reason", required=True)

    for name, help_text in [("new-cycle", "Open a research cycle."), ("enqueue", "Queue a research cycle.")]:
        cyc = sub.add_parser(name, help=help_text)
        cyc.add_argument("--hypothesis", required=True)
        cyc.add_argument("--cycle-type", required=True)
        cyc.add_argument("--failure-class", required=True)
        cyc.add_argument("--gate", required=True)

    close = sub.add_parser("close-cycle", help="Close the active research cycle.")
    close.add_argument("--run-id", required=True)
    close.add_argument("--outcome", required=True)
    close.add_argument("--failure-class", required=True)
    close.add_argument("--summary", required=True)
    close.add_argument("--metrics-path", default=None)

    gate = sub.add_parser("run-gate", help="Run a configured gate (executes its commands).")
    gate.add_argument("gate")
    gate.add_argument("--run-dir", default=None)

    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.command == "status":
        _emit(session.readiness_report(root))
    elif args.command == "drift-status":
        _emit(checks.research_drift_status(root))
    elif args.command == "synthesis-status":
        _emit(checks.synthesis_status(root))
    elif args.command == "circuit-check":
        status = checks.circuit_breaker_status(root, max_consecutive_failures=args.max_failures)
        if status["halt"]:
            session.stop_path(root).touch()  # trip the breaker
            status["tripped_stop"] = True
        _emit(status)
    elif args.command == "is-protected":
        _emit({"path": args.path, "protected": surfaces.is_protected(args.path)})
    elif args.command == "set-incumbent":
        _emit(session.set_incumbent(root, args.checkpoint, reason=args.reason))
    elif args.command == "new-cycle":
        _emit(session.new_cycle(root, hypothesis=args.hypothesis, cycle_type=args.cycle_type,
                                failure_class=args.failure_class, gate=args.gate))
    elif args.command == "enqueue":
        _emit(session.enqueue_cycle(root, hypothesis=args.hypothesis, cycle_type=args.cycle_type,
                                    failure_class=args.failure_class, gate=args.gate))
    elif args.command == "close-cycle":
        _emit(session.close_cycle(root, args.run_id, outcome=args.outcome,
                                  failure_class=args.failure_class, summary=args.summary,
                                  metrics_path=args.metrics_path))
    elif args.command == "run-gate":
        _emit(checks.run_gate(root, args.gate, run_dir=args.run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
