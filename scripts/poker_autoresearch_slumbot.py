#!/usr/bin/env python3
"""Run a small Slumbot smoke and emit parsed JSON metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.slumbot_eval import add_runtime_metrics, parse_slumbot_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Slumbot smoke evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hands", type=int, default=5)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-allin", action="store_true")
    parser.add_argument("--no-solver", action="store_true")
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy"),
        default="regret",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)

    command = [
        sys.executable,
        "scripts/play_slumbot.py",
        "--model",
        args.model,
        "--hands",
        str(args.hands),
    ]
    if args.greedy:
        command.append("--greedy")
    if args.no_allin:
        command.append("--no-allin")
    if args.no_solver:
        command.append("--no-solver")
    if not args.no_solver and args.solver_backend != "auto":
        command.extend(["--solver-backend", args.solver_backend])
    if args.strategy_source != "regret":
        command.extend(["--strategy-source", args.strategy_source])

    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    elapsed = time.monotonic() - started
    metrics = add_runtime_metrics(
        parse_slumbot_summary(completed.stdout),
        elapsed_seconds=elapsed,
    )
    metrics.update(
        {
            "command": command,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        }
    )
    if completed.returncode != 0:
        metrics["passed"] = False
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
