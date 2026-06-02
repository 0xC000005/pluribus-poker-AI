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
    parser.add_argument("--model")
    parser.add_argument(
        "--model-kind",
        choices=("auto", "deep-cfr", "tianshou-rainbow"),
        default="auto",
    )
    parser.add_argument(
        "--model-glob",
        action="append",
        default=[],
        help="Opt-in SD-CFR mixture checkpoint glob forwarded to play_slumbot.py.",
    )
    parser.add_argument(
        "--model-checkpoint",
        action="append",
        default=[],
        help="Opt-in explicit SD-CFR mixture checkpoint path forwarded to play_slumbot.py.",
    )
    parser.add_argument("--mixture-seed", type=int, default=20260521)
    parser.add_argument("--hands", type=int, default=5)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-allin", action="store_true")
    parser.add_argument("--no-solver", action="store_true")
    parser.add_argument(
        "--solver-backend",
        choices=(
            "auto",
            "cpu",
            "torch-cuda",
            "torch-cpu",
            "torch-levelsync-cuda",
            "torch-levelsync-cpu",
            "segmented-cuda",
            "segmented-cpu",
        ),
        default="auto",
    )
    parser.add_argument(
        "--solver-budget-profile",
        choices=("live", "frontier-live", "fast-live", "selective-fast-live"),
        default="live",
    )
    parser.add_argument("--solver-budget-policy")
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    parser.add_argument("--trace-jsonl")
    parser.add_argument("--api-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if not args.model and not args.model_glob and not args.model_checkpoint:
        parser.error("either --model or --model-glob/--model-checkpoint is required")

    command = [
        sys.executable,
        "scripts/play_slumbot.py",
        "--hands",
        str(args.hands),
    ]
    if args.model:
        command.extend(["--model", args.model])
        if args.model_kind != "auto":
            command.extend(["--model-kind", args.model_kind])
    for pattern in args.model_glob:
        command.extend(["--model-glob", pattern])
    for checkpoint in args.model_checkpoint:
        command.extend(["--model-checkpoint", checkpoint])
    if args.model_glob or args.model_checkpoint:
        command.extend(["--mixture-seed", str(args.mixture_seed)])
    if args.greedy:
        command.append("--greedy")
    if args.no_allin:
        command.append("--no-allin")
    if args.no_solver:
        command.append("--no-solver")
    if not args.no_solver and args.solver_backend != "auto":
        command.extend(["--solver-backend", args.solver_backend])
    if not args.no_solver and args.solver_budget_profile != "live":
        command.extend(["--solver-budget-profile", args.solver_budget_profile])
    if not args.no_solver and args.solver_budget_policy:
        command.extend(["--solver-budget-policy", args.solver_budget_policy])
    if args.strategy_source != "regret":
        command.extend(["--strategy-source", args.strategy_source])
    if args.trace_jsonl:
        command.extend(["--trace-jsonl", args.trace_jsonl])
    command.extend(["--api-timeout-seconds", str(args.api_timeout_seconds)])

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
    if args.trace_jsonl:
        trace_path = Path(args.trace_jsonl)
        metrics["trace_jsonl"] = str(trace_path)
        if trace_path.exists():
            metrics["trace_records"] = sum(1 for _ in trace_path.open(encoding="utf-8"))
    if completed.returncode != 0:
        metrics["passed"] = False
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
