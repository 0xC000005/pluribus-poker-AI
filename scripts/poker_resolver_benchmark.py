#!/usr/bin/env python3
"""Run fixed public-state turn/river resolver diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.evaluation import load_value_network_checkpoint  # noqa: E402
from poker_ai.research.resolver_benchmark import (  # noqa: E402
    default_benchmark_cases,
    load_cases_json,
    run_resolver_benchmark,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark blueprint-vs-resolver behavior on fixed public states."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    parser.add_argument(
        "--cases-json",
        help="Optional JSON list or {'cases': [...]} of fixed resolver benchmark cases.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Limit number of cases for smoke tests.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--enforce-policy-head-behavior-gate", action="store_true")
    parser.add_argument("--max-policy-head-allin-rate", type=float, default=0.05)
    parser.add_argument(
        "--max-policy-head-solver-allin-gap",
        type=float,
        help="Use a solver-relative all-in-rate gate instead of an absolute cap.",
    )
    parser.add_argument(
        "--max-policy-head-solver-allin-prob-gap",
        type=float,
        help="Use a solver-relative mean all-in-probability gate instead of a top-action cap.",
    )
    parser.add_argument("--max-policy-head-mean-l1-drift", type=float, default=0.75)
    args = parser.parse_args(argv)

    device = _device(args.device)
    loaded = load_value_network_checkpoint(args.checkpoint, device)
    cases = load_cases_json(args.cases_json) if args.cases_json else default_benchmark_cases()
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    metrics = run_resolver_benchmark(
        loaded.value_net,
        device,
        cases=cases,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        checkpoint_metadata=loaded.metadata,
        enforce_policy_head_behavior_gate=args.enforce_policy_head_behavior_gate,
        max_policy_head_allin_rate=args.max_policy_head_allin_rate,
        max_policy_head_solver_allin_gap=args.max_policy_head_solver_allin_gap,
        max_policy_head_solver_allin_prob_gap=args.max_policy_head_solver_allin_prob_gap,
        max_policy_head_mean_l1_drift=args.max_policy_head_mean_l1_drift,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
