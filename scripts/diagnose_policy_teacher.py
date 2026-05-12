#!/usr/bin/env python3
"""Diagnose policy-calibration teacher collapse on public-state hand sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report collapse metrics for a policy-calibration teacher."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--cases-json")
    parser.add_argument("--sampled-cases", type=int, default=0)
    parser.add_argument("--hands-per-case", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy-source", choices=("regret", "policy-head"), default="regret")
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    from poker_ai.research.policy_calibration import diagnose_policy_calibration_teacher
    from poker_ai.research.resolver_benchmark import load_cases_json
    from poker_ai.research.search_targets import sample_resolver_cases

    if args.cases_json:
        cases = load_cases_json(args.cases_json)
    elif args.sampled_cases > 0:
        cases = sample_resolver_cases(args.sampled_cases, seed=args.seed)
    else:
        parser.error("requires --cases-json or --sampled-cases")

    metrics = diagnose_policy_calibration_teacher(
        cases,
        checkpoint=args.checkpoint,
        seed=args.seed,
        strategy_source=args.strategy_source,
        device=args.device,
        hands_per_case=args.hands_per_case,
        target_temperature=args.target_temperature,
    )
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
