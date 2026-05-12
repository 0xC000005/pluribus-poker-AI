#!/usr/bin/env python3
"""Build bounded resolver policy targets for search-consistency training."""

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
        description="Generate .npz search-consistency targets from resolver cases."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases-json")
    parser.add_argument(
        "--sampled-cases",
        type=int,
        default=0,
        help="Generate this many sampled turn/river cases instead of fixed defaults.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    parser.add_argument(
        "--range-checkpoint",
        help="Optional blueprint checkpoint used to generate belief-conditioned ranges.",
    )
    parser.add_argument(
        "--range-strategy-source",
        choices=("regret", "policy-head"),
        default="regret",
        help="Network source used by the range tracker when --range-checkpoint is set.",
    )
    parser.add_argument(
        "--range-device",
        default="auto",
        help="Device for range-tracker network inference: auto, cpu, cuda, etc.",
    )
    parser.add_argument(
        "--range-prune-threshold",
        type=float,
        default=1e-4,
        help="Relative hand-range pruning threshold passed to the street solver.",
    )
    args = parser.parse_args(argv)

    from poker_ai.research.search_targets import save_resolver_policy_targets

    metadata = save_resolver_policy_targets(
        args.output,
        cases_json=args.cases_json,
        sampled_cases=args.sampled_cases,
        seed=args.seed,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        range_checkpoint=args.range_checkpoint,
        range_strategy_source=args.range_strategy_source,
        range_device=args.range_device,
        range_prune_threshold=args.range_prune_threshold,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
