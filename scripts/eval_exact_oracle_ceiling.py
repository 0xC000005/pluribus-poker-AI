#!/usr/bin/env python3
"""Evaluate a native one-step exact-oracle ceiling for a parent checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.exact_oracle_ceiling import (  # noqa: E402
    ExactOracleCeilingConfig,
    evaluate_exact_oracle_ceiling,
    load_exact_oracle_ceiling_records,
    summarize_exact_oracle_ceiling_records,
)
from poker_ai.research.mixed_policy_h2h import SUPPORTED_POLICY_KINDS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", help="Summarize existing rows instead of running a checkpoint.")
    parser.add_argument("--checkpoint", help="Parent checkpoint to evaluate.")
    parser.add_argument("--kind", choices=SUPPORTED_POLICY_KINDS, default="tianshou-rainbow")
    parser.add_argument("--n-roots", type=int, default=8)
    parser.add_argument("--n-worlds", type=int, default=16)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sample-parent", action="store_true")
    parser.add_argument("--sample-continuation", action="store_true")
    parser.add_argument("--min-roots", type=int, default=1)
    parser.add_argument("--min-mean-oracle-gap", type=float, default=0.0)
    parser.add_argument("--require-positive-ceiling", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.records_json:
        rows = load_exact_oracle_ceiling_records(args.records_json)
        metrics = summarize_exact_oracle_ceiling_records(
            rows,
            min_roots=args.min_roots,
            min_mean_oracle_gap=args.min_mean_oracle_gap,
        )
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --records-json is supplied")
        metrics = evaluate_exact_oracle_ceiling(
            ExactOracleCeilingConfig(
                checkpoint=args.checkpoint,
                kind=args.kind,
                n_roots=args.n_roots,
                n_worlds=args.n_worlds,
                initial_chips=args.initial_chips,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
                max_steps_per_hand=args.max_steps_per_hand,
                seed=args.seed,
                device=args.device,
                greedy_parent=not args.sample_parent,
                greedy_continuation=not args.sample_continuation,
                min_roots=args.min_roots,
                min_mean_oracle_gap=args.min_mean_oracle_gap,
            )
        )
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.require_positive_ceiling and not metrics["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
