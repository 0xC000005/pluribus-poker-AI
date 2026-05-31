#!/usr/bin/env python3
"""Evaluate an exact restricted response oracle against a local incumbent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from poker_ai.research.exact_response_oracle import (  # noqa: E402
    ExactResponseH2HConfig,
    PolicySpec,
    evaluate_exact_response_h2h,
    summarize_exact_response_h2h,
)


def _parse_policy_spec(raw: str) -> PolicySpec:
    parts = str(raw).split(":")
    if len(parts) == 2:
        kind, checkpoint = parts
        weight = 1.0
    elif len(parts) == 3:
        kind, checkpoint, weight_raw = parts
        weight = float(weight_raw)
    else:
        raise argparse.ArgumentTypeError(
            "policy specs must be KIND:CHECKPOINT or KIND:CHECKPOINT:WEIGHT"
        )
    if not kind or not checkpoint:
        raise argparse.ArgumentTypeError("policy spec kind and checkpoint must be non-empty")
    return PolicySpec(kind=kind, checkpoint=checkpoint, weight=float(weight))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-json",
        help="Summarize a previously recorded exact-response payload instead of running H2H.",
    )
    parser.add_argument("--baseline", help="Baseline checkpoint for live H2H mode.")
    parser.add_argument("--baseline-kind", default="tianshou-rainbow")
    parser.add_argument(
        "--continuation",
        action="append",
        type=_parse_policy_spec,
        default=[],
        help="Continuation population member as KIND:CHECKPOINT[:WEIGHT]. Defaults to baseline.",
    )
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--n-worlds", type=int, default=4)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sample-baseline", action="store_true")
    parser.add_argument("--sample-continuation", action="store_true")
    parser.add_argument("--max-decision-records", type=int, default=512)
    parser.add_argument("--min-games", type=int, default=1)
    parser.add_argument("--min-lower95-response-payoff", type=float, default=0.0)
    parser.add_argument("--output-json")
    parser.add_argument("--require-pass", action="store_true")
    return parser


def _load_records_summary(args: argparse.Namespace) -> dict:
    payload = json.loads(Path(args.records_json).read_text(encoding="utf-8"))
    response_payoffs = payload.get("response_payoffs")
    if response_payoffs is None:
        response_payoffs = payload.get("paired_deltas", [])
    decision_records = payload.get("decision_records", [])
    return summarize_exact_response_h2h(
        response_payoffs,
        decision_records,
        min_games=args.min_games,
        min_lower95_response_payoff=args.min_lower95_response_payoff,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.records_json:
        metrics = _load_records_summary(args)
    else:
        if not args.baseline:
            parser.error("--baseline is required unless --records-json is provided")
        metrics = evaluate_exact_response_h2h(
            ExactResponseH2HConfig(
                baseline_checkpoint=str(args.baseline),
                baseline_kind=str(args.baseline_kind),
                continuation_specs=tuple(args.continuation),
                n_games=int(args.n_games),
                n_worlds=int(args.n_worlds),
                initial_chips=int(args.initial_chips),
                small_blind=int(args.small_blind),
                big_blind=int(args.big_blind),
                max_steps_per_hand=int(args.max_steps_per_hand),
                seed=int(args.seed),
                device=str(args.device),
                greedy_baseline=not bool(args.sample_baseline),
                greedy_continuation=not bool(args.sample_continuation),
                max_decision_records=int(args.max_decision_records),
                min_games=int(args.min_games),
                min_lower95_response_payoff=float(args.min_lower95_response_payoff),
            )
        )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.require_pass and not bool(metrics.get("passed", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
