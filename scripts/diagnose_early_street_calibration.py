#!/usr/bin/env python3
"""Diagnose early-street blueprint action drift before live Slumbot spend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from poker_ai.research.early_street_calibration import (  # noqa: E402
    evaluate_early_street_calibration,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--state-source-checkpoint")
    parser.add_argument("--n-states", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-strategy-source", default="regret")
    parser.add_argument("--reference-strategy-source", default="regret")
    parser.add_argument("--state-source-strategy-source", default="regret")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-hands", type=int)
    parser.add_argument("--max-candidate-preflop-allin-rate", type=float, default=0.10)
    parser.add_argument("--max-candidate-flop-allin-rate", type=float, default=0.20)
    parser.add_argument("--max-mean-l1-to-reference", type=float, default=0.75)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = evaluate_early_street_calibration(
        candidate_checkpoint=args.candidate_checkpoint,
        reference_checkpoint=args.reference_checkpoint,
        state_source_checkpoint=args.state_source_checkpoint,
        n_states=args.n_states,
        seed=args.seed,
        candidate_strategy_source=args.candidate_strategy_source,
        reference_strategy_source=args.reference_strategy_source,
        state_source_strategy_source=args.state_source_strategy_source,
        device=args.device,
        max_hands=args.max_hands,
        max_candidate_preflop_allin_rate=args.max_candidate_preflop_allin_rate,
        max_candidate_flop_allin_rate=args.max_candidate_flop_allin_rate,
        max_mean_l1_to_reference=args.max_mean_l1_to_reference,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
