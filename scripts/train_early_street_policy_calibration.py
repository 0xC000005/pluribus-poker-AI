#!/usr/bin/env python3
"""Train a candidate policy head on learned early-street reference targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from poker_ai.research.early_street_policy_calibration import (  # noqa: E402
    train_early_street_policy_calibration,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--state-source-checkpoint")
    parser.add_argument("--n-states", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-strategy-source", default="regret")
    parser.add_argument("--state-source-strategy-source", default="regret")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-hands", type=int)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.25)
    parser.add_argument("--rank-confidence-weighted", action="store_true")
    parser.add_argument("--diagnostic-states", type=int, default=0)
    parser.add_argument("--diagnostic-seed", type=int)
    parser.add_argument("--max-candidate-preflop-allin-rate", type=float, default=0.10)
    parser.add_argument("--max-candidate-flop-allin-rate", type=float, default=0.20)
    parser.add_argument("--max-mean-l1-to-reference", type=float, default=0.75)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = train_early_street_policy_calibration(
        candidate_checkpoint=args.candidate_checkpoint,
        reference_checkpoint=args.reference_checkpoint,
        output_checkpoint=args.output_checkpoint,
        state_source_checkpoint=args.state_source_checkpoint,
        n_states=args.n_states,
        seed=args.seed,
        reference_strategy_source=args.reference_strategy_source,
        state_source_strategy_source=args.state_source_strategy_source,
        device=args.device,
        max_hands=args.max_hands,
        target_temperature=args.target_temperature,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        rank_loss_weight=args.rank_loss_weight,
        rank_margin=args.rank_margin,
        rank_confidence_weighted=args.rank_confidence_weighted,
        diagnostic_states=args.diagnostic_states,
        diagnostic_seed=args.diagnostic_seed,
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
