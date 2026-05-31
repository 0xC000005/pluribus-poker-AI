#!/usr/bin/env python3
"""Evaluate duplicate-swapped H2H across supported native policy formats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.mixed_policy_h2h import (  # noqa: E402
    SUPPORTED_POLICY_KINDS,
    evaluate_mixed_policy_head_to_head,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-kind", choices=SUPPORTED_POLICY_KINDS, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-kind", choices=SUPPORTED_POLICY_KINDS, required=True)
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--min-lower95-candidate-payoff", type=float)
    parser.add_argument(
        "--eval-state-backend",
        choices=("full-deck", "fast-state-canonical-deal"),
        default="full-deck",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = evaluate_mixed_policy_head_to_head(
        candidate_checkpoint=args.candidate,
        candidate_kind=args.candidate_kind,
        baseline_checkpoint=args.baseline,
        baseline_kind=args.baseline_kind,
        n_games=args.n_games,
        device=args.device,
        seed=args.seed,
        min_lower95_candidate_payoff=args.min_lower95_candidate_payoff,
        eval_state_backend=args.eval_state_backend,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
