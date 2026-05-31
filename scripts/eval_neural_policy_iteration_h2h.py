#!/usr/bin/env python3
"""Evaluate duplicate-swapped H2H for neural policy-iteration checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.neural_policy_iteration import (  # noqa: E402
    evaluate_neural_policy_iteration_head_to_head,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--min-lower95-candidate-payoff", type=float)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = evaluate_neural_policy_iteration_head_to_head(
        args.candidate,
        args.baseline,
        n_games=args.n_games,
        device=args.device,
        seed=args.seed,
        min_lower95_candidate_payoff=args.min_lower95_candidate_payoff,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
