#!/usr/bin/env python3
"""Evaluate whether candidate evidence clears pre-Slumbot promotion gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.candidate_promotion_gate import (  # noqa: E402
    evaluate_candidate_promotion_evidence,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlcard-reference-json", required=True, type=Path)
    parser.add_argument("--native-h2h-json", required=True, type=Path)
    parser.add_argument("--empirical-game-json", type=Path)
    parser.add_argument("--native-candidate-checkpoint")
    parser.add_argument("--min-lower95", type=float, default=0.0)
    parser.add_argument("--min-candidate-support", type=float, default=1.0e-9)
    parser.add_argument(
        "--allow-missing-empirical-game",
        action="store_true",
        help="Diagnostic-only mode; promotion should normally require empirical-game support.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = evaluate_candidate_promotion_evidence(
        rlcard_reference_json=args.rlcard_reference_json,
        native_h2h_json=args.native_h2h_json,
        empirical_game_json=args.empirical_game_json,
        native_candidate_checkpoint=args.native_candidate_checkpoint,
        min_lower95=float(args.min_lower95),
        require_empirical=not bool(args.allow_missing_empirical_game),
        min_candidate_support=float(args.min_candidate_support),
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
