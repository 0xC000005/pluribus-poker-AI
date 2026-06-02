#!/usr/bin/env python3
"""Evaluate native-only local evidence for a poker candidate checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.native_candidate_precheck import (  # noqa: E402
    evaluate_native_candidate_local_precheck,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--native-h2h-json", action="append", required=True, type=Path)
    parser.add_argument("--empirical-game-json", required=True, type=Path)
    parser.add_argument("--min-lower95", type=float, default=0.0)
    parser.add_argument("--min-candidate-support", type=float, default=1.0e-9)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = evaluate_native_candidate_local_precheck(
        candidate_checkpoint=args.candidate_checkpoint,
        native_h2h_jsons=list(args.native_h2h_json),
        empirical_game_json=args.empirical_game_json,
        min_lower95=float(args.min_lower95),
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
