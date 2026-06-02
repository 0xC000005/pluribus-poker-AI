#!/usr/bin/env python3
"""Gate a solved empirical-game meta-strategy using H2H artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.empirical_meta_strategy_gate import (  # noqa: E402
    evaluate_empirical_meta_strategy_h2h_gate,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empirical-game-json", required=True)
    parser.add_argument(
        "--h2h-record",
        action="append",
        default=[],
        help="Meta-strategy-vs-pure H2H JSON artifact. Repeat for every policy.",
    )
    parser.add_argument(
        "--strategy-key",
        choices=("row_strategy", "column_strategy"),
        default="row_strategy",
    )
    parser.add_argument("--min-support-weight", type=float, default=1e-12)
    parser.add_argument("--max-support-abs-mean", type=float, default=0.01)
    parser.add_argument("--max-support-lower95-loss", type=float, default=0.01)
    parser.add_argument("--min-off-support-lower95", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = evaluate_empirical_meta_strategy_h2h_gate(
        args.empirical_game_json,
        args.h2h_record,
        strategy_key=args.strategy_key,
        min_support_weight=args.min_support_weight,
        max_support_abs_mean=args.max_support_abs_mean,
        max_support_lower95_loss=args.max_support_lower95_loss,
        min_off_support_lower95=args.min_off_support_lower95,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
