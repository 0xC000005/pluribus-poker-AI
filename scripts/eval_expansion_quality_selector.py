#!/usr/bin/env python3
"""Evaluate a fixed-slate native population expansion selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.expansion_quality_selector import (  # noqa: E402
    evaluate_expansion_quality_selector,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-policy", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--h2h-json", action="append", required=True, type=Path)
    parser.add_argument("--incumbent-checkpoint", required=True)
    parser.add_argument("--min-candidate-support", type=float, default=1.0e-9)
    parser.add_argument("--min-lower95", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = evaluate_expansion_quality_selector(
        support_policies=list(args.support_policy),
        candidate_checkpoints=list(args.candidate),
        h2h_records=list(args.h2h_json),
        incumbent_checkpoint=args.incumbent_checkpoint,
        min_candidate_support=float(args.min_candidate_support),
        min_lower95=float(args.min_lower95),
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
