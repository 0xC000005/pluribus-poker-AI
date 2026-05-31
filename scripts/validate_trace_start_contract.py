#!/usr/bin/env python3
"""Validate Slumbot trace-start observation contracts before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.trace_start_contract import (  # noqa: E402
    load_trace_start_observations,
    summarize_trace_start_observations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Resolver-case JSON from Slumbot traces.")
    parser.add_argument("--output-json", help="Optional metrics JSON output path.")
    parser.add_argument(
        "--source-flag",
        type=float,
        default=1.0,
        help="Explicit provenance scalar appended by features_with_context.",
    )
    args = parser.parse_args(argv)

    observations = load_trace_start_observations(args.cases, source_flag=args.source_flag)
    metrics = summarize_trace_start_observations(observations, source_flag=args.source_flag)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
