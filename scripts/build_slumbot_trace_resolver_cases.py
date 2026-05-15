#!/usr/bin/env python3
"""Extract fixed resolver cases from Slumbot JSONL traces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.slumbot_trace_cases import extract_resolver_cases_from_trace  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build turn/river resolver benchmark cases from a Slumbot trace."
    )
    parser.add_argument("--trace", required=True, help="Input Slumbot trace JSONL.")
    parser.add_argument("--output", required=True, help="Output cases JSON.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of cases.")
    args = parser.parse_args(argv)

    cases = extract_resolver_cases_from_trace(args.trace, limit=args.limit)
    payload = {
        "mode": "slumbot_trace_resolver_cases",
        "trace": str(args.trace),
        "n_cases": len(cases),
        "cases": [
            {
                **asdict(case),
                "hole_cards": list(case.hole_cards),
                "board": list(case.board),
            }
            for case in cases
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "n_cases": len(cases)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
