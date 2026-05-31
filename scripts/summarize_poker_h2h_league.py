#!/usr/bin/env python3
"""Summarize local duplicate-swapped H2H artifacts into one league gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.h2h_league_summary import summarize_h2h_league  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", help="H2H JSON artifacts to summarize.")
    parser.add_argument("--min-lower95", type=float, default=0.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    summary = summarize_h2h_league(args.records, min_lower95=args.min_lower95)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
