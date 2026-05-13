#!/usr/bin/env python3
"""Validate a poker autoresearch failure-synthesis bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.autoresearch import validate_failure_synthesis  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate causal-model synthesis after repeated failures."
    )
    parser.add_argument("--synthesis-dir", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    result = validate_failure_synthesis(args.synthesis_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_complete and not result["passed"]:
        print("\n".join(result["errors"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
