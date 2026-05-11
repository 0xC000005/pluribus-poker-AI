#!/usr/bin/env python3
"""Validate poker autoresearch methodology review artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.autoresearch import validate_methodology_review  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate independent-verifier and related-work artifacts."
    )
    parser.add_argument("--review-dir", required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return non-zero if any required artifact is missing or pending.",
    )
    args = parser.parse_args(argv)

    result = validate_methodology_review(args.review_dir)
    if args.require_complete and not result["passed"]:
        print("\n".join(result["errors"]), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
