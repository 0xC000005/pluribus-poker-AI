#!/usr/bin/env python3
"""Audit autoresearch changes for objective drift and benchmark hacking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.autoresearch import (  # noqa: E402
    audit_objective_alignment,
    git_changed_paths,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject protected evaluation-surface changes without review."
    )
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--base-ref", help="Optional git ref to diff against.")
    parser.add_argument("--review-dir", help="Completed methodology review bundle.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow an audit with no changed paths. Without this, empty audits fail.",
    )
    parser.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="Explicit changed path. If omitted, paths are read from git diff.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    changed_paths = args.changed_path or git_changed_paths(root, args.base_ref)
    if not changed_paths and not args.allow_empty:
        print(
            "No changed paths found; pass --allow-empty for an explicit no-op audit.",
            file=sys.stderr,
        )
        return 2
    result = audit_objective_alignment(
        root,
        changed_paths=changed_paths,
        review_dir=args.review_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
