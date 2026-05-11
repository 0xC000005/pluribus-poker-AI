#!/usr/bin/env python3
"""Audit autoresearch changes for objective drift and benchmark hacking."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.autoresearch import audit_objective_alignment  # noqa: E402


def _git_changed_paths(root: Path, base_ref: str | None) -> list[str]:
    commands = [
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
    ]
    if base_ref:
        commands.append(["git", "diff", "--name-only", base_ref, "--"])

    paths: set[str] = set()
    for command in commands:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"git command failed: {command}")
        paths.update(path.strip() for path in result.stdout.splitlines() if path.strip())
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject protected evaluation-surface changes without review."
    )
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--base-ref", help="Optional git ref to diff against.")
    parser.add_argument("--review-dir", help="Completed methodology review bundle.")
    parser.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="Explicit changed path. If omitted, paths are read from git diff.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    changed_paths = args.changed_path or _git_changed_paths(root, args.base_ref)
    result = audit_objective_alignment(
        root,
        changed_paths=changed_paths,
        review_dir=args.review_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
