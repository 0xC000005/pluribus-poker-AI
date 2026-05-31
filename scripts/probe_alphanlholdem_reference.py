#!/usr/bin/env python3
"""Probe the ignored AlphaNLHoldem reference checkout contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.alphanlholdem_reference import (  # noqa: E402
    inspect_alphanlholdem_reference,
)


def _repo_root_for_reference(reference_dir: Path) -> Path | None:
    try:
        reference_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    return REPO_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the unofficial AlphaNLHoldem checkout without importing "
            "AGPL reference code, and report whether an isolated RLCard "
            "benchmark can be built."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REPO_ROOT / "reference_code" / "AlphaNLHoldem",
    )
    parser.add_argument("--native-num-actions", type=int, default=9)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    summary = inspect_alphanlholdem_reference(
        args.reference_dir,
        native_num_actions=args.native_num_actions,
        repo_root=_repo_root_for_reference(args.reference_dir),
    )
    if args.require_checkpoint and not summary.get("bundled_checkpoint_present", False):
        summary["ready_for_rlcard_benchmark"] = False
        if "bundled checkpoint required by CLI" not in summary["blockers"]:
            summary["blockers"].append("bundled checkpoint required by CLI")

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary.get("ready_for_rlcard_benchmark") else 1


if __name__ == "__main__":
    raise SystemExit(main())
