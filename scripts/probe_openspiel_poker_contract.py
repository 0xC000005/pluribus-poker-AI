#!/usr/bin/env python3
"""Probe whether installed OpenSpiel poker games preserve the native contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.openspiel_contract import probe_installed_openspiel  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether OpenSpiel poker can be used as a native 9-action "
            "backend or only as reference/evaluation machinery."
        )
    )
    parser.add_argument("--native-num-actions", type=int, default=9)
    parser.add_argument(
        "--game",
        action="append",
        dest="games",
        help="OpenSpiel game short name to probe. May be supplied multiple times.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    summary = probe_installed_openspiel(
        native_num_actions=args.native_num_actions,
        game_names=tuple(args.games) if args.games else ("kuhn_poker", "leduc_poker", "universal_poker"),
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
