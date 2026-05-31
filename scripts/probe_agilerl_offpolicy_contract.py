#!/usr/bin/env python3
"""Probe AgileRL off-policy MARL algorithms on native poker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.agilerl_contract import probe_agilerl_offpolicy_algorithms  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", default="MADDPG,MATD3")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--evo-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = probe_agilerl_offpolicy_algorithms(
        algorithms=tuple(
            item.strip() for item in args.algorithms.split(",") if item.strip()
        ),
        seed=args.seed,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        max_steps=args.max_steps,
        evo_steps=args.evo_steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
