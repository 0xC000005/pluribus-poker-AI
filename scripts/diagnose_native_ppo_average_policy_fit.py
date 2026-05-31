#!/usr/bin/env python3
"""Diagnose actor-vs-average policy fit for native PPO FSP checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.native_ppo_average_policy_fit import diagnose_average_policy_fit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--rollout-source", choices=("actor", "average"), default="actor")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = diagnose_average_policy_fit(
        args.checkpoint,
        n_games=args.n_games,
        device=args.device,
        seed=args.seed,
        rollout_source=args.rollout_source,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
