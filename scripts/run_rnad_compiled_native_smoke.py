#!/usr/bin/env python3
"""Smoke-test R-NaD updates on compiled native 9-action self-play trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.native_rnad import run_compiled_native_rnad_smoke  # noqa: E402


def _write_metrics(metrics: dict[str, Any], output_json: Path | None) -> dict[str, Any]:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(output_json)
        output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-games", type=int, default=32)
    parser.add_argument("--collector-batch-size", type=int, default=16)
    parser.add_argument("--max-steps-per-game", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_compiled_native_rnad_smoke(
        n_games=args.n_games,
        collector_batch_size=args.collector_batch_size,
        max_steps_per_game=args.max_steps_per_game,
        initial_chips=args.initial_chips,
        updates=args.updates,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    metrics = _write_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
