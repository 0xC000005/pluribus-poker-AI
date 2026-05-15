#!/usr/bin/env python3
"""Run the research-only sampled Deep CFR traversal probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.sampled_deep_cfr_traversal_probe import run_probe  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare exhaustive and sampled traverser regrets on a tiny full-deck state."
    )
    parser.add_argument("--n-repeats", type=int, default=64)
    parser.add_argument("--n-reference-repeats", type=int)
    parser.add_argument("--initial-chips", type=int, default=300)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument(
        "--sampling-mode",
        choices=("with-replacement", "without-replacement", "priority-without-replacement"),
        default="with-replacement",
    )
    parser.add_argument("--priority-forced-count", type=int, default=0)
    parser.add_argument(
        "--priority-source",
        choices=("strategy", "advantage", "abs-advantage"),
        default="strategy",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--uniform-mix", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_probe(
        n_repeats=args.n_repeats,
        n_reference_repeats=args.n_reference_repeats,
        initial_chips=args.initial_chips,
        sample_count=args.sample_count,
        sampling_mode=args.sampling_mode,
        priority_forced_count=args.priority_forced_count,
        priority_source=args.priority_source,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        uniform_mix=args.uniform_mix,
        seed=args.seed,
        device=args.device,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
