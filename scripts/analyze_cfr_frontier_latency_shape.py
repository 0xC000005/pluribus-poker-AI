#!/usr/bin/env python3
"""Join CFR budget-frontier latency with solver tree-shape footprint metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.cfr_frontier_latency_shape import (  # noqa: E402
    analyze_frontier_latency_shape,
)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze which public-state tree shapes dominate CFR frontier latency."
    )
    parser.add_argument("--frontier-json", required=True)
    parser.add_argument("--footprint-json", required=True)
    parser.add_argument("--budget")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = analyze_frontier_latency_shape(
        _load_json(args.frontier_json),
        _load_json(args.footprint_json),
        budget=args.budget,
        top_k=args.top_k,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
