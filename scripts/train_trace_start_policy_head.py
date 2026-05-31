#!/usr/bin/env python3
"""Train a diagnostic policy head on trace-start counterfactual values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.trace_start_policy import train_trace_start_policy_head  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ev-gate-json", required=True, action="append")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--source-flag", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=500.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.33)
    parser.add_argument(
        "--holdout-source",
        help="Optional basename of one --ev-gate-json to hold out completely.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    records = []
    sources = []
    for ev_gate_json in args.ev_gate_json:
        path = Path(ev_gate_json)
        data = json.loads(path.read_text(encoding="utf-8"))
        source_name = path.name
        sources.append(source_name)
        for record in data.get("records") or []:
            item = dict(record)
            item["_trace_source"] = source_name
            records.append(item)
    data = {"records": records}
    policy_net, metrics = train_trace_start_policy_head(
        data,
        source_flag=args.source_flag,
        target_temperature=args.target_temperature,
        holdout_fraction=args.holdout_fraction,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        holdout_source=args.holdout_source,
    )
    metrics["ev_gate_json"] = [str(path) for path in args.ev_gate_json]
    metrics["trace_sources"] = sources
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_checkpoint:
        checkpoint = Path(args.output_checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": policy_net.state_dict(),
                "metadata": metrics,
                "input_dim": metrics["feature_context_dim"],
                "hidden_dim": int(args.hidden_dim),
                "n_layers": int(args.n_layers),
            },
            checkpoint,
        )
        metrics["output_checkpoint"] = str(checkpoint)
        output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
