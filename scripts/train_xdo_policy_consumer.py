#!/usr/bin/env python3
"""Train an NPI-compatible policy consumer from XDO-lite targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.xdo_policy_consumer import train_xdo_policy_consumer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--min-training-action-agreement", type=float, default=0.0)
    parser.add_argument("--parent-checkpoint", default="")
    parser.add_argument("--parent-kind", default="")
    parser.add_argument("--metrics-output")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)

    metrics = train_xdo_policy_consumer(
        args.targets,
        args.output,
        hidden_dim=args.hidden_dim,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        min_training_action_agreement=args.min_training_action_agreement,
        parent_checkpoint=args.parent_checkpoint,
        parent_kind=args.parent_kind,
        metrics_output=args.metrics_output,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.require_pass and not bool(metrics.get("passed", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
