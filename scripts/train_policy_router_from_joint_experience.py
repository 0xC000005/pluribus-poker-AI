#!/usr/bin/env python3
"""Train a state-level router over a local policy population."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.policy_router import (  # noqa: E402
    train_policy_router_from_joint_experience,
)


def _parse_member_policy(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError("member policy must use KIND:CHECKPOINT")
    kind, checkpoint = spec.split(":", 1)
    kind = kind.strip()
    checkpoint = checkpoint.strip()
    if not kind or not checkpoint:
        raise argparse.ArgumentTypeError("member policy kind and checkpoint must be non-empty")
    return kind, checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument(
        "--member-policy",
        action="append",
        required=True,
        help="Member policy in KIND:CHECKPOINT format, matching dataset policy order.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260990)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = train_policy_router_from_joint_experience(
        args.dataset_npz,
        member_specs=[_parse_member_policy(spec) for spec in args.member_policy],
        hidden_dim=args.hidden_dim,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
