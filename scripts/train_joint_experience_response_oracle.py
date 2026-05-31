#!/usr/bin/env python3
"""Train a Tianshou Rainbow response oracle from local joint experience."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.joint_experience_response_oracle import (  # noqa: E402
    train_joint_experience_rainbow_response_oracle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = train_joint_experience_rainbow_response_oracle(
        args.dataset_npz,
        updates=args.updates,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_atoms=args.num_atoms,
        lr=args.lr,
        gamma=args.gamma,
        n_step=args.n_step,
        target_update_freq=args.target_update_freq,
        seed=args.seed,
        device=args.device,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    if args.output_json is not None and not args.output_json.exists():
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
