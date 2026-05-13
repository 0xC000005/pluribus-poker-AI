#!/usr/bin/env python3
"""Train only the ValueNetwork policy head on calibration targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the policy head from targets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.25)
    parser.add_argument("--metrics-output")
    args = parser.parse_args(argv)

    from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
    from poker_ai.research.policy_calibration import train_policy_head_calibration

    targets = PolicyTargetBuffer.from_npz(args.targets)
    metrics = train_policy_head_calibration(
        args.checkpoint,
        targets,
        args.output,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        rank_loss_weight=args.rank_loss_weight,
        rank_margin=args.rank_margin,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.metrics_output:
        output = Path(args.metrics_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
