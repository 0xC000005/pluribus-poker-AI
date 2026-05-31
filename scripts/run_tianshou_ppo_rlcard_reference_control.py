#!/usr/bin/env python3
"""Run a Tianshou PPO control on RLCard's no-limit Hold'em surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.rlcard_tianshou_ppo import run_control  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--opponent-checkpoint")
    parser.add_argument(
        "--opponent-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for an optional frozen RLCard PPO opponent checkpoint.",
    )
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_control(
        rollout_steps=args.rollout_steps,
        updates=args.updates,
        repeat=args.repeat,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        replay_size=args.replay_size,
        eval_games=args.eval_games,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        seed=args.seed,
        device=args.device,
        max_steps_per_hand=args.max_steps_per_hand,
        opponent_checkpoint=args.opponent_checkpoint,
        opponent_device=args.opponent_device,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
