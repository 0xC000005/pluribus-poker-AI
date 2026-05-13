#!/usr/bin/env python3
"""Train a saved dual-player public-belief hand-CFV checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and save the separate-head dual-player hand-CFV model."
    )
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--train-cfv-cache", required=True)
    parser.add_argument("--holdout-cases", required=True)
    parser.add_argument("--holdout-cfv-cache", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--train-start-index", type=int, default=0)
    parser.add_argument("--holdout-start-index", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--holdout-limit", type=int, default=64)
    parser.add_argument("--train-dual-cache")
    parser.add_argument("--holdout-dual-cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--head-mode",
        choices=("shared", "separate"),
        default="separate",
    )
    parser.add_argument("--belief-bottleneck-dim", type=int, default=32)
    parser.add_argument(
        "--card-encoder",
        choices=("flat", "deepset"),
        default="flat",
    )
    parser.add_argument(
        "--value-factorization",
        choices=("direct", "state-player-offset"),
        default="direct",
    )
    parser.add_argument("--label-jobs", type=int, default=1)
    parser.add_argument("--loss-kind", choices=("mse", "smooth-l1"), default="mse")
    parser.add_argument(
        "--weight-mode",
        choices=("uniform", "opponent-reach"),
        default="uniform",
    )
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--weight-floor", type=float, default=0.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    from eval_public_belief_dual_hand_cfv_probe import (
        save_metrics,
        train_public_belief_dual_hand_cfv_checkpoint,
    )

    metrics = train_public_belief_dual_hand_cfv_checkpoint(
        train_cases_json=args.train_cases,
        train_cfv_cache=args.train_cfv_cache,
        holdout_cases_json=args.holdout_cases,
        holdout_cfv_cache=args.holdout_cfv_cache,
        output_checkpoint=args.output_checkpoint,
        train_start_index=args.train_start_index,
        holdout_start_index=args.holdout_start_index,
        train_limit=args.train_limit,
        holdout_limit=args.holdout_limit,
        train_dual_cache=args.train_dual_cache,
        holdout_dual_cache=args.holdout_dual_cache,
        device=args.device,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        head_mode=args.head_mode,
        belief_bottleneck_dim=args.belief_bottleneck_dim,
        card_encoder=args.card_encoder,
        value_factorization=args.value_factorization,
        label_jobs=args.label_jobs,
        loss_kind=args.loss_kind,
        weight_mode=args.weight_mode,
        weight_power=args.weight_power,
        weight_floor=args.weight_floor,
        smooth_l1_beta=args.smooth_l1_beta,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
