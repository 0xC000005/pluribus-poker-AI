#!/usr/bin/env python3
"""Run an aligned depth-limited resolving pilot.

This pilot keeps the learned object and inference boundary matched:

1. collect dual CFV labels from states actually queried by the CFR leaf callback;
2. train the public-belief dual hand-CFV network on that callback distribution;
3. evaluate by inserting the checkpoint back into the same resolver leaf callback.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_public_belief_callback_leaf_targets import build_callback_leaf_targets  # noqa: E402
from eval_public_belief_dcvn_leaf_ab import eval_public_belief_dcvn_leaf_ab  # noqa: E402
from eval_public_belief_dual_hand_cfv_probe import (  # noqa: E402
    train_public_belief_dual_hand_cfv_checkpoint,
)
from poker_ai.research.belief_probe import N_HANDS  # noqa: E402
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402


@dataclass(frozen=True)
class DepthLimitedContractConfig:
    cases_json: str | Path
    cfv_cache: str | Path
    output_dir: str | Path
    prefix: str = "contract"
    train_start_index: int = 0
    train_limit: int = 16
    holdout_start_index: int = 128
    holdout_limit: int = 8
    solver_iterations: int = 5
    solver_backend: str = "cpu"
    value_scale: float = 20000.0
    max_states_per_case: int = 256
    seed: int = 20260515
    device: str = "auto"
    hidden_dim: int = 64
    epochs: int = 30
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-3
    head_mode: str = "separate"
    belief_bottleneck_dim: int = 32
    card_encoder: str = "deepset"
    value_factorization: str = "direct"
    loss_kind: str = "mse"
    weight_mode: str = "uniform"
    state_batch_size: int = 16
    hand_batch_size: int = N_HANDS
    project_zero_sum: bool = True
    min_action_agreement: float = 0.5
    max_mean_l1_drift: float = 0.75


def _artifact_paths(output_dir: str | Path, prefix: str) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "train_cache": root / f"{prefix}_train_callback_leaf.npz",
        "holdout_cache": root / f"{prefix}_holdout_callback_leaf.npz",
        "train_collection_json": root / f"{prefix}_train_callback_leaf.json",
        "holdout_collection_json": root / f"{prefix}_holdout_callback_leaf.json",
        "checkpoint": root / f"{prefix}_model.pt",
        "train_json": root / f"{prefix}_train.json",
        "leaf_ab_json": root / f"{prefix}_leaf_ab.json",
    }


def _compact_train_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(metrics.get("passed", False)),
        "checkpoint": metrics.get("checkpoint"),
        "device": metrics.get("device"),
        "train_size": metrics.get("train_size"),
        "holdout_size": metrics.get("holdout_size"),
        "belief_holdout": metrics.get("belief_holdout"),
        "zero_baseline": metrics.get("zero_baseline"),
        "best_constant_mae": metrics.get("best_constant_mae"),
        "best_constant_rmse": metrics.get("best_constant_rmse"),
    }


def _compact_leaf_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(metrics.get("passed", False)),
        "n_evaluated": metrics.get("n_evaluated"),
        "n_leaf_applied": metrics.get("n_leaf_applied"),
        "leaf_action_agreement_rate": metrics.get("leaf_action_agreement_rate"),
        "leaf_mean_action_l1_drift": metrics.get("leaf_mean_action_l1_drift"),
        "leaf_max_action_l1_drift": metrics.get("leaf_max_action_l1_drift"),
        "min_action_agreement": metrics.get("min_action_agreement"),
        "max_mean_l1_drift": metrics.get("max_mean_l1_drift"),
    }


def run_depth_limited_resolving_contract_pilot(
    config: DepthLimitedContractConfig,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_dir, config.prefix)

    train_collection = build_callback_leaf_targets(
        cases_json=config.cases_json,
        cfv_cache=config.cfv_cache,
        output=paths["train_cache"],
        start_index=config.train_start_index,
        limit=config.train_limit,
        solver_iterations=config.solver_iterations,
        solver_backend=config.solver_backend,
        value_scale=config.value_scale,
        max_states_per_case=config.max_states_per_case,
        seed=config.seed,
        output_json=paths["train_collection_json"],
    )
    holdout_collection = build_callback_leaf_targets(
        cases_json=config.cases_json,
        cfv_cache=config.cfv_cache,
        output=paths["holdout_cache"],
        start_index=config.holdout_start_index,
        limit=config.holdout_limit,
        solver_iterations=config.solver_iterations,
        solver_backend=config.solver_backend,
        value_scale=config.value_scale,
        max_states_per_case=config.max_states_per_case,
        seed=config.seed + 100000,
        output_json=paths["holdout_collection_json"],
    )

    train_metrics = train_public_belief_dual_hand_cfv_checkpoint(
        train_cases_json=config.cases_json,
        train_cfv_cache=config.cfv_cache,
        holdout_cases_json=config.cases_json,
        holdout_cfv_cache=config.cfv_cache,
        output_checkpoint=paths["checkpoint"],
        train_start_index=config.train_start_index,
        holdout_start_index=config.holdout_start_index,
        train_limit=config.train_limit,
        holdout_limit=config.holdout_limit,
        train_dual_cache=paths["train_cache"],
        holdout_dual_cache=paths["holdout_cache"],
        device=config.device,
        solver_iterations=config.solver_iterations,
        solver_backend=config.solver_backend,
        value_scale=config.value_scale,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        seed=config.seed,
        head_mode=config.head_mode,
        belief_bottleneck_dim=config.belief_bottleneck_dim,
        card_encoder=config.card_encoder,
        value_factorization=config.value_factorization,
        label_jobs=1,
        loss_kind=config.loss_kind,
        weight_mode=config.weight_mode,
    )
    save_metrics(train_metrics, paths["train_json"])

    leaf_metrics = eval_public_belief_dcvn_leaf_ab(
        checkpoint=paths["checkpoint"],
        cases_json=config.cases_json,
        cfv_cache=config.cfv_cache,
        device=config.device,
        start_index=config.holdout_start_index,
        limit=config.holdout_limit,
        solver_iterations=config.solver_iterations,
        solver_backend=config.solver_backend,
        value_scale=config.value_scale,
        state_batch_size=config.state_batch_size,
        hand_batch_size=config.hand_batch_size,
        project_zero_sum=config.project_zero_sum,
        min_action_agreement=config.min_action_agreement,
        max_mean_l1_drift=config.max_mean_l1_drift,
    )
    save_metrics(leaf_metrics, paths["leaf_ab_json"])

    train_supervised_fit_passed = bool(train_metrics.get("passed", False))
    leaf_eval_passed = bool(leaf_metrics.get("passed", False))
    passed = bool(
        train_collection.get("passed", False)
        and holdout_collection.get("passed", False)
        and leaf_eval_passed
    )
    return {
        "mode": "depth_limited_resolving_contract_pilot",
        "contract": "callback_leaf_train_to_callback_leaf_inference",
        "passed": passed,
        "promotion": False,
        "promotion_blockers": [
            "pilot_scale_only",
            "must_pass_root_disjoint_leaf_ab_before_live_use",
            "cpu_callback_leaf_path_not_live_latency_ready",
        ],
        "cases_json": str(config.cases_json),
        "cfv_cache": str(config.cfv_cache),
        "output_dir": str(output_dir),
        "train_start_index": int(config.train_start_index),
        "train_limit": int(config.train_limit),
        "holdout_start_index": int(config.holdout_start_index),
        "holdout_limit": int(config.holdout_limit),
        "solver_iterations": int(config.solver_iterations),
        "solver_backend": config.solver_backend,
        "device": config.device,
        "train_callback_cache": str(paths["train_cache"]),
        "holdout_callback_cache": str(paths["holdout_cache"]),
        "checkpoint": str(paths["checkpoint"]),
        "train_collection_passed": bool(train_collection.get("passed", False)),
        "holdout_collection_passed": bool(holdout_collection.get("passed", False)),
        "train_supervised_fit_passed": train_supervised_fit_passed,
        "leaf_eval_passed": leaf_eval_passed,
        "train_collection": {
            "n_cases": train_collection.get("n_cases"),
            "n_states": train_collection.get("n_states"),
            "n_skipped": train_collection.get("n_skipped"),
            "hero_label_count": train_collection.get("hero_label_count"),
            "villain_label_count": train_collection.get("villain_label_count"),
        },
        "holdout_collection": {
            "n_cases": holdout_collection.get("n_cases"),
            "n_states": holdout_collection.get("n_states"),
            "n_skipped": holdout_collection.get("n_skipped"),
            "hero_label_count": holdout_collection.get("hero_label_count"),
            "villain_label_count": holdout_collection.get("villain_label_count"),
        },
        "train_metrics": _compact_train_metrics(train_metrics),
        "leaf_eval_metrics": _compact_leaf_metrics(leaf_metrics),
        "artifact_paths": {key: str(value) for key, value in paths.items()},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="contract")
    parser.add_argument("--train-start-index", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--holdout-start-index", type=int, default=128)
    parser.add_argument("--holdout-limit", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--max-states-per-case", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--head-mode", choices=("shared", "separate"), default="separate")
    parser.add_argument("--belief-bottleneck-dim", type=int, default=32)
    parser.add_argument("--card-encoder", choices=("flat", "deepset"), default="deepset")
    parser.add_argument(
        "--value-factorization",
        choices=("direct", "state-player-offset"),
        default="direct",
    )
    parser.add_argument("--loss-kind", choices=("mse", "smooth-l1"), default="mse")
    parser.add_argument(
        "--weight-mode",
        choices=("uniform", "opponent-reach"),
        default="uniform",
    )
    parser.add_argument("--state-batch-size", type=int, default=16)
    parser.add_argument("--hand-batch-size", type=int, default=N_HANDS)
    parser.add_argument("--no-project-zero-sum", action="store_true")
    parser.add_argument("--min-action-agreement", type=float, default=0.5)
    parser.add_argument("--max-mean-l1-drift", type=float, default=0.75)
    parser.add_argument("--output-json")
    return parser


def _config_from_args(args: argparse.Namespace) -> DepthLimitedContractConfig:
    return DepthLimitedContractConfig(
        cases_json=args.cases_json,
        cfv_cache=args.cfv_cache,
        output_dir=args.output_dir,
        prefix=args.prefix,
        train_start_index=args.train_start_index,
        train_limit=args.train_limit,
        holdout_start_index=args.holdout_start_index,
        holdout_limit=args.holdout_limit,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        max_states_per_case=args.max_states_per_case,
        seed=args.seed,
        device=args.device,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        head_mode=args.head_mode,
        belief_bottleneck_dim=args.belief_bottleneck_dim,
        card_encoder=args.card_encoder,
        value_factorization=args.value_factorization,
        loss_kind=args.loss_kind,
        weight_mode=args.weight_mode,
        state_batch_size=args.state_batch_size,
        hand_batch_size=args.hand_batch_size,
        project_zero_sum=not args.no_project_zero_sum,
        min_action_agreement=args.min_action_agreement,
        max_mean_l1_drift=args.max_mean_l1_drift,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics = run_depth_limited_resolving_contract_pilot(_config_from_args(args))
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
