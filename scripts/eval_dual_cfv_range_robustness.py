#!/usr/bin/env python3
"""Evaluate dual-player hand-CFV checkpoints under perturbed public beliefs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_probe import N_HANDS, _resolve_device
from poker_ai.research.belief_value_probe import (
    load_public_belief_cfv_dataset_cache,
    save_metrics,
)
from poker_ai.research.resolver_benchmark import load_cases_json

from eval_hand_cfv_range_robustness import _legal_global_mask, _mix_with_uniform
from eval_public_belief_dual_cfv_checkpoint import _zero_sum_residuals
from eval_public_belief_dual_hand_cfv_probe import (
    DualCFVDataset,
    _case_dual_cfv_target,
    _constant_dual_prediction,
    _metrics,
    _zero_dual_prediction,
    load_public_belief_dual_hand_cfv_ensemble,
    predict_public_belief_dual_hand_cfv_model,
)
from play_slumbot import card_str_to_index, parse_action


def _predict_loaded_ensemble(
    loaded,
    dataset: DualCFVDataset,
    *,
    device,
    batch_size: int,
) -> np.ndarray:
    preds = [
        predict_public_belief_dual_hand_cfv_model(
            model,
            payload,
            dataset.features,
            dataset.belief,
            dataset.hero_masks,
            dataset.villain_masks,
            device=device,
            batch_size=batch_size,
        )
        for model, payload in loaded
    ]
    return np.mean(np.stack(preds, axis=0), axis=0, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Perturb turn/river public beliefs and compare a dual-CFV checkpoint "
            "ensemble to re-solved dual-player CFV labels."
        )
    )
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--mix-uniform", type=float, default=0.5)
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    cases = load_cases_json(args.cases)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(args.cfv_cache)
    if len(cases) != base_dataset.features.shape[0]:
        raise ValueError("case count does not match CFV cache rows")

    loaded = load_public_belief_dual_hand_cfv_ensemble(args.checkpoint, device=device)
    payload = loaded[0][1]
    limit = min(max(1, int(args.limit)), len(cases))
    features = []
    beliefs = []
    hero_values = []
    villain_values = []
    hero_masks = []
    villain_masks = []
    labels = []
    records = []
    for idx, case in enumerate(cases[:limit]):
        parsed = parse_action(case.action_str)
        street = int(parsed.get("st", -1))
        if street not in (2, 3):
            continue
        n_board = 4 if street == 2 else 5
        board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
        legal = _legal_global_mask(board_idx)
        base_belief = base_dataset.belief[idx]
        hero = _mix_with_uniform(base_belief[:N_HANDS], legal, args.mix_uniform)
        villain = _mix_with_uniform(base_belief[N_HANDS:], legal, args.mix_uniform)
        belief_row = np.concatenate([hero, villain]).astype(np.float32, copy=False)
        hero_value, villain_value, hero_mask, villain_mask, record = _case_dual_cfv_target(
            case,
            belief_row,
            solver_iterations=args.solver_iterations,
            solver_backend=args.solver_backend,
            value_scale=args.value_scale,
        )
        features.append(base_dataset.features[idx])
        beliefs.append(belief_row)
        hero_values.append(hero_value)
        villain_values.append(villain_value)
        hero_masks.append(hero_mask)
        villain_masks.append(villain_mask)
        labels.append(case.label)
        records.append(record)

    if not records:
        raise ValueError("no river cases were evaluated")
    dataset = DualCFVDataset(
        features=np.stack(features).astype(np.float32, copy=False),
        belief=np.stack(beliefs).astype(np.float32, copy=False),
        hero_values=np.stack(hero_values).astype(np.float32, copy=False),
        villain_values=np.stack(villain_values).astype(np.float32, copy=False),
        hero_masks=np.stack(hero_masks).astype(np.float32, copy=False),
        villain_masks=np.stack(villain_masks).astype(np.float32, copy=False),
        labels=tuple(labels),
    )
    pred = _predict_loaded_ensemble(
        loaded,
        dataset,
        device=device,
        batch_size=args.batch_size,
    )
    model_metrics = _metrics(pred, dataset)
    zero_metrics = _metrics(_zero_dual_prediction(dataset), dataset)
    constant_baselines = {"zero": zero_metrics}
    if "target_mean" in payload:
        constant_baselines["train_mean"] = _metrics(
            _constant_dual_prediction(dataset, float(payload["target_mean"])),
            dataset,
        )
    if "target_median" in payload:
        constant_baselines["train_median"] = _metrics(
            _constant_dual_prediction(dataset, float(payload["target_median"])),
            dataset,
        )
    best_constant_mae = min(item["mae"] for item in constant_baselines.values())
    best_constant_rmse = min(item["rmse"] for item in constant_baselines.values())
    value_sum = _zero_sum_residuals(pred, dataset, n_hands=N_HANDS)
    metrics = {
        "mode": "public_belief_dual_hand_cfv_range_robustness",
        "passed": bool(
            np.isfinite(model_metrics["mae"])
            and model_metrics["mae"] < zero_metrics["mae"]
            and model_metrics["mae"] < best_constant_mae
            and model_metrics["rmse"] <= best_constant_rmse
            and value_sum["pred_minus_target_abs_mean"] < value_sum["target_abs_mean"]
        ),
        "pass_criteria": (
            "perturbed-range predictions must beat zero-CFV and train-constant "
            "MAE/RMSE, and improve range-weighted value-sum residual over zero"
        ),
        "checkpoints": [str(checkpoint) for checkpoint in args.checkpoint],
        "checkpoint_count": len(args.checkpoint),
        "cases": str(args.cases),
        "cfv_cache": str(args.cfv_cache),
        "device": str(device),
        "limit": int(args.limit),
        "n_evaluated": len(records),
        "mix_uniform": float(args.mix_uniform),
        "solver_iterations": int(args.solver_iterations),
        "solver_backend": args.solver_backend,
        "value_scale": float(args.value_scale),
        "n_labels": int(dataset.hero_masks.sum() + dataset.villain_masks.sum()),
        "model_holdout": model_metrics,
        "zero_baseline": zero_metrics,
        "constant_baselines": constant_baselines,
        "best_constant_mae": round(float(best_constant_mae), 8),
        "best_constant_rmse": round(float(best_constant_rmse), 8),
        "value_sum_residual": value_sum,
        "solver_mean_ms": round(float(np.mean([r["solver_latency_ms"] for r in records])), 3),
        "records": records,
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
