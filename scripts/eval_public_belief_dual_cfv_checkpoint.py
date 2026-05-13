#!/usr/bin/env python3
"""Evaluate a saved dual-player public-belief hand-CFV checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _zero_sum_residuals(pred: np.ndarray, dataset, *, n_hands: int) -> dict[str, float]:
    target = np.stack([dataset.hero_values, dataset.villain_values], axis=0)
    residual_pred: list[float] = []
    residual_target: list[float] = []
    for idx in range(dataset.features.shape[0]):
        hero_range = np.maximum(dataset.belief[idx, :n_hands], 0.0) * dataset.hero_masks[idx]
        villain_range = (
            np.maximum(dataset.belief[idx, n_hands : 2 * n_hands], 0.0)
            * dataset.villain_masks[idx]
        )
        hero_total = float(hero_range.sum())
        villain_total = float(villain_range.sum())
        if hero_total <= 1e-12 or villain_total <= 1e-12:
            continue
        hero_range = hero_range / hero_total
        villain_range = villain_range / villain_total
        residual_pred.append(
            float(hero_range @ pred[0, idx] + villain_range @ pred[1, idx])
        )
        residual_target.append(
            float(hero_range @ target[0, idx] + villain_range @ target[1, idx])
        )
    if not residual_pred:
        return {
            "pred_abs_mean": 0.0,
            "target_abs_mean": 0.0,
            "pred_minus_target_abs_mean": 0.0,
        }
    pred_arr = np.asarray(residual_pred, dtype=np.float64)
    target_arr = np.asarray(residual_target, dtype=np.float64)
    return {
        "pred_abs_mean": round(float(np.mean(np.abs(pred_arr))), 8),
        "target_abs_mean": round(float(np.mean(np.abs(target_arr))), 8),
        "pred_minus_target_abs_mean": round(float(np.mean(np.abs(pred_arr - target_arr))), 8),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved dual-player hand-CFV checkpoint on a dual cache."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dual-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    from poker_ai.research.belief_probe import N_HANDS, _resolve_device
    from poker_ai.research.belief_value_probe import save_metrics
    from eval_public_belief_dual_hand_cfv_probe import (
        _constant_dual_prediction,
        _load_dual_cache,
        _metrics,
        _zero_dual_prediction,
        load_public_belief_dual_hand_cfv_checkpoint,
        predict_public_belief_dual_hand_cfv_model,
    )

    device = _resolve_device(args.device)
    dataset, records = _load_dual_cache(args.dual_cache)
    model, payload = load_public_belief_dual_hand_cfv_checkpoint(
        args.checkpoint,
        device=device,
    )
    for _ in range(max(0, int(args.warmup))):
        predict_public_belief_dual_hand_cfv_model(
            model,
            payload,
            dataset.features,
            dataset.belief,
            dataset.hero_masks,
            dataset.villain_masks,
            device=device,
            batch_size=args.batch_size,
        )
    _sync_if_needed(device)

    timings_ms = []
    pred = None
    for _ in range(max(1, int(args.repeats))):
        started = time.perf_counter()
        pred = predict_public_belief_dual_hand_cfv_model(
            model,
            payload,
            dataset.features,
            dataset.belief,
            dataset.hero_masks,
            dataset.villain_masks,
            device=device,
            batch_size=args.batch_size,
        )
        _sync_if_needed(device)
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    assert pred is not None

    n_states = int(dataset.features.shape[0])
    n_labels = int(dataset.hero_masks.sum() + dataset.villain_masks.sum())
    mean_ms = float(statistics.mean(timings_ms))
    p50_ms = float(statistics.median(timings_ms))
    p95_ms = float(sorted(timings_ms)[max(0, int(0.95 * len(timings_ms)) - 1)])
    solver_ms = [
        float(record["solver_latency_ms"])
        for record in records
        if "solver_latency_ms" in record
    ]
    solver_mean_ms = float(statistics.mean(solver_ms)) if solver_ms else None
    model_ms_per_state = mean_ms / max(n_states, 1)
    speedup = (
        solver_mean_ms / model_ms_per_state
        if solver_mean_ms is not None and model_ms_per_state > 0
        else None
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
    zero_sum = _zero_sum_residuals(pred, dataset, n_hands=N_HANDS)
    metrics = {
        "mode": "public_belief_dual_hand_cfv_checkpoint_eval",
        "passed": bool(
            n_states > 0
            and n_labels > 0
            and model_metrics["mae"] < zero_metrics["mae"]
            and model_metrics["mae"] < best_constant_mae
            and model_metrics["rmse"] <= best_constant_rmse
            and np.isfinite(zero_sum["pred_abs_mean"])
        ),
        "checkpoint": str(args.checkpoint),
        "dual_cache": str(args.dual_cache),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "n_states": n_states,
        "n_labels": n_labels,
        "model_holdout": model_metrics,
        "zero_baseline": zero_metrics,
        "constant_baselines": constant_baselines,
        "best_constant_mae": round(float(best_constant_mae), 8),
        "best_constant_rmse": round(float(best_constant_rmse), 8),
        "zero_sum_residual": zero_sum,
        "inference_mean_ms": round(mean_ms, 6),
        "inference_p50_ms": round(p50_ms, 6),
        "inference_p95_ms": round(p95_ms, 6),
        "model_ms_per_state": round(model_ms_per_state, 6),
        "labels_per_second": round(n_labels / max(mean_ms / 1000.0, 1e-12), 3),
        "solver_mean_ms_per_state": round(solver_mean_ms, 6)
        if solver_mean_ms is not None
        else None,
        "speedup_vs_solver_mean_per_state": round(float(speedup), 3)
        if speedup is not None
        else None,
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
