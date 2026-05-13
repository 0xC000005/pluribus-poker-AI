#!/usr/bin/env python3
"""Benchmark saved public-belief hand-CFV inference against solver-label cost."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure hand-CFV checkpoint batch inference throughput."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    from poker_ai.research.belief_probe import _resolve_device
    from poker_ai.research.belief_value_probe import (
        load_public_belief_cfv_dataset_cache,
        load_public_belief_hand_cfv_checkpoint,
        predict_public_belief_hand_cfv_model,
        save_metrics,
    )

    device = _resolve_device(args.device)
    dataset, records = load_public_belief_cfv_dataset_cache(args.cfv_cache)
    model, payload = load_public_belief_hand_cfv_checkpoint(args.checkpoint, device=device)

    for _ in range(max(0, int(args.warmup))):
        predict_public_belief_hand_cfv_model(
            model,
            payload,
            dataset.features,
            dataset.belief,
            dataset.value_masks,
            device=device,
            batch_size=args.batch_size,
        )
    _sync_if_needed(device)

    timings_ms = []
    for _ in range(max(1, int(args.repeats))):
        started = time.perf_counter()
        predict_public_belief_hand_cfv_model(
            model,
            payload,
            dataset.features,
            dataset.belief,
            dataset.value_masks,
            device=device,
            batch_size=args.batch_size,
        )
        _sync_if_needed(device)
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    solver_ms = [
        float(record["solver_latency_ms"])
        for record in records
        if "solver_latency_ms" in record
    ]
    n_states = int(dataset.features.shape[0])
    n_labels = int(dataset.value_masks.sum())
    mean_ms = float(statistics.mean(timings_ms))
    p50_ms = float(statistics.median(timings_ms))
    p95_ms = float(sorted(timings_ms)[max(0, int(0.95 * len(timings_ms)) - 1)])
    model_ms_per_state = mean_ms / max(n_states, 1)
    solver_mean_ms = float(statistics.mean(solver_ms)) if solver_ms else None
    speedup = (
        solver_mean_ms / model_ms_per_state
        if solver_mean_ms is not None and model_ms_per_state > 0
        else None
    )
    metrics = {
        "mode": "public_belief_hand_cfv_inference_benchmark",
        "passed": bool(n_states > 0 and n_labels > 0 and mean_ms > 0),
        "checkpoint": str(args.checkpoint),
        "cfv_cache": str(args.cfv_cache),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "n_states": n_states,
        "n_labels": n_labels,
        "inference_mean_ms": round(mean_ms, 6),
        "inference_p50_ms": round(p50_ms, 6),
        "inference_p95_ms": round(p95_ms, 6),
        "model_ms_per_state": round(model_ms_per_state, 6),
        "labels_per_second": round(n_labels / max(mean_ms / 1000.0, 1e-12), 3),
        "solver_mean_ms_per_state": round(solver_mean_ms, 6)
        if solver_mean_ms is not None
        else None,
        "solver_total_ms": round(float(sum(solver_ms)), 6) if solver_ms else None,
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
