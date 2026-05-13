#!/usr/bin/env python3
"""Attribute dual-CFV checkpoint errors on an existing exact-label cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_probe import N_HANDS, _resolve_device
from poker_ai.research.belief_value_probe import save_metrics

from eval_public_belief_dual_cfv_checkpoint import _zero_sum_residuals
from eval_public_belief_dual_hand_cfv_probe import (  # noqa: E402
    _load_dual_cache,
    _metrics,
    _zero_dual_prediction,
    load_public_belief_dual_hand_cfv_ensemble,
    predict_public_belief_dual_hand_cfv_model,
    project_dual_cfv_zero_sum,
)


def _predict_loaded_ensemble(
    loaded: list[tuple[Any, dict[str, Any]]],
    dataset: Any,
    *,
    device: Any,
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


def _state_errors(pred: np.ndarray, dataset: Any) -> list[dict[str, float]]:
    target = np.stack([dataset.hero_values, dataset.villain_values], axis=0)
    mask = np.stack([dataset.hero_masks, dataset.villain_masks], axis=0)
    records: list[dict[str, float]] = []
    for state_idx in range(dataset.features.shape[0]):
        selected = mask[:, state_idx, :] > 0
        err = pred[:, state_idx, :][selected].astype(np.float64) - target[:, state_idx, :][
            selected
        ].astype(np.float64)
        if err.size == 0:
            records.append({"mae": 0.0, "rmse": 0.0, "bias": 0.0, "n_labels": 0.0})
            continue
        records.append(
            {
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "n_labels": float(err.size),
            }
        )
    return records


def _subset_dataset(dataset: Any, indices: list[int]) -> Any:
    return type(dataset)(
        features=dataset.features[indices],
        belief=dataset.belief[indices],
        hero_values=dataset.hero_values[indices],
        villain_values=dataset.villain_values[indices],
        hero_masks=dataset.hero_masks[indices],
        villain_masks=dataset.villain_masks[indices],
        labels=tuple(dataset.labels[idx] for idx in indices),
    )


def _residual_for_indices(pred: np.ndarray, dataset: Any, indices: list[int]) -> dict[str, float]:
    if not indices:
        return {
            "pred_abs_mean": 0.0,
            "target_abs_mean": 0.0,
            "pred_minus_target_abs_mean": 0.0,
        }
    subset = _subset_dataset(dataset, indices)
    return _zero_sum_residuals(pred[:, indices, :], subset, n_hands=N_HANDS)


def _metric_for_indices(pred: np.ndarray, dataset: Any, indices: list[int]) -> dict[str, float]:
    if not indices:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0}
    subset = _subset_dataset(dataset, indices)
    return _metrics(pred[:, indices, :], subset)


def _group_key(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts = [str(record.get(field, "<missing>")) for field in fields]
    return "|".join(parts)


def merge_record_metadata(
    records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge leaf-export metadata into exact-cache records by stable label."""
    metadata_by_label = {
        str(record.get("label", "")): record
        for record in metadata_records
        if record.get("label") is not None
    }
    merged: list[dict[str, Any]] = []
    for record in records:
        label = str(record.get("label", ""))
        metadata = metadata_by_label.get(label, {})
        combined = dict(metadata)
        combined.update(record)
        merged.append(combined)
    return merged


def group_error_records(
    pred: np.ndarray,
    dataset: Any,
    records: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return groups sorted by worst MAE, using only cached exact labels."""
    per_state = _state_errors(pred, dataset)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in range(dataset.features.shape[0]):
        record = records[idx] if idx < len(records) else {}
        groups[_group_key(record, group_by)].append(idx)

    summaries: list[dict[str, Any]] = []
    for key, indices in groups.items():
        metrics = _metric_for_indices(pred, dataset, indices)
        residual = _residual_for_indices(pred, dataset, indices)
        subset = _subset_dataset(dataset, indices)
        zero_metrics = _metrics(_zero_dual_prediction(subset), subset)
        worst = sorted(indices, key=lambda idx: per_state[idx]["mae"], reverse=True)[:top_k]
        summaries.append(
            {
                "key": key,
                "group_by": list(group_by),
                "n_states": int(len(indices)),
                "n_labels": int(sum(per_state[idx]["n_labels"] for idx in indices)),
                "model": metrics,
                "zero_baseline": zero_metrics,
                "value_sum_residual": residual,
                "worst_states": [
                    {
                        "index": int(idx),
                        "label": str(dataset.labels[idx]),
                        "mae": round(float(per_state[idx]["mae"]), 8),
                        "rmse": round(float(per_state[idx]["rmse"]), 8),
                        "record": records[idx] if idx < len(records) else {},
                    }
                    for idx in worst
                ],
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -float(item["model"]["mae"]),
            -float(item["value_sum_residual"]["pred_minus_target_abs_mean"]),
            str(item["key"]),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Group exact dual-CFV cache errors by source/terminal metadata."
    )
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--dual-cache", required=True)
    parser.add_argument(
        "--metadata-json",
        help="Optional leaf-export JSON whose records are merged into cache records by label.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--project-zero-sum", action="store_true")
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["leaf_action_str"],
        help="Record fields used to group cache rows.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    dataset, records = _load_dual_cache(args.dual_cache)
    if args.metadata_json:
        metadata_payload = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
        records = merge_record_metadata(records, list(metadata_payload.get("records", [])))
    loaded = load_public_belief_dual_hand_cfv_ensemble(args.checkpoint, device=device)
    pred = _predict_loaded_ensemble(
        loaded,
        dataset,
        device=device,
        batch_size=args.batch_size,
    )
    if args.project_zero_sum:
        pred = project_dual_cfv_zero_sum(
            pred,
            dataset.belief,
            dataset.hero_masks,
            dataset.villain_masks,
        )

    groups = group_error_records(
        pred,
        dataset,
        records,
        group_by=tuple(str(item) for item in args.group_by),
        top_k=max(1, int(args.top_k)),
    )
    model_metrics = _metrics(pred, dataset)
    zero_metrics = _metrics(_zero_dual_prediction(dataset), dataset)
    residual = _zero_sum_residuals(pred, dataset, n_hands=N_HANDS)
    metrics = {
        "mode": "dual_cfv_cache_error_attribution",
        "passed": True,
        "dual_cache": str(args.dual_cache),
        "metadata_json": str(args.metadata_json) if args.metadata_json else None,
        "checkpoints": [str(path) for path in args.checkpoint],
        "checkpoint_count": len(args.checkpoint),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "project_zero_sum": bool(args.project_zero_sum),
        "group_by": [str(item) for item in args.group_by],
        "n_states": int(dataset.features.shape[0]),
        "n_records": int(len(records)),
        "n_labels": int(dataset.hero_masks.sum() + dataset.villain_masks.sum()),
        "model_holdout": model_metrics,
        "zero_baseline": zero_metrics,
        "value_sum_residual": residual,
        "groups": groups,
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
