#!/usr/bin/env python3
"""Group joint PBS value prediction errors by exported target metadata."""

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

from poker_ai.research.belief_value_probe import save_metrics

from eval_joint_pbs_continuation_probe import (  # noqa: E402
    load_joint_pbs_continuation_checkpoint,
    load_joint_pbs_dataset,
    predict_joint_pbs_cfv_model,
)


def _state_errors(pred: np.ndarray, dataset: Any) -> list[dict[str, Any]]:
    target = np.stack([dataset.hero_values, dataset.villain_values], axis=0)
    mask = np.stack([dataset.hero_masks, dataset.villain_masks], axis=0) > 0
    records = []
    for idx, label in enumerate(dataset.labels):
        selected = mask[:, idx, :]
        if not np.any(selected):
            mae = 0.0
            rmse = 0.0
            bias = 0.0
            count = 0
        else:
            err = pred[:, idx, :][selected].astype(np.float64) - target[:, idx, :][
                selected
            ].astype(np.float64)
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            bias = float(np.mean(err))
            count = int(err.size)
        records.append(
            {
                "label": str(label),
                "mae": round(mae, 8),
                "rmse": round(rmse, 8),
                "bias": round(bias, 8),
                "label_count": count,
            }
        )
    return records


def _mean_record(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {"n": 0, "mae": 0.0, "rmse": 0.0, "bias": 0.0, "label_count": 0}
    weights = np.asarray([max(int(record["label_count"]), 1) for record in records], dtype=np.float64)
    mae = np.asarray([float(record["mae"]) for record in records], dtype=np.float64)
    rmse = np.asarray([float(record["rmse"]) for record in records], dtype=np.float64)
    bias = np.asarray([float(record["bias"]) for record in records], dtype=np.float64)
    return {
        "n": int(len(records)),
        "mae": round(float(np.average(mae, weights=weights)), 8),
        "rmse": round(float(np.average(rmse, weights=weights)), 8),
        "bias": round(float(np.average(bias, weights=weights)), 8),
        "label_count": int(sum(int(record["label_count"]) for record in records)),
    }


def _group_records(
    records: list[dict[str, Any]],
    metadata_by_label: dict[str, dict[str, Any]],
    group_by: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        meta = metadata_by_label.get(str(record["label"]), {})
        key = tuple(meta.get(field, "missing") for field in group_by)
        buckets[key].append(record)
    out = []
    for key, bucket in buckets.items():
        row = {field: value for field, value in zip(group_by, key, strict=True)}
        row.update(_mean_record(bucket))
        out.append(row)
    return sorted(out, key=lambda item: (float(item["mae"]), int(item["n"])), reverse=True)


def analyze_joint_pbs_value_errors(
    *,
    checkpoint: str | Path,
    joint_npz: str | Path,
    metadata_json: str | Path | None = None,
    group_by: tuple[str, ...] = ("actor_to_act", "bet_count"),
    batch_size: int = 8192,
    device: str = "auto",
) -> dict[str, Any]:
    dataset = load_joint_pbs_dataset(joint_npz, metadata_json=metadata_json)
    model, payload = load_joint_pbs_continuation_checkpoint(checkpoint, device=device)
    pred = predict_joint_pbs_cfv_model(
        model,
        payload,
        dataset.features,
        dataset.belief,
        dataset.hero_masks,
        dataset.villain_masks,
        action_tokens=dataset.action_tokens,
        action_amounts=dataset.action_amounts,
        device=device,
        batch_size=batch_size,
    )
    state_errors = _state_errors(pred, dataset)
    metadata_path = Path(metadata_json) if metadata_json else Path(joint_npz).with_suffix(".json")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_by_label = {
        str(item.get("label")): item
        for item in metadata.get("cut_records", [])
        if item.get("label") is not None
    }
    return {
        "mode": "joint_pbs_value_error_attribution",
        "checkpoint": str(checkpoint),
        "joint_npz": str(joint_npz),
        "metadata_json": str(metadata_path) if metadata_path.exists() else None,
        "device": str(device),
        "group_by": list(group_by),
        "global": _mean_record(state_errors),
        "groups": _group_records(state_errors, metadata_by_label, group_by),
        "state_errors": state_errors,
        "worst_states": sorted(
            state_errors,
            key=lambda item: float(item["mae"]),
            reverse=True,
        )[:10],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Group joint PBS value prediction errors by target metadata."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--metadata-json")
    parser.add_argument("--group-by", default="actor_to_act,bet_count")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    group_by = tuple(part.strip() for part in args.group_by.split(",") if part.strip())
    metrics = analyze_joint_pbs_value_errors(
        checkpoint=args.checkpoint,
        joint_npz=args.joint,
        metadata_json=args.metadata_json,
        group_by=group_by,
        batch_size=args.batch_size,
        device=args.device,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
