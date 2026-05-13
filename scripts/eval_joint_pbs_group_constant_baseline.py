#!/usr/bin/env python3
"""Evaluate metadata-grouped constant baselines for joint PBS value targets."""

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

from eval_joint_pbs_continuation_probe import load_joint_pbs_dataset  # noqa: E402


def _load_metadata_by_label(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    metadata_path = Path(path)
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for field in ("cut_records", "records"):
        for item in metadata.get(field, []):
            label = item.get("label")
            if label is not None:
                out[str(label)] = dict(item)
    return out


def _group_key(
    label: str,
    metadata_by_label: dict[str, dict[str, Any]],
    group_by: tuple[str, ...],
) -> tuple[Any, ...]:
    meta = metadata_by_label.get(str(label), {})
    return tuple(meta.get(field, "missing") for field in group_by)


def _state_values(dataset: Any, idx: int) -> np.ndarray:
    hero_mask = dataset.hero_masks[idx] > 0
    villain_mask = dataset.villain_masks[idx] > 0
    values = [
        dataset.hero_values[idx, hero_mask],
        dataset.villain_values[idx, villain_mask],
    ]
    selected = [np.asarray(item, dtype=np.float64) for item in values if item.size]
    return np.concatenate(selected) if selected else np.zeros(0, dtype=np.float64)


def _fit_group_means(
    dataset: Any,
    metadata_by_label: dict[str, dict[str, Any]],
    group_by: tuple[str, ...],
) -> tuple[float, dict[tuple[Any, ...], float]]:
    all_values: list[np.ndarray] = []
    buckets: dict[tuple[Any, ...], list[np.ndarray]] = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        values = _state_values(dataset, idx)
        if not values.size:
            continue
        all_values.append(values)
        buckets[_group_key(label, metadata_by_label, group_by)].append(values)
    if not all_values:
        return 0.0, {}
    global_mean = float(np.concatenate(all_values).mean())
    group_means = {
        key: float(np.concatenate(parts).mean())
        for key, parts in buckets.items()
        if parts
    }
    return global_mean, group_means


def _value_metrics(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0}
    return {
        "mae": round(float(np.mean(np.abs(errors))), 8),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 8),
        "bias": round(float(np.mean(errors)), 8),
    }


def _eval_group_means(
    dataset: Any,
    metadata_by_label: dict[str, dict[str, Any]],
    group_by: tuple[str, ...],
    global_mean: float,
    group_means: dict[tuple[Any, ...], float],
) -> dict[str, Any]:
    errors = []
    fallback_count = 0
    group_rows: dict[tuple[Any, ...], list[np.ndarray]] = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        values = _state_values(dataset, idx)
        if not values.size:
            continue
        key = _group_key(label, metadata_by_label, group_by)
        pred = group_means.get(key)
        if pred is None:
            pred = global_mean
            fallback_count += 1
        err = np.full(values.shape, float(pred), dtype=np.float64) - values
        errors.append(err)
        group_rows[key].append(err)
    flat = np.concatenate(errors) if errors else np.zeros(0, dtype=np.float64)
    groups = []
    for key, parts in group_rows.items():
        row = {field: value for field, value in zip(group_by, key, strict=True)}
        merged = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
        row.update(_value_metrics(merged))
        row["label_count"] = int(merged.size)
        groups.append(row)
    return {
        **_value_metrics(flat),
        "fallback_states": int(fallback_count),
        "label_count": int(flat.size),
        "groups": sorted(groups, key=lambda item: float(item["mae"]), reverse=True),
    }


def eval_joint_pbs_group_constant_baseline(
    *,
    train_joint_npz: str | Path,
    holdout_joint_npz: str | Path,
    train_metadata_json: str | Path | None = None,
    holdout_metadata_json: str | Path | None = None,
    group_by: tuple[str, ...] = ("action_shape",),
) -> dict[str, Any]:
    train = load_joint_pbs_dataset(train_joint_npz, metadata_json=train_metadata_json)
    holdout = load_joint_pbs_dataset(holdout_joint_npz, metadata_json=holdout_metadata_json)
    train_meta = _load_metadata_by_label(train_metadata_json or Path(train_joint_npz).with_suffix(".json"))
    holdout_meta = _load_metadata_by_label(
        holdout_metadata_json or Path(holdout_joint_npz).with_suffix(".json")
    )
    global_mean, group_means = _fit_group_means(train, train_meta, group_by)
    grouped = _eval_group_means(
        holdout,
        holdout_meta,
        group_by,
        global_mean,
        group_means,
    )
    global_baseline = _eval_group_means(
        holdout,
        holdout_meta,
        group_by,
        global_mean,
        {},
    )
    return {
        "mode": "joint_pbs_group_constant_baseline",
        "train_joint_npz": str(train_joint_npz),
        "holdout_joint_npz": str(holdout_joint_npz),
        "train_metadata_json": str(train_metadata_json) if train_metadata_json else None,
        "holdout_metadata_json": str(holdout_metadata_json) if holdout_metadata_json else None,
        "group_by": list(group_by),
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "global_mean": round(float(global_mean), 8),
        "n_train_groups": int(len(group_means)),
        "grouped_constant": grouped,
        "global_train_mean_constant": {
            key: value
            for key, value in global_baseline.items()
            if key != "groups"
        },
        "grouped_beats_global": bool(grouped["mae"] < global_baseline["mae"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate train-metadata grouped constants on joint PBS values."
    )
    parser.add_argument("--train-joint", required=True)
    parser.add_argument("--holdout-joint", required=True)
    parser.add_argument("--train-metadata-json")
    parser.add_argument("--holdout-metadata-json")
    parser.add_argument("--group-by", default="action_shape")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    group_by = tuple(part.strip() for part in args.group_by.split(",") if part.strip())
    metrics = eval_joint_pbs_group_constant_baseline(
        train_joint_npz=args.train_joint,
        holdout_joint_npz=args.holdout_joint,
        train_metadata_json=args.train_metadata_json,
        holdout_metadata_json=args.holdout_metadata_json,
        group_by=group_by,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["grouped_beats_global"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
