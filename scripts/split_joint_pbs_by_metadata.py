#!/usr/bin/env python3
"""Split joint PBS targets while preserving successor-frontier metadata coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics


_ARRAY_KEYS = (
    "features",
    "policy_features",
    "belief",
    "legal_masks",
    "target_probs",
    "policy_weights",
    "hero_values",
    "villain_values",
    "hero_masks",
    "villain_masks",
    "labels",
)

_REACH_FIELDS = (
    "hero_reach_top10_mass",
    "villain_reach_top10_mass",
    "hero_reach_normalized_entropy",
    "villain_reach_normalized_entropy",
)


def _load_payload(joint_npz: str | Path, metadata_json: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    data = np.load(Path(joint_npz), allow_pickle=False)
    missing = [key for key in _ARRAY_KEYS if key not in data]
    if missing:
        raise ValueError(f"joint dataset is missing arrays: {missing}")
    arrays = {key: data[key] for key in _ARRAY_KEYS}
    labels = [str(item) for item in arrays["labels"].tolist()]
    metadata = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    records = [dict(item) for item in metadata.get("cut_records", [])]
    if len(records) != len(labels):
        raise ValueError(
            f"cut_records length {len(records)} does not match labels length {len(labels)}"
        )
    record_labels = [str(item.get("label")) for item in records]
    if record_labels != labels:
        by_label = {str(item.get("label")): dict(item) for item in records}
        if len(by_label) != len(records):
            raise ValueError("cut_records labels must be unique when metadata order differs")
        try:
            records = [by_label[label] for label in labels]
        except KeyError as exc:
            raise ValueError(f"metadata is missing cut_record for label {exc.args[0]!r}") from exc
    return arrays, metadata, records


def _shape_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(record.get("action_shape", "missing")) for record in records)
    return [{"key": key, "count": int(count)} for key, count in counts.most_common()]


def _numeric_summary(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [
        float(record[field])
        for record in records
        if isinstance(record.get(field), (int, float)) and not isinstance(record.get(field), bool)
    ]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 8),
        "std": round(float(arr.std()), 8),
        "min": round(float(arr.min()), 8),
        "max": round(float(arr.max()), 8),
    }


def _reach_thresholds(records: list[dict[str, Any]], reach_bins: int) -> dict[str, np.ndarray]:
    if reach_bins <= 1:
        return {}
    quantiles = np.linspace(0.0, 1.0, int(reach_bins) + 1, dtype=np.float64)[1:-1]
    thresholds: dict[str, np.ndarray] = {}
    for field in _REACH_FIELDS:
        values = [
            float(record[field])
            for record in records
            if isinstance(record.get(field), (int, float)) and not isinstance(record.get(field), bool)
        ]
        if values:
            thresholds[field] = np.quantile(np.asarray(values, dtype=np.float64), quantiles)
    return thresholds


def _reach_key(record: dict[str, Any], thresholds: dict[str, np.ndarray]) -> tuple[int, ...]:
    bins = []
    for field in _REACH_FIELDS:
        cuts = thresholds.get(field)
        value = record.get(field)
        if cuts is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            bins.append(0)
        else:
            bins.append(int(np.searchsorted(cuts, float(value), side="right")))
    return tuple(bins)


def _choose_holdout_indices(
    records: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
    reach_bins: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("--holdout-fraction must be between 0 and 1")
    rng = np.random.default_rng(int(seed))
    thresholds = _reach_thresholds(records, int(reach_bins))
    by_shape: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_shape[str(record.get("action_shape", "missing"))].append(idx)

    train_indices: set[int] = set()
    holdout_indices: set[int] = set()
    unsplittable_shapes: list[str] = []
    per_shape: list[dict[str, Any]] = []

    for shape, indices in sorted(by_shape.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train_indices.update(shuffled)
            unsplittable_shapes.append(shape)
            per_shape.append({"action_shape": shape, "total": len(shuffled), "train": len(shuffled), "holdout": 0})
            continue

        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx in shuffled:
            groups[_reach_key(records[idx], thresholds)].append(idx)

        target = int(round(len(shuffled) * float(holdout_fraction)))
        target = max(1, min(len(shuffled) - 1, target))
        shape_holdout: list[int] = []
        for key in sorted(groups):
            group = list(groups[key])
            rng.shuffle(group)
            group_target = int(round(len(group) * float(holdout_fraction)))
            if len(group) >= 2 and group_target == 0 and len(shape_holdout) < target:
                group_target = 1
            group_target = max(0, min(len(group) - 1, group_target))
            shape_holdout.extend(group[:group_target])

        if len(shape_holdout) < target:
            remaining = [idx for idx in shuffled if idx not in shape_holdout]
            rng.shuffle(remaining)
            shape_holdout.extend(remaining[: target - len(shape_holdout)])
        elif len(shape_holdout) > target:
            rng.shuffle(shape_holdout)
            shape_holdout = shape_holdout[:target]

        holdout_set = set(shape_holdout)
        train_set = set(shuffled) - holdout_set
        if not train_set:
            restored = shape_holdout.pop()
            holdout_set.remove(restored)
            train_set.add(restored)
        train_indices.update(train_set)
        holdout_indices.update(holdout_set)
        per_shape.append(
            {
                "action_shape": shape,
                "total": int(len(shuffled)),
                "train": int(len(train_set)),
                "holdout": int(len(holdout_set)),
            }
        )

    train = sorted(train_indices)
    holdout = sorted(holdout_indices)
    if set(train) & set(holdout):
        raise AssertionError("split produced overlapping train and holdout rows")
    if len(train) + len(holdout) != len(records):
        raise AssertionError("split did not account for every row")
    audit = {
        "holdout_fraction": float(holdout_fraction),
        "seed": int(seed),
        "reach_bins": int(reach_bins),
        "reach_fields": list(_REACH_FIELDS),
        "unsplittable_train_only_shapes": unsplittable_shapes,
        "per_shape_split": per_shape,
    }
    return train, holdout, audit


def _subset_arrays(arrays: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    return {key: np.asarray(value)[idx] for key, value in arrays.items()}


def _save_subset(path: str | Path, arrays: dict[str, np.ndarray], indices: list[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_subset_arrays(arrays, indices))


def _subset_metadata(
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    indices: list[int],
    *,
    output: str | Path,
    split_name: str,
    source_joint_npz: str | Path,
    source_metadata_json: str | Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    subset_records = [records[idx] for idx in indices]
    policy_weights = [float(record.get("policy_weight", 0.0)) for record in subset_records]
    hero_masks = [float(record.get("hero_mask_count", 0.0)) for record in subset_records]
    villain_masks = [float(record.get("villain_mask_count", 0.0)) for record in subset_records]
    payload = {
        "mode": "joint_pbs_metadata_split",
        "split": split_name,
        "source_mode": metadata.get("mode"),
        "source_joint_npz": str(source_joint_npz),
        "source_metadata_json": str(source_metadata_json),
        "output": str(output),
        "n_targets": int(len(indices)),
        "policy_target_count": int(sum(weight > 0.0 for weight in policy_weights)),
        "value_label_count": int(sum(hero_masks) + sum(villain_masks)),
        "action_shape_counts": _shape_counts(subset_records),
        "reach_summary": {
            field: _numeric_summary(subset_records, field)
            for field in _REACH_FIELDS
        },
        "split_audit": audit,
        "cut_records": subset_records,
    }
    return payload


def split_joint_pbs_by_metadata(
    *,
    joint_npz: str | Path,
    metadata_json: str | Path,
    train_output: str | Path,
    holdout_output: str | Path,
    holdout_fraction: float = 0.25,
    seed: int = 0,
    reach_bins: int = 4,
) -> dict[str, Any]:
    arrays, metadata, records = _load_payload(joint_npz, metadata_json)
    train_indices, holdout_indices, audit = _choose_holdout_indices(
        records,
        holdout_fraction=holdout_fraction,
        seed=seed,
        reach_bins=reach_bins,
    )
    _save_subset(train_output, arrays, train_indices)
    _save_subset(holdout_output, arrays, holdout_indices)
    train_metadata = _subset_metadata(
        metadata,
        records,
        train_indices,
        output=train_output,
        split_name="train",
        source_joint_npz=joint_npz,
        source_metadata_json=metadata_json,
        audit=audit,
    )
    holdout_metadata = _subset_metadata(
        metadata,
        records,
        holdout_indices,
        output=holdout_output,
        split_name="holdout",
        source_joint_npz=joint_npz,
        source_metadata_json=metadata_json,
        audit=audit,
    )
    train_json = Path(train_output).with_suffix(".json")
    holdout_json = Path(holdout_output).with_suffix(".json")
    save_metrics(train_metadata, train_json)
    save_metrics(holdout_metadata, holdout_json)

    train_shapes = {row["key"] for row in train_metadata["action_shape_counts"]}
    holdout_shapes = {row["key"] for row in holdout_metadata["action_shape_counts"]}
    metrics = {
        "mode": "joint_pbs_metadata_split_summary",
        "joint_npz": str(joint_npz),
        "metadata_json": str(metadata_json),
        "train_output": str(train_output),
        "holdout_output": str(holdout_output),
        "train_metadata_json": str(train_json),
        "holdout_metadata_json": str(holdout_json),
        "train_size": int(len(train_indices)),
        "holdout_size": int(len(holdout_indices)),
        "holdout_fraction": float(len(holdout_indices) / max(len(records), 1)),
        "holdout_missing_in_train_shapes": sorted(holdout_shapes - train_shapes),
        "train_only_shapes": sorted(train_shapes - holdout_shapes),
        "train_shape_counts": train_metadata["action_shape_counts"],
        "holdout_shape_counts": holdout_metadata["action_shape_counts"],
        "split_audit": audit,
    }
    save_metrics(metrics, Path(train_output).with_name(Path(train_output).stem + "_split_summary.json"))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split joint PBS targets with action-shape and reach-coverage metadata audits."
    )
    parser.add_argument("--joint", required=True, help="Input joint PBS .npz file.")
    parser.add_argument("--metadata-json", required=True, help="Input joint PBS metadata JSON.")
    parser.add_argument("--train-output", required=True, help="Output train .npz file.")
    parser.add_argument("--holdout-output", required=True, help="Output holdout .npz file.")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reach-bins", type=int, default=4)
    args = parser.parse_args(argv)
    metrics = split_joint_pbs_by_metadata(
        joint_npz=args.joint,
        metadata_json=args.metadata_json,
        train_output=args.train_output,
        holdout_output=args.holdout_output,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        reach_bins=args.reach_bins,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
