#!/usr/bin/env python3
"""Reweight joint-PBS targets by exact-vs-student resolver drift."""

from __future__ import annotations

import argparse
import json
import sys
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

_OPTIONAL_ARRAY_KEYS = (
    "hero_value_weights",
    "villain_value_weights",
    "action_tokens",
    "action_amounts",
)


def _root_label_from_record(record: dict[str, Any], label: str) -> str:
    root = record.get("root_label")
    if root is not None:
        return str(root)
    marker = "-cut"
    return label.split(marker, 1)[0] if marker in label else label


def _load_search_weights(search_json: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(search_json).read_text(encoding="utf-8"))
    by_label: dict[str, dict[str, Any]] = {}
    for record in payload.get("records", []):
        if "action_l1_drift" not in record:
            continue
        label = str(record.get("label", ""))
        if not label:
            continue
        drift = float(record["action_l1_drift"])
        if not np.isfinite(drift):
            continue
        action_agreement = bool(record.get("action_agreement", False))
        # Fixed mechanism-specific weighting: focus learning where the student
        # search perturbed the exact root policy, with a modest extra penalty
        # when the top action changed.
        weight = 1.0 + drift + (1.0 if not action_agreement else 0.0)
        by_label[label] = {
            "search_consistency_weight": float(weight),
            "teacher_student_action_l1_drift": float(drift),
            "teacher_student_action_agreement": action_agreement,
            "cut_applied": bool(record.get("cut_applied", False)),
        }
    return by_label, payload


def _load_joint_arrays(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    missing = [key for key in _ARRAY_KEYS if key not in data]
    if missing:
        raise ValueError(f"joint dataset is missing arrays: {missing}")
    arrays = {key: data[key] for key in _ARRAY_KEYS}
    for key in _OPTIONAL_ARRAY_KEYS:
        if key in data:
            arrays[key] = data[key]
    return arrays


def reweight_joint_pbs_by_search_drift(
    *,
    joint_npz: str | Path,
    metadata_json: str | Path,
    search_json: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    arrays = _load_joint_arrays(joint_npz)
    labels = [str(label) for label in arrays["labels"].tolist()]
    metadata = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    cut_records = [dict(record) for record in metadata.get("cut_records", [])]
    if len(cut_records) != len(labels):
        raise ValueError("cut_records length does not match joint labels")
    drift_by_root, search_payload = _load_search_weights(search_json)

    hero_weights = (
        np.asarray(arrays["hero_value_weights"], dtype=np.float32).copy()
        if "hero_value_weights" in arrays
        else np.asarray(arrays["hero_masks"], dtype=np.float32).copy()
    )
    villain_weights = (
        np.asarray(arrays["villain_value_weights"], dtype=np.float32).copy()
        if "villain_value_weights" in arrays
        else np.asarray(arrays["villain_masks"], dtype=np.float32).copy()
    )
    row_weights = np.ones(len(labels), dtype=np.float32)
    matched_roots: set[str] = set()
    for idx, (label, record) in enumerate(zip(labels, cut_records, strict=True)):
        root_label = _root_label_from_record(record, label)
        drift_record = drift_by_root.get(root_label)
        if drift_record is None:
            continue
        row_weight = float(drift_record["search_consistency_weight"])
        row_weights[idx] = row_weight
        hero_weights[idx] *= row_weight
        villain_weights[idx] *= row_weight
        matched_roots.add(root_label)
        record["search_consistency_weight"] = round(row_weight, 8)
        record["teacher_student_action_l1_drift"] = round(
            float(drift_record["teacher_student_action_l1_drift"]),
            8,
        )
        record["teacher_student_action_agreement"] = bool(
            drift_record["teacher_student_action_agreement"]
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_out = dict(arrays)
    arrays_out["hero_value_weights"] = hero_weights.astype(np.float32, copy=False)
    arrays_out["villain_value_weights"] = villain_weights.astype(np.float32, copy=False)
    arrays_out["search_consistency_weights"] = row_weights.astype(np.float32, copy=False)
    np.savez_compressed(output, **arrays_out)

    weighted_rows = int(np.count_nonzero(row_weights > 1.0))
    metrics = {
        "mode": "joint_pbs_search_drift_reweight",
        "joint_npz": str(joint_npz),
        "metadata_json": str(metadata_json),
        "search_json": str(search_json),
        "output": str(output),
        "source_search_mode": search_payload.get("mode"),
        "n_targets": int(len(labels)),
        "weighted_rows": weighted_rows,
        "matched_root_count": int(len(matched_roots)),
        "mean_search_consistency_weight": round(float(row_weights.mean()), 8),
        "max_search_consistency_weight": round(float(row_weights.max()), 8),
        "cut_records": cut_records,
    }
    save_metrics(metrics, output.with_suffix(".json"))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multiply joint-PBS value weights by exact-vs-student root search drift."
    )
    parser.add_argument("--joint", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--search-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    metrics = reweight_joint_pbs_by_search_drift(
        joint_npz=args.joint,
        metadata_json=args.metadata_json,
        search_json=args.search_json,
        output=args.output,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
