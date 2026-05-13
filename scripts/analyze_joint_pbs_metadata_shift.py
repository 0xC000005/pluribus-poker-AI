#!/usr/bin/env python3
"""Analyze train/holdout metadata shift and error correlations for joint PBS targets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.belief_value_probe import save_metrics


def _records(path: str | Path) -> list[dict[str, Any]]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict(item) for item in metadata.get("cut_records", [])]


def _numeric_fields(train: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> list[str]:
    fields = set(train[0]) if train else set()
    for record in train[1:] + holdout:
        fields &= set(record)
    out = []
    for field in sorted(fields):
        values = [record.get(field) for record in train + holdout]
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            out.append(field)
    return out


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(arr.mean()), 8),
        "std": round(float(arr.std()), 8),
        "min": round(float(arr.min()), 8),
        "max": round(float(arr.max()), 8),
    }


def _numeric_shift(
    train: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for field in _numeric_fields(train, holdout):
        train_values = [float(record[field]) for record in train]
        holdout_values = [float(record[field]) for record in holdout]
        train_summary = _summary(train_values)
        holdout_summary = _summary(holdout_values)
        pooled = math.sqrt(
            max(float(train_summary["std"]) ** 2 + float(holdout_summary["std"]) ** 2, 1e-12)
            / 2.0
        )
        rows.append(
            {
                "field": field,
                "train": train_summary,
                "holdout": holdout_summary,
                "standardized_mean_diff": round(
                    (float(holdout_summary["mean"]) - float(train_summary["mean"])) / pooled,
                    8,
                ),
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["standardized_mean_diff"])), reverse=True)


def _categorical_shift(
    train: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for field in fields:
        train_counts = Counter(str(record.get(field, "missing")) for record in train)
        holdout_counts = Counter(str(record.get(field, "missing")) for record in holdout)
        missing_in_train = sorted(key for key in holdout_counts if key not in train_counts)
        rows.append(
            {
                "field": field,
                "train_unique": int(len(train_counts)),
                "holdout_unique": int(len(holdout_counts)),
                "holdout_missing_in_train": missing_in_train,
                "train_top": [
                    {"key": key, "count": int(count)}
                    for key, count in train_counts.most_common(8)
                ],
                "holdout_top": [
                    {"key": key, "count": int(count)}
                    for key, count in holdout_counts.most_common(8)
                ],
            }
        )
    return rows


def _error_correlations(
    holdout: list[dict[str, Any]],
    error_json: str | Path | None,
) -> list[dict[str, Any]]:
    if error_json is None:
        return []
    payload = json.loads(Path(error_json).read_text(encoding="utf-8"))
    error_by_label = {
        str(record.get("label")): float(record.get("mae", 0.0))
        for record in payload.get("state_errors", [])
    }
    joined = [record for record in holdout if str(record.get("label")) in error_by_label]
    rows = []
    for field in _numeric_fields(joined, []):
        x = np.asarray([float(record[field]) for record in joined], dtype=np.float64)
        y = np.asarray([error_by_label[str(record.get("label"))] for record in joined], dtype=np.float64)
        if x.size < 3 or float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        rows.append({"field": field, "pearson_mae": round(corr, 8)})
    return sorted(rows, key=lambda row: abs(float(row["pearson_mae"])), reverse=True)


def analyze_joint_pbs_metadata_shift(
    *,
    train_metadata_json: str | Path,
    holdout_metadata_json: str | Path,
    error_json: str | Path | None = None,
) -> dict[str, Any]:
    train = _records(train_metadata_json)
    holdout = _records(holdout_metadata_json)
    return {
        "mode": "joint_pbs_metadata_shift",
        "train_metadata_json": str(train_metadata_json),
        "holdout_metadata_json": str(holdout_metadata_json),
        "error_json": str(error_json) if error_json else None,
        "train_size": int(len(train)),
        "holdout_size": int(len(holdout)),
        "numeric_shift": _numeric_shift(train, holdout),
        "categorical_shift": _categorical_shift(
            train,
            holdout,
            ("action_shape", "actor_to_act", "bet_count"),
        ),
        "error_correlations": _error_correlations(holdout, error_json),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze joint PBS metadata shift and error correlations."
    )
    parser.add_argument("--train-metadata-json", required=True)
    parser.add_argument("--holdout-metadata-json", required=True)
    parser.add_argument("--error-json")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = analyze_joint_pbs_metadata_shift(
        train_metadata_json=args.train_metadata_json,
        holdout_metadata_json=args.holdout_metadata_json,
        error_json=args.error_json,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
