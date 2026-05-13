#!/usr/bin/env python3
"""Fit affine calibration for joint PBS value predictions."""

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
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_value_probe import save_metrics

from eval_joint_pbs_continuation_probe import (  # noqa: E402
    load_joint_pbs_continuation_checkpoint,
    load_joint_pbs_dataset,
    predict_joint_pbs_cfv_model,
)


def fit_affine_calibration(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(mask, dtype=bool)
    if int(selected.sum()) < 2:
        return {"scale": 1.0, "bias": 0.0, "n": int(selected.sum())}
    x = np.asarray(pred, dtype=np.float64)[selected]
    y = np.asarray(target, dtype=np.float64)[selected]
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"scale": round(float(scale), 8), "bias": round(float(bias), 8), "n": int(x.size)}


def apply_value_calibration(pred: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float32).copy()
    players = calibration.get("players", calibration)
    hero = players.get("hero", {})
    villain = players.get("villain", {})
    out[0] = out[0] * float(hero.get("scale", 1.0)) + float(hero.get("bias", 0.0))
    out[1] = out[1] * float(villain.get("scale", 1.0)) + float(villain.get("bias", 0.0))
    return out


def _value_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    target = np.stack([dataset.hero_values, dataset.villain_values], axis=0)
    mask = np.stack([dataset.hero_masks, dataset.villain_masks], axis=0) > 0
    return target.astype(np.float32, copy=False), mask


def _error_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "label_count": 0}
    err = np.asarray(pred, dtype=np.float64)[selected] - np.asarray(target, dtype=np.float64)[selected]
    return {
        "mae": round(float(np.mean(np.abs(err))), 8),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 8),
        "bias": round(float(np.mean(err)), 8),
        "label_count": int(err.size),
    }


def _predict_dataset(
    *,
    checkpoint: str | Path,
    joint_npz: str | Path,
    metadata_json: str | Path | None,
    device: str,
    batch_size: int,
) -> tuple[Any, np.ndarray]:
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
    return dataset, pred


def fit_joint_pbs_value_calibration(
    *,
    checkpoint: str | Path,
    train_joint_npz: str | Path,
    holdout_joint_npz: str | Path,
    train_metadata_json: str | Path | None = None,
    holdout_metadata_json: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 8192,
) -> dict[str, Any]:
    train, train_pred = _predict_dataset(
        checkpoint=checkpoint,
        joint_npz=train_joint_npz,
        metadata_json=train_metadata_json,
        device=device,
        batch_size=batch_size,
    )
    holdout, holdout_pred = _predict_dataset(
        checkpoint=checkpoint,
        joint_npz=holdout_joint_npz,
        metadata_json=holdout_metadata_json,
        device=device,
        batch_size=batch_size,
    )
    train_target, train_mask = _value_arrays(train)
    holdout_target, holdout_mask = _value_arrays(holdout)
    calibration = {
        "mode": "joint_pbs_value_affine_calibration",
        "value_units": "normalized_by_checkpoint_value_scale",
        "players": {
            "hero": fit_affine_calibration(train_pred[0], train.hero_values, train.hero_masks > 0),
            "villain": fit_affine_calibration(
                train_pred[1],
                train.villain_values,
                train.villain_masks > 0,
            ),
        },
    }
    train_calibrated = apply_value_calibration(train_pred, calibration)
    holdout_calibrated = apply_value_calibration(holdout_pred, calibration)
    return {
        "mode": "joint_pbs_value_calibration_fit",
        "checkpoint": str(checkpoint),
        "train_joint_npz": str(train_joint_npz),
        "holdout_joint_npz": str(holdout_joint_npz),
        "train_metadata_json": str(train_metadata_json) if train_metadata_json else None,
        "holdout_metadata_json": str(holdout_metadata_json) if holdout_metadata_json else None,
        "device": str(device),
        "batch_size": int(batch_size),
        "calibration": calibration,
        "train_raw": _error_metrics(train_pred, train_target, train_mask),
        "train_calibrated": _error_metrics(train_calibrated, train_target, train_mask),
        "holdout_raw": _error_metrics(holdout_pred, holdout_target, holdout_mask),
        "holdout_calibrated": _error_metrics(holdout_calibrated, holdout_target, holdout_mask),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit affine calibration for joint PBS values.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-joint", required=True)
    parser.add_argument("--holdout-joint", required=True)
    parser.add_argument("--train-metadata-json")
    parser.add_argument("--holdout-metadata-json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    metrics = fit_joint_pbs_value_calibration(
        checkpoint=args.checkpoint,
        train_joint_npz=args.train_joint,
        holdout_joint_npz=args.holdout_joint,
        train_metadata_json=args.train_metadata_json,
        holdout_metadata_json=args.holdout_metadata_json,
        device=args.device,
        batch_size=args.batch_size,
    )
    save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
