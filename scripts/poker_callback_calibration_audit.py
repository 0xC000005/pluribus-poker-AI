#!/usr/bin/env python3
"""Audit callback-state dual-CFV target calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(payload: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    raw = payload.get("records_json")
    if raw is None:
        return []
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return []


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    q = np.quantile(values.astype(np.float64), [0.5, 0.9, 0.99])
    return {
        "mean": round(float(np.mean(values)), 8),
        "std": round(float(np.std(values)), 8),
        "p50": round(float(q[0]), 8),
        "p90": round(float(q[1]), 8),
        "p99": round(float(q[2]), 8),
    }


def _reach_summary(reach: np.ndarray) -> dict[str, float]:
    if reach.size == 0:
        return {}
    sums = np.sum(reach, axis=1)
    probs = np.divide(
        reach,
        np.maximum(sums[:, None], 1e-12),
        out=np.zeros_like(reach, dtype=np.float64),
        where=sums[:, None] > 1e-12,
    )
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    entropy = -np.sum(
        np.where(probs > 0.0, probs * np.log(np.maximum(probs, 1e-12)), 0.0),
        axis=1,
    )
    eff_support = np.exp(entropy)
    return {
        "sum_mean": round(float(np.mean(sums)), 8),
        "sum_min": round(float(np.min(sums)), 8),
        "entropy_mean": round(float(np.mean(entropy)), 8),
        "effective_support_mean": round(float(np.mean(eff_support)), 8),
        "top1_mass_mean": round(float(np.mean(sorted_probs[:, 0])), 8),
        "top10_mass_mean": round(float(np.mean(np.sum(sorted_probs[:, :10], axis=1))), 8),
        "tiny_reach_mass_mean": round(float(np.mean(np.sum(probs * (probs <= 1e-6), axis=1))), 8),
    }


def _masked_values(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return values[masks > 0.5].astype(np.float32, copy=False)


def _cache_summary(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        hero_values = np.asarray(payload["hero_values"], dtype=np.float32)
        villain_values = np.asarray(payload["villain_values"], dtype=np.float32)
        hero_masks = np.asarray(payload["hero_masks"], dtype=np.float32)
        villain_masks = np.asarray(payload["villain_masks"], dtype=np.float32)
        belief = np.asarray(payload["belief"], dtype=np.float32)
        records = _records(payload)
    n_states, n_hands = hero_values.shape
    hero_reach = belief[:, :n_hands] if belief.shape[1] >= n_hands else np.zeros((n_states, 0))
    villain_reach = (
        belief[:, n_hands : 2 * n_hands]
        if belief.shape[1] >= 2 * n_hands
        else np.zeros((n_states, 0))
    )
    hero_masked = _masked_values(hero_values, hero_masks)
    villain_masked = _masked_values(villain_values, villain_masks)
    all_masked = np.concatenate([hero_masked, villain_masked])
    roots = sorted({str(record.get("root_label", "")) for record in records if record.get("root_label")})
    action_shapes = sorted({str(record.get("action_str", "")) for record in records if record.get("action_str")})
    zero_sum_overlap = (hero_masks > 0.5) & (villain_masks > 0.5)
    zero_sum_sum = (hero_values + villain_values)[zero_sum_overlap]
    return {
        "path": str(path),
        "n_states": int(n_states),
        "n_hands": int(n_hands),
        "n_roots": len(roots),
        "n_action_shapes": len(action_shapes),
        "hero_label_count": int(np.sum(hero_masks > 0.5)),
        "villain_label_count": int(np.sum(villain_masks > 0.5)),
        "hero_mask_count_mean": round(float(np.mean(np.sum(hero_masks > 0.5, axis=1))), 4),
        "villain_mask_count_mean": round(float(np.mean(np.sum(villain_masks > 0.5, axis=1))), 4),
        "target_abs": _quantiles(np.abs(all_masked)),
        "target_signed": _quantiles(all_masked),
        "zero_sum_pair_sum_abs": _quantiles(np.abs(zero_sum_sum.astype(np.float32))),
        "hero_reach": _reach_summary(hero_reach),
        "villain_reach": _reach_summary(villain_reach),
        "roots": roots[:20],
    }


def _smd(a: float, b: float, sa: float, sb: float) -> float:
    pooled = ((sa * sa + sb * sb) / 2.0) ** 0.5
    if pooled <= 1e-12:
        return 0.0
    return round(float((b - a) / pooled), 8)


def _shift_summary(train: dict[str, Any], holdout: dict[str, Any]) -> dict[str, float]:
    return {
        "target_abs_mean_smd": _smd(
            train["target_abs"]["mean"],
            holdout["target_abs"]["mean"],
            train["target_abs"]["std"],
            holdout["target_abs"]["std"],
        ),
        "hero_entropy_mean_delta": round(
            float(holdout["hero_reach"].get("entropy_mean", 0.0) - train["hero_reach"].get("entropy_mean", 0.0)),
            8,
        ),
        "villain_entropy_mean_delta": round(
            float(holdout["villain_reach"].get("entropy_mean", 0.0) - train["villain_reach"].get("entropy_mean", 0.0)),
            8,
        ),
    }


def run_audit(
    *,
    train_dual_cache: str | Path,
    holdout_dual_cache: str | Path,
    supervised_metrics: str | Path | None = None,
    leaf_ab: str | Path | None = None,
) -> dict[str, Any]:
    train = _cache_summary(train_dual_cache)
    holdout = _cache_summary(holdout_dual_cache)
    supervised = _load_json(supervised_metrics)
    leaf = _load_json(leaf_ab)
    warnings: list[str] = []
    if supervised:
        belief = supervised.get("belief_holdout", {})
        zero = supervised.get("zero_baseline", {})
        if belief.get("mae", 0.0) >= zero.get("mae", float("inf")):
            warnings.append("belief_holdout_mae_not_better_than_zero")
        if belief.get("rmse", 0.0) >= zero.get("rmse", float("inf")):
            warnings.append("belief_holdout_rmse_not_better_than_zero")
        if belief.get("mae", 0.0) >= supervised.get("best_constant_mae", float("inf")):
            warnings.append("belief_holdout_mae_not_better_than_train_constant")
    if leaf and leaf.get("mean_action_l1_drift") is not None:
        if float(leaf["mean_action_l1_drift"]) > float(leaf.get("max_mean_l1_drift", 0.4)):
            warnings.append("learned_leaf_action_drift_above_gate")
    if holdout["target_abs"]["p99"] > max(1e-12, 3.0 * train["target_abs"]["p50"]):
        warnings.append("heavy_tailed_holdout_targets")
    train_roots = set(train.get("roots", []))
    holdout_roots = set(holdout.get("roots", []))
    return {
        "mode": "callback_state_calibration_audit",
        "passed": True,
        "promotion": False,
        "train": train,
        "holdout": holdout,
        "shift": _shift_summary(train, holdout),
        "root_overlap_in_first20": sorted(train_roots & holdout_roots),
        "supervised_metrics": {
            key: supervised.get(key)
            for key in ("passed", "belief_holdout", "zero_baseline", "best_constant_mae", "best_constant_rmse")
            if key in supervised
        },
        "leaf_ab_metrics": {
            key: leaf.get(key)
            for key in ("passed", "action_agreement_rate", "mean_action_l1_drift", "max_mean_l1_drift", "project_zero_sum")
            if key in leaf
        },
        "warnings": warnings,
        "next_step": (
            "Use these summaries to choose a calibration fix; do not add model "
            "capacity or Slumbot runs until supervised baselines and leaf A/B "
            "agree on a root-disjoint holdout."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit callback-state DCVN target calibration and reach skew."
    )
    parser.add_argument("--train-dual-cache", required=True)
    parser.add_argument("--holdout-dual-cache", required=True)
    parser.add_argument("--supervised-metrics")
    parser.add_argument("--leaf-ab")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_audit(
        train_dual_cache=args.train_dual_cache,
        holdout_dual_cache=args.holdout_dual_cache,
        supervised_metrics=args.supervised_metrics,
        leaf_ab=args.leaf_ab,
    )
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
