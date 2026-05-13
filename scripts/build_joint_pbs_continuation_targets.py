#!/usr/bin/env python3
"""Combine policy and dual-CFV targets for joint PBS continuation probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.belief_value_probe import save_metrics

from eval_public_belief_dual_hand_cfv_probe import _load_dual_cache  # noqa: E402


def _policy_summary(target_probs: np.ndarray) -> dict[str, float]:
    top_actions = np.argmax(target_probs, axis=1)
    entropies = []
    for row in target_probs:
        positive = row[row > 0]
        entropies.append(float(-(positive * np.log(positive)).sum()) if positive.size else 0.0)
    return {
        "policy_target_allin_rate": round(float(np.mean(top_actions == 8)), 8),
        "policy_target_mean_allin_prob": round(float(np.mean(target_probs[:, 8])), 8),
        "policy_target_mean_entropy": round(float(np.mean(entropies)), 8),
    }


def _public_features(features: np.ndarray) -> np.ndarray:
    public = np.asarray(features, dtype=np.float32).copy()
    public[..., :52] = 0.0
    return public


def build_joint_payload(
    policy_targets: PolicyTargetBuffer,
    dual_dataset,
    *,
    labels: tuple[str, ...],
    feature_atol: float,
) -> tuple[dict[str, np.ndarray], dict]:
    if policy_targets.size != dual_dataset.features.shape[0]:
        raise ValueError("policy target and dual-CFV row counts differ")
    if policy_targets.features.shape != dual_dataset.features.shape:
        raise ValueError("policy target and dual-CFV feature shapes differ")
    policy_public_features = _public_features(policy_targets.features)
    feature_abs_diff = np.abs(policy_public_features - dual_dataset.features)
    max_feature_abs_diff = float(feature_abs_diff.max()) if feature_abs_diff.size else 0.0
    if max_feature_abs_diff > float(feature_atol):
        raise ValueError(
            "policy target public features and dual-CFV features are misaligned: "
            f"max abs diff {max_feature_abs_diff:.8g} > {feature_atol:.8g}"
        )

    payload = {
        "features": dual_dataset.features.astype(np.float32, copy=False),
        "policy_features": policy_targets.features.astype(np.float32, copy=False),
        "belief": dual_dataset.belief.astype(np.float32, copy=False),
        "legal_masks": policy_targets.legal_masks.astype(np.float32, copy=False),
        "target_probs": policy_targets.target_probs.astype(np.float32, copy=False),
        "policy_weights": policy_targets.weights.astype(np.float32, copy=False),
        "hero_values": dual_dataset.hero_values.astype(np.float32, copy=False),
        "villain_values": dual_dataset.villain_values.astype(np.float32, copy=False),
        "hero_masks": dual_dataset.hero_masks.astype(np.float32, copy=False),
        "villain_masks": dual_dataset.villain_masks.astype(np.float32, copy=False),
        "labels": np.asarray(labels),
    }
    metadata = {
        "mode": "joint_pbs_continuation_targets",
        "feature_alignment": "public_features_match;policy_features_preserve_private_cards",
        "n_states": int(policy_targets.size),
        "feature_dim": int(dual_dataset.features.shape[1]),
        "belief_dim": int(dual_dataset.belief.shape[1]),
        "max_feature_abs_diff": round(max_feature_abs_diff, 10),
        "hero_label_count": int(dual_dataset.hero_masks.sum()),
        "villain_label_count": int(dual_dataset.villain_masks.sum()),
        "label_count": int(dual_dataset.hero_masks.sum() + dual_dataset.villain_masks.sum()),
        **_policy_summary(policy_targets.target_probs),
    }
    return payload, metadata


def save_joint_payload(payload: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge aligned policy targets and dual-CFV labels for joint probes."
    )
    parser.add_argument("--policy-targets", required=True)
    parser.add_argument("--dual-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-atol", type=float, default=1e-6)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    policy_targets = PolicyTargetBuffer.from_npz(args.policy_targets)
    dual_dataset, _records = _load_dual_cache(args.dual_cache)
    payload, metadata = build_joint_payload(
        policy_targets,
        dual_dataset,
        labels=dual_dataset.labels,
        feature_atol=args.feature_atol,
    )
    save_joint_payload(payload, args.output)
    metrics = {
        **metadata,
        "policy_targets": str(args.policy_targets),
        "dual_cache": str(args.dual_cache),
        "output": str(args.output),
        "feature_atol": float(args.feature_atol),
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
