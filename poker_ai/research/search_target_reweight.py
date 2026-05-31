"""Reweight exact-search policy targets by incumbent disagreement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.search_target_eval import _policy_probs


def disagreement_reweighted_targets(
    *,
    policy_probs: np.ndarray,
    target_probs: np.ndarray,
    legal_masks: np.ndarray,
    base_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return weights proportional to policy-vs-search L1 disagreement.

    The returned weights preserve the mean of ``base_weights``. When there is no
    disagreement signal, the original weights are returned unchanged.
    """
    policy = np.asarray(policy_probs, dtype=np.float64)
    targets = np.asarray(target_probs, dtype=np.float64)
    legal = (np.asarray(legal_masks, dtype=np.float64) > 0).astype(np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    if policy.shape != targets.shape or policy.shape != legal.shape:
        raise ValueError("policy_probs, target_probs, and legal_masks must share shape")
    if base.shape != (policy.shape[0],):
        raise ValueError("base_weights must have shape (N,)")
    if policy.ndim != 2:
        raise ValueError("policy_probs must be a 2D array")

    l1 = np.abs((policy - targets) * legal).sum(axis=1)
    mean_l1 = float(l1.mean()) if l1.size else 0.0
    base_mean = float(base.mean()) if base.size else 0.0
    fallback = mean_l1 <= 1e-12 or base_mean <= 1e-12
    if fallback:
        weights = base.astype(np.float32, copy=True)
    else:
        weights = base * (l1 / mean_l1)
        weight_mean = float(weights.mean())
        if weight_mean > 1e-12:
            weights *= base_mean / weight_mean
        weights = weights.astype(np.float32)

    policy_top = policy.argmax(axis=1)
    target_top = targets.argmax(axis=1)
    metrics = {
        "mode": "search_target_disagreement_reweight",
        "n_targets": int(policy.shape[0]),
        "fallback_used": bool(fallback),
        "mean_l1_disagreement": round(mean_l1, 6),
        "max_l1_disagreement": round(float(l1.max()) if l1.size else 0.0, 6),
        "top1_mismatch_rate": round(float((policy_top != target_top).mean()), 6)
        if policy.shape[0]
        else 0.0,
        "base_weight_mean": round(base_mean, 6),
        "new_weight_mean": round(float(weights.mean()) if weights.size else 0.0, 6),
        "new_weight_max": round(float(weights.max()) if weights.size else 0.0, 6),
    }
    return weights, metrics


def reweight_targets_by_checkpoint_disagreement(
    *,
    checkpoint: str | Path,
    targets: str | Path,
    output: str | Path,
    strategy_source: str = "regret",
    device: str = "auto",
) -> dict[str, Any]:
    """Write a target NPZ whose weights emphasize exact-search disagreement."""
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    target_buffer = PolicyTargetBuffer.from_npz(targets)
    loaded = load_value_network_checkpoint(checkpoint, torch_device)
    assert_strategy_source_supported(loaded, strategy_source)
    policy = _policy_probs(
        loaded.value_net,
        target_buffer.features,
        target_buffer.legal_masks,
        torch_device,
        strategy_source=strategy_source,
    )
    weights, metrics = disagreement_reweighted_targets(
        policy_probs=policy,
        target_probs=target_buffer.target_probs,
        legal_masks=target_buffer.legal_masks,
        base_weights=target_buffer.weights,
    )
    reweighted = PolicyTargetBuffer(
        target_buffer.features,
        target_buffer.legal_masks,
        target_buffer.target_probs,
        weights,
    )
    reweighted.save_npz(output)
    return {
        **metrics,
        "checkpoint": str(checkpoint),
        "targets": str(targets),
        "output": str(output),
        "strategy_source": strategy_source,
        "device": str(torch_device),
    }
