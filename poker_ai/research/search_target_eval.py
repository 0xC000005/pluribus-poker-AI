"""Evaluate learned policies against search-generated policy targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)


def _policy_probs(
    value_net,
    features: np.ndarray,
    legal_masks: np.ndarray,
    device: torch.device,
    *,
    strategy_source: str = "policy-head",
) -> np.ndarray:
    features_t = torch.from_numpy(features).to(device)
    with torch.no_grad():
        if strategy_source == "policy-head":
            _, logits_t = value_net.forward_with_policy(features_t)
            scores = logits_t.cpu().numpy().astype(np.float64)
            masked = np.where(legal_masks > 0, scores, -1e9)
            shifted = masked - masked.max(axis=1, keepdims=True)
            probs = np.exp(shifted) * legal_masks
        elif strategy_source == "regret":
            advantages = value_net(features_t).cpu().numpy().astype(np.float64)
            probs = np.maximum(advantages, 0.0) * legal_masks
        else:
            raise ValueError(f"unknown strategy source: {strategy_source}")
    totals = probs.sum(axis=1, keepdims=True)
    legal_totals = legal_masks.sum(axis=1, keepdims=True).clip(min=1.0)
    fallback = legal_masks / legal_totals
    return np.where(totals > 1e-8, probs / np.maximum(totals, 1e-8), fallback)


def evaluate_search_targets(
    checkpoint: str | Path,
    targets: str | Path,
    *,
    device: str = "auto",
    strategy_source: str = "policy-head",
) -> dict:
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    target_buffer = PolicyTargetBuffer.from_npz(targets)
    loaded = load_value_network_checkpoint(checkpoint, torch_device)
    assert_strategy_source_supported(loaded, strategy_source)
    legal_masks = target_buffer.legal_masks.astype(np.float64)
    target_probs = target_buffer.target_probs.astype(np.float64)
    policy = _policy_probs(
        loaded.value_net,
        target_buffer.features,
        legal_masks,
        torch_device,
        strategy_source=strategy_source,
    )
    eps = 1e-9
    l1 = np.abs(policy - target_probs).sum(axis=1)
    kl = (target_probs * (np.log(target_probs + eps) - np.log(policy + eps))).sum(axis=1)
    policy_top = policy.argmax(axis=1)
    target_top = target_probs.argmax(axis=1)
    return {
        "passed": True,
        "mode": "search_target_eval",
        "checkpoint": str(checkpoint),
        "targets": str(targets),
        "strategy_source": strategy_source,
        "n_targets": int(target_buffer.size),
        "mean_l1": round(float(l1.mean()), 6),
        "mean_kl": round(float(kl.mean()), 6),
        "top1_match_rate": round(float((policy_top == target_top).mean()), 6),
        "policy_allin_rate": round(float((policy_top == 8).mean()), 6),
        "target_allin_rate": round(float((target_top == 8).mean()), 6),
        **loaded.metadata,
    }
