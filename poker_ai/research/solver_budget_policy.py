"""Utilities for selective live-solver budget policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROB_EPS = 1e-12


def normalized_strategy(strategy: list[float] | np.ndarray, width: int) -> np.ndarray:
    probs = np.asarray(strategy, dtype=np.float64).reshape(-1)
    if probs.shape[0] > width:
        raise ValueError("strategy width exceeds requested feature width")
    padded = np.zeros((width,), dtype=np.float64)
    padded[: probs.shape[0]] = np.clip(probs, 0.0, None)
    total = float(padded.sum())
    if total <= PROB_EPS:
        return padded
    return padded / total


def solver_native_vector(
    strategy: list[float] | np.ndarray,
    *,
    action: int,
    width: int,
) -> np.ndarray:
    """Build deployment-available budget features from a cheap solver strategy."""

    probs = normalized_strategy(strategy, width)
    legal = probs > PROB_EPS
    sorted_probs = np.sort(probs)[::-1]
    top = float(sorted_probs[0]) if sorted_probs.size else 0.0
    second = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
    entropy = float(-(probs[legal] * np.log(probs[legal] + PROB_EPS)).sum())
    legal_count = int(legal.sum())
    normalized_entropy = entropy / float(np.log(max(legal_count, 2)))
    concentration = float(np.square(probs).sum())
    action_scale = float(max(width - 1, 1))
    return np.concatenate(
        [
            probs,
            np.asarray(
                [
                    top,
                    second,
                    top - second,
                    entropy,
                    normalized_entropy,
                    concentration,
                    float(legal_count) / float(max(width, 1)),
                    float(action) / action_scale,
                ],
                dtype=np.float64,
            ),
        ],
        axis=0,
    )


def score_selective_policy(policy: dict[str, Any], feature: np.ndarray) -> float:
    feature_arr = np.asarray(feature, dtype=np.float64).reshape(1, -1)
    mean = np.asarray(policy["feature_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(policy["feature_std"], dtype=np.float64).reshape(1, -1)
    weights = np.asarray(policy["ridge_weights"], dtype=np.float64)
    feature_std = (feature_arr - mean) / std
    x_aug = np.concatenate([np.ones((feature_std.shape[0], 1), dtype=np.float64), feature_std], axis=1)
    return float((x_aug @ weights)[0])


def load_selective_policy(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("policy_type") != "solver_budget_selective_profile":
        raise ValueError(f"Unsupported solver budget policy: {policy.get('policy_type')}")
    if policy.get("feature_source") != "solver-native":
        raise ValueError("Only solver-native selective budget policies are supported for live play")
    return policy
