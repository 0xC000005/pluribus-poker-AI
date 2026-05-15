"""Small sampled-action MCCFR estimator diagnostics.

This module is intentionally standalone. It verifies the local regret-estimator
math before any full poker traversal or CUDA integration uses sampled traverser
actions.
"""

from __future__ import annotations

import numpy as np


def _legal_arrays(
    action_values: np.ndarray,
    strategy: np.ndarray,
    legal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(action_values, dtype=np.float64)
    sigma = np.asarray(strategy, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=np.float64) > 0.0
    if values.shape != sigma.shape or values.shape != legal.shape:
        raise ValueError("action_values, strategy, and legal_mask must match")
    if not np.any(legal):
        raise ValueError("at least one legal action is required")
    sigma = np.where(legal, sigma, 0.0)
    total = float(sigma.sum())
    if total <= 0.0:
        sigma = legal.astype(np.float64) / float(legal.sum())
    else:
        sigma = sigma / total
    return values, sigma, legal.astype(np.float64), legal


def full_regret(
    action_values: np.ndarray,
    strategy: np.ndarray,
    legal_mask: np.ndarray,
) -> np.ndarray:
    """Return exact immediate regrets for one traverser infoset."""
    values, sigma, _, legal_bool = _legal_arrays(action_values, strategy, legal_mask)
    state_value = float(np.dot(sigma, values))
    regret = values - state_value
    regret[~legal_bool] = 0.0
    return regret.astype(np.float32)


def sampled_action_regret_estimate(
    action_values: np.ndarray,
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    *,
    sampled_actions: np.ndarray,
    sample_probs: np.ndarray,
) -> np.ndarray:
    """Unbiased single-infoset regret estimate from sampled traverser actions.

    ``sampled_actions`` are assumed to be drawn with replacement from
    ``sample_probs``. Each sampled action contributes an inverse-probability
    weighted action-value estimate and the corresponding strategy-weighted
    state-value estimate.
    """
    values, sigma, _, legal_bool = _legal_arrays(action_values, strategy, legal_mask)
    q = np.asarray(sample_probs, dtype=np.float64)
    if q.shape != values.shape:
        raise ValueError("sample_probs must match action_values")
    if np.any(q[legal_bool] <= 0.0):
        raise ValueError("every legal action needs positive sampling probability")
    q = np.where(legal_bool, q, 0.0)
    q_total = float(q.sum())
    if q_total <= 0.0:
        raise ValueError("sample_probs must assign mass to legal actions")
    q = q / q_total

    actions = np.asarray(sampled_actions, dtype=np.int64).reshape(-1)
    if actions.size == 0:
        raise ValueError("sampled_actions must be non-empty")

    estimated_values = np.zeros_like(values, dtype=np.float64)
    estimated_state_value = 0.0
    sample_count = float(actions.size)
    for action in actions:
        if action < 0 or action >= values.shape[0] or not legal_bool[action]:
            raise ValueError("sampled action must be legal")
        weight = 1.0 / (sample_count * q[action])
        estimated_values[action] += values[action] * weight
        estimated_state_value += sigma[action] * values[action] * weight

    regret = estimated_values - estimated_state_value
    regret[~legal_bool] = 0.0
    return regret.astype(np.float32)
