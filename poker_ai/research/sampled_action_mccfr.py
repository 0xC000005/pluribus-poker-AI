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
    baseline_values: np.ndarray | None = None,
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

    if baseline_values is None:
        baseline = np.zeros_like(values, dtype=np.float64)
    else:
        baseline = np.asarray(baseline_values, dtype=np.float64)
        if baseline.shape != values.shape:
            raise ValueError("baseline_values must match action_values")
        baseline = np.where(legal_bool, baseline, 0.0)

    estimated_values = baseline.astype(np.float64, copy=True)
    estimated_state_value = float(np.dot(sigma, baseline))
    sample_count = float(actions.size)
    for action in actions:
        if action < 0 or action >= values.shape[0] or not legal_bool[action]:
            raise ValueError("sampled action must be legal")
        weight = 1.0 / (sample_count * q[action])
        residual = values[action] - baseline[action]
        estimated_values[action] += residual * weight
        estimated_state_value += sigma[action] * residual * weight

    regret = estimated_values - estimated_state_value
    regret[~legal_bool] = 0.0
    return regret.astype(np.float32)


def pps_without_replacement_inclusion_probs(
    sample_probs: np.ndarray,
    legal_mask: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    """Exact inclusion probabilities for sequential PPS without replacement.

    The action space is tiny in this project, so exact recursion is acceptable
    for research probes. ``sample_probs`` defines the sequential draw weights
    over legal actions; selected actions are not replaced.
    """
    legal = np.asarray(legal_mask, dtype=np.float64) > 0.0
    q = np.asarray(sample_probs, dtype=np.float64)
    if q.shape != legal.shape:
        raise ValueError("sample_probs must match legal_mask")
    if not np.any(legal):
        raise ValueError("at least one legal action is required")
    if np.any(q[legal] <= 0.0):
        raise ValueError("every legal action needs positive sampling probability")
    q = np.where(legal, q, 0.0)
    q = q / float(q.sum())
    legal_indices = tuple(int(idx) for idx in np.flatnonzero(legal))
    k = min(max(1, int(sample_count)), len(legal_indices))
    inclusion = np.zeros_like(q, dtype=np.float64)
    if k >= len(legal_indices):
        inclusion[list(legal_indices)] = 1.0
        return inclusion.astype(np.float32)

    def visit(
        remaining: tuple[int, ...],
        draws_left: int,
        probability: float,
        selected: tuple[int, ...],
    ) -> None:
        if draws_left == 0:
            for action in selected:
                inclusion[action] += probability
            return
        denom = float(sum(q[action] for action in remaining))
        for action in remaining:
            next_probability = probability * float(q[action]) / denom
            next_remaining = tuple(item for item in remaining if item != action)
            visit(
                next_remaining,
                draws_left - 1,
                next_probability,
                selected + (action,),
            )

    visit(legal_indices, k, 1.0, ())
    return inclusion.astype(np.float32)


def sample_pps_without_replacement(
    rng: np.random.Generator,
    sample_probs: np.ndarray,
    legal_mask: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    """Draw unique legal actions by sequential PPS without replacement."""
    legal = np.asarray(legal_mask, dtype=np.float64) > 0.0
    q = np.asarray(sample_probs, dtype=np.float64)
    if q.shape != legal.shape:
        raise ValueError("sample_probs must match legal_mask")
    if np.any(q[legal] <= 0.0):
        raise ValueError("every legal action needs positive sampling probability")
    remaining = [int(idx) for idx in np.flatnonzero(legal)]
    k = min(max(1, int(sample_count)), len(remaining))
    if k >= len(remaining):
        return np.asarray(remaining, dtype=np.int64)
    selected: list[int] = []
    for _ in range(k):
        weights = np.asarray([q[action] for action in remaining], dtype=np.float64)
        weights = weights / float(weights.sum())
        local_idx = int(rng.choice(len(remaining), p=weights))
        selected.append(remaining.pop(local_idx))
    return np.asarray(selected, dtype=np.int64)


def sampled_action_regret_estimate_without_replacement(
    action_values: np.ndarray,
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    *,
    sampled_actions: np.ndarray,
    inclusion_probs: np.ndarray,
    baseline_values: np.ndarray | None = None,
) -> np.ndarray:
    """Unbiased regret estimate from actions sampled without replacement."""
    values, sigma, _, legal_bool = _legal_arrays(action_values, strategy, legal_mask)
    inclusion = np.asarray(inclusion_probs, dtype=np.float64)
    if inclusion.shape != values.shape:
        raise ValueError("inclusion_probs must match action_values")
    if np.any(inclusion[legal_bool] <= 0.0):
        raise ValueError("every legal action needs positive inclusion probability")
    actions = np.asarray(sampled_actions, dtype=np.int64).reshape(-1)
    if actions.size == 0:
        raise ValueError("sampled_actions must be non-empty")
    if len(set(int(action) for action in actions)) != actions.size:
        raise ValueError("without-replacement sampled actions must be unique")

    if baseline_values is None:
        baseline = np.zeros_like(values, dtype=np.float64)
    else:
        baseline = np.asarray(baseline_values, dtype=np.float64)
        if baseline.shape != values.shape:
            raise ValueError("baseline_values must match action_values")
        baseline = np.where(legal_bool, baseline, 0.0)

    estimated_values = baseline.astype(np.float64, copy=True)
    estimated_state_value = float(np.dot(sigma, baseline))
    for action in actions:
        if action < 0 or action >= values.shape[0] or not legal_bool[action]:
            raise ValueError("sampled action must be legal")
        residual = values[action] - baseline[action]
        weight = 1.0 / float(inclusion[action])
        estimated_values[action] += residual * weight
        estimated_state_value += sigma[action] * residual * weight

    regret = estimated_values - estimated_state_value
    regret[~legal_bool] = 0.0
    return regret.astype(np.float32)


def sampled_toy_traversal_regret_estimate(
    payoff_matrix: np.ndarray,
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    opponent_strategy: np.ndarray,
    *,
    sampled_actions: np.ndarray,
    sampled_opponent_actions: np.ndarray,
    sample_probs: np.ndarray,
) -> np.ndarray:
    """Toy one-step traversal estimate with sampled traverser and opponent actions.

    The matrix rows are traverser actions and columns are opponent responses.
    Opponent actions are assumed to be sampled from ``opponent_strategy``. This
    mirrors external-sampling's sampled opponent branch while testing the
    inverse-probability correction for sampled traverser actions.
    """
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("payoff_matrix must be two-dimensional")
    values, sigma, _, legal_bool = _legal_arrays(
        matrix @ np.asarray(opponent_strategy, dtype=np.float64),
        strategy,
        legal_mask,
    )
    del values

    opp_sigma = np.asarray(opponent_strategy, dtype=np.float64)
    if opp_sigma.ndim != 1 or opp_sigma.shape[0] != matrix.shape[1]:
        raise ValueError("opponent_strategy must match payoff_matrix columns")
    if np.any(opp_sigma < 0.0) or float(opp_sigma.sum()) <= 0.0:
        raise ValueError("opponent_strategy must have positive probability mass")
    opp_sigma = opp_sigma / float(opp_sigma.sum())

    q = np.asarray(sample_probs, dtype=np.float64)
    if q.shape[0] != matrix.shape[0]:
        raise ValueError("sample_probs must match payoff_matrix rows")
    if np.any(q[legal_bool] <= 0.0):
        raise ValueError("every legal action needs positive sampling probability")
    q = np.where(legal_bool, q, 0.0)
    q = q / float(q.sum())

    actions = np.asarray(sampled_actions, dtype=np.int64).reshape(-1)
    opp_actions = np.asarray(sampled_opponent_actions, dtype=np.int64).reshape(-1)
    if actions.size == 0 or actions.size != opp_actions.size:
        raise ValueError("sampled action arrays must be non-empty and equal length")

    estimated_values = np.zeros(matrix.shape[0], dtype=np.float64)
    estimated_state_value = 0.0
    sample_count = float(actions.size)
    for action, opp_action in zip(actions, opp_actions):
        if action < 0 or action >= matrix.shape[0] or not legal_bool[action]:
            raise ValueError("sampled action must be legal")
        if (
            opp_action < 0
            or opp_action >= matrix.shape[1]
            or opp_sigma[opp_action] <= 0.0
        ):
            raise ValueError("sampled opponent action must have positive probability")
        weight = 1.0 / (sample_count * q[action])
        payoff = matrix[action, opp_action]
        estimated_values[action] += payoff * weight
        estimated_state_value += sigma[action] * payoff * weight

    regret = estimated_values - estimated_state_value
    regret[~legal_bool] = 0.0
    return regret.astype(np.float32)
