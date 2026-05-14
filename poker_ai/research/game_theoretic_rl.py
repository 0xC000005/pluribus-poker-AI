"""Small game-theoretic RL primitives for poker autoresearch.

These helpers deliberately do not implement a full trainer. They provide the
first shared contract for testing NFSP/RM-FSP style policy mixing and for
classifying generic RL algorithms as controls rather than equilibrium methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlgorithmProfile:
    name: str
    primary_role: str
    equilibrium_mechanism: str
    search_dependency: str
    candidate_frameworks: tuple[str, ...]
    warning: str = ""


@dataclass(frozen=True)
class BaselinePlan:
    profile: AlgorithmProfile
    allowed_role: str
    requires_equilibrium_layer: bool


_PROFILES: dict[str, AlgorithmProfile] = {
    "nfsp": AlgorithmProfile(
        name="nfsp",
        primary_role="game_theoretic_rl_candidate",
        equilibrium_mechanism="average-policy fictitious self-play",
        search_dependency="none",
        candidate_frameworks=("rlcard", "native"),
    ),
    "rm_fsp": AlgorithmProfile(
        name="rm_fsp",
        primary_role="game_theoretic_rl_candidate",
        equilibrium_mechanism="regret-minimizing fictitious self-play",
        search_dependency="none",
        candidate_frameworks=("native",),
    ),
    "rnad": AlgorithmProfile(
        name="rnad",
        primary_role="game_theoretic_rl_candidate",
        equilibrium_mechanism="regularized Nash dynamics",
        search_dependency="none",
        candidate_frameworks=("native",),
        warning="R-NaD-style training is a mechanism target, not a drop-in trainer yet.",
    ),
    "rebel": AlgorithmProfile(
        name="rebel",
        primary_role="game_theoretic_rl_candidate",
        equilibrium_mechanism="public-belief self-play with value/policy learning",
        search_dependency="public-belief search",
        candidate_frameworks=("native",),
        warning="ReBeL-style work must pass public-belief/resolver gates before gameplay use.",
    ),
    "student_of_games": AlgorithmProfile(
        name="student_of_games",
        primary_role="game_theoretic_rl_candidate",
        equilibrium_mechanism="self-play learning with game-theoretic search",
        search_dependency="guided search",
        candidate_frameworks=("native",),
        warning="Student-of-Games-style work needs a bounded local search/evaluation contract.",
    ),
    "rainbow_dqn": AlgorithmProfile(
        name="rainbow_dqn",
        primary_role="rl_control_baseline",
        equilibrium_mechanism="none",
        search_dependency="none",
        candidate_frameworks=("tianshou", "rlcard", "native"),
        warning="Rainbow DQN is a value-learning control, not an equilibrium method by itself.",
    ),
    "dqn": AlgorithmProfile(
        name="dqn",
        primary_role="rl_control_baseline",
        equilibrium_mechanism="none",
        search_dependency="none",
        candidate_frameworks=("rlcard", "native"),
        warning="DQN is a value-learning control, not an equilibrium method by itself.",
    ),
    "ppo": AlgorithmProfile(
        name="ppo",
        primary_role="rl_control_baseline",
        equilibrium_mechanism="none",
        search_dependency="none",
        candidate_frameworks=("tianshou", "stable-baselines3", "cleanrl"),
        warning="PPO is a policy-gradient control, not an equilibrium method by itself.",
    ),
}


def _legal_mask_array(legal_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    if mask.size == 0 or not np.any(mask > 0):
        raise ValueError("legal_mask must contain at least one legal action")
    return (mask > 0).astype(np.float32)


def _normalize_legal(probs: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    mask = _legal_mask_array(legal_mask)
    values = np.asarray(probs, dtype=np.float32).reshape(mask.shape) * mask
    total = float(values.sum())
    if total > 1e-8:
        return values / total
    return mask / float(mask.sum())


def legal_softmax(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Softmax over legal actions with zero probability on illegal actions."""
    mask = _legal_mask_array(legal_mask)
    values = np.asarray(logits, dtype=np.float32).reshape(mask.shape)
    temperature = max(float(temperature), 1e-6)
    legal_values = values[mask > 0] / temperature
    shifted = legal_values - np.max(legal_values)
    exp_values = np.exp(shifted).astype(np.float32)
    probs = np.zeros_like(mask, dtype=np.float32)
    probs[mask > 0] = exp_values / float(exp_values.sum())
    return probs


def epsilon_greedy_distribution(
    q_values: np.ndarray,
    legal_mask: np.ndarray,
    *,
    epsilon: float = 0.06,
) -> np.ndarray:
    """Legal-mask-safe epsilon-greedy distribution for DQN-like controls."""
    mask = _legal_mask_array(legal_mask)
    q = np.asarray(q_values, dtype=np.float32).reshape(mask.shape)
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    legal_indices = np.flatnonzero(mask > 0)
    greedy_index = legal_indices[np.argmax(q[legal_indices])]
    probs = mask * (epsilon / float(legal_indices.size))
    probs[greedy_index] += 1.0 - epsilon
    return probs.astype(np.float32)


def anticipatory_mixture(
    best_response_policy: np.ndarray,
    average_policy: np.ndarray,
    legal_mask: np.ndarray,
    *,
    anticipatory_param: float = 0.1,
) -> np.ndarray:
    """NFSP-style mixture of best-response and average policies.

    `anticipatory_param` is the probability of acting from the best-response
    learner. The remaining probability uses the supervised average policy.
    """
    eta = float(np.clip(anticipatory_param, 0.0, 1.0))
    br = _normalize_legal(best_response_policy, legal_mask)
    avg = _normalize_legal(average_policy, legal_mask)
    return _normalize_legal(eta * br + (1.0 - eta) * avg, legal_mask)


def regularized_policy_update(
    current_policy: np.ndarray,
    advantages: np.ndarray,
    legal_mask: np.ndarray,
    *,
    reference_policy: np.ndarray | None = None,
    step_size: float = 0.1,
    regularization_strength: float = 0.1,
) -> np.ndarray:
    """One legal-mask-safe regularized exponentiated policy update.

    This is a small R-NaD-inspired primitive, not a full R-NaD trainer. The
    update improves actions with positive advantage while adding a KL-style
    pull toward a full-support reference policy.
    """
    mask = _legal_mask_array(legal_mask)
    current = _normalize_legal(current_policy, mask)
    if reference_policy is None:
        reference = mask / float(mask.sum())
    else:
        reference = _normalize_legal(reference_policy, mask)
    adv = np.asarray(advantages, dtype=np.float32).reshape(mask.shape)
    eps = 1e-8
    current_safe = np.clip(current, eps, 1.0)
    reference_safe = np.clip(reference, eps, 1.0)
    regularized_advantage = adv - float(regularization_strength) * (
        np.log(current_safe) - np.log(reference_safe)
    )
    logits = np.log(current_safe) + float(step_size) * regularized_advantage
    return legal_softmax(logits, mask)


def algorithm_profile(name: str) -> AlgorithmProfile:
    """Return a normalized profile for a game-theoretic RL candidate/control."""
    key = name.strip().lower().replace("-", "_")
    if key == "rainbow":
        key = "rainbow_dqn"
    if key in {"deepnash", "regularized_nash_dynamics"}:
        key = "rnad"
    if key in {"sog", "student_of_game", "player_of_games"}:
        key = "student_of_games"
    if key not in _PROFILES:
        known = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown algorithm profile '{name}'; known: {known}")
    return _PROFILES[key]


def validate_baseline_plan(
    name: str,
    *,
    allow_control_baseline: bool = False,
) -> BaselinePlan:
    """Guard against promoting generic RL algorithms as the mainline method."""
    profile = algorithm_profile(name)
    is_control = profile.primary_role == "rl_control_baseline"
    if is_control and not allow_control_baseline:
        raise ValueError(
            f"{profile.name} cannot be the mainline without an equilibrium "
            "layer; use it only as a control baseline."
        )
    return BaselinePlan(
        profile=profile,
        allowed_role="control_baseline" if is_control else "mainline_candidate",
        requires_equilibrium_layer=is_control,
    )
