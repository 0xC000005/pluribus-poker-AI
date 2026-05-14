import numpy as np
import pytest

from scripts.poker_rl_baseline_plan import build_plan_payload
from poker_ai.research.game_theoretic_rl import (
    algorithm_profile,
    anticipatory_mixture,
    epsilon_greedy_distribution,
    legal_softmax,
    regularized_policy_update,
    validate_baseline_plan,
)


def test_legal_softmax_masks_illegal_actions_and_normalizes():
    logits = np.array([10.0, 1.0, -5.0, 3.0], dtype=np.float32)
    legal_mask = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    probs = legal_softmax(logits, legal_mask)

    assert probs[0] == 0.0
    assert probs[2] == 0.0
    np.testing.assert_allclose(probs.sum(), 1.0)
    assert probs[3] > probs[1]


def test_epsilon_greedy_distribution_keeps_exploration_on_legal_actions_only():
    q_values = np.array([0.0, 5.0, 3.0, 9.0], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    probs = epsilon_greedy_distribution(q_values, legal_mask, epsilon=0.2)

    np.testing.assert_allclose(probs, np.array([0.1, 0.9, 0.0, 0.0]))


def test_anticipatory_mixture_combines_best_response_and_average_policy():
    best_response = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    average_policy = np.array([0.6, 0.2, 0.0, 0.2], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    probs = anticipatory_mixture(
        best_response,
        average_policy,
        legal_mask,
        anticipatory_param=0.25,
    )

    np.testing.assert_allclose(probs, np.array([0.45, 0.4, 0.0, 0.15]))


def test_regularized_policy_update_is_legal_and_advantage_sensitive():
    current = np.array([0.45, 0.45, 0.10], dtype=np.float32)
    reference = np.array([0.70, 0.20, 0.10], dtype=np.float32)
    advantages = np.array([0.0, 1.0, 10.0], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    updated = regularized_policy_update(
        current,
        advantages,
        legal_mask,
        reference_policy=reference,
        step_size=0.5,
        regularization_strength=0.2,
    )

    assert updated[2] == 0.0
    np.testing.assert_allclose(updated.sum(), 1.0)
    assert updated[1] > current[1]


def test_regularized_policy_update_moves_toward_reference_without_advantage():
    current = np.array([0.1, 0.9], dtype=np.float32)
    reference = np.array([0.8, 0.2], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0], dtype=np.float32)

    updated = regularized_policy_update(
        current,
        np.zeros(2, dtype=np.float32),
        legal_mask,
        reference_policy=reference,
        step_size=1.0,
        regularization_strength=0.5,
    )

    assert updated[0] > current[0]
    assert updated[1] < current[1]


def test_algorithm_profile_classifies_framework_controls_separately_from_nfsp():
    nfsp = algorithm_profile("nfsp")
    rnad = algorithm_profile("rnad")
    rebel = algorithm_profile("rebel")
    student = algorithm_profile("student_of_games")
    rainbow = algorithm_profile("rainbow_dqn")
    ppo = algorithm_profile("ppo")

    assert nfsp.equilibrium_mechanism == "average-policy fictitious self-play"
    assert nfsp.primary_role == "game_theoretic_rl_candidate"
    assert rnad.primary_role == "game_theoretic_rl_candidate"
    assert "regularized" in rnad.equilibrium_mechanism
    assert rebel.search_dependency == "public-belief search"
    assert student.search_dependency == "guided search"
    assert rainbow.primary_role == "rl_control_baseline"
    assert ppo.primary_role == "rl_control_baseline"
    assert "not an equilibrium method by itself" in ppo.warning


def test_validate_baseline_plan_rejects_framework_algorithm_as_mainline_without_equilibrium_layer():
    with pytest.raises(ValueError, match="cannot be the mainline"):
        validate_baseline_plan("ppo", allow_control_baseline=False)

    plan = validate_baseline_plan("ppo", allow_control_baseline=True)
    assert plan.profile.name == "ppo"
    assert plan.requires_equilibrium_layer is True
    assert plan.allowed_role == "control_baseline"


def test_build_plan_payload_serializes_control_baseline_warning():
    payload = build_plan_payload("rainbow", allow_control_baseline=True)

    assert payload["algorithm"] == "rainbow_dqn"
    assert payload["allowed_role"] == "control_baseline"
    assert payload["requires_equilibrium_layer"] is True
    assert "not an equilibrium method" in payload["warning"]
