import numpy as np

from poker_ai.research.sampled_action_mccfr import (
    full_regret,
    pps_without_replacement_inclusion_probs,
    sample_pps_without_replacement,
    sampled_action_regret_estimate,
    sampled_action_regret_estimate_without_replacement,
    sampled_toy_traversal_regret_estimate,
)


def test_full_regret_is_action_value_minus_strategy_value():
    values = np.array([3.0, -1.0, 0.5], dtype=np.float32)
    strategy = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    regret = full_regret(values, strategy, legal_mask)

    assert np.allclose(regret, values - np.dot(strategy, values))


def test_sampled_action_regret_estimator_is_unbiased_for_single_draw():
    values = np.array([3.0, -1.0, 0.5], dtype=np.float32)
    strategy = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    sample_probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)

    expected = np.zeros_like(values)
    for action, prob in enumerate(sample_probs):
        expected += prob * sampled_action_regret_estimate(
            values,
            strategy,
            legal_mask,
            sampled_actions=np.array([action], dtype=np.int64),
            sample_probs=sample_probs,
        )

    assert np.allclose(expected, full_regret(values, strategy, legal_mask))


def test_sampled_action_regret_estimator_with_baseline_is_still_unbiased():
    values = np.array([3.0, -1.0, 0.5], dtype=np.float32)
    strategy = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    sample_probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    baseline = np.array([1.0, -2.0, 0.25], dtype=np.float32)

    expected = np.zeros_like(values)
    for action, prob in enumerate(sample_probs):
        expected += prob * sampled_action_regret_estimate(
            values,
            strategy,
            legal_mask,
            sampled_actions=np.array([action], dtype=np.int64),
            sample_probs=sample_probs,
            baseline_values=baseline,
        )

    assert np.allclose(expected, full_regret(values, strategy, legal_mask))


def test_pps_without_replacement_inclusion_probs_sum_to_sample_count():
    legal_mask = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    sample_probs = np.array([0.2, 0.3, 0.0, 0.5], dtype=np.float32)

    inclusion = pps_without_replacement_inclusion_probs(
        sample_probs,
        legal_mask,
        sample_count=2,
    )

    assert np.all(inclusion[legal_mask > 0.0] > 0.0)
    assert np.isclose(inclusion.sum(), 2.0)
    assert inclusion[2] == 0.0


def test_sample_pps_without_replacement_returns_unique_legal_actions():
    rng = np.random.default_rng(5)
    legal_mask = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    sample_probs = np.array([0.2, 0.3, 0.0, 0.5], dtype=np.float32)

    sampled = sample_pps_without_replacement(
        rng,
        sample_probs,
        legal_mask,
        sample_count=2,
    )

    assert sampled.shape == (2,)
    assert len(set(sampled.tolist())) == 2
    assert all(legal_mask[action] > 0.0 for action in sampled)


def test_without_replacement_regret_estimator_is_unbiased_by_enumeration():
    values = np.array([3.0, -1.0, 0.5], dtype=np.float32)
    strategy = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    sample_probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    inclusion = pps_without_replacement_inclusion_probs(
        sample_probs,
        legal_mask,
        sample_count=2,
    )

    expected = np.zeros_like(values)
    for first in range(3):
        p_first = sample_probs[first]
        remaining_prob = sample_probs.sum() - p_first
        for second in range(3):
            if second == first:
                continue
            prob = p_first * sample_probs[second] / remaining_prob
            expected += prob * sampled_action_regret_estimate_without_replacement(
                values,
                strategy,
                legal_mask,
                sampled_actions=np.array([first, second], dtype=np.int64),
                inclusion_probs=inclusion,
            )

    assert np.allclose(expected, full_regret(values, strategy, legal_mask))


def test_sampled_action_regret_estimator_rejects_zero_sampling_probability():
    values = np.array([1.0, 2.0], dtype=np.float32)
    strategy = np.array([0.5, 0.5], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0], dtype=np.float32)
    sample_probs = np.array([1.0, 0.0], dtype=np.float32)

    try:
        sampled_action_regret_estimate(
            values,
            strategy,
            legal_mask,
            sampled_actions=np.array([1], dtype=np.int64),
            sample_probs=sample_probs,
        )
    except ValueError as exc:
        assert "positive sampling probability" in str(exc)
    else:
        raise AssertionError("expected zero-probability sampled action to fail")


def test_sampled_toy_traversal_estimator_matches_exhaustive_expectation():
    payoff_matrix = np.array(
        [
            [2.0, -1.0],
            [0.5, 1.0],
            [-2.0, 3.0],
        ],
        dtype=np.float32,
    )
    strategy = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    opponent_strategy = np.array([0.7, 0.3], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    sample_probs = np.array([0.25, 0.25, 0.5], dtype=np.float32)
    action_values = payoff_matrix @ opponent_strategy

    expected = np.zeros(3, dtype=np.float32)
    for action, action_prob in enumerate(sample_probs):
        for opp_action, opp_prob in enumerate(opponent_strategy):
            expected += action_prob * opp_prob * sampled_toy_traversal_regret_estimate(
                payoff_matrix,
                strategy,
                legal_mask,
                opponent_strategy,
                sampled_actions=np.array([action], dtype=np.int64),
                sampled_opponent_actions=np.array([opp_action], dtype=np.int64),
                sample_probs=sample_probs,
            )

    assert np.allclose(expected, full_regret(action_values, strategy, legal_mask))
