import numpy as np

from poker_ai.research.sampled_action_mccfr import (
    full_regret,
    sampled_action_regret_estimate,
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
