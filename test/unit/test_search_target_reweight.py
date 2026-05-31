import numpy as np

from poker_ai.research.search_target_reweight import disagreement_reweighted_targets


def test_disagreement_reweighted_targets_preserves_mean_weight_and_prioritizes_disagreement():
    legal = np.ones((3, 4), dtype=np.float32)
    policy = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.0, 0.0, 1.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    base = np.ones(3, dtype=np.float32)

    weights, metrics = disagreement_reweighted_targets(
        policy_probs=policy,
        target_probs=targets,
        legal_masks=legal,
        base_weights=base,
    )

    assert np.isclose(weights.mean(), base.mean())
    assert weights[0] == 0.0
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]
    assert np.isclose(metrics["top1_mismatch_rate"], 2 / 3)
    assert metrics["mean_l1_disagreement"] > 0.0


def test_disagreement_reweighted_targets_falls_back_when_policy_matches_target():
    legal = np.ones((2, 3), dtype=np.float32)
    policy = np.array(
        [[0.2, 0.8, 0.0], [0.4, 0.1, 0.5]],
        dtype=np.float32,
    )
    targets = policy.copy()
    base = np.array([0.5, 2.0], dtype=np.float32)

    weights, metrics = disagreement_reweighted_targets(
        policy_probs=policy,
        target_probs=targets,
        legal_masks=legal,
        base_weights=base,
    )

    np.testing.assert_allclose(weights, base)
    assert metrics["fallback_used"] is True
