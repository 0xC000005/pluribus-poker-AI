import numpy as np

from scripts.diagnose_cfr_trace_state import (
    action_policy_from_field,
    action_advantage_from_trace,
    belief_summary_features,
    summarize_trace_records,
)


def test_action_policy_from_field_uses_legal_actions_and_uniform_fallback():
    values = np.array([0.0, 3.0, 99.0, 0.0], dtype=np.float32)

    policy = action_policy_from_field(values, legal_actions=(0, 1, 3))

    np.testing.assert_allclose(policy, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    fallback = action_policy_from_field(np.zeros(4, dtype=np.float32), legal_actions=(0, 3))
    np.testing.assert_allclose(fallback, np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32))


def test_action_advantage_from_trace_uses_current_player_values():
    hero_action_values = np.array(
        [
            [1.0, 2.0],
            [3.0, 5.0],
            [7.0, 11.0],
            [13.0, 17.0],
        ],
        dtype=np.float32,
    )
    villain_action_values = -hero_action_values
    hero_values = np.array([0.5, 4.0], dtype=np.float32)
    villain_values = np.array([-0.5, -4.0], dtype=np.float32)

    hero_advantage = action_advantage_from_trace(
        hero_action_values=hero_action_values,
        villain_action_values=villain_action_values,
        hero_values=hero_values,
        villain_values=villain_values,
        player=0,
        hand_idx=1,
        legal_actions=(0, 1, 3),
    )
    villain_advantage = action_advantage_from_trace(
        hero_action_values=hero_action_values,
        villain_action_values=villain_action_values,
        hero_values=hero_values,
        villain_values=villain_values,
        player=1,
        hand_idx=1,
        legal_actions=(0, 1, 3),
    )

    np.testing.assert_allclose(hero_advantage, np.array([-2.0, 1.0, 0.0, 13.0], dtype=np.float32))
    np.testing.assert_allclose(villain_advantage, np.array([2.0, -1.0, 0.0, -13.0], dtype=np.float32))


def test_summarize_trace_records_reports_iteration_l1_curve():
    records = [
        {"iteration": 0, "l1_to_final_strategy": 1.0},
        {"iteration": 0, "l1_to_final_strategy": 0.5},
        {"iteration": 1, "l1_to_final_strategy": 0.25},
    ]

    summary = summarize_trace_records(records)

    assert summary["n_trace_records"] == 3
    assert summary["mean_l1_to_final_by_iteration"] == {"0": 0.75, "1": 0.25}


def test_belief_summary_features_report_range_shape():
    hero = np.array([0.5, 0.25, 0.25, 0.0], dtype=np.float32)
    villain = np.array([0.7, 0.2, 0.1, 0.0], dtype=np.float32)

    features = belief_summary_features(hero, villain)

    assert len(features) == 8
    np.testing.assert_allclose(features[0], 1.0397208, rtol=1e-5)
    np.testing.assert_allclose(features[1], 0.80181855, rtol=1e-5)
    assert features[2] == 0.5
    assert features[3] == 1.0
    np.testing.assert_allclose(features[6], 0.7, rtol=1e-5)
    assert features[7] == 1.0
