import numpy as np

from poker_ai.research.slumbot_response_ev_gate import (
    counterfactual_ev_gate_succeeded,
    one_hot_truth_range,
    strategy_ev,
    summarize_counterfactual_ev_records,
)


def test_one_hot_truth_range_scores_revealed_bot_hand_only():
    solver_hands = ((0, 1), (2, 3), (4, 5))

    truth = one_hot_truth_range(solver_hands, bot_hand=(3, 2))

    assert truth.tolist() == [0.0, 1.0, 0.0]
    assert float(truth.sum()) == 1.0


def test_strategy_ev_compares_mixed_policies_on_same_action_values():
    action_values = np.array([0.0, 1.0, -1.0, 3.0], dtype=np.float64)
    baseline = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    response = np.array([0.0, 0.25, 0.0, 0.75], dtype=np.float64)

    assert strategy_ev(response, action_values) > strategy_ev(baseline, action_values)


def test_counterfactual_ev_summary_blocks_negative_ev_despite_range_lift():
    records = [
        {
            "passed": True,
            "baseline_strategy_ev": 2.0,
            "response_strategy_ev": 1.0,
            "baseline_selected_action_ev": 2.0,
            "response_selected_action_ev": 1.0,
            "delta_true_hand_log_lift": 3.0,
        },
        {
            "passed": True,
            "baseline_strategy_ev": 0.0,
            "response_strategy_ev": -1.0,
            "baseline_selected_action_ev": 0.0,
            "response_selected_action_ev": -1.0,
            "delta_true_hand_log_lift": 2.0,
        },
    ]

    summary = summarize_counterfactual_ev_records(
        records,
        min_evaluated=2,
        min_mean_strategy_ev_delta=0.0,
        min_response_beats_rate=0.5,
    )

    assert summary["passed"] is True
    assert summary["ev_gate_passed"] is False
    assert summary["mean_delta_true_hand_log_lift"] == 2.5
    assert summary["mean_strategy_ev_delta"] == -1.0
    assert "positive_counterfactual_ev_not_demonstrated" in summary["promotion_blockers"]


def test_counterfactual_ev_gate_success_requires_ev_gate_not_just_mechanics():
    assert counterfactual_ev_gate_succeeded({"passed": True, "ev_gate_passed": False}) is False
    assert counterfactual_ev_gate_succeeded({"passed": True, "ev_gate_passed": True}) is True
