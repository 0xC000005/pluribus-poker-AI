import numpy as np

from scripts.diagnose_cfr_trace_state import (
    action_policy_from_field,
    summarize_trace_records,
)


def test_action_policy_from_field_uses_legal_actions_and_uniform_fallback():
    values = np.array([0.0, 3.0, 99.0, 0.0], dtype=np.float32)

    policy = action_policy_from_field(values, legal_actions=(0, 1, 3))

    np.testing.assert_allclose(policy, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    fallback = action_policy_from_field(np.zeros(4, dtype=np.float32), legal_actions=(0, 3))
    np.testing.assert_allclose(fallback, np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32))


def test_summarize_trace_records_reports_iteration_l1_curve():
    records = [
        {"iteration": 0, "l1_to_final_strategy": 1.0},
        {"iteration": 0, "l1_to_final_strategy": 0.5},
        {"iteration": 1, "l1_to_final_strategy": 0.25},
    ]

    summary = summarize_trace_records(records)

    assert summary["n_trace_records"] == 3
    assert summary["mean_l1_to_final_by_iteration"] == {"0": 0.75, "1": 0.25}
