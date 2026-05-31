import numpy as np
import pytest

from scripts.eval_same_topology_batched_cfr import (
    actual_hand_root_policy,
    summarize_actual_hand_root_parity,
    summarize_teacher_relative_root_decisions,
)


def test_actual_hand_root_policy_normalizes_only_legal_actions():
    strategy_sum = np.zeros((2, 4, 3), dtype=np.float32)
    strategy_sum[0, 1, 2] = 2.0
    strategy_sum[0, 3, 2] = 6.0

    policy = actual_hand_root_policy(
        strategy_sum,
        hand_index=2,
        legal_actions=[1, 3],
        n_actions=4,
    )

    np.testing.assert_allclose(policy, [0.0, 0.25, 0.0, 0.75])


def test_actual_hand_root_parity_reports_l1_and_top_match():
    serial = np.zeros((1, 3, 2), dtype=np.float32)
    batched = np.zeros_like(serial)
    serial[0, :, 1] = [0.0, 4.0, 6.0]
    batched[0, :, 1] = [0.0, 5.0, 5.0]

    summary = summarize_actual_hand_root_parity(
        labels=["root-a"],
        serial_strategy_sums=[serial],
        batched_strategy_sums=np.asarray([batched]),
        hand_indices=[1],
        legal_actions_by_root=[[1, 2]],
        n_actions=3,
    )

    assert summary["top_matches"] == 0
    assert summary["max_actual_root_l1"] == pytest.approx(0.2)
    assert summary["roots"][0]["serial_top"] == 2
    assert summary["roots"][0]["batched_top"] == 1


def test_teacher_relative_root_decisions_compare_low_budget_modes():
    teacher = [np.asarray([0.1, 0.7, 0.2]), np.asarray([0.2, 0.3, 0.5])]
    serial = [np.asarray([0.2, 0.6, 0.2]), np.asarray([0.2, 0.6, 0.2])]
    batched = [np.asarray([0.2, 0.5, 0.3]), np.asarray([0.2, 0.2, 0.6])]

    summary = summarize_teacher_relative_root_decisions(
        labels=["a", "b"],
        teacher_policies=teacher,
        serial_policies=serial,
        batched_policies=batched,
    )

    assert summary["serial_teacher_top_matches"] == 1
    assert summary["batched_teacher_top_matches"] == 2
    assert summary["teacher_top_match_delta"] == 1
    assert summary["max_batched_excess_l1_to_teacher"] == pytest.approx(0.2)
    assert summary["roots"][1]["serial_matches_teacher"] is False
    assert summary["roots"][1]["batched_matches_teacher"] is True
