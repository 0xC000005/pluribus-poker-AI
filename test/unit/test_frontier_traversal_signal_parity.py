import numpy as np

from poker_ai.research.frontier_traversal_signal_parity import (
    compare_signal_groups,
    summarize_signal_samples,
)


def test_summarize_signal_samples_reports_count_and_regret_mean():
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    regrets = np.array([[0.5, -0.5, 0.0], [1.0, -1.0, 0.5]], dtype=np.float32)

    summary = summarize_signal_samples(features, regrets)

    assert summary["count"] == 2
    np.testing.assert_allclose(summary["feature_mean"], [2.0, 3.0])
    np.testing.assert_allclose(summary["regret_mean"], [0.75, -0.75, 0.25])


def test_compare_signal_groups_fails_when_between_gap_exceeds_within_variance():
    baseline = [
        {"count": 100, "regret_mean": [0.0, 1.0]},
        {"count": 102, "regret_mean": [0.1, 0.9]},
    ]
    frontier = [
        {"count": 80, "regret_mean": [0.8, 0.2]},
        {"count": 82, "regret_mean": [0.9, 0.1]},
    ]

    report = compare_signal_groups(
        baseline,
        frontier,
        max_count_rel_gap=0.05,
        max_regret_l1_ratio=2.0,
    )

    assert not report["passed"]
    assert any("count_rel_gap" in failure for failure in report["gate_failures"])
    assert any("regret_l1_ratio" in failure for failure in report["gate_failures"])


def test_compare_signal_groups_passes_for_small_between_gap():
    baseline = [
        {"count": 100, "regret_mean": [0.0, 1.0]},
        {"count": 102, "regret_mean": [0.2, 0.8]},
    ]
    frontier = [
        {"count": 101, "regret_mean": [0.1, 0.9]},
        {"count": 103, "regret_mean": [0.2, 0.8]},
    ]

    report = compare_signal_groups(
        baseline,
        frontier,
        max_count_rel_gap=0.05,
        max_regret_l1_ratio=2.0,
    )

    assert report["passed"]
    assert report["gate_failures"] == []
