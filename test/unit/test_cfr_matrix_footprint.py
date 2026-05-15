import numpy as np

from poker_ai.research.cfr_matrix_footprint import summarize_tree_for_matrix_cfr
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase
from scripts.analyze_cfr_matrix_footprint import select_case_window


def test_matrix_cfr_footprint_counts_levels_edges_and_memory():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [
                [-1, 1, -1],
                [2, -1, 3],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            dtype=np.int32,
        ),
        "decision_idx": np.asarray([0, 1], dtype=np.int32),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }

    summary = summarize_tree_for_matrix_cfr(tree, n_hands=5)

    assert summary["n_nodes"] == 4
    assert summary["n_edges"] == 3
    assert summary["n_decision_nodes"] == 2
    assert summary["n_terminal_nodes"] == 2
    assert summary["max_depth"] == 2
    assert summary["level_widths"] == [1, 1, 2]
    assert summary["level_edges"] == [1, 2]
    assert summary["memory_bytes"]["cfr_state"] == 4 * 3 * 5 * 4 * 2
    assert summary["memory_bytes"]["reach_value_working"] == 4 * 5 * 4 * 4
    assert summary["memory_bytes"]["terminal_payoff_matrices"] == 4 * 5 * 5 * 4
    assert summary["memory_bytes"]["dense_level_transitions"] == (1 * 1 + 1 * 2) * 4
    assert summary["memory_bytes"]["sparse_level_transitions"] == 3 * (4 + 4 + 4)
    assert summary["sparse_to_dense_transition_ratio"] == 3.0


def test_matrix_footprint_case_window_supports_holdout_offsets():
    cases = [
        ResolverBenchmarkCase(
            label=f"case-{idx}",
            hole_cards=("Ac", "Kd"),
            board=("2c", "7d", "Jh", "4s"),
            action_str="ck/kk/",
            client_pos=0,
        )
        for idx in range(5)
    ]

    selected = select_case_window(cases, start_index=2, max_cases=2)

    assert [case.label for case in selected] == ["case-2", "case-3"]
    assert [case.label for case in select_case_window(cases, start_index=4, max_cases=None)] == [
        "case-4"
    ]
