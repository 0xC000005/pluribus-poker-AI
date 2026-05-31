import pytest
import numpy as np

from poker_ai.research.cfr_matrix_footprint import (
    plan_memory_capped_chunks,
    plan_ragged_terminal_chunks,
    plan_shape_compatible_chunks,
    summarize_ragged_batching_potential,
    summarize_tree_for_matrix_cfr,
)
from poker_ai.research.cfr_frontier_latency_shape import analyze_frontier_latency_shape
from poker_ai.research.cfr_solver_warmup import summarize_warmup_records
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


def test_matrix_chunk_planner_respects_memory_cap_and_preserves_order():
    records = [
        {"label": "a", "memory_bytes": {"solver_state_total": 60}},
        {"label": "b", "memory_bytes": {"solver_state_total": 70}},
        {"label": "c", "memory_bytes": {"solver_state_total": 30}},
        {"label": "d", "memory_bytes": {"solver_state_total": 40}},
    ]

    plan = plan_memory_capped_chunks(records, memory_cap_bytes=100)

    assert plan["passed"] is True
    assert plan["n_chunks"] == 3
    assert [chunk["labels"] for chunk in plan["chunks"]] == [["a"], ["b", "c"], ["d"]]
    assert plan["max_chunk_bytes"] == 100
    assert plan["total_bytes"] == 200


def test_matrix_chunk_planner_rejects_single_case_over_cap():
    records = [{"label": "too-large", "memory_bytes": {"solver_state_total": 101}}]

    with pytest.raises(ValueError, match="exceeds memory cap"):
        plan_memory_capped_chunks(records, memory_cap_bytes=100)


def test_shape_compatible_chunk_planner_groups_by_tree_shape_then_memory():
    records = [
        {
            "label": "a",
            "street": 2,
            "n_actions": 9,
            "n_hands": 100,
            "level_widths": [1, 2, 3],
            "level_edges": [2, 3],
            "memory_bytes": {"solver_state_total": 60},
        },
        {
            "label": "b",
            "street": 2,
            "n_actions": 9,
            "n_hands": 100,
            "level_widths": [1, 2, 3],
            "level_edges": [2, 3],
            "memory_bytes": {"solver_state_total": 30},
        },
        {
            "label": "c",
            "street": 3,
            "n_actions": 9,
            "n_hands": 90,
            "level_widths": [1, 1],
            "level_edges": [1],
            "memory_bytes": {"solver_state_total": 25},
        },
        {
            "label": "d",
            "street": 2,
            "n_actions": 9,
            "n_hands": 100,
            "level_widths": [1, 2, 3],
            "level_edges": [2, 3],
            "memory_bytes": {"solver_state_total": 50},
        },
    ]

    plan = plan_shape_compatible_chunks(records, memory_cap_bytes=100)

    assert plan["passed"] is True
    assert plan["n_shape_groups"] == 2
    assert plan["largest_shape_group_cases"] == 3
    assert plan["same_shape_case_fraction"] == 0.75
    assert [chunk["labels"] for chunk in plan["shape_groups"][0]["chunks"]] == [["a", "b"], ["d"]]
    assert plan["shape_groups"][0]["n_chunks"] == 2


def test_shape_compatible_chunk_planner_respects_topology_hash_when_present():
    common = {
        "street": 2,
        "n_actions": 9,
        "n_hands": 100,
        "level_widths": [1, 2, 3],
        "level_edges": [2, 3],
    }
    records = [
        {
            **common,
            "label": "a",
            "topology_hash": "hash-a",
            "memory_bytes": {"solver_state_total": 10},
        },
        {
            **common,
            "label": "b",
            "topology_hash": "hash-b",
            "memory_bytes": {"solver_state_total": 10},
        },
    ]

    plan = plan_shape_compatible_chunks(records, memory_cap_bytes=100)

    assert plan["n_shape_groups"] == 2
    assert plan["largest_shape_group_cases"] == 1


def test_ragged_batching_potential_summarizes_level_and_terminal_work():
    records = [
        {
            "label": "a",
            "passed": True,
            "level_widths": [1, 2, 3],
            "level_edges": [2, 4],
            "n_showdown_nodes": 2,
            "n_hero_fold_nodes": 1,
            "n_villain_fold_nodes": 0,
        },
        {
            "label": "b",
            "passed": True,
            "level_widths": [1, 1],
            "level_edges": [1],
            "n_showdown_nodes": 1,
            "n_hero_fold_nodes": 0,
            "n_villain_fold_nodes": 1,
        },
        {"label": "skipped", "passed": False},
    ]

    metrics = summarize_ragged_batching_potential(records)

    assert metrics["passed"] is True
    assert metrics["n_cases"] == 2
    assert metrics["max_depth"] == 2
    assert metrics["level_batches"][0]["n_cases"] == 2
    assert metrics["level_batches"][0]["total_nodes"] == 2
    assert metrics["level_batches"][1]["total_edges"] == 4
    assert metrics["level_batches"][2]["n_cases"] == 1
    assert metrics["terminal_batches"]["showdown"]["total_nodes"] == 3
    assert metrics["terminal_batches"]["hero_fold"]["active_case_fraction"] == 0.5
    assert metrics["total_terminal_nodes"] == 5


def test_ragged_terminal_chunk_planner_bounds_padding_fraction():
    records = [
        {"label": "a", "n_showdown_nodes": 10},
        {"label": "b", "n_showdown_nodes": 8},
        {"label": "c", "n_showdown_nodes": 2},
        {"label": "d", "n_showdown_nodes": 2},
    ]

    plan = plan_ragged_terminal_chunks(
        records,
        terminal_key="n_showdown_nodes",
        max_padding_fraction=0.25,
        max_cases_per_chunk=4,
    )

    assert plan["passed"] is True
    assert [chunk["labels"] for chunk in plan["chunks"]] == [["a", "b"], ["c", "d"]]
    assert plan["chunks"][0]["padding_fraction"] == 0.1
    assert plan["chunks"][1]["padding_fraction"] == 0.0
    assert plan["max_padding_fraction_observed"] == 0.1


def test_ragged_terminal_chunk_planner_can_sort_by_terminal_count():
    records = [
        {"label": "a", "n_showdown_nodes": 10},
        {"label": "b", "n_showdown_nodes": 2},
        {"label": "c", "n_showdown_nodes": 8},
        {"label": "d", "n_showdown_nodes": 2},
    ]

    plan = plan_ragged_terminal_chunks(
        records,
        terminal_key="n_showdown_nodes",
        max_padding_fraction=0.25,
        max_cases_per_chunk=4,
        sort_by_count=True,
    )

    assert [chunk["labels"] for chunk in plan["chunks"]] == [["a", "c"], ["b", "d"]]
    assert plan["chunks"][0]["indices"] == [0, 2]


def test_frontier_latency_shape_joins_budget_records_to_footprint():
    frontier = {
        "best_l1_budget": 20,
        "records": [
            {
                "label": "root-a",
                "passed": True,
                "budgets": {
                    "20": {
                        "latency_ms": 200.0,
                        "l1_to_reference": 0.2,
                        "kl_to_reference": 0.03,
                    }
                },
            },
            {
                "label": "root-b",
                "passed": True,
                "budgets": {
                    "20": {
                        "latency_ms": 600.0,
                        "l1_to_reference": 0.1,
                        "kl_to_reference": 0.01,
                    }
                },
            },
        ],
    }
    footprint = {
        "records": [
            {
                "label": "root-a",
                "passed": True,
                "n_nodes": 100,
                "n_edges": 120,
                "max_depth": 2,
                "memory_mib": {"solver_state_total": 10.0},
            },
            {
                "label": "root-b",
                "passed": True,
                "n_nodes": 300,
                "n_edges": 360,
                "max_depth": 4,
                "memory_mib": {"solver_state_total": 30.0},
            },
        ]
    }

    metrics = analyze_frontier_latency_shape(frontier, footprint, top_k=1)

    assert metrics["passed"] is True
    assert metrics["budget"] == "20"
    assert metrics["n_joined"] == 2
    assert metrics["mean_latency_ms"] == 400.0
    assert metrics["mean_latency_ms_per_1k_nodes"] == 2000.0
    assert metrics["correlations"]["latency_vs_n_nodes"] == 1.0
    assert metrics["top_roots_by_latency"][0]["label"] == "root-b"


def test_solver_warmup_summary_reports_cold_to_warm_speedup():
    metrics = summarize_warmup_records(
        [
            {
                "label": "root-a",
                "passed": True,
                "construction_ms": 20.0,
                "cold_solve_ms": 100.0,
                "warm_solve_mean_ms": 50.0,
            },
            {
                "label": "root-b",
                "passed": True,
                "construction_ms": 40.0,
                "cold_solve_ms": 120.0,
                "warm_solve_mean_ms": 80.0,
            },
        ]
    )

    assert metrics["passed"] is True
    assert metrics["mean_construction_ms"] == 30.0
    assert metrics["mean_cold_solve_ms"] == 110.0
    assert metrics["mean_warm_solve_ms"] == 65.0
    assert metrics["mean_cold_to_warm_speedup"] == 1.75
    assert metrics["max_cold_to_warm_speedup"] == 2.0
