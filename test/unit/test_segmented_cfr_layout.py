import numpy as np

from poker_ai.research.segmented_cfr_layout import (
    build_segmented_cfr_layout,
    edge_tensors_from_node_action,
    materialize_segmented_cfr_tensors,
    scatter_edge_tensors_to_node_action,
    segmented_cfr_iterations,
    segmented_cfr_single_iteration,
    segmented_edge_cfr_update,
    segmented_edge_value_backward,
    segmented_edge_regret_matched_reach_forward,
    segmented_regret_matched_reach_forward,
    segmented_terminal_values,
    segmented_uniform_reach_forward,
    summarize_segmented_cfr_layout,
)


def test_segmented_cfr_layout_packs_heterogeneous_tree_offsets_and_levels():
    first = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    second = {
        "n_nodes": 3,
        "n_actions": 3,
        "player": np.asarray([1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 0], dtype=np.int32),
        "children": np.asarray(
            [[1, -1, 2], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([1], dtype=np.int32),
        "villain_fold_idx": np.asarray([], dtype=np.int32),
    }

    layout = build_segmented_cfr_layout([first, second])

    assert layout["passed"] is True
    assert layout["n_roots"] == 2
    assert layout["node_offsets"] == [0, 4]
    assert layout["n_nodes_total"] == 7
    assert layout["level_segments"][0]["depth"] == 0
    assert layout["level_segments"][0]["parent_global"].tolist() == [0, 4, 4]
    assert layout["level_segments"][0]["child_global"].tolist() == [1, 5, 6]
    assert layout["level_segments"][0]["action"].tolist() == [1, 0, 2]
    assert layout["level_segments"][0]["parent_pos"].tolist() == [0, 1, 1]
    assert layout["level_segments"][0]["strategy_prob"].tolist() == [1.0, 0.5, 0.5]
    assert layout["level_segments"][1]["parent_global"].tolist() == [1, 1]
    assert layout["terminal_segments"]["showdown"]["global_indices"].tolist() == [2, 6]
    assert layout["terminal_segments"]["hero_fold"]["global_indices"].tolist() == [5]
    assert layout["terminal_segments"]["villain_fold"]["global_indices"].tolist() == [3]

    summary = summarize_segmented_cfr_layout(layout)
    assert summary["n_roots"] == 2
    assert summary["n_nodes_total"] == 7
    assert summary["max_level_edges"] == 3
    assert summary["level_segments"][0]["n_edges"] == 3
    assert summary["terminal_segments"]["showdown"]["n_nodes"] == 2


def test_segmented_uniform_reach_forward_matches_tree_semantics():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([tree])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    hero = np.asarray([[1.0, 2.0]], dtype=np.float32)
    villain = np.asarray([[3.0, 4.0]], dtype=np.float32)

    hero_reach, villain_reach = segmented_uniform_reach_forward(
        tensors,
        n_hands=2,
        hero_ranges=hero,
        villain_ranges=villain,
    )

    np.testing.assert_allclose(hero_reach.numpy()[0], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[0], [3.0, 4.0])
    np.testing.assert_allclose(hero_reach.numpy()[1], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[1], [3.0, 4.0])
    np.testing.assert_allclose(hero_reach.numpy()[2], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[2], [1.5, 2.0])
    np.testing.assert_allclose(hero_reach.numpy()[3], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[3], [1.5, 2.0])


def test_segmented_regret_matched_reach_forward_uses_positive_regrets():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([tree])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    regrets = np.zeros((layout["n_nodes_total"], 3, 2), dtype=np.float32)
    regrets[1, 0, :] = [1.0, 2.0]
    regrets[1, 2, :] = [3.0, 2.0]

    hero_reach, villain_reach = segmented_regret_matched_reach_forward(
        tensors,
        regrets,
        n_hands=2,
        hero_ranges=np.asarray([[1.0, 2.0]], dtype=np.float32),
        villain_ranges=np.asarray([[4.0, 8.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(hero_reach.numpy()[1], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[1], [4.0, 8.0])
    np.testing.assert_allclose(hero_reach.numpy()[2], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[2], [1.0, 4.0])
    np.testing.assert_allclose(hero_reach.numpy()[3], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[3], [3.0, 4.0])


def test_segmented_edge_regret_forward_uses_edge_aligned_regrets():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([tree])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    edge_regrets = [
        np.zeros((1, 2), dtype=np.float32),
        np.asarray([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32),
    ]

    hero_reach, villain_reach = segmented_edge_regret_matched_reach_forward(
        tensors,
        edge_regrets,
        n_hands=2,
        hero_ranges=np.asarray([[1.0, 2.0]], dtype=np.float32),
        villain_ranges=np.asarray([[4.0, 8.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(hero_reach.numpy()[2], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[2], [1.0, 4.0])
    np.testing.assert_allclose(hero_reach.numpy()[3], [1.0, 2.0])
    np.testing.assert_allclose(villain_reach.numpy()[3], [3.0, 4.0])


def test_segmented_regret_matching_normalizes_subunit_positive_regrets():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([tree])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    dense_regrets = np.zeros((layout["n_nodes_total"], 3, 2), dtype=np.float32)
    dense_regrets[1, 0, :] = [0.25, 0.2]
    dense_regrets[1, 2, :] = [0.75, 0.0]
    edge_regrets = [
        np.zeros((1, 2), dtype=np.float32),
        np.asarray([[0.25, 0.2], [0.75, 0.0]], dtype=np.float32),
    ]
    ranges_h = np.asarray([[1.0, 2.0]], dtype=np.float32)
    ranges_v = np.asarray([[4.0, 8.0]], dtype=np.float32)

    dense_hero, dense_villain = segmented_regret_matched_reach_forward(
        tensors,
        dense_regrets,
        n_hands=2,
        hero_ranges=ranges_h,
        villain_ranges=ranges_v,
    )
    edge_hero, edge_villain = segmented_edge_regret_matched_reach_forward(
        tensors,
        edge_regrets,
        n_hands=2,
        hero_ranges=ranges_h,
        villain_ranges=ranges_v,
    )

    np.testing.assert_allclose(dense_villain.numpy()[2], [1.0, 8.0])
    np.testing.assert_allclose(dense_villain.numpy()[3], [3.0, 0.0])
    np.testing.assert_allclose(edge_hero.numpy(), dense_hero.numpy())
    np.testing.assert_allclose(edge_villain.numpy(), dense_villain.numpy())


def test_segmented_edge_value_backward_reduces_values_in_reverse_levels():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([tree])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    edge_strategy = [
        np.ones((1, 2), dtype=np.float32),
        np.asarray([[0.25, 0.5], [0.75, 0.5]], dtype=np.float32),
    ]
    hero_leaf_values = np.zeros((layout["n_nodes_total"], 2), dtype=np.float32)
    villain_leaf_values = np.zeros_like(hero_leaf_values)
    hero_leaf_values[2] = [2.0, 4.0]
    hero_leaf_values[3] = [10.0, 20.0]
    villain_leaf_values[2] = [3.0, 5.0]
    villain_leaf_values[3] = [7.0, 11.0]

    hero_values, villain_values = segmented_edge_value_backward(
        tensors,
        edge_strategy,
        hero_leaf_values,
        villain_leaf_values,
        n_hands=2,
    )

    np.testing.assert_allclose(hero_values.numpy()[1], [8.0, 12.0])
    np.testing.assert_allclose(villain_values.numpy()[1], [6.0, 8.0])
    np.testing.assert_allclose(hero_values.numpy()[0], [8.0, 12.0])
    np.testing.assert_allclose(villain_values.numpy()[0], [6.0, 8.0])


def test_segmented_terminal_values_match_solver_showdown_and_fold_formulas():
    first = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "stacks_h": np.asarray([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        "stacks_v": np.asarray([10.0, 10.0, 8.0, 7.0], dtype=np.float32),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    second = {
        "n_nodes": 3,
        "n_actions": 3,
        "player": np.asarray([1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 0], dtype=np.int32),
        "children": np.asarray(
            [[1, -1, 2], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "stacks_h": np.asarray([10.0, 6.0, 9.0], dtype=np.float32),
        "stacks_v": np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([1], dtype=np.int32),
        "villain_fold_idx": np.asarray([], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([first, second])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    hero_reach = np.zeros((layout["n_nodes_total"], 2), dtype=np.float32)
    villain_reach = np.zeros_like(hero_reach)
    hero_reach[2] = [2.0, 4.0]
    villain_reach[2] = [3.0, 5.0]
    hero_reach[3] = [13.0, 17.0]
    villain_reach[3] = [7.0, 11.0]
    hero_reach[5] = [29.0, 31.0]
    villain_reach[5] = [19.0, 23.0]
    hero_reach[6] = [5.0, 7.0]
    villain_reach[6] = [2.0, 3.0]

    win = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    lose = np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    tie = np.zeros_like(win)
    valid = np.asarray(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    hero_values, villain_values = segmented_terminal_values(
        tensors,
        hero_reach,
        villain_reach,
        win,
        lose,
        tie,
        valid,
        pot_start=np.asarray([10.0, 20.0], dtype=np.float32),
        hero_stack_start=np.asarray([10.0, 10.0], dtype=np.float32),
        villain_stack_start=np.asarray([10.0, 10.0], dtype=np.float32),
        n_hands=2,
    )

    np.testing.assert_allclose(hero_values.numpy()[2], [36.0, 60.0])
    np.testing.assert_allclose(villain_values.numpy()[2], [36.0, 12.0])
    np.testing.assert_allclose(hero_values.numpy()[3], [234.0, 234.0])
    np.testing.assert_allclose(villain_values.numpy()[3], [-90.0, -90.0])
    np.testing.assert_allclose(hero_values.numpy()[5], [-76.0, -92.0])
    np.testing.assert_allclose(villain_values.numpy()[5], [696.0, 744.0])
    np.testing.assert_allclose(hero_values.numpy()[6], [-2.0, -3.0])
    np.testing.assert_allclose(villain_values.numpy()[6], [105.0, 147.0])


def test_segmented_edge_cfr_update_applies_regret_and_strategy_mass():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([tree])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    edge_strategy = [
        np.ones((1, 2), dtype=np.float32),
        np.asarray([[0.25, 0.5], [0.75, 0.5]], dtype=np.float32),
    ]
    edge_regrets = [
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    ]
    edge_strategy_sum = [
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ]
    hero_reach = np.zeros((layout["n_nodes_total"], 2), dtype=np.float32)
    villain_reach = np.zeros_like(hero_reach)
    hero_reach[0] = [1.0, 2.0]
    hero_reach[1] = [1.0, 2.0]
    villain_reach[0] = [4.0, 8.0]
    villain_reach[1] = [4.0, 8.0]
    hero_values = np.zeros_like(hero_reach)
    villain_values = np.zeros_like(villain_reach)
    hero_values[0] = [8.0, 12.0]
    hero_values[1] = [8.0, 12.0]
    hero_values[2] = [2.0, 4.0]
    hero_values[3] = [10.0, 20.0]
    villain_values[0] = [6.0, 8.0]
    villain_values[1] = [6.0, 8.0]
    villain_values[2] = [3.0, 5.0]
    villain_values[3] = [7.0, 11.0]

    updated_regrets, updated_strategy_sum = segmented_edge_cfr_update(
        tensors,
        edge_strategy,
        hero_reach,
        villain_reach,
        hero_values,
        villain_values,
        edge_regrets,
        edge_strategy_sum,
        n_hands=2,
    )

    np.testing.assert_allclose(updated_regrets[0].numpy(), [[2.0, 0.0]])
    np.testing.assert_allclose(updated_regrets[1].numpy(), [[0.0, 0.0], [1.0, 4.0]])
    np.testing.assert_allclose(updated_strategy_sum[0].numpy(), [[1.0, 2.0]])
    np.testing.assert_allclose(updated_strategy_sum[1].numpy(), [[1.0, 4.0], [3.0, 4.0]])


def test_segmented_cfr_single_iteration_matches_toy_reference_semantics():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "stacks_h": np.asarray([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        "stacks_v": np.asarray([10.0, 10.0, 8.0, 7.0], dtype=np.float32),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([tree])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    edge_regrets = [np.zeros((1, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]
    edge_strategy_sum = [np.zeros_like(edge_regrets[0]), np.zeros_like(edge_regrets[1])]
    win = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    lose = np.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    tie = np.zeros_like(win)
    valid = np.ones_like(win)

    result = segmented_cfr_single_iteration(
        tensors,
        edge_regrets,
        edge_strategy_sum,
        win,
        lose,
        tie,
        valid,
        pot_start=10.0,
        hero_stack_start=10.0,
        villain_stack_start=10.0,
        n_hands=2,
        hero_ranges=np.asarray([[1.0, 2.0]], dtype=np.float32),
        villain_ranges=np.asarray([[4.0, 8.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(result["hero_values"].numpy()[2], [24.0, 48.0])
    np.testing.assert_allclose(result["villain_values"].numpy()[2], [18.0, 6.0])
    np.testing.assert_allclose(result["hero_values"].numpy()[3], [78.0, 78.0])
    np.testing.assert_allclose(result["villain_values"].numpy()[3], [-9.0, -9.0])
    np.testing.assert_allclose(result["edge_regret_sums"][0].numpy(), [[0.0, 0.0]])
    np.testing.assert_allclose(result["edge_regret_sums"][1].numpy(), [[13.5, 7.5], [0.0, 0.0]])
    np.testing.assert_allclose(result["edge_strategy_sums"][0].numpy(), [[1.0, 2.0]])
    np.testing.assert_allclose(result["edge_strategy_sums"][1].numpy(), [[2.0, 4.0], [2.0, 4.0]])

    dense_regrets = scatter_edge_tensors_to_node_action(
        tensors,
        result["edge_regret_sums"],
        n_actions=3,
        n_hands=2,
    )
    np.testing.assert_allclose(dense_regrets.numpy()[1, 0], [13.5, 7.5])
    np.testing.assert_allclose(dense_regrets.numpy()[1, 2], [0.0, 0.0])
    edge_roundtrip = edge_tensors_from_node_action(tensors, dense_regrets, n_hands=2)
    np.testing.assert_allclose(edge_roundtrip[1].numpy(), result["edge_regret_sums"][1].numpy())


def test_segmented_cfr_iterations_matches_repeated_single_iteration():
    tree = {
        "n_nodes": 4,
        "n_actions": 3,
        "player": np.asarray([0, 1, -1, -1], dtype=np.int32),
        "parent_idx": np.asarray([-1, 0, 1, 1], dtype=np.int32),
        "children": np.asarray(
            [[-1, 1, -1], [2, -1, 3], [-1, -1, -1], [-1, -1, -1]],
            dtype=np.int32,
        ),
        "stacks_h": np.asarray([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        "stacks_v": np.asarray([10.0, 10.0, 8.0, 7.0], dtype=np.float32),
        "showdown_idx": np.asarray([2], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([3], dtype=np.int32),
    }
    layout = build_segmented_cfr_layout([tree])
    tensors = materialize_segmented_cfr_tensors(layout, device="cpu")
    edge_regrets = [np.zeros((1, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]
    edge_strategy_sum = [np.zeros_like(edge_regrets[0]), np.zeros_like(edge_regrets[1])]
    win = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    lose = np.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    tie = np.zeros_like(win)
    valid = np.ones_like(win)
    kwargs = {
        "pot_start": 10.0,
        "hero_stack_start": 10.0,
        "villain_stack_start": 10.0,
        "n_hands": 2,
        "hero_ranges": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "villain_ranges": np.asarray([[4.0, 8.0]], dtype=np.float32),
    }

    first = segmented_cfr_single_iteration(
        tensors,
        edge_regrets,
        edge_strategy_sum,
        win,
        lose,
        tie,
        valid,
        **kwargs,
    )
    second = segmented_cfr_single_iteration(
        tensors,
        first["edge_regret_sums"],
        first["edge_strategy_sums"],
        win,
        lose,
        tie,
        valid,
        **kwargs,
    )
    looped = segmented_cfr_iterations(
        tensors,
        edge_regrets,
        edge_strategy_sum,
        win,
        lose,
        tie,
        valid,
        n_iterations=2,
        **kwargs,
    )

    np.testing.assert_allclose(looped["edge_regret_sums"][0].numpy(), second["edge_regret_sums"][0].numpy())
    np.testing.assert_allclose(looped["edge_regret_sums"][1].numpy(), second["edge_regret_sums"][1].numpy())
    np.testing.assert_allclose(looped["edge_strategy_sums"][0].numpy(), second["edge_strategy_sums"][0].numpy())
    np.testing.assert_allclose(looped["edge_strategy_sums"][1].numpy(), second["edge_strategy_sums"][1].numpy())
    assert looped["n_iterations"] == 2


def test_segmented_cfr_layout_rejects_action_count_mismatch():
    base = {
        "n_nodes": 1,
        "n_actions": 2,
        "player": np.asarray([-1], dtype=np.int32),
        "parent_idx": np.asarray([-1], dtype=np.int32),
        "children": np.asarray([[-1, -1]], dtype=np.int32),
        "showdown_idx": np.asarray([], dtype=np.int32),
        "hero_fold_idx": np.asarray([], dtype=np.int32),
        "villain_fold_idx": np.asarray([], dtype=np.int32),
    }
    other = {**base, "n_actions": 3, "children": np.asarray([[-1, -1, -1]], dtype=np.int32)}

    try:
        build_segmented_cfr_layout([base, other])
    except ValueError as exc:
        assert "action count" in str(exc)
    else:
        raise AssertionError("expected action count mismatch to fail")
