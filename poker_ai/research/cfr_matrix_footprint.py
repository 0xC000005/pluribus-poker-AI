"""Diagnostics for matrix/fused CFR feasibility.

The goal is not to solve CFR here. This module measures whether a street
subgame is large and regular enough to justify moving the recurrence to a
coarser sparse/dense matrix boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _level_widths(parent_idx: np.ndarray) -> tuple[list[int], np.ndarray]:
    parent_idx = np.asarray(parent_idx, dtype=np.int32)
    n_nodes = int(parent_idx.shape[0])
    depths = np.zeros(n_nodes, dtype=np.int32)
    for idx in range(n_nodes):
        parent = int(parent_idx[idx])
        if parent >= 0:
            depths[idx] = depths[parent] + 1
    max_depth = int(depths.max()) if n_nodes else 0
    widths = [int(np.count_nonzero(depths == depth)) for depth in range(max_depth + 1)]
    return widths, depths


def _level_edges(children: np.ndarray, depths: np.ndarray) -> list[int]:
    children = np.asarray(children, dtype=np.int32)
    if children.size == 0 or depths.size == 0:
        return []
    max_depth = int(depths.max())
    edges = []
    for depth in range(max_depth):
        node_indices = np.nonzero(depths == depth)[0]
        if node_indices.size == 0:
            edges.append(0)
            continue
        edges.append(int(np.count_nonzero(children[node_indices] >= 0)))
    return edges


def _sum_dense_transition_cells(level_widths: list[int]) -> int:
    return int(
        sum(
            int(level_widths[idx]) * int(level_widths[idx + 1])
            for idx in range(max(0, len(level_widths) - 1))
        )
    )


def summarize_tree_for_matrix_cfr(
    tree: dict[str, Any],
    *,
    n_hands: int,
    dtype_bytes: int = 4,
    index_bytes: int = 4,
) -> dict[str, Any]:
    """Summarize a flattened CFR tree's matrix-CFR footprint."""
    n_nodes = int(tree["n_nodes"])
    n_actions = int(tree["n_actions"])
    n_hands = int(n_hands)
    dtype_bytes = int(dtype_bytes)
    index_bytes = int(index_bytes)
    player = np.asarray(tree["player"], dtype=np.int32)
    parent_idx = np.asarray(tree["parent_idx"], dtype=np.int32)
    children = np.asarray(tree["children"], dtype=np.int32)

    level_widths, depths = _level_widths(parent_idx)
    level_edges = _level_edges(children, depths)
    n_edges = int(np.count_nonzero(children >= 0))
    dense_cells = _sum_dense_transition_cells(level_widths)
    sparse_bytes_per_edge = (2 * index_bytes) + dtype_bytes
    dense_transition_bytes = int(dense_cells * dtype_bytes)
    sparse_transition_bytes = int(n_edges * sparse_bytes_per_edge)
    cfr_state_bytes = int(n_nodes * n_actions * n_hands * dtype_bytes * 2)
    reach_value_bytes = int(n_nodes * n_hands * dtype_bytes * 4)
    terminal_payoff_bytes = int(4 * n_hands * n_hands * dtype_bytes)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_actions": n_actions,
        "n_hands": n_hands,
        "n_decision_nodes": int(np.count_nonzero(player >= 0)),
        "n_terminal_nodes": int(np.count_nonzero(player < 0)),
        "n_showdown_nodes": int(len(tree.get("showdown_idx", []))),
        "n_hero_fold_nodes": int(len(tree.get("hero_fold_idx", []))),
        "n_villain_fold_nodes": int(len(tree.get("villain_fold_idx", []))),
        "max_depth": int(depths.max()) if depths.size else 0,
        "max_level_width": max(level_widths, default=0),
        "max_level_edges": max(level_edges, default=0),
        "level_widths": level_widths,
        "level_edges": level_edges,
        "dense_transition_cells": dense_cells,
        "sparse_to_dense_transition_ratio": round(
            float(sparse_transition_bytes / max(dense_transition_bytes, 1)),
            8,
        ),
        "memory_bytes": {
            "cfr_state": cfr_state_bytes,
            "reach_value_working": reach_value_bytes,
            "terminal_payoff_matrices": terminal_payoff_bytes,
            "dense_level_transitions": dense_transition_bytes,
            "sparse_level_transitions": sparse_transition_bytes,
            "solver_state_total": int(
                cfr_state_bytes + reach_value_bytes + terminal_payoff_bytes
            ),
        },
    }
