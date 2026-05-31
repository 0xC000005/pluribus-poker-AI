"""Segmented layout builders for heterogeneous CFR executor experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _tree_depths(parent_idx: np.ndarray) -> np.ndarray:
    parents = np.asarray(parent_idx, dtype=np.int32)
    depths = np.zeros(int(parents.shape[0]), dtype=np.int32)
    for index in range(int(parents.shape[0])):
        parent = int(parents[index])
        while parent >= 0:
            depths[index] += 1
            parent = int(parents[parent])
    return depths


def _terminal_global_indices(
    trees: list[dict[str, Any]],
    offsets: list[int],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root_indices: list[int] = []
    global_indices: list[int] = []
    local_indices: list[int] = []
    stack_h: list[float] = []
    stack_v: list[float] = []
    for root_index, (tree, offset) in enumerate(zip(trees, offsets, strict=True)):
        stacks_h = np.asarray(tree.get("stacks_h", []), dtype=np.float32)
        stacks_v = np.asarray(tree.get("stacks_v", []), dtype=np.float32)
        for local_index in np.asarray(tree.get(key, []), dtype=np.int64):
            local = int(local_index)
            root_indices.append(int(root_index))
            global_indices.append(int(offset + local))
            local_indices.append(local)
            if stacks_h.size and stacks_v.size:
                stack_h.append(float(stacks_h[local]))
                stack_v.append(float(stacks_v[local]))
            else:
                stack_h.append(0.0)
                stack_v.append(0.0)
    return (
        np.asarray(root_indices, dtype=np.int32),
        np.asarray(global_indices, dtype=np.int64),
        np.asarray(local_indices, dtype=np.int64),
        np.asarray(stack_h, dtype=np.float32),
        np.asarray(stack_v, dtype=np.float32),
    )


def build_segmented_cfr_layout(trees: list[dict[str, Any]]) -> dict[str, Any]:
    """Pack heterogeneous CFR trees into depth-wise edge segments.

    The returned arrays are intentionally numpy-only metadata. They are the
    contract for later torch/Numba/CUDA kernels that keep CFR state resident on
    device while iterating through ragged levels.
    """
    if not trees:
        raise ValueError("trees must be non-empty")
    normalized = list(trees)
    n_actions = int(normalized[0]["n_actions"])
    node_offsets: list[int] = []
    offset = 0
    max_depth = 0
    per_tree_depths: list[np.ndarray] = []
    for root_index, tree in enumerate(normalized):
        if int(tree["n_actions"]) != n_actions:
            raise ValueError(f"tree {root_index} action count differs")
        n_nodes = int(tree["n_nodes"])
        children = np.asarray(tree["children"], dtype=np.int64)
        expected_children_shape = (n_nodes, n_actions)
        if children.shape != expected_children_shape:
            raise ValueError(
                f"tree {root_index} children shape {children.shape} != {expected_children_shape}"
            )
        depths = _tree_depths(np.asarray(tree["parent_idx"], dtype=np.int32))
        per_tree_depths.append(depths)
        max_depth = max(max_depth, int(depths.max(initial=0)))
        node_offsets.append(int(offset))
        offset += n_nodes

    level_segments: list[dict[str, Any]] = []
    for depth in range(max_depth):
        root_indices: list[int] = []
        parent_global: list[int] = []
        child_global: list[int] = []
        actions: list[int] = []
        parent_pos: list[int] = []
        players: list[int] = []
        strategy_probs: list[float] = []
        segment_parent_count = 0
        for root_index, (tree, node_offset, depths) in enumerate(
            zip(normalized, node_offsets, per_tree_depths, strict=True)
        ):
            children = np.asarray(tree["children"], dtype=np.int64)
            player = np.asarray(tree["player"], dtype=np.int32)
            parents = [
                int(node_index)
                for node_index in np.where(depths == int(depth))[0]
                if np.any(children[int(node_index)] >= 0)
            ]
            for parent in parents:
                legal_actions = np.where(children[parent] >= 0)[0]
                strategy_prob = 1.0 / float(max(int(legal_actions.size), 1))
                current_parent_pos = int(segment_parent_count)
                segment_parent_count += 1
                for action in legal_actions:
                    root_indices.append(int(root_index))
                    parent_global.append(int(node_offset + parent))
                    child_global.append(int(node_offset + int(children[parent, int(action)])))
                    actions.append(int(action))
                    parent_pos.append(current_parent_pos)
                    players.append(int(player[parent]))
                    strategy_probs.append(float(strategy_prob))
        if parent_global:
            level_segments.append(
                {
                    "depth": int(depth),
                    "root_index": np.asarray(root_indices, dtype=np.int32),
                    "parent_global": np.asarray(parent_global, dtype=np.int64),
                    "child_global": np.asarray(child_global, dtype=np.int64),
                    "action": np.asarray(actions, dtype=np.int32),
                    "parent_pos": np.asarray(parent_pos, dtype=np.int32),
                    "player": np.asarray(players, dtype=np.int32),
                    "strategy_prob": np.asarray(strategy_probs, dtype=np.float32),
                    "n_edges": int(len(parent_global)),
                    "n_distinct_parents": int(segment_parent_count),
                }
            )

    terminal_segments = {}
    for name, key in (
        ("showdown", "showdown_idx"),
        ("hero_fold", "hero_fold_idx"),
        ("villain_fold", "villain_fold_idx"),
    ):
        root_indices, global_indices, local_indices, stack_h, stack_v = _terminal_global_indices(
            normalized,
            node_offsets,
            key,
        )
        terminal_segments[name] = {
            "root_index": root_indices,
            "global_indices": global_indices,
            "local_indices": local_indices,
            "stacks_h": stack_h,
            "stacks_v": stack_v,
            "n_nodes": int(global_indices.size),
        }

    return {
        "mode": "segmented_cfr_layout",
        "passed": True,
        "n_roots": int(len(normalized)),
        "n_actions": int(n_actions),
        "node_offsets": node_offsets,
        "n_nodes_total": int(offset),
        "max_depth": int(max_depth),
        "level_segments": level_segments,
        "terminal_segments": terminal_segments,
    }


def summarize_segmented_cfr_layout(layout: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-serializable summary for a segmented layout."""
    level_segments = [
        {
            "depth": int(segment["depth"]),
            "n_edges": int(segment["n_edges"]),
            "n_distinct_parents": int(segment["n_distinct_parents"]),
            "n_roots": int(np.unique(segment["root_index"]).size),
            "hero_edges": int(np.sum(segment["player"] == 0)),
            "villain_edges": int(np.sum(segment["player"] == 1)),
        }
        for segment in layout.get("level_segments", [])
    ]
    terminal_segments = {
        name: {
            "n_nodes": int(segment["n_nodes"]),
            "n_roots": int(np.unique(segment["root_index"]).size)
            if int(segment["n_nodes"]) > 0
            else 0,
        }
        for name, segment in layout.get("terminal_segments", {}).items()
    }
    return {
        "mode": "segmented_cfr_layout_summary",
        "passed": bool(layout.get("passed", False)),
        "n_roots": int(layout.get("n_roots", 0)),
        "n_actions": int(layout.get("n_actions", 0)),
        "n_nodes_total": int(layout.get("n_nodes_total", 0)),
        "max_depth": int(layout.get("max_depth", 0)),
        "n_level_segments": int(len(level_segments)),
        "max_level_edges": max((segment["n_edges"] for segment in level_segments), default=0),
        "total_edges": int(sum(segment["n_edges"] for segment in level_segments)),
        "level_segments": level_segments,
        "terminal_segments": terminal_segments,
    }


def materialize_segmented_cfr_tensors(
    layout: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Move segmented layout arrays to torch tensors for executor prototypes."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda is unavailable")
    levels = []
    for segment in layout.get("level_segments", []):
        levels.append(
            {
                "depth": int(segment["depth"]),
                "root_index": torch.as_tensor(segment["root_index"], dtype=torch.long, device=torch_device),
                "parent_global": torch.as_tensor(segment["parent_global"], dtype=torch.long, device=torch_device),
                "child_global": torch.as_tensor(segment["child_global"], dtype=torch.long, device=torch_device),
                "action": torch.as_tensor(segment["action"], dtype=torch.long, device=torch_device),
                "parent_pos": torch.as_tensor(segment["parent_pos"], dtype=torch.long, device=torch_device),
                "player": torch.as_tensor(segment["player"], dtype=torch.long, device=torch_device),
                "strategy_prob": torch.as_tensor(
                    segment["strategy_prob"],
                    dtype=torch.float32,
                    device=torch_device,
                ),
            }
        )
    terminal_segments = {}
    for name, segment in layout.get("terminal_segments", {}).items():
        terminal_segments[name] = {
            "root_index": torch.as_tensor(segment["root_index"], dtype=torch.long, device=torch_device),
            "global_indices": torch.as_tensor(segment["global_indices"], dtype=torch.long, device=torch_device),
            "local_indices": torch.as_tensor(segment["local_indices"], dtype=torch.long, device=torch_device),
            "stacks_h": torch.as_tensor(segment["stacks_h"], dtype=torch.float32, device=torch_device),
            "stacks_v": torch.as_tensor(segment["stacks_v"], dtype=torch.float32, device=torch_device),
        }
    return {
        "device": str(torch_device),
        "n_roots": int(layout["n_roots"]),
        "n_nodes_total": int(layout["n_nodes_total"]),
        "node_offsets": torch.as_tensor(layout["node_offsets"], dtype=torch.long, device=torch_device),
        "level_segments": levels,
        "terminal_segments": terminal_segments,
    }


def edge_tensors_from_node_action(
    layout_tensors: dict[str, Any],
    node_action_values: np.ndarray | torch.Tensor,
    *,
    n_hands: int,
) -> list[torch.Tensor]:
    """Gather active-action edge tensors from a dense node-action-hand tensor."""
    device = torch.device(layout_tensors["device"])
    n = int(n_hands)
    values = torch.as_tensor(node_action_values, dtype=torch.float32, device=device)
    if values.ndim != 3:
        raise ValueError("node_action_values must have shape (nodes, actions, hands)")
    if int(values.shape[0]) != int(layout_tensors["n_nodes_total"]) or int(values.shape[2]) != n:
        raise ValueError(
            "node_action_values must match layout n_nodes_total and requested n_hands"
        )
    edge_values: list[torch.Tensor] = []
    for segment in layout_tensors["level_segments"]:
        edge_values.append(values[segment["parent_global"], segment["action"], :].clone())
    return edge_values


def scatter_edge_tensors_to_node_action(
    layout_tensors: dict[str, Any],
    edge_values: list[np.ndarray | torch.Tensor],
    *,
    n_actions: int,
    n_hands: int,
) -> torch.Tensor:
    """Scatter per-level active-action edge tensors into dense node-action form."""
    device = torch.device(layout_tensors["device"])
    levels = layout_tensors["level_segments"]
    if len(edge_values) != len(levels):
        raise ValueError("edge_values must have one tensor per level segment")
    dense = torch.zeros(
        (int(layout_tensors["n_nodes_total"]), int(n_actions), int(n_hands)),
        dtype=torch.float32,
        device=device,
    )
    for segment, raw_edges in zip(levels, edge_values, strict=True):
        parents = segment["parent_global"]
        expected = (int(parents.numel()), int(n_hands))
        edges = torch.as_tensor(raw_edges, dtype=torch.float32, device=device)
        if tuple(edges.shape) != expected:
            raise ValueError(f"edge tensor shape {tuple(edges.shape)} != {expected}")
        dense[parents, segment["action"], :] = edges
    return dense


def segmented_uniform_reach_forward(
    layout_tensors: dict[str, Any],
    *,
    n_hands: int,
    hero_ranges: np.ndarray | torch.Tensor | None = None,
    villain_ranges: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a uniform-strategy forward reach pass over a segmented layout."""
    device = torch.device(layout_tensors["device"])
    n_roots = int(layout_tensors["n_roots"])
    n = int(n_hands)
    if hero_ranges is None:
        hero_init = torch.ones((n_roots, n), dtype=torch.float32, device=device)
    else:
        hero_init = torch.as_tensor(hero_ranges, dtype=torch.float32, device=device)
    if villain_ranges is None:
        villain_init = torch.ones((n_roots, n), dtype=torch.float32, device=device)
    else:
        villain_init = torch.as_tensor(villain_ranges, dtype=torch.float32, device=device)
    expected = (n_roots, n)
    if tuple(hero_init.shape) != expected or tuple(villain_init.shape) != expected:
        raise ValueError(f"hero_ranges and villain_ranges must have shape {expected}")

    hero_reach = torch.zeros(
        (int(layout_tensors["n_nodes_total"]), n),
        dtype=torch.float32,
        device=device,
    )
    villain_reach = torch.zeros_like(hero_reach)
    offsets = layout_tensors["node_offsets"]
    hero_reach[offsets] = hero_init
    villain_reach[offsets] = villain_init

    for segment in layout_tensors["level_segments"]:
        parents = segment["parent_global"]
        children = segment["child_global"]
        probs = segment["strategy_prob"].reshape(-1, 1)
        hero_edges = segment["player"] == 0
        villain_edges = segment["player"] != 0
        if bool(hero_edges.any()):
            hero_children = children[hero_edges]
            hero_parents = parents[hero_edges]
            hero_reach[hero_children] = hero_reach[hero_parents] * probs[hero_edges]
            villain_reach[hero_children] = villain_reach[hero_parents]
        if bool(villain_edges.any()):
            villain_children = children[villain_edges]
            villain_parents = parents[villain_edges]
            hero_reach[villain_children] = hero_reach[villain_parents]
            villain_reach[villain_children] = villain_reach[villain_parents] * probs[villain_edges]
    return hero_reach, villain_reach


def segmented_regret_matched_reach_forward(
    layout_tensors: dict[str, Any],
    regret_sum: np.ndarray | torch.Tensor,
    *,
    n_hands: int,
    hero_ranges: np.ndarray | torch.Tensor | None = None,
    villain_ranges: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a regret-matched forward reach pass over segmented edge lists."""
    device = torch.device(layout_tensors["device"])
    n_roots = int(layout_tensors["n_roots"])
    n = int(n_hands)
    regrets = torch.as_tensor(regret_sum, dtype=torch.float32, device=device)
    if tuple(regrets.shape[:1]) != (int(layout_tensors["n_nodes_total"]),):
        raise ValueError("regret_sum first dimension must match n_nodes_total")

    if hero_ranges is None:
        hero_init = torch.ones((n_roots, n), dtype=torch.float32, device=device)
    else:
        hero_init = torch.as_tensor(hero_ranges, dtype=torch.float32, device=device)
    if villain_ranges is None:
        villain_init = torch.ones((n_roots, n), dtype=torch.float32, device=device)
    else:
        villain_init = torch.as_tensor(villain_ranges, dtype=torch.float32, device=device)
    expected = (n_roots, n)
    if tuple(hero_init.shape) != expected or tuple(villain_init.shape) != expected:
        raise ValueError(f"hero_ranges and villain_ranges must have shape {expected}")

    hero_reach = torch.zeros(
        (int(layout_tensors["n_nodes_total"]), n),
        dtype=torch.float32,
        device=device,
    )
    villain_reach = torch.zeros_like(hero_reach)
    offsets = layout_tensors["node_offsets"]
    hero_reach[offsets] = hero_init
    villain_reach[offsets] = villain_init

    for segment in layout_tensors["level_segments"]:
        parents = segment["parent_global"]
        children = segment["child_global"]
        actions = segment["action"]
        parent_pos = segment["parent_pos"]
        positive = torch.clamp(regrets[parents, actions, :], min=0.0)
        denom = torch.zeros(
            (int(parent_pos.max().item()) + 1 if parent_pos.numel() else 0, n),
            dtype=torch.float32,
            device=device,
        )
        denom.index_add_(0, parent_pos, positive)
        parent_denom = denom.index_select(0, parent_pos)
        fallback = segment["strategy_prob"].reshape(-1, 1)
        safe_denom = torch.where(parent_denom > 0.0, parent_denom, torch.ones_like(parent_denom))
        strategy = torch.where(parent_denom > 0.0, positive / safe_denom, fallback)
        hero_edges = segment["player"] == 0
        villain_edges = segment["player"] != 0
        if bool(hero_edges.any()):
            hero_children = children[hero_edges]
            hero_parents = parents[hero_edges]
            hero_reach[hero_children] = hero_reach[hero_parents] * strategy[hero_edges]
            villain_reach[hero_children] = villain_reach[hero_parents]
        if bool(villain_edges.any()):
            villain_children = children[villain_edges]
            villain_parents = parents[villain_edges]
            hero_reach[villain_children] = hero_reach[villain_parents]
            villain_reach[villain_children] = villain_reach[villain_parents] * strategy[villain_edges]
    return hero_reach, villain_reach


def segmented_edge_regret_matched_reach_forward(
    layout_tensors: dict[str, Any],
    edge_regret_sums: list[np.ndarray | torch.Tensor],
    *,
    n_hands: int,
    hero_ranges: np.ndarray | torch.Tensor | None = None,
    villain_ranges: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run regret-matched forward propagation from edge-aligned regrets."""
    edge_strategy = segmented_edge_regret_matching_strategy(
        layout_tensors,
        edge_regret_sums,
        n_hands=n_hands,
    )
    return segmented_edge_strategy_reach_forward(
        layout_tensors,
        edge_strategy,
        n_hands=n_hands,
        hero_ranges=hero_ranges,
        villain_ranges=villain_ranges,
    )


def segmented_edge_regret_matching_strategy(
    layout_tensors: dict[str, Any],
    edge_regret_sums: list[np.ndarray | torch.Tensor],
    *,
    n_hands: int,
) -> list[torch.Tensor]:
    """Compute per-edge regret-matched strategy from active-action regrets."""
    device = torch.device(layout_tensors["device"])
    n = int(n_hands)
    levels = layout_tensors["level_segments"]
    if len(edge_regret_sums) != len(levels):
        raise ValueError("edge_regret_sums must have one tensor per level segment")

    edge_strategy: list[torch.Tensor] = []
    for segment, raw_edge_regrets in zip(levels, edge_regret_sums, strict=True):
        parents = segment["parent_global"]
        parent_pos = segment["parent_pos"]
        edge_regrets = torch.as_tensor(raw_edge_regrets, dtype=torch.float32, device=device)
        expected_edge_shape = (int(parents.numel()), n)
        if tuple(edge_regrets.shape) != expected_edge_shape:
            raise ValueError(f"edge regret shape {tuple(edge_regrets.shape)} != {expected_edge_shape}")
        positive = torch.clamp(edge_regrets, min=0.0)
        denom = torch.zeros(
            (int(parent_pos.max().item()) + 1 if parent_pos.numel() else 0, n),
            dtype=torch.float32,
            device=device,
        )
        denom.index_add_(0, parent_pos, positive)
        parent_denom = denom.index_select(0, parent_pos)
        fallback = segment["strategy_prob"].reshape(-1, 1)
        safe_denom = torch.where(parent_denom > 0.0, parent_denom, torch.ones_like(parent_denom))
        edge_strategy.append(torch.where(parent_denom > 0.0, positive / safe_denom, fallback))
    return edge_strategy


def segmented_edge_strategy_reach_forward(
    layout_tensors: dict[str, Any],
    edge_strategy: list[np.ndarray | torch.Tensor],
    *,
    n_hands: int,
    hero_ranges: np.ndarray | torch.Tensor | None = None,
    villain_ranges: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward reach pass from explicit active-action edge strategies."""
    device = torch.device(layout_tensors["device"])
    n_roots = int(layout_tensors["n_roots"])
    n = int(n_hands)
    levels = layout_tensors["level_segments"]
    if len(edge_strategy) != len(levels):
        raise ValueError("edge_strategy must have one tensor per level segment")

    if hero_ranges is None:
        hero_init = torch.ones((n_roots, n), dtype=torch.float32, device=device)
    else:
        hero_init = torch.as_tensor(hero_ranges, dtype=torch.float32, device=device)
    if villain_ranges is None:
        villain_init = torch.ones((n_roots, n), dtype=torch.float32, device=device)
    else:
        villain_init = torch.as_tensor(villain_ranges, dtype=torch.float32, device=device)
    expected = (n_roots, n)
    if tuple(hero_init.shape) != expected or tuple(villain_init.shape) != expected:
        raise ValueError(f"hero_ranges and villain_ranges must have shape {expected}")

    hero_reach = torch.zeros(
        (int(layout_tensors["n_nodes_total"]), n),
        dtype=torch.float32,
        device=device,
    )
    villain_reach = torch.zeros_like(hero_reach)
    offsets = layout_tensors["node_offsets"]
    hero_reach[offsets] = hero_init
    villain_reach[offsets] = villain_init

    for segment, raw_strategy in zip(levels, edge_strategy, strict=True):
        parents = segment["parent_global"]
        children = segment["child_global"]
        strategy = torch.as_tensor(raw_strategy, dtype=torch.float32, device=device)
        expected_edge_shape = (int(parents.numel()), n)
        if tuple(strategy.shape) != expected_edge_shape:
            raise ValueError(f"edge strategy shape {tuple(strategy.shape)} != {expected_edge_shape}")
        hero_edges = segment["player"] == 0
        villain_edges = segment["player"] != 0
        if bool(hero_edges.any()):
            hero_children = children[hero_edges]
            hero_parents = parents[hero_edges]
            hero_reach[hero_children] = hero_reach[hero_parents] * strategy[hero_edges]
            villain_reach[hero_children] = villain_reach[hero_parents]
        if bool(villain_edges.any()):
            villain_children = children[villain_edges]
            villain_parents = parents[villain_edges]
            hero_reach[villain_children] = hero_reach[villain_parents]
            villain_reach[villain_children] = villain_reach[villain_parents] * strategy[villain_edges]
    return hero_reach, villain_reach


def segmented_edge_value_backward(
    layout_tensors: dict[str, Any],
    edge_strategy: list[np.ndarray | torch.Tensor],
    hero_leaf_values: np.ndarray | torch.Tensor,
    villain_leaf_values: np.ndarray | torch.Tensor,
    *,
    n_hands: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-node values backward through segmented edge strategies."""
    device = torch.device(layout_tensors["device"])
    levels = layout_tensors["level_segments"]
    if len(edge_strategy) != len(levels):
        raise ValueError("edge_strategy must have one tensor per level segment")
    n = int(n_hands)
    expected_value_shape = (int(layout_tensors["n_nodes_total"]), n)
    hero_values = torch.as_tensor(hero_leaf_values, dtype=torch.float32, device=device).clone()
    villain_values = torch.as_tensor(villain_leaf_values, dtype=torch.float32, device=device).clone()
    if tuple(hero_values.shape) != expected_value_shape or tuple(villain_values.shape) != expected_value_shape:
        raise ValueError(f"leaf value tensors must have shape {expected_value_shape}")

    for segment, raw_strategy in zip(reversed(levels), reversed(edge_strategy), strict=True):
        parents = segment["parent_global"]
        children = segment["child_global"]
        strategy = torch.as_tensor(raw_strategy, dtype=torch.float32, device=device)
        expected_strategy_shape = (int(parents.numel()), n)
        if tuple(strategy.shape) != expected_strategy_shape:
            raise ValueError(f"edge strategy shape {tuple(strategy.shape)} != {expected_strategy_shape}")
        parent_rows = torch.unique(parents)
        hero_values[parent_rows] = 0.0
        villain_values[parent_rows] = 0.0
        hero_values.index_add_(0, parents, hero_values[children] * strategy)
        villain_values.index_add_(0, parents, villain_values[children] * strategy)
    return hero_values, villain_values


def segmented_edge_cfr_update(
    layout_tensors: dict[str, Any],
    edge_strategy: list[np.ndarray | torch.Tensor],
    hero_reach: np.ndarray | torch.Tensor,
    villain_reach: np.ndarray | torch.Tensor,
    hero_values: np.ndarray | torch.Tensor,
    villain_values: np.ndarray | torch.Tensor,
    edge_regret_sums: list[np.ndarray | torch.Tensor],
    edge_strategy_sums: list[np.ndarray | torch.Tensor],
    *,
    n_hands: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Apply one CFR+ regret and average-strategy update on edge-aligned state."""
    device = torch.device(layout_tensors["device"])
    levels = layout_tensors["level_segments"]
    if len(edge_strategy) != len(levels):
        raise ValueError("edge_strategy must have one tensor per level segment")
    if len(edge_regret_sums) != len(levels):
        raise ValueError("edge_regret_sums must have one tensor per level segment")
    if len(edge_strategy_sums) != len(levels):
        raise ValueError("edge_strategy_sums must have one tensor per level segment")

    n = int(n_hands)
    expected_node_shape = (int(layout_tensors["n_nodes_total"]), n)
    hr = torch.as_tensor(hero_reach, dtype=torch.float32, device=device)
    vr = torch.as_tensor(villain_reach, dtype=torch.float32, device=device)
    hv = torch.as_tensor(hero_values, dtype=torch.float32, device=device)
    vv = torch.as_tensor(villain_values, dtype=torch.float32, device=device)
    if tuple(hr.shape) != expected_node_shape or tuple(vr.shape) != expected_node_shape:
        raise ValueError(f"reach tensors must have shape {expected_node_shape}")
    if tuple(hv.shape) != expected_node_shape or tuple(vv.shape) != expected_node_shape:
        raise ValueError(f"value tensors must have shape {expected_node_shape}")

    updated_regrets: list[torch.Tensor] = []
    updated_strategy_sums: list[torch.Tensor] = []
    for segment, raw_strategy, raw_regrets, raw_strategy_sums in zip(
        levels,
        edge_strategy,
        edge_regret_sums,
        edge_strategy_sums,
        strict=True,
    ):
        parents = segment["parent_global"]
        children = segment["child_global"]
        expected_edge_shape = (int(parents.numel()), n)
        strategy = torch.as_tensor(raw_strategy, dtype=torch.float32, device=device)
        regrets = torch.as_tensor(raw_regrets, dtype=torch.float32, device=device).clone()
        strategy_sums = torch.as_tensor(raw_strategy_sums, dtype=torch.float32, device=device).clone()
        if tuple(strategy.shape) != expected_edge_shape:
            raise ValueError(f"edge strategy shape {tuple(strategy.shape)} != {expected_edge_shape}")
        if tuple(regrets.shape) != expected_edge_shape:
            raise ValueError(f"edge regret shape {tuple(regrets.shape)} != {expected_edge_shape}")
        if tuple(strategy_sums.shape) != expected_edge_shape:
            raise ValueError(
                f"edge strategy sum shape {tuple(strategy_sums.shape)} != {expected_edge_shape}"
            )

        hero_edges = segment["player"] == 0
        villain_edges = segment["player"] != 0
        if bool(hero_edges.any()):
            hero_edge_indices = torch.nonzero(hero_edges, as_tuple=False).reshape(-1)
            hero_parents = parents.index_select(0, hero_edge_indices)
            hero_children = children.index_select(0, hero_edge_indices)
            hero_instant = hv.index_select(0, hero_children) - hv.index_select(0, hero_parents)
            regrets[hero_edge_indices] = torch.clamp(regrets[hero_edge_indices] + hero_instant, min=0.0)
            strategy_sums[hero_edge_indices] = (
                strategy_sums[hero_edge_indices]
                + hr.index_select(0, hero_parents) * strategy.index_select(0, hero_edge_indices)
            )
        if bool(villain_edges.any()):
            villain_edge_indices = torch.nonzero(villain_edges, as_tuple=False).reshape(-1)
            villain_parents = parents.index_select(0, villain_edge_indices)
            villain_children = children.index_select(0, villain_edge_indices)
            villain_instant = vv.index_select(0, villain_children) - vv.index_select(0, villain_parents)
            regrets[villain_edge_indices] = torch.clamp(
                regrets[villain_edge_indices] + villain_instant,
                min=0.0,
            )
            strategy_sums[villain_edge_indices] = (
                strategy_sums[villain_edge_indices]
                + vr.index_select(0, villain_parents) * strategy.index_select(0, villain_edge_indices)
            )

        updated_regrets.append(regrets)
        updated_strategy_sums.append(strategy_sums)
    return updated_regrets, updated_strategy_sums


def _batched_terminal_matrix(
    raw_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    *,
    n_roots: int,
    n_hands: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(raw_matrices, (list, tuple)):
        if raw_matrices and all(torch.is_tensor(matrix) for matrix in raw_matrices):
            matrix = torch.stack(
                [torch.as_tensor(item, dtype=torch.float32, device=device) for item in raw_matrices],
                dim=0,
            )
        else:
            matrix = torch.as_tensor(np.asarray(raw_matrices, dtype=np.float32), dtype=torch.float32, device=device)
    else:
        matrix = torch.as_tensor(raw_matrices, dtype=torch.float32, device=device)
    expected = (int(n_roots), int(n_hands), int(n_hands))
    if tuple(matrix.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}")
    return matrix


def _root_vector(
    raw_values: float | Sequence[float] | np.ndarray | torch.Tensor,
    *,
    n_roots: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    values = torch.as_tensor(raw_values, dtype=torch.float32, device=device)
    if values.ndim == 0:
        return values.reshape(1).expand(int(n_roots))
    expected = (int(n_roots),)
    if tuple(values.shape) != expected:
        raise ValueError(f"{name} must be a scalar or have shape {expected}")
    return values


def segmented_terminal_values(
    layout_tensors: dict[str, Any],
    hero_reach: np.ndarray | torch.Tensor,
    villain_reach: np.ndarray | torch.Tensor,
    win_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    lose_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    tie_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    valid_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    *,
    pot_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    hero_stack_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    villain_stack_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    n_hands: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct terminal counterfactual numerator values for a segmented layout."""
    device = torch.device(layout_tensors["device"])
    n_roots = int(layout_tensors["n_roots"])
    n = int(n_hands)
    expected_reach_shape = (int(layout_tensors["n_nodes_total"]), n)
    hr = torch.as_tensor(hero_reach, dtype=torch.float32, device=device)
    vr = torch.as_tensor(villain_reach, dtype=torch.float32, device=device)
    if tuple(hr.shape) != expected_reach_shape or tuple(vr.shape) != expected_reach_shape:
        raise ValueError(f"reach tensors must have shape {expected_reach_shape}")

    win = _batched_terminal_matrix(
        win_matrices,
        n_roots=n_roots,
        n_hands=n,
        device=device,
        name="win_matrices",
    )
    lose = _batched_terminal_matrix(
        lose_matrices,
        n_roots=n_roots,
        n_hands=n,
        device=device,
        name="lose_matrices",
    )
    tie = _batched_terminal_matrix(
        tie_matrices,
        n_roots=n_roots,
        n_hands=n,
        device=device,
        name="tie_matrices",
    )
    valid = _batched_terminal_matrix(
        valid_matrices,
        n_roots=n_roots,
        n_hands=n,
        device=device,
        name="valid_matrices",
    )
    pot_by_root = _root_vector(
        pot_start,
        n_roots=n_roots,
        device=device,
        name="pot_start",
    )
    hero_stack_by_root = _root_vector(
        hero_stack_start,
        n_roots=n_roots,
        device=device,
        name="hero_stack_start",
    )
    villain_stack_by_root = _root_vector(
        villain_stack_start,
        n_roots=n_roots,
        device=device,
        name="villain_stack_start",
    )

    hero_values = torch.zeros_like(hr)
    villain_values = torch.zeros_like(vr)
    terminal_segments = layout_tensors.get("terminal_segments", {})

    show = terminal_segments.get("showdown")
    if show is not None and show["global_indices"].numel() > 0:
        indices = show["global_indices"]
        roots = show["root_index"]
        for root in torch.unique(roots).tolist():
            root_int = int(root)
            mask = roots == root_int
            root_indices = indices[mask]
            stacks_h = show["stacks_h"][mask].reshape(-1, 1)
            stacks_v = show["stacks_v"][mask].reshape(-1, 1)
            pot = pot_by_root[root_int].reshape(1, 1)
            hero_invested = hero_stack_by_root[root_int].reshape(1, 1) - stacks_h
            villain_invested = villain_stack_by_root[root_int].reshape(1, 1) - stacks_v
            vr_show = vr.index_select(0, root_indices)
            hr_show = hr.index_select(0, root_indices)
            h_win = vr_show @ win[root_int].T
            h_lose = vr_show @ lose[root_int].T
            h_tie = vr_show @ tie[root_int].T
            v_win = hr_show @ win[root_int]
            v_lose = hr_show @ lose[root_int]
            v_tie = hr_show @ tie[root_int]
            hero_values[root_indices] = (
                (pot + villain_invested) * h_win
                + (-hero_invested) * h_lose
                + ((pot + villain_invested - hero_invested) / 2.0) * h_tie
            )
            villain_values[root_indices] = (
                (pot + hero_invested) * v_lose
                + (-villain_invested) * v_win
                + ((pot + hero_invested - villain_invested) / 2.0) * v_tie
            )

    hero_fold = terminal_segments.get("hero_fold")
    if hero_fold is not None and hero_fold["global_indices"].numel() > 0:
        indices = hero_fold["global_indices"]
        roots = hero_fold["root_index"]
        for root in torch.unique(roots).tolist():
            root_int = int(root)
            mask = roots == root_int
            root_indices = indices[mask]
            hero_invested = hero_stack_by_root[root_int].reshape(1, 1) - hero_fold["stacks_h"][mask].reshape(-1, 1)
            pot = pot_by_root[root_int].reshape(1, 1)
            vr_fold = vr.index_select(0, root_indices)
            hr_fold = hr.index_select(0, root_indices)
            hero_values[root_indices] = (-hero_invested) * (vr_fold @ valid[root_int].T)
            villain_values[root_indices] = (pot + hero_invested) * (hr_fold @ valid[root_int])

    villain_fold = terminal_segments.get("villain_fold")
    if villain_fold is not None and villain_fold["global_indices"].numel() > 0:
        indices = villain_fold["global_indices"]
        roots = villain_fold["root_index"]
        for root in torch.unique(roots).tolist():
            root_int = int(root)
            mask = roots == root_int
            root_indices = indices[mask]
            villain_invested = villain_stack_by_root[root_int].reshape(1, 1) - villain_fold["stacks_v"][mask].reshape(-1, 1)
            pot = pot_by_root[root_int].reshape(1, 1)
            vr_fold = vr.index_select(0, root_indices)
            hr_fold = hr.index_select(0, root_indices)
            hero_values[root_indices] = (pot + villain_invested) * (vr_fold @ valid[root_int].T)
            villain_values[root_indices] = (-villain_invested) * (hr_fold @ valid[root_int])

    return hero_values, villain_values


def segmented_cfr_single_iteration(
    layout_tensors: dict[str, Any],
    edge_regret_sums: list[np.ndarray | torch.Tensor],
    edge_strategy_sums: list[np.ndarray | torch.Tensor],
    win_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    lose_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    tie_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    valid_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    *,
    pot_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    hero_stack_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    villain_stack_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    n_hands: int,
    hero_ranges: np.ndarray | torch.Tensor | None = None,
    villain_ranges: np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run one composed segmented CFR+ iteration over edge-aligned state."""
    edge_strategy = segmented_edge_regret_matching_strategy(
        layout_tensors,
        edge_regret_sums,
        n_hands=n_hands,
    )
    hero_reach, villain_reach = segmented_edge_strategy_reach_forward(
        layout_tensors,
        edge_strategy,
        n_hands=n_hands,
        hero_ranges=hero_ranges,
        villain_ranges=villain_ranges,
    )
    hero_leaf_values, villain_leaf_values = segmented_terminal_values(
        layout_tensors,
        hero_reach,
        villain_reach,
        win_matrices,
        lose_matrices,
        tie_matrices,
        valid_matrices,
        pot_start=pot_start,
        hero_stack_start=hero_stack_start,
        villain_stack_start=villain_stack_start,
        n_hands=n_hands,
    )
    hero_values, villain_values = segmented_edge_value_backward(
        layout_tensors,
        edge_strategy,
        hero_leaf_values,
        villain_leaf_values,
        n_hands=n_hands,
    )
    updated_regrets, updated_strategy_sums = segmented_edge_cfr_update(
        layout_tensors,
        edge_strategy,
        hero_reach,
        villain_reach,
        hero_values,
        villain_values,
        edge_regret_sums,
        edge_strategy_sums,
        n_hands=n_hands,
    )
    return {
        "edge_strategy": edge_strategy,
        "hero_reach": hero_reach,
        "villain_reach": villain_reach,
        "hero_leaf_values": hero_leaf_values,
        "villain_leaf_values": villain_leaf_values,
        "hero_values": hero_values,
        "villain_values": villain_values,
        "edge_regret_sums": updated_regrets,
        "edge_strategy_sums": updated_strategy_sums,
    }


def segmented_cfr_iterations(
    layout_tensors: dict[str, Any],
    edge_regret_sums: list[np.ndarray | torch.Tensor],
    edge_strategy_sums: list[np.ndarray | torch.Tensor],
    win_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    lose_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    tie_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    valid_matrices: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor],
    *,
    pot_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    hero_stack_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    villain_stack_start: float | Sequence[float] | np.ndarray | torch.Tensor,
    n_hands: int,
    n_iterations: int,
    hero_ranges: np.ndarray | torch.Tensor | None = None,
    villain_ranges: np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run multiple segmented CFR+ iterations over edge-aligned state."""
    total_iterations = int(n_iterations)
    if total_iterations <= 0:
        raise ValueError("n_iterations must be positive")
    current_regrets = edge_regret_sums
    current_strategy_sums = edge_strategy_sums
    result: dict[str, Any] | None = None
    for _ in range(total_iterations):
        result = segmented_cfr_single_iteration(
            layout_tensors,
            current_regrets,
            current_strategy_sums,
            win_matrices,
            lose_matrices,
            tie_matrices,
            valid_matrices,
            pot_start=pot_start,
            hero_stack_start=hero_stack_start,
            villain_stack_start=villain_stack_start,
            n_hands=n_hands,
            hero_ranges=hero_ranges,
            villain_ranges=villain_ranges,
        )
        current_regrets = result["edge_regret_sums"]
        current_strategy_sums = result["edge_strategy_sums"]
    assert result is not None
    result["n_iterations"] = total_iterations
    return result
