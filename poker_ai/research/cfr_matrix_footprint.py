"""Diagnostics for matrix/fused CFR feasibility.

The goal is not to solve CFR here. This module measures whether a street
subgame is large and regular enough to justify moving the recurrence to a
coarser sparse/dense matrix boundary.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np


def _case_solver_state_bytes(record: dict[str, Any]) -> int:
    try:
        return int(record["memory_bytes"]["solver_state_total"])
    except KeyError as exc:
        label = record.get("label", "<unknown>")
        raise KeyError(f"record {label!r} is missing memory_bytes.solver_state_total") from exc


def plan_memory_capped_chunks(
    records: list[dict[str, Any]],
    *,
    memory_cap_bytes: int,
) -> dict[str, Any]:
    """Greedily partition footprint records into order-preserving memory chunks."""
    cap = int(memory_cap_bytes)
    if cap <= 0:
        raise ValueError("memory_cap_bytes must be positive")

    chunks: list[dict[str, Any]] = []
    current_labels: list[str] = []
    current_indices: list[int] = []
    current_bytes = 0
    total_bytes = 0
    max_case_bytes = 0
    for index, record in enumerate(records):
        size = _case_solver_state_bytes(record)
        label = str(record.get("label", index))
        if size > cap:
            raise ValueError(f"case {label!r} requires {size} bytes and exceeds memory cap {cap}")
        if current_labels and current_bytes + size > cap:
            chunks.append(
                {
                    "indices": current_indices,
                    "labels": current_labels,
                    "n_cases": len(current_labels),
                    "bytes": int(current_bytes),
                }
            )
            current_labels = []
            current_indices = []
            current_bytes = 0
        current_labels.append(label)
        current_indices.append(int(index))
        current_bytes += size
        total_bytes += size
        max_case_bytes = max(max_case_bytes, size)

    if current_labels:
        chunks.append(
            {
                "indices": current_indices,
                "labels": current_labels,
                "n_cases": len(current_labels),
                "bytes": int(current_bytes),
            }
        )

    max_chunk_bytes = max((int(chunk["bytes"]) for chunk in chunks), default=0)
    return {
        "mode": "cfr_matrix_chunk_plan",
        "passed": True,
        "memory_cap_bytes": cap,
        "n_cases": int(len(records)),
        "n_chunks": int(len(chunks)),
        "max_case_bytes": int(max_case_bytes),
        "max_chunk_bytes": int(max_chunk_bytes),
        "total_bytes": int(total_bytes),
        "chunks": chunks,
    }


def _shape_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(record.get("street", -1)),
        int(record.get("n_actions", -1)),
        int(record.get("n_hands", -1)),
        str(record.get("topology_hash", "")),
        tuple(int(value) for value in record.get("level_widths", [])),
        tuple(int(value) for value in record.get("level_edges", [])),
    )


def plan_shape_compatible_chunks(
    records: list[dict[str, Any]],
    *,
    memory_cap_bytes: int,
) -> dict[str, Any]:
    """Group roots by exact tree shape, then create memory-capped chunks."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_shape_signature(record), []).append(record)

    shape_groups = []
    for group_index, (signature, group_records) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), str(item[0])),
        )
    ):
        chunk_plan = plan_memory_capped_chunks(
            group_records,
            memory_cap_bytes=memory_cap_bytes,
        )
        shape_groups.append(
            {
                "shape_id": f"shape-{group_index:03d}",
                "signature": {
                    "street": int(signature[0]),
                    "n_actions": int(signature[1]),
                    "n_hands": int(signature[2]),
                    "topology_hash": str(signature[3]),
                    "level_widths": list(signature[4]),
                    "level_edges": list(signature[5]),
                },
                "n_cases": int(len(group_records)),
                "n_chunks": int(chunk_plan["n_chunks"]),
                "max_chunk_bytes": int(chunk_plan["max_chunk_bytes"]),
                "chunks": chunk_plan["chunks"],
            }
        )

    largest = max((int(group["n_cases"]) for group in shape_groups), default=0)
    return {
        "mode": "cfr_shape_compatible_chunk_plan",
        "passed": True,
        "memory_cap_bytes": int(memory_cap_bytes),
        "n_cases": int(len(records)),
        "n_shape_groups": int(len(shape_groups)),
        "largest_shape_group_cases": int(largest),
        "same_shape_case_fraction": round(float(largest / max(len(records), 1)), 8),
        "shape_groups": shape_groups,
    }


def _terminal_batch_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    counts = [int(record.get(key, 0)) for record in records]
    active = [count for count in counts if count > 0]
    return {
        "total_nodes": int(sum(counts)),
        "active_cases": int(len(active)),
        "active_case_fraction": round(float(len(active) / max(len(records), 1)), 8),
        "max_nodes_per_case": int(max(counts, default=0)),
    }


def summarize_ragged_batching_potential(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize heterogeneous level/terminal work available to ragged batching."""
    evaluated = [record for record in records if bool(record.get("passed", True))]
    max_depth = max((len(record.get("level_widths", [])) - 1 for record in evaluated), default=0)
    level_batches: list[dict[str, Any]] = []
    for depth in range(max_depth + 1):
        widths = [
            int(record.get("level_widths", [])[depth])
            for record in evaluated
            if depth < len(record.get("level_widths", []))
        ]
        edges = [
            int(record.get("level_edges", [])[depth])
            for record in evaluated
            if depth < len(record.get("level_edges", []))
        ]
        level_batches.append(
            {
                "depth": int(depth),
                "n_cases": int(len(widths)),
                "active_case_fraction": round(float(len(widths) / max(len(evaluated), 1)), 8),
                "total_nodes": int(sum(widths)),
                "max_nodes_per_case": int(max(widths, default=0)),
                "distinct_widths": sorted({int(width) for width in widths}),
                "total_edges": int(sum(edges)),
                "max_edges_per_case": int(max(edges, default=0)),
                "distinct_edges": sorted({int(edge) for edge in edges}),
            }
        )

    terminal_batches = {
        "showdown": _terminal_batch_summary(evaluated, "n_showdown_nodes"),
        "hero_fold": _terminal_batch_summary(evaluated, "n_hero_fold_nodes"),
        "villain_fold": _terminal_batch_summary(evaluated, "n_villain_fold_nodes"),
    }
    total_terminal_nodes = int(
        sum(batch["total_nodes"] for batch in terminal_batches.values())
    )
    return {
        "mode": "cfr_ragged_batching_potential",
        "passed": True,
        "n_cases": int(len(evaluated)),
        "max_depth": int(max_depth),
        "level_batches": level_batches,
        "terminal_batches": terminal_batches,
        "total_terminal_nodes": total_terminal_nodes,
        "max_level_total_nodes": max(
            (int(level["total_nodes"]) for level in level_batches),
            default=0,
        ),
        "max_level_total_edges": max(
            (int(level["total_edges"]) for level in level_batches),
            default=0,
        ),
    }


def _ragged_padding_fraction(counts: list[int]) -> float:
    if not counts:
        return 0.0
    padded = int(max(counts)) * int(len(counts))
    if padded <= 0:
        return 0.0
    return float(1.0 - (sum(counts) / padded))


def plan_ragged_terminal_chunks(
    records: list[dict[str, Any]],
    *,
    terminal_key: str,
    max_padding_fraction: float,
    max_cases_per_chunk: int,
    sort_by_count: bool = False,
) -> dict[str, Any]:
    """Greedily chunk terminal rows so padded ragged batches stay efficient."""
    if max_cases_per_chunk <= 0:
        raise ValueError("max_cases_per_chunk must be positive")
    if not 0.0 <= float(max_padding_fraction) < 1.0:
        raise ValueError("max_padding_fraction must be in [0, 1)")

    chunks: list[dict[str, Any]] = []
    current_records: list[dict[str, Any]] = []
    current_counts: list[int] = []
    candidates = [
        {
            **record,
            "_index": index,
            "_terminal_count": int(record.get(terminal_key, 0)),
        }
        for index, record in enumerate(records)
        if bool(record.get("passed", True))
    ]
    if sort_by_count:
        candidates = sorted(
            candidates,
            key=lambda record: (
                -int(record["_terminal_count"]),
                str(record.get("label", record["_index"])),
            ),
        )

    def flush() -> None:
        if not current_records:
            return
        padded_rows = int(max(current_counts, default=0) * len(current_counts))
        total_rows = int(sum(current_counts))
        chunks.append(
            {
                "indices": [int(record.get("_index", idx)) for idx, record in enumerate(current_records)],
                "labels": [
                    str(record.get("label", record.get("_index", idx)))
                    for idx, record in enumerate(current_records)
                ],
                "n_cases": int(len(current_records)),
                "total_rows": total_rows,
                "max_rows": int(max(current_counts, default=0)),
                "padded_rows": padded_rows,
                "padding_fraction": round(_ragged_padding_fraction(current_counts), 8),
            }
        )

    for candidate_record in candidates:
        candidate_count = int(candidate_record["_terminal_count"])
        next_counts = [*current_counts, candidate_count]
        would_exceed_padding = (
            current_records
            and _ragged_padding_fraction(next_counts) > float(max_padding_fraction)
        )
        would_exceed_size = current_records and len(next_counts) > int(max_cases_per_chunk)
        if would_exceed_padding or would_exceed_size:
            flush()
            current_records = []
            current_counts = []
        current_records.append(candidate_record)
        current_counts.append(candidate_count)
    flush()

    return {
        "mode": "ragged_terminal_chunk_plan",
        "passed": True,
        "terminal_key": str(terminal_key),
        "max_padding_fraction": float(max_padding_fraction),
        "max_cases_per_chunk": int(max_cases_per_chunk),
        "sort_by_count": bool(sort_by_count),
        "n_cases": int(sum(chunk["n_cases"] for chunk in chunks)),
        "n_chunks": int(len(chunks)),
        "max_padding_fraction_observed": max(
            (float(chunk["padding_fraction"]) for chunk in chunks),
            default=0.0,
        ),
        "max_chunk_cases": max((int(chunk["n_cases"]) for chunk in chunks), default=0),
        "chunks": chunks,
    }


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


def _topology_hash(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        arr = np.asarray(array)
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


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
    terminal_type = np.asarray(
        tree.get("terminal_type", np.zeros(n_nodes, dtype=np.int32)),
        dtype=np.int32,
    )

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
        "topology_hash": _topology_hash(player, parent_idx, children, terminal_type),
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
