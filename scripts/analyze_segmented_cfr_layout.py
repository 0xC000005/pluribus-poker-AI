#!/usr/bin/env python3
"""Summarize a segmented ragged CFR layout for resolver cases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_ragged_terminal_cfr_smoke import _select_cases  # noqa: E402
from eval_same_topology_batched_cfr import (  # noqa: E402
    _actual_hand_index,
    _make_solver,
    actual_hand_root_policy,
    summarize_actual_hand_root_parity,
)
from fast_cfr import solve_cfr  # noqa: E402
from poker_ai.research.segmented_cfr_layout import (  # noqa: E402
    build_segmented_cfr_layout,
    scatter_edge_tensors_to_node_action,
    segmented_cfr_iterations,
    materialize_segmented_cfr_tensors,
    segmented_edge_cfr_update,
    segmented_edge_regret_matching_strategy,
    segmented_edge_value_backward,
    segmented_edge_regret_matched_reach_forward,
    segmented_regret_matched_reach_forward,
    segmented_uniform_reach_forward,
    summarize_segmented_cfr_layout,
)


def _edge_strategy_from_regrets(
    layout_tensors: dict[str, object],
    edge_regrets: list[np.ndarray],
    *,
    n_hands: int,
) -> list[torch.Tensor]:
    return segmented_edge_regret_matching_strategy(
        layout_tensors,
        edge_regrets,
        n_hands=n_hands,
    )


def _compare_single_iteration_parity(
    *,
    solvers,
    cases,
    layout: dict,
    tensors: dict,
    device: str,
    iterations: int,
) -> dict:
    n_hands = int(solvers[0].n)
    edge_regrets = [
        np.zeros((int(segment["n_edges"]), n_hands), dtype=np.float32)
        for segment in layout["level_segments"]
    ]
    edge_strategy_sums = [np.zeros_like(regrets, dtype=np.float32) for regrets in edge_regrets]
    started = time.perf_counter()
    segmented = segmented_cfr_iterations(
        tensors,
        edge_regrets,
        edge_strategy_sums,
        [solver.win_m for solver in solvers],
        [solver.lose_m for solver in solvers],
        [solver.tie_m for solver in solvers],
        [solver.valid for solver in solvers],
        pot_start=[float(solver.pot_start) for solver in solvers],
        hero_stack_start=[float(solver.hero_stack_start) for solver in solvers],
        villain_stack_start=[float(solver.villain_stack_start) for solver in solvers],
        n_hands=n_hands,
        n_iterations=int(iterations),
    )
    dense_regrets = scatter_edge_tensors_to_node_action(
        tensors,
        segmented["edge_regret_sums"],
        n_actions=int(layout["n_actions"]),
        n_hands=n_hands,
    )
    dense_strategy_sums = scatter_edge_tensors_to_node_action(
        tensors,
        segmented["edge_strategy_sums"],
        n_actions=int(layout["n_actions"]),
        n_hands=n_hands,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    segmented_sec = time.perf_counter() - started

    serial_started = time.perf_counter()
    serial = [
        solve_cfr(
            solver._tree,
            solver.n,
            solver.win_m,
            solver.lose_m,
            solver.tie_m,
            solver.valid,
            solver.pot_start,
            solver.hero_stack_start,
            solver.villain_stack_start,
            n_iterations=int(iterations),
        )
        for solver in solvers
    ]
    serial_sec = time.perf_counter() - serial_started

    dense_regrets_cpu = dense_regrets.detach().cpu().numpy()
    dense_strategy_sums_cpu = dense_strategy_sums.detach().cpu().numpy()
    max_regret_abs_diff = 0.0
    max_regret_scaled_diff = 0.0
    max_strategy_abs_diff = 0.0
    segmented_strategy_sums = []
    for root_index, (solver, (serial_regrets, serial_strategy)) in enumerate(
        zip(solvers, serial, strict=True)
    ):
        offset = int(layout["node_offsets"][root_index])
        n_nodes = int(solver._tree["n_nodes"])
        segmented_regret = dense_regrets_cpu[offset : offset + n_nodes]
        segmented_strategy = dense_strategy_sums_cpu[offset : offset + n_nodes]
        regret_abs_diff = np.abs(segmented_regret - serial_regrets)
        max_regret_abs_diff = max(max_regret_abs_diff, float(np.max(regret_abs_diff)))
        max_regret_scaled_diff = max(
            max_regret_scaled_diff,
            float(np.max(regret_abs_diff / np.maximum(np.abs(serial_regrets), 100000.0))),
        )
        max_strategy_abs_diff = max(
            max_strategy_abs_diff,
            float(np.max(np.abs(segmented_strategy - serial_strategy))),
        )
        segmented_strategy_sums.append(segmented_strategy)

    labels = [str(case.label) for case in cases]
    hand_indices = [_actual_hand_index(solver, case) for solver, case in zip(solvers, cases, strict=True)]
    legal_actions_by_root = [sorted(solver.root.children.keys()) for solver in solvers]
    parity = summarize_actual_hand_root_parity(
        labels=labels,
        serial_strategy_sums=[strategy for _, strategy in serial],
        batched_strategy_sums=np.asarray(segmented_strategy_sums, dtype=object),
        hand_indices=hand_indices,
        legal_actions_by_root=legal_actions_by_root,
        n_actions=int(layout["n_actions"]),
    )
    max_regret_abs_threshold = 128.0
    max_regret_scaled_threshold = 2e-4
    max_strategy_abs_threshold = 1e-4
    max_root_l1_threshold = 1e-4
    internal_state_passed = (
        max_regret_abs_diff <= max_regret_abs_threshold
        and max_regret_scaled_diff <= max_regret_scaled_threshold
        and max_strategy_abs_diff <= max_strategy_abs_threshold
    )
    decision_passed = (
        int(parity["top_matches"]) == len(solvers)
        and float(parity["max_actual_root_l1"]) <= max_root_l1_threshold
    )
    passed = (
        internal_state_passed
        and decision_passed
    )
    return {
        "passed": bool(passed),
        "decision_passed": bool(decision_passed),
        "internal_state_passed": bool(internal_state_passed),
        "device": str(device),
        "n_roots": int(len(solvers)),
        "iterations": int(iterations),
        "segmented_sec": float(segmented_sec),
        "serial_sec": float(serial_sec),
        "segmented_ms_per_root": float(1000.0 * segmented_sec / max(len(solvers), 1)),
        "serial_ms_per_root": float(1000.0 * serial_sec / max(len(solvers), 1)),
        "speedup": float(serial_sec / segmented_sec) if segmented_sec > 0.0 else 0.0,
        "max_regret_abs_diff": round(float(max_regret_abs_diff), 10),
        "max_regret_abs_threshold": float(max_regret_abs_threshold),
        "max_regret_scaled_diff_floor_100k": round(float(max_regret_scaled_diff), 10),
        "max_regret_scaled_threshold": float(max_regret_scaled_threshold),
        "max_strategy_abs_diff": round(float(max_strategy_abs_diff), 10),
        "max_strategy_abs_threshold": float(max_strategy_abs_threshold),
        "max_root_l1_threshold": float(max_root_l1_threshold),
        **parity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument(
        "--case-indices",
        help="Comma-separated case indices relative to --start-index. Overrides --limit.",
    )
    parser.add_argument("--run-uniform-forward", action="store_true")
    parser.add_argument("--run-regret-forward", action="store_true")
    parser.add_argument("--run-edge-regret-forward", action="store_true")
    parser.add_argument("--run-edge-update", action="store_true")
    parser.add_argument("--run-single-iteration-parity", action="store_true")
    parser.add_argument("--parity-iterations", type=int, default=1)
    parser.add_argument("--forward-repeats", type=int, default=3)
    parser.add_argument("--update-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    cases = _select_cases(
        args.cases_json,
        start_index=args.start_index,
        limit=args.limit,
        case_indices=args.case_indices,
    )
    solvers = [_make_solver(case) for case in cases]
    layout = build_segmented_cfr_layout([solver._tree for solver in solvers])
    summary = summarize_segmented_cfr_layout(layout)
    summary["labels"] = [str(case.label) for case in cases]
    if (
        args.run_uniform_forward
        or args.run_regret_forward
        or args.run_edge_regret_forward
        or args.run_edge_update
        or args.run_single_iteration_parity
    ):
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        tensors = materialize_segmented_cfr_tensors(layout, device=device)
        n_hands = int(solvers[0].n)
    if args.run_uniform_forward:
        started = time.perf_counter()
        hero_reach = None
        villain_reach = None
        for _ in range(int(args.forward_repeats)):
            hero_reach, villain_reach = segmented_uniform_reach_forward(
                tensors,
                n_hands=n_hands,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert hero_reach is not None and villain_reach is not None
        summary["uniform_forward"] = {
            "device": device,
            "repeats": int(args.forward_repeats),
            "seconds": float(elapsed),
            "ms_per_repeat": float(1000.0 * elapsed / max(int(args.forward_repeats), 1)),
            "hero_reach_sum": float(hero_reach.sum().detach().cpu().item()),
            "villain_reach_sum": float(villain_reach.sum().detach().cpu().item()),
        }
    if args.run_regret_forward:
        rng = np.random.default_rng(int(args.seed))
        regrets = rng.standard_normal(
            (int(layout["n_nodes_total"]), int(layout["n_actions"]), int(solvers[0].n)),
            dtype=np.float32,
        )
        started = time.perf_counter()
        hero_reach = None
        villain_reach = None
        for _ in range(int(args.forward_repeats)):
            hero_reach, villain_reach = segmented_regret_matched_reach_forward(
                tensors,
                regrets,
                n_hands=int(solvers[0].n),
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert hero_reach is not None and villain_reach is not None
        summary["regret_forward"] = {
            "device": device,
            "repeats": int(args.forward_repeats),
            "seconds": float(elapsed),
            "ms_per_repeat": float(1000.0 * elapsed / max(int(args.forward_repeats), 1)),
            "hero_reach_sum": float(hero_reach.sum().detach().cpu().item()),
            "villain_reach_sum": float(villain_reach.sum().detach().cpu().item()),
            "seed": int(args.seed),
        }
    if args.run_edge_regret_forward:
        rng = np.random.default_rng(int(args.seed))
        edge_regrets = [
            rng.standard_normal(
                (int(segment["n_edges"]), int(solvers[0].n)),
                dtype=np.float32,
            )
            for segment in summary["level_segments"]
        ]
        started = time.perf_counter()
        hero_reach = None
        villain_reach = None
        for _ in range(int(args.forward_repeats)):
            hero_reach, villain_reach = segmented_edge_regret_matched_reach_forward(
                tensors,
                edge_regrets,
                n_hands=int(solvers[0].n),
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert hero_reach is not None and villain_reach is not None
        summary["edge_regret_forward"] = {
            "device": device,
            "repeats": int(args.forward_repeats),
            "seconds": float(elapsed),
            "ms_per_repeat": float(1000.0 * elapsed / max(int(args.forward_repeats), 1)),
            "hero_reach_sum": float(hero_reach.sum().detach().cpu().item()),
            "villain_reach_sum": float(villain_reach.sum().detach().cpu().item()),
            "seed": int(args.seed),
        }
    if args.run_edge_update:
        rng = np.random.default_rng(int(args.seed))
        edge_regrets = [
            rng.standard_normal(
                (int(segment["n_edges"]), int(solvers[0].n)),
                dtype=np.float32,
            )
            for segment in summary["level_segments"]
        ]
        edge_strategy_sums = [np.zeros_like(regrets, dtype=np.float32) for regrets in edge_regrets]
        edge_strategy = _edge_strategy_from_regrets(
            tensors,
            edge_regrets,
            n_hands=int(solvers[0].n),
        )
        hero_reach, villain_reach = segmented_edge_regret_matched_reach_forward(
            tensors,
            edge_regrets,
            n_hands=int(solvers[0].n),
        )
        leaf_hero_values = np.zeros(
            (int(layout["n_nodes_total"]), int(solvers[0].n)),
            dtype=np.float32,
        )
        leaf_villain_values = np.zeros_like(leaf_hero_values)
        for segment_name in ("showdown", "hero_fold", "villain_fold"):
            terminal_indices = tensors["terminal_segments"][segment_name]["global_indices"].detach().cpu().numpy()
            if terminal_indices.size:
                leaf_hero_values[terminal_indices] = rng.standard_normal(
                    (int(terminal_indices.size), int(solvers[0].n)),
                    dtype=np.float32,
                )
                leaf_villain_values[terminal_indices] = rng.standard_normal(
                    (int(terminal_indices.size), int(solvers[0].n)),
                    dtype=np.float32,
                )
        hero_values, villain_values = segmented_edge_value_backward(
            tensors,
            edge_strategy,
            leaf_hero_values,
            leaf_villain_values,
            n_hands=int(solvers[0].n),
        )
        started = time.perf_counter()
        updated_regrets = None
        updated_strategy_sums = None
        for _ in range(int(args.update_repeats)):
            updated_regrets, updated_strategy_sums = segmented_edge_cfr_update(
                tensors,
                edge_strategy,
                hero_reach,
                villain_reach,
                hero_values,
                villain_values,
                edge_regrets,
                edge_strategy_sums,
                n_hands=int(solvers[0].n),
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert updated_regrets is not None and updated_strategy_sums is not None
        summary["edge_update"] = {
            "device": device,
            "repeats": int(args.update_repeats),
            "seconds": float(elapsed),
            "ms_per_repeat": float(1000.0 * elapsed / max(int(args.update_repeats), 1)),
            "regret_checksum": float(sum(t.sum().detach().cpu().item() for t in updated_regrets)),
            "strategy_sum_checksum": float(sum(t.sum().detach().cpu().item() for t in updated_strategy_sums)),
            "seed": int(args.seed),
        }
    if args.run_single_iteration_parity:
        summary["single_iteration_parity"] = _compare_single_iteration_parity(
            solvers=solvers,
            cases=cases,
            layout=layout,
            tensors=tensors,
            device=device,
            iterations=int(args.parity_iterations),
        )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
