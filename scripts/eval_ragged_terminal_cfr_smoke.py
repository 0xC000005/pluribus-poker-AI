#!/usr/bin/env python3
"""Evaluate ragged-terminal CFR on heterogeneous resolver roots."""

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

from eval_same_topology_batched_cfr import (  # noqa: E402
    _actual_hand_index,
    _make_solver,
    actual_hand_root_policy,
    summarize_actual_hand_root_parity,
)
from fast_cfr import solve_cfr, solve_cfr_levelsync_torch_ragged_terminals  # noqa: E402
from poker_ai.research.resolver_benchmark import load_cases_json  # noqa: E402


def _parse_case_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _select_cases(cases_json: str, *, start_index: int, limit: int, case_indices: str | None):
    cases = load_cases_json(cases_json)[int(start_index) :]
    indices = _parse_case_indices(case_indices)
    if indices is not None:
        return [cases[index] for index in indices]
    return cases[: int(limit)]


def evaluate_ragged_terminal_cfr(
    *,
    cases_json: str,
    start_index: int,
    limit: int,
    case_indices: str | None,
    iterations: int,
    device: str,
    max_root_l1: float,
    min_speedup: float,
) -> dict:
    torch_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if torch_device == "auto":
        torch_device = "cpu"
    cases = _select_cases(
        cases_json,
        start_index=int(start_index),
        limit=int(limit),
        case_indices=case_indices,
    )
    solvers = [_make_solver(case) for case in cases]
    if not solvers:
        raise ValueError("selected case set is empty")
    n_hands = int(solvers[0].n)
    if any(int(solver.n) != n_hands for solver in solvers):
        raise ValueError("all selected roots must have the same hand dimension")

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

    ragged_started = time.perf_counter()
    ragged = solve_cfr_levelsync_torch_ragged_terminals(
        [solver._tree for solver in solvers],
        n_hands,
        [solver.win_m for solver in solvers],
        [solver.lose_m for solver in solvers],
        [solver.tie_m for solver in solvers],
        [solver.valid for solver in solvers],
        [solver.pot_start for solver in solvers],
        [solver.hero_stack_start for solver in solvers],
        [solver.villain_stack_start for solver in solvers],
        n_iterations=int(iterations),
        device=torch_device,
    )
    ragged_sec = time.perf_counter() - ragged_started

    if torch_device == "cuda":
        torch.cuda.synchronize()
    labels = [str(case.label) for case in cases]
    hand_indices = [_actual_hand_index(solver, case) for solver, case in zip(solvers, cases, strict=True)]
    legal_actions_by_root = [sorted(solver.root.children.keys()) for solver in solvers]
    n_actions = max(int(solver._tree["n_actions"]) for solver in solvers)
    parity = summarize_actual_hand_root_parity(
        labels=labels,
        serial_strategy_sums=[strategy for _, strategy in serial],
        batched_strategy_sums=np.asarray([strategy for _, strategy in ragged], dtype=object),
        hand_indices=hand_indices,
        legal_actions_by_root=legal_actions_by_root,
        n_actions=n_actions,
    )
    serial_policies = [
        actual_hand_root_policy(
            strategy,
            hand_index=hand_index,
            legal_actions=legal_actions,
            n_actions=n_actions,
        )
        for (_, strategy), hand_index, legal_actions in zip(
            serial,
            hand_indices,
            legal_actions_by_root,
            strict=True,
        )
    ]
    ragged_policies = [
        actual_hand_root_policy(
            strategy,
            hand_index=hand_index,
            legal_actions=legal_actions,
            n_actions=n_actions,
        )
        for (_, strategy), hand_index, legal_actions in zip(
            ragged,
            hand_indices,
            legal_actions_by_root,
            strict=True,
        )
    ]
    root_policy_linf = [
        float(np.max(np.abs(serial_policy - ragged_policy)))
        for serial_policy, ragged_policy in zip(serial_policies, ragged_policies, strict=True)
    ]
    speedup = float(serial_sec / ragged_sec) if ragged_sec > 0.0 else 0.0
    passed = (
        parity["top_matches"] == len(solvers)
        and parity["max_actual_root_l1"] <= float(max_root_l1)
        and speedup >= float(min_speedup)
    )
    return {
        "mode": "ragged_terminal_full_cfr_smoke",
        "passed": bool(passed),
        "device": str(torch_device),
        "n_roots": int(len(solvers)),
        "iterations": int(iterations),
        "n_hands": n_hands,
        "serial_sec": float(serial_sec),
        "ragged_sec": float(ragged_sec),
        "serial_ms_per_root": float(1000.0 * serial_sec / max(len(solvers), 1)),
        "ragged_ms_per_root": float(1000.0 * ragged_sec / max(len(solvers), 1)),
        "speedup": speedup,
        "max_root_linf": round(float(max(root_policy_linf, default=0.0)), 10),
        "max_root_l1_threshold": float(max_root_l1),
        "min_speedup": float(min_speedup),
        **parity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--case-indices",
        help="Comma-separated case indices relative to --start-index. Overrides --limit.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-root-l1", type=float, default=1e-3)
    parser.add_argument("--min-speedup", type=float, default=0.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = evaluate_ragged_terminal_cfr(
        cases_json=args.cases_json,
        start_index=args.start_index,
        limit=args.limit,
        case_indices=args.case_indices,
        iterations=args.iterations,
        device=args.device,
        max_root_l1=args.max_root_l1,
        min_speedup=args.min_speedup,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
