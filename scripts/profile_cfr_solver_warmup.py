#!/usr/bin/env python3
"""Profile cold versus warm exact CFR solve latency on fixed public states."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache  # noqa: E402
from poker_ai.research.cfr_solver_warmup import summarize_warmup_records  # noqa: E402
from poker_ai.research.resolver_benchmark import load_cases_json  # noqa: E402
from eval_joint_pbs_policy_warm_start import _case_slice, _solver_context  # noqa: E402
from eval_joint_pbs_resolver_leaf_ab import _local_ranges_from_belief  # noqa: E402
from play_slumbot import parse_action  # noqa: E402
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402


def _solve_once(
    solver: StreetSolver,
    *,
    iterations: int,
    hero_range: np.ndarray,
    villain_range: np.ndarray,
    backend: str,
    device: str | None,
) -> float:
    solver.solve(
        n_iterations=int(iterations),
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=device,
    )
    return float(getattr(solver, "last_solve_ms", 0.0))


def _profile_case(
    *,
    case: Any,
    parsed: dict[str, Any],
    belief_row: np.ndarray,
    iterations: int,
    warm_repeats: int,
    solver_backend: str,
) -> dict[str, Any]:
    street = int(parsed["st"])
    if street not in (2, 3):
        return {"label": case.label, "passed": False, "skipped": "unsupported_street"}
    board_idx, _our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action = (
        _solver_context(case, parsed)
    )
    started = time.perf_counter()
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    construction_ms = (time.perf_counter() - started) * 1000.0
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None or getattr(node, "is_terminal", False):
        return {"label": case.label, "passed": False, "skipped": "solver_skipped"}
    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    backend, backend_device = resolve_solver_backend(solver_backend)
    cold_ms = _solve_once(
        solver,
        iterations=iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
    )
    warm_latencies = [
        _solve_once(
            solver,
            iterations=iterations,
            hero_range=hero_range,
            villain_range=villain_range,
            backend=backend,
            device=backend_device,
        )
        for _ in range(int(warm_repeats))
    ]
    return {
        "label": case.label,
        "passed": True,
        "street": street,
        "n_nodes": int(solver._tree["n_nodes"]),
        "construction_ms": round(float(construction_ms), 3),
        "cold_solve_ms": round(float(cold_ms), 3),
        "warm_solve_mean_ms": round(float(np.mean(warm_latencies)), 3)
        if warm_latencies
        else 0.0,
        "warm_solve_min_ms": round(float(np.min(warm_latencies)), 3)
        if warm_latencies
        else 0.0,
        "warm_solve_max_ms": round(float(np.max(warm_latencies)), 3)
        if warm_latencies
        else 0.0,
        "cold_to_warm_speedup": round(
            float(cold_ms) / max(float(np.mean(warm_latencies)), 1e-9),
            8,
        )
        if warm_latencies
        else 0.0,
    }


def profile_cfr_solver_warmup(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    start_index: int,
    limit: int,
    iterations: int,
    warm_repeats: int,
    solver_backend: str,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)
    records = []
    for local_idx, case in enumerate(selected_cases):
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append({"label": case.label, "passed": False, "skipped": parsed["error"]})
            continue
        records.append(
            _profile_case(
                case=case,
                parsed=parsed,
                belief_row=base_dataset.belief[int(start_index) + int(local_idx)],
                iterations=iterations,
                warm_repeats=warm_repeats,
                solver_backend=solver_backend,
            )
        )
    summary = summarize_warmup_records(records)
    return {
        **summary,
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "start_index": int(start_index),
        "limit": int(limit),
        "iterations": int(iterations),
        "warm_repeats": int(warm_repeats),
        "solver_backend": solver_backend,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure cold and warm CFR solve latency on the same fixed roots."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=125)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument(
        "--solver-backend",
        choices=(
            "cpu",
            "cpu-levelsync",
            "auto",
            "torch-cuda",
            "torch-cpu",
            "torch-levelsync-cuda",
            "torch-levelsync-cpu",
        ),
        default="torch-levelsync-cuda",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = profile_cfr_solver_warmup(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        start_index=args.start_index,
        limit=args.limit,
        iterations=args.iterations,
        warm_repeats=args.warm_repeats,
        solver_backend=args.solver_backend,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
