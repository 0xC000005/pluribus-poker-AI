#!/usr/bin/env python3
"""Compare exact resolver backends on identical fixed public states."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import SolverDecision, load_cases_json

from eval_joint_pbs_policy_warm_start import (  # noqa: E402
    _case_slice,
    _illegal_mass,
    _kl_to_reference,
    _mean,
    _rate,
    _solver_context,
    _strategy_decision,
)
from eval_joint_pbs_resolver_leaf_ab import _local_ranges_from_belief  # noqa: E402
from play_slumbot import parse_action  # noqa: E402
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402


def summarize_backend_parity_records(
    records: list[dict[str, Any]],
    *,
    min_evaluated: int = 1,
    max_mean_l1: float = 1e-4,
    max_kl: float = 1e-5,
    max_illegal_mass: float = 1e-6,
    require_speedup: bool = True,
) -> dict[str, Any]:
    evaluated = [record for record in records if record.get("passed")]
    l1_values = [float(record["strategy_l1"]) for record in evaluated]
    kl_values = [float(record["strategy_kl"]) for record in evaluated]
    action_matches = [
        int(record["reference_action"]) == int(record["candidate_action"])
        for record in evaluated
    ]
    reference_latency = [float(record["reference_latency_ms"]) for record in evaluated]
    candidate_latency = [float(record["candidate_latency_ms"]) for record in evaluated]
    illegal_values = []
    for record in evaluated:
        illegal_values.append(float(record.get("reference_illegal_mass", 0.0)))
        illegal_values.append(float(record.get("candidate_illegal_mass", 0.0)))
    mean_reference_latency = _mean(reference_latency)
    mean_candidate_latency = _mean(candidate_latency)
    latency_ratio = round(
        float(mean_candidate_latency / max(mean_reference_latency, 1e-9)),
        8,
    ) if evaluated else 0.0
    mean_l1 = _mean(l1_values)
    max_l1 = round(float(max(l1_values)), 8) if l1_values else 0.0
    max_observed_kl = round(float(max(kl_values)), 8) if kl_values else 0.0
    max_observed_illegal = round(float(max(illegal_values)), 8) if illegal_values else 0.0
    action_agreement = _rate(action_matches)
    speedup_passed = latency_ratio < 1.0 if require_speedup else True
    passed = (
        len(evaluated) >= int(min_evaluated)
        and action_agreement == 1.0
        and mean_l1 <= float(max_mean_l1)
        and max_observed_kl <= float(max_kl)
        and max_observed_illegal <= float(max_illegal_mass)
        and speedup_passed
    )
    return {
        "passed": bool(passed),
        "n_evaluated": int(len(evaluated)),
        "min_evaluated": int(min_evaluated),
        "mean_strategy_l1": mean_l1,
        "max_strategy_l1": max_l1,
        "max_strategy_kl": max_observed_kl,
        "action_agreement": action_agreement,
        "mean_reference_latency_ms": mean_reference_latency,
        "mean_candidate_latency_ms": mean_candidate_latency,
        "latency_ratio_candidate_to_reference": latency_ratio,
        "speedup": round(float(1.0 / latency_ratio), 8) if latency_ratio > 0.0 else 0.0,
        "require_speedup": bool(require_speedup),
        "max_illegal_mass": max_observed_illegal,
        "thresholds": {
            "max_mean_l1": float(max_mean_l1),
            "max_kl": float(max_kl),
            "max_illegal_mass": float(max_illegal_mass),
        },
    }


def _solve_case_backend(
    *,
    case: Any,
    parsed: dict[str, Any],
    belief_row: np.ndarray,
    iterations: int,
    solver_backend: str,
) -> tuple[StreetSolver, Any, SolverDecision] | None:
    street = int(parsed["st"])
    if street not in (2, 3):
        return None
    board_idx, _our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action = _solver_context(
        case,
        parsed,
    )
    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    backend, backend_device = resolve_solver_backend(solver_backend)
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    nav = _parse_nav(street_action, solver)
    node = solver.navigate(nav)
    if node is None or node.is_terminal:
        return None
    solver.solve(
        n_iterations=int(iterations),
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
    )
    solved_node = solver.navigate(nav)
    if solved_node is None or solved_node.is_terminal:
        return None
    return solver, solved_node, _strategy_decision(
        case,
        parsed,
        solver,
        solved_node,
        latency_ms=float(getattr(solver, "last_solve_ms", 0.0)),
    )


def eval_solver_backend_parity(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    reference_backend: str = "torch-levelsync-cuda",
    candidate_backend: str = "segmented-cuda",
    iterations: int = 25,
    start_index: int = 128,
    limit: int = 16,
    min_evaluated: int = 1,
    max_mean_l1: float = 1e-4,
    max_kl: float = 1e-5,
    require_speedup: bool = True,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    records: list[dict[str, Any]] = []
    for local_idx, case in enumerate(_case_slice(cases, start_index=start_index, limit=limit)):
        case_idx = int(start_index) + int(local_idx)
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append({"label": case.label, "passed": False, "skipped": parsed["error"]})
            continue
        belief_row = np.asarray(base_dataset.belief[case_idx], dtype=np.float32)
        reference = _solve_case_backend(
            case=case,
            parsed=parsed,
            belief_row=belief_row,
            iterations=iterations,
            solver_backend=reference_backend,
        )
        candidate = _solve_case_backend(
            case=case,
            parsed=parsed,
            belief_row=belief_row,
            iterations=iterations,
            solver_backend=candidate_backend,
        )
        if reference is None or candidate is None:
            records.append({"label": case.label, "passed": False, "skipped": "solver_skipped"})
            continue
        _reference_solver, reference_node, reference_decision = reference
        _candidate_solver, candidate_node, candidate_decision = candidate
        l1_value = float(np.abs(candidate_decision.strategy - reference_decision.strategy).sum())
        records.append(
            {
                "label": case.label,
                "case_index": int(case_idx),
                "passed": True,
                "strategy_l1": round(l1_value, 8),
                "strategy_kl": _kl_to_reference(
                    candidate_decision.strategy,
                    reference_decision.strategy,
                ),
                "reference_action": int(reference_decision.action),
                "candidate_action": int(candidate_decision.action),
                "reference_increment": reference_decision.increment,
                "candidate_increment": candidate_decision.increment,
                "reference_latency_ms": round(float(reference_decision.latency_ms), 3),
                "candidate_latency_ms": round(float(candidate_decision.latency_ms), 3),
                "reference_illegal_mass": _illegal_mass(reference_decision.strategy, reference_node),
                "candidate_illegal_mass": _illegal_mass(candidate_decision.strategy, candidate_node),
            }
        )
    summary = summarize_backend_parity_records(
        records,
        min_evaluated=min_evaluated,
        max_mean_l1=max_mean_l1,
        max_kl=max_kl,
        require_speedup=require_speedup,
    )
    return {
        "mode": "solver_backend_parity",
        "promotion": False,
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "reference_backend": reference_backend,
        "candidate_backend": candidate_backend,
        "iterations": int(iterations),
        "start_index": int(start_index),
        "limit": int(limit),
        **summary,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--reference-backend", default="torch-levelsync-cuda")
    parser.add_argument("--candidate-backend", default="segmented-cuda")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--max-mean-l1", type=float, default=1e-4)
    parser.add_argument("--max-kl", type=float, default=1e-5)
    parser.add_argument("--allow-slower", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_solver_backend_parity(
        cases_json=args.cases_json,
        cfv_cache=args.cfv_cache,
        reference_backend=args.reference_backend,
        candidate_backend=args.candidate_backend,
        iterations=args.iterations,
        start_index=args.start_index,
        limit=args.limit,
        min_evaluated=args.min_evaluated,
        max_mean_l1=args.max_mean_l1,
        max_kl=args.max_kl,
        require_speedup=not args.allow_slower,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
