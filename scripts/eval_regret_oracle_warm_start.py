#!/usr/bin/env python3
"""Evaluate teacher-regret CFR+ warm starts as an oracle upper bound."""

from __future__ import annotations

import argparse
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
from poker_ai.research.resolver_benchmark import load_cases_json

from eval_joint_pbs_policy_warm_start import (  # noqa: E402
    _case_slice,
    _illegal_mass,
    _kl_to_reference,
    _solve_with_belief,
    _solver_context,
    _strategy_decision,
    _summarize_records,
)
from play_slumbot import parse_action  # noqa: E402
from solver import StreetSolver, _parse_nav  # noqa: E402


def build_regret_oracle_warm_start(
    *,
    reference_solver: StreetSolver,
    reference_node: Any,
    target_solver: StreetSolver,
    target_node: Any,
    regret_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy the teacher's selected-node CFR+ regret/policy field into a new solve."""
    if reference_solver._regret_sum is None:
        raise ValueError("reference solver has no regret_sum; solve it before building an oracle")
    if reference_solver._strategy_sum is None:
        raise ValueError("reference solver has no strategy_sum; solve it before building an oracle")
    if reference_solver.n != target_solver.n:
        raise ValueError("reference and target solvers use different hand counts")
    if reference_solver._tree["n_actions"] != target_solver._tree["n_actions"]:
        raise ValueError("reference and target solvers use different action dimensions")
    ref_idx = reference_solver._tree["all_nodes"].index(reference_node)
    target_idx = target_solver._tree["all_nodes"].index(target_node)
    shape = (
        target_solver._tree["n_nodes"],
        target_solver._tree["n_actions"],
        target_solver.n,
    )
    initial_regret = np.zeros(shape, dtype=np.float32)
    initial_strategy = np.zeros(shape, dtype=np.float32)
    selected = np.maximum(reference_solver._regret_sum[ref_idx], 0.0)
    initial_regret[target_idx] = (float(regret_scale) * selected).astype(np.float32, copy=False)
    initial_strategy[target_idx] = np.maximum(
        reference_solver._strategy_sum[ref_idx],
        0.0,
    ).astype(np.float32, copy=False)
    return initial_regret, initial_strategy


def _oracle_warm_start_decision(
    case: Any,
    parsed: dict[str, Any],
    *,
    belief_row: np.ndarray,
    reference_solver: StreetSolver,
    reference_node: Any,
    solver_iterations: int,
    solver_backend: str,
    regret_scale: float,
) -> tuple[StreetSolver, Any, Any] | None:
    board_idx, _our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action = _solver_context(
        case,
        parsed,
    )
    target_solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    target_node = target_solver.navigate(_parse_nav(street_action, target_solver))
    if target_node is None or target_node.is_terminal:
        return None
    initial_regret, initial_strategy = build_regret_oracle_warm_start(
        reference_solver=reference_solver,
        reference_node=reference_node,
        target_solver=target_solver,
        target_node=target_node,
        regret_scale=regret_scale,
    )
    solved = _solve_with_belief(
        case,
        parsed,
        belief_row=belief_row,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        initial_regret_sum=initial_regret,
        initial_strategy_sum=initial_strategy,
    )
    if solved is None:
        return None
    solver, node, decision = solved
    return solver, node, type(decision)(
        decision.action,
        decision.increment,
        decision.strategy,
        float(decision.latency_ms),
        decision.node_terminal,
    )


def eval_regret_oracle_warm_start(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    start_index: int = 128,
    limit: int = 64,
    low_iterations: int = 5,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    regret_scale: float = 1.0,
    min_evaluated: int = 1,
    max_oracle_latency_ratio: float = 2.0,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)
    records: list[dict[str, Any]] = []
    for local_idx, case in enumerate(selected_cases):
        case_idx = int(start_index) + int(local_idx)
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append({"label": case.label, "passed": False, "skipped": parsed["error"]})
            continue
        belief_row = base_dataset.belief[case_idx]
        low_solved = _solve_with_belief(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
        )
        reference_solved = _solve_with_belief(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=reference_iterations,
            solver_backend=solver_backend,
        )
        if low_solved is None or reference_solved is None:
            records.append({"label": case.label, "passed": False, "skipped": "solver_skipped"})
            continue
        reference_solver, reference_node, reference = reference_solved
        oracle_solved = _oracle_warm_start_decision(
            case,
            parsed,
            belief_row=belief_row,
            reference_solver=reference_solver,
            reference_node=reference_node,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
            regret_scale=regret_scale,
        )
        if oracle_solved is None:
            records.append({"label": case.label, "passed": False, "skipped": "oracle_skipped"})
            continue
        _oracle_solver, oracle_node, oracle = oracle_solved
        low_node = low_solved[1]
        low = low_solved[2]
        records.append(
            {
                "label": case.label,
                "passed": True,
                "low_action": int(np.argmax(low.strategy)),
                "reference_action": int(np.argmax(reference.strategy)),
                "warm_action": int(np.argmax(oracle.strategy)),
                "low_l1_to_reference": round(float(np.abs(low.strategy - reference.strategy).sum()), 8),
                "warm_l1_to_reference": round(float(np.abs(oracle.strategy - reference.strategy).sum()), 8),
                "low_kl_to_reference": _kl_to_reference(low.strategy, reference.strategy),
                "warm_kl_to_reference": _kl_to_reference(oracle.strategy, reference.strategy),
                "low_allin_prob": round(float(low.strategy[8]), 8),
                "reference_allin_prob": round(float(reference.strategy[8]), 8),
                "warm_allin_prob": round(float(oracle.strategy[8]), 8),
                "low_allin_selected": bool(int(np.argmax(low.strategy)) == 8),
                "reference_allin_selected": bool(int(np.argmax(reference.strategy)) == 8),
                "warm_allin_selected": bool(int(np.argmax(oracle.strategy)) == 8),
                "low_illegal_mass": _illegal_mass(low.strategy, low_node),
                "reference_illegal_mass": _illegal_mass(reference.strategy, reference_node),
                "warm_illegal_mass": _illegal_mass(oracle.strategy, oracle_node),
                "low_latency_ms": round(float(low.latency_ms), 3),
                "reference_latency_ms": round(float(reference.latency_ms), 3),
                "warm_latency_ms": round(float(oracle.latency_ms), 3),
                "low_agrees_with_reference": bool(
                    int(np.argmax(low.strategy)) == int(np.argmax(reference.strategy))
                ),
                "warm_agrees_with_reference": bool(
                    int(np.argmax(oracle.strategy)) == int(np.argmax(reference.strategy))
                ),
            }
        )
    summary = _summarize_records(
        records,
        min_evaluated=min_evaluated,
        max_warm_latency_ratio=max_oracle_latency_ratio,
    )
    return {
        "mode": "regret_oracle_warm_start_solver_budget",
        "oracle_upper_bound": True,
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "start_index": int(start_index),
        "limit": int(limit),
        "low_iterations": int(low_iterations),
        "reference_iterations": int(reference_iterations),
        "solver_backend": solver_backend,
        "regret_scale": float(regret_scale),
        "max_oracle_latency_ratio": float(max_oracle_latency_ratio),
        "n_cases": int(len(records)),
        **summary,
        "mean_oracle_l1_to_reference": summary["mean_warm_l1_to_reference"],
        "mean_oracle_kl_to_reference": summary["mean_warm_kl_to_reference"],
        "oracle_action_agreement": summary["warm_action_agreement"],
        "mean_oracle_latency_ms": summary["mean_warm_latency_ms"],
        "oracle_to_low_latency_ratio": summary["warm_to_low_latency_ratio"],
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare low-budget CFR+ to teacher-regret oracle warm-start CFR+."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--regret-scale", type=float, default=1.0)
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--max-oracle-latency-ratio", type=float, default=2.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_regret_oracle_warm_start(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        start_index=args.start_index,
        limit=args.limit,
        low_iterations=args.low_iterations,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
        regret_scale=args.regret_scale,
        min_evaluated=args.min_evaluated,
        max_oracle_latency_ratio=args.max_oracle_latency_ratio,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
