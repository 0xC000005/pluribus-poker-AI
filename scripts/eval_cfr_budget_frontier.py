#!/usr/bin/env python3
"""Measure CFR+ budget quality/latency frontier against one teacher solve."""

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
from poker_ai.research.resolver_benchmark import load_cases_json

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


def _unique_budgets(budgets: list[int]) -> list[int]:
    unique = sorted({int(value) for value in budgets})
    if not unique or unique[0] <= 0:
        raise ValueError("budgets must contain positive iteration counts")
    return unique


def _gap(value: float, reference: float) -> float:
    return round(abs(float(value) - float(reference)), 8)


def summarize_budget_frontier_records(
    records: list[dict[str, Any]],
    *,
    budgets: list[int],
    min_evaluated: int = 1,
) -> dict[str, Any]:
    budget_ids = [str(budget) for budget in _unique_budgets(budgets)]
    evaluated = [record for record in records if record.get("passed")]
    min_budget = budget_ids[0]
    min_latency = _mean(
        [
            float(record["budgets"][min_budget]["latency_ms"])
            for record in evaluated
            if min_budget in record.get("budgets", {})
        ]
    )
    reference_illegal = [float(record.get("reference_illegal_mass", 0.0)) for record in evaluated]
    max_illegal_mass = max(reference_illegal) if reference_illegal else 0.0
    by_budget: dict[str, dict[str, Any]] = {}
    for budget_id in budget_ids:
        pairs = [
            (record, record["budgets"][budget_id])
            for record in evaluated
            if budget_id in record.get("budgets", {})
        ]
        values = [value for _record, value in pairs]
        reference_allin_prob = [
            float(record["reference_allin_prob"])
            for record, _value in pairs
        ]
        action_agreement = _rate(
            [
                int(value["action"]) == int(record["reference_action"])
                for record, value in pairs
            ]
        )
        allin_rate = _rate([bool(value["allin_selected"]) for value in values])
        reference_allin_rate = _rate(
            [bool(record["reference_allin_selected"]) for record, _value in pairs]
        )
        mean_latency = _mean([float(value["latency_ms"]) for value in values])
        budget_illegal = [float(value.get("illegal_mass", 0.0)) for value in values]
        if budget_illegal:
            max_illegal_mass = max(max_illegal_mass, max(budget_illegal))
        mean_allin_prob = _mean([float(value["allin_prob"]) for value in values])
        mean_reference_allin_prob = _mean(reference_allin_prob)
        by_budget[budget_id] = {
            "n_evaluated": int(len(values)),
            "mean_l1_to_reference": _mean([float(value["l1_to_reference"]) for value in values]),
            "mean_kl_to_reference": _mean([float(value["kl_to_reference"]) for value in values]),
            "action_agreement": action_agreement,
            "allin_rate": allin_rate,
            "reference_allin_rate": reference_allin_rate,
            "allin_gap": _gap(allin_rate, reference_allin_rate),
            "mean_allin_prob": mean_allin_prob,
            "mean_reference_allin_prob": mean_reference_allin_prob,
            "allin_prob_gap": _gap(mean_allin_prob, mean_reference_allin_prob),
            "mean_latency_ms": mean_latency,
            "latency_ratio_to_min_budget": round(
                float(mean_latency / max(min_latency, 1e-9)),
                8,
            )
            if values
            else 0.0,
            "max_illegal_mass": round(float(max(budget_illegal)), 8) if budget_illegal else 0.0,
        }
    complete_budgets = [
        budget_id
        for budget_id, metrics in by_budget.items()
        if int(metrics["n_evaluated"]) >= int(min_evaluated)
    ]
    best_l1_budget = min(
        complete_budgets,
        key=lambda budget_id: by_budget[budget_id]["mean_l1_to_reference"],
        default=None,
    )
    fastest_budget = min(
        complete_budgets,
        key=lambda budget_id: by_budget[budget_id]["mean_latency_ms"],
        default=None,
    )
    return {
        "n_evaluated": int(len(evaluated)),
        "min_evaluated": int(min_evaluated),
        "budgets": by_budget,
        "best_l1_budget": int(best_l1_budget) if best_l1_budget is not None else None,
        "fastest_budget": int(fastest_budget) if fastest_budget is not None else None,
        "max_illegal_mass": round(float(max_illegal_mass), 8),
        "passed": bool(
            len(evaluated) >= int(min_evaluated)
            and len(complete_budgets) == len(budget_ids)
            and max_illegal_mass <= 1e-6
        ),
    }


def _solve_case_budget_frontier(
    *,
    case: Any,
    parsed: dict[str, Any],
    belief_row: np.ndarray,
    budgets: list[int],
    reference_iterations: int,
    solver_backend: str,
    solver_update: str,
    solver_factory: Any = StreetSolver,
) -> dict[str, Any]:
    street = int(parsed["st"])
    if street not in (2, 3):
        return {"label": case.label, "passed": False, "skipped": "unsupported_street"}
    board_idx, _our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action = _solver_context(
        case,
        parsed,
    )
    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    backend, backend_device = resolve_solver_backend(solver_backend)
    solver = solver_factory(board_idx, pot, hero_stack, villain_stack, hero_first)
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None or getattr(node, "is_terminal", False):
        return {"label": case.label, "passed": False, "skipped": "solver_skipped"}

    decisions: dict[int, Any] = {}
    nodes: dict[int, Any] = {}
    for iterations in [int(reference_iterations), *budgets]:
        solver.solve(
            n_iterations=iterations,
            hero_range=hero_range,
            villain_range=villain_range,
            backend=backend,
            device=backend_device,
            solver_update=solver_update,
        )
        current_node = solver.navigate(_parse_nav(street_action, solver))
        if current_node is None or getattr(current_node, "is_terminal", False):
            return {"label": case.label, "passed": False, "skipped": f"budget_{iterations}_skipped"}
        decisions[int(iterations)] = _strategy_decision(
            case,
            parsed,
            solver,
            current_node,
            latency_ms=float(getattr(solver, "last_solve_ms", 0.0)),
        )
        nodes[int(iterations)] = current_node

    reference = decisions[int(reference_iterations)]
    reference_node = nodes[int(reference_iterations)]
    budget_metrics: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        decision = decisions[int(budget)]
        current_node = nodes[int(budget)]
        budget_metrics[str(budget)] = {
            "action": int(np.argmax(decision.strategy)),
            "l1_to_reference": round(
                float(np.abs(decision.strategy - reference.strategy).sum()),
                8,
            ),
            "kl_to_reference": _kl_to_reference(decision.strategy, reference.strategy),
            "allin_prob": round(float(decision.strategy[8]), 8),
            "allin_selected": bool(int(np.argmax(decision.strategy)) == 8),
            "illegal_mass": _illegal_mass(decision.strategy, current_node),
            "latency_ms": round(float(decision.latency_ms), 3),
        }
    return {
        "label": case.label,
        "passed": True,
        "reference_action": int(np.argmax(reference.strategy)),
        "reference_allin_prob": round(float(reference.strategy[8]), 8),
        "reference_allin_selected": bool(int(np.argmax(reference.strategy)) == 8),
        "reference_illegal_mass": _illegal_mass(reference.strategy, reference_node),
        "reference_latency_ms": round(float(reference.latency_ms), 3),
        "budgets": budget_metrics,
    }


def eval_cfr_budget_frontier(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    budgets: list[int],
    start_index: int = 128,
    limit: int = 64,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    solver_update: str = "cfr_plus",
    min_evaluated: int = 1,
) -> dict[str, Any]:
    budgets = _unique_budgets(budgets)
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
        records.append(
            _solve_case_budget_frontier(
                case=case,
                parsed=parsed,
                belief_row=base_dataset.belief[case_idx],
                budgets=budgets,
                reference_iterations=reference_iterations,
                solver_backend=solver_backend,
                solver_update=solver_update,
            )
        )
    summary = summarize_budget_frontier_records(
        records,
        budgets=budgets,
        min_evaluated=min_evaluated,
    )
    return {
        "mode": "cfr_budget_frontier",
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "start_index": int(start_index),
        "limit": int(limit),
        "budgets": budgets,
        "reference_iterations": int(reference_iterations),
        "solver_backend": solver_backend,
        "solver_update": solver_update,
        "n_cases": int(len(records)),
        "promotion": False,
        **summary,
        "records": records,
    }


def _parse_budget_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure low CFR+ budget decisions against one higher-budget teacher per public state."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--budgets", default="5,10")
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument(
        "--solver-backend",
        choices=("cpu", "auto", "torch-cuda", "torch-cpu"),
        default="cpu",
    )
    parser.add_argument(
        "--solver-update",
        choices=("cfr_plus", "dcfr_plus", "pdcfr_plus"),
        default="cfr_plus",
    )
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_cfr_budget_frontier(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        budgets=_parse_budget_list(args.budgets),
        start_index=args.start_index,
        limit=args.limit,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
        solver_update=args.solver_update,
        min_evaluated=args.min_evaluated,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
