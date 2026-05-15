#!/usr/bin/env python3
"""Compare named live resolver budget profiles on identical public states."""

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
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    _get_our_street_bet,
    _solver_iterations_for_profile,
    parse_action,
)
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402


DEFAULT_PROFILES = ("live", "fast-live")


def profile_iterations_for_case(
    case: Any,
    parsed: dict[str, Any],
    *,
    profiles: tuple[str, ...] = DEFAULT_PROFILES,
) -> dict[str, int]:
    """Return the Slumbot live-iteration profile budgets for one case."""
    street = int(parsed["st"])
    action_str = str(case.action_str)
    streets = action_str.split("/")
    current_street = streets[-1] if streets else ""
    our_street_bet = _get_our_street_bet(current_street, int(case.client_pos), street)
    to_call = int(parsed.get("street_last_bet_to", 0)) - int(our_street_bet)
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        action_str,
        int(case.client_pos),
        target_street=street,
    )
    pot = int(our_bet_pre + opp_bet_pre)
    hero_stack = 20000 - int(our_bet_pre)
    villain_stack = 20000 - int(opp_bet_pre)
    return {
        profile: _solver_iterations_for_profile(
            profile,
            to_call=to_call,
            pot=pot,
            hero_stack=hero_stack,
            villain_stack=villain_stack,
        )
        for profile in profiles
    }


def summarize_profile_records(
    records: list[dict[str, Any]],
    *,
    reference_profile: str = "live",
    candidate_profile: str = "fast-live",
    min_evaluated: int = 1,
) -> dict[str, Any]:
    evaluated = [
        record
        for record in records
        if record.get("passed")
        and reference_profile in record.get("profiles", {})
        and candidate_profile in record.get("profiles", {})
    ]
    l1_values: list[float] = []
    kl_values: list[float] = []
    action_matches: list[bool] = []
    illegal_masses: list[float] = []
    live_latencies: list[float] = []
    candidate_latencies: list[float] = []
    live_iterations: list[float] = []
    candidate_iterations: list[float] = []
    action_disagreements: list[dict[str, Any]] = []
    for record in evaluated:
        reference = record["profiles"][reference_profile]
        candidate = record["profiles"][candidate_profile]
        reference_strategy = np.asarray(reference["strategy"], dtype=np.float64)
        candidate_strategy = np.asarray(candidate["strategy"], dtype=np.float64)
        l1 = float(np.abs(candidate_strategy - reference_strategy).sum())
        l1_values.append(l1)
        kl_values.append(_kl_to_reference(candidate_strategy, reference_strategy))
        actions_match = int(candidate["action"]) == int(reference["action"])
        action_matches.append(actions_match)
        if not actions_match:
            action_disagreements.append(
                {
                    "label": str(record.get("label", "")),
                    "reference_action": int(reference["action"]),
                    "candidate_action": int(candidate["action"]),
                    "reference_increment": str(reference.get("increment", "")),
                    "candidate_increment": str(candidate.get("increment", "")),
                    "reference_iterations": int(reference["iterations"]),
                    "candidate_iterations": int(candidate["iterations"]),
                    "l1_to_reference": round(float(l1), 8),
                }
            )
        illegal_masses.extend([
            float(reference.get("illegal_mass", 0.0)),
            float(candidate.get("illegal_mass", 0.0)),
        ])
        live_latencies.append(float(reference["latency_ms"]))
        candidate_latencies.append(float(candidate["latency_ms"]))
        live_iterations.append(float(reference["iterations"]))
        candidate_iterations.append(float(candidate["iterations"]))

    live_latency = _mean(live_latencies)
    candidate_latency = _mean(candidate_latencies)
    max_illegal_mass = max(illegal_masses) if illegal_masses else 0.0
    return {
        "reference_profile": reference_profile,
        "candidate_profile": candidate_profile,
        "n_evaluated": int(len(evaluated)),
        "min_evaluated": int(min_evaluated),
        "fast_live_mean_l1_to_live": _mean(l1_values),
        "fast_live_mean_kl_to_live": _mean(kl_values),
        "fast_live_action_agreement": _rate(action_matches),
        "n_action_disagreements": int(len(action_disagreements)),
        "action_disagreements": action_disagreements,
        "live_mean_iterations": _mean(live_iterations),
        "fast_live_mean_iterations": _mean(candidate_iterations),
        "live_mean_latency_ms": live_latency,
        "fast_live_mean_latency_ms": candidate_latency,
        "fast_live_latency_ratio_to_live": round(
            float(candidate_latency / max(live_latency, 1e-9)),
            8,
        )
        if evaluated
        else 0.0,
        "max_illegal_mass": round(float(max_illegal_mass), 8),
        "passed": bool(len(evaluated) >= int(min_evaluated) and max_illegal_mass <= 1e-6),
    }


def _solve_case_profiles(
    *,
    case: Any,
    parsed: dict[str, Any],
    belief_row: np.ndarray,
    solver_backend: str,
    profiles: tuple[str, ...] = DEFAULT_PROFILES,
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
    nav = _parse_nav(street_action, solver)
    node = solver.navigate(nav)
    if node is None or getattr(node, "is_terminal", False):
        return {"label": case.label, "passed": False, "skipped": "solver_skipped"}

    profile_iters = profile_iterations_for_case(case, parsed, profiles=profiles)
    decisions_by_iter: dict[int, dict[str, Any]] = {}
    for iterations in sorted(set(profile_iters.values())):
        solver.solve(
            n_iterations=int(iterations),
            hero_range=hero_range,
            villain_range=villain_range,
            backend=backend,
            device=backend_device,
        )
        current_node = solver.navigate(nav)
        if current_node is None or getattr(current_node, "is_terminal", False):
            return {"label": case.label, "passed": False, "skipped": f"profile_{iterations}_skipped"}
        decision = _strategy_decision(
            case,
            parsed,
            solver,
            current_node,
            latency_ms=float(getattr(solver, "last_solve_ms", 0.0)),
        )
        decisions_by_iter[int(iterations)] = {
            "action": int(decision.action),
            "increment": decision.increment,
            "strategy": decision.strategy.tolist(),
            "latency_ms": round(float(decision.latency_ms), 3),
            "iterations": int(iterations),
            "illegal_mass": _illegal_mass(decision.strategy, current_node),
        }

    return {
        "label": str(case.label),
        "passed": True,
        "iterations": profile_iters,
        "profiles": {
            profile: decisions_by_iter[int(iterations)]
            for profile, iterations in profile_iters.items()
        },
    }


def eval_solver_budget_profiles(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    start_index: int = 128,
    limit: int = 64,
    solver_backend: str = "torch-levelsync-cuda",
    min_evaluated: int = 1,
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
        records.append(
            _solve_case_profiles(
                case=case,
                parsed=parsed,
                belief_row=base_dataset.belief[case_idx],
                solver_backend=solver_backend,
            )
        )

    summary = summarize_profile_records(records, min_evaluated=min_evaluated)
    summary.update(
        {
            "mode": "solver_budget_profile_compare",
            "solver_backend": solver_backend,
            "cases_json": str(cases_json),
            "cfv_cache": str(cfv_cache),
            "start_index": int(start_index),
            "limit": int(limit),
            "records": records,
            "promotion": False,
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare live and fast-live resolver budget profiles on fixed public states."
    )
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--solver-backend", default="torch-levelsync-cuda")
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    metrics = eval_solver_budget_profiles(
        cases_json=args.cases_json,
        cfv_cache=args.cfv_cache,
        start_index=args.start_index,
        limit=args.limit,
        solver_backend=args.solver_backend,
        min_evaluated=args.min_evaluated,
    )
    save_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
