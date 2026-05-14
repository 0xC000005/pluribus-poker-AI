#!/usr/bin/env python3
"""Compare low-budget solver update rules against a higher-budget teacher."""

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


def _solve_with_update(
    case: Any,
    parsed: dict[str, Any],
    *,
    belief_row: np.ndarray,
    solver_iterations: int,
    solver_backend: str,
    solver_update: str,
) -> tuple[StreetSolver, Any, Any] | None:
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
    solver.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
        solver_update=solver_update,
    )
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None or node.is_terminal:
        return None
    return solver, node, _strategy_decision(case, parsed, solver, node)


def _gap(value: float, reference: float) -> float:
    return round(abs(float(value) - float(reference)), 8)


def summarize_solver_update_records(
    records: list[dict[str, Any]],
    *,
    candidate_update: str,
    min_evaluated: int = 1,
    max_candidate_latency_ratio: float = 2.0,
) -> dict[str, Any]:
    evaluated = [record for record in records if record.get("passed")]
    low_l1 = [float(record["low_l1_to_reference"]) for record in evaluated]
    cand_l1 = [float(record["candidate_l1_to_reference"]) for record in evaluated]
    low_kl = [float(record["low_kl_to_reference"]) for record in evaluated]
    cand_kl = [float(record["candidate_kl_to_reference"]) for record in evaluated]
    low_agreement = _rate([bool(record["low_agrees_with_reference"]) for record in evaluated])
    cand_agreement = _rate([bool(record["candidate_agrees_with_reference"]) for record in evaluated])
    low_allin = _rate([bool(record["low_allin_selected"]) for record in evaluated])
    cand_allin = _rate([bool(record["candidate_allin_selected"]) for record in evaluated])
    reference_allin = _rate([bool(record["reference_allin_selected"]) for record in evaluated])
    low_allin_prob = [float(record["low_allin_prob"]) for record in evaluated]
    cand_allin_prob = [float(record["candidate_allin_prob"]) for record in evaluated]
    reference_allin_prob = [float(record["reference_allin_prob"]) for record in evaluated]
    low_latency = _mean([float(record["low_latency_ms"]) for record in evaluated])
    cand_latency = _mean([float(record["candidate_latency_ms"]) for record in evaluated])
    reference_latency = _mean([float(record["reference_latency_ms"]) for record in evaluated])
    latency_ratio = round(float(cand_latency / max(low_latency, 1e-9)), 8) if evaluated else 0.0
    illegal_masses = [
        float(record.get(key, 0.0))
        for record in evaluated
        for key in ("low_illegal_mass", "candidate_illegal_mass", "reference_illegal_mass")
    ]
    low_allin_prob_gap = _gap(_mean(low_allin_prob), _mean(reference_allin_prob))
    cand_allin_prob_gap = _gap(_mean(cand_allin_prob), _mean(reference_allin_prob))
    return {
        "candidate_update": candidate_update,
        "n_evaluated": int(len(evaluated)),
        "min_evaluated": int(min_evaluated),
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_candidate_l1_to_reference": _mean(cand_l1),
        "mean_low_kl_to_reference": _mean(low_kl),
        "mean_candidate_kl_to_reference": _mean(cand_kl),
        "low_action_agreement": low_agreement,
        "candidate_action_agreement": cand_agreement,
        "low_allin_rate": low_allin,
        "candidate_allin_rate": cand_allin,
        "reference_allin_rate": reference_allin,
        "mean_low_allin_prob": _mean(low_allin_prob),
        "mean_candidate_allin_prob": _mean(cand_allin_prob),
        "mean_reference_allin_prob": _mean(reference_allin_prob),
        "low_allin_gap": _gap(low_allin, reference_allin),
        "candidate_allin_gap": _gap(cand_allin, reference_allin),
        "low_allin_prob_gap": low_allin_prob_gap,
        "candidate_allin_prob_gap": cand_allin_prob_gap,
        "mean_low_latency_ms": low_latency,
        "mean_candidate_latency_ms": cand_latency,
        "mean_reference_latency_ms": reference_latency,
        "candidate_to_low_latency_ratio": latency_ratio,
        "max_candidate_latency_ratio": float(max_candidate_latency_ratio),
        "max_illegal_mass": round(float(max(illegal_masses)), 8) if illegal_masses else 0.0,
        "passed": bool(
            len(evaluated) >= int(min_evaluated)
            and _mean(cand_l1) < _mean(low_l1)
            and _mean(cand_kl) <= _mean(low_kl)
            and cand_agreement >= low_agreement
            and _gap(cand_allin, reference_allin) <= _gap(low_allin, reference_allin)
            and cand_allin_prob_gap <= low_allin_prob_gap
            and (not illegal_masses or max(illegal_masses) <= 1e-6)
            and latency_ratio <= float(max_candidate_latency_ratio)
        ),
    }


def eval_solver_update_gate(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    start_index: int = 128,
    limit: int = 64,
    low_iterations: int = 5,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    baseline_update: str = "cfr_plus",
    candidate_update: str = "dcfr_plus",
    min_evaluated: int = 1,
    max_candidate_latency_ratio: float = 2.0,
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
        low_solved = _solve_with_update(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
            solver_update=baseline_update,
        )
        candidate_solved = _solve_with_update(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
            solver_update=candidate_update,
        )
        reference_solved = _solve_with_update(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=reference_iterations,
            solver_backend=solver_backend,
            solver_update=baseline_update,
        )
        if low_solved is None or candidate_solved is None or reference_solved is None:
            records.append({"label": case.label, "passed": False, "skipped": "solver_skipped"})
            continue
        low_node = low_solved[1]
        low = low_solved[2]
        candidate_node = candidate_solved[1]
        candidate = candidate_solved[2]
        reference_node = reference_solved[1]
        reference = reference_solved[2]
        records.append(
            {
                "label": case.label,
                "passed": True,
                "low_action": int(np.argmax(low.strategy)),
                "candidate_action": int(np.argmax(candidate.strategy)),
                "reference_action": int(np.argmax(reference.strategy)),
                "low_l1_to_reference": round(float(np.abs(low.strategy - reference.strategy).sum()), 8),
                "candidate_l1_to_reference": round(
                    float(np.abs(candidate.strategy - reference.strategy).sum()),
                    8,
                ),
                "low_kl_to_reference": _kl_to_reference(low.strategy, reference.strategy),
                "candidate_kl_to_reference": _kl_to_reference(candidate.strategy, reference.strategy),
                "low_allin_prob": round(float(low.strategy[8]), 8),
                "candidate_allin_prob": round(float(candidate.strategy[8]), 8),
                "reference_allin_prob": round(float(reference.strategy[8]), 8),
                "low_allin_selected": bool(int(np.argmax(low.strategy)) == 8),
                "candidate_allin_selected": bool(int(np.argmax(candidate.strategy)) == 8),
                "reference_allin_selected": bool(int(np.argmax(reference.strategy)) == 8),
                "low_illegal_mass": _illegal_mass(low.strategy, low_node),
                "candidate_illegal_mass": _illegal_mass(candidate.strategy, candidate_node),
                "reference_illegal_mass": _illegal_mass(reference.strategy, reference_node),
                "low_latency_ms": round(float(low.latency_ms), 3),
                "candidate_latency_ms": round(float(candidate.latency_ms), 3),
                "reference_latency_ms": round(float(reference.latency_ms), 3),
                "low_agrees_with_reference": bool(
                    int(np.argmax(low.strategy)) == int(np.argmax(reference.strategy))
                ),
                "candidate_agrees_with_reference": bool(
                    int(np.argmax(candidate.strategy)) == int(np.argmax(reference.strategy))
                ),
            }
        )
    summary = summarize_solver_update_records(
        records,
        candidate_update=candidate_update,
        min_evaluated=min_evaluated,
        max_candidate_latency_ratio=max_candidate_latency_ratio,
    )
    return {
        "mode": "solver_update_ab_gate",
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "start_index": int(start_index),
        "limit": int(limit),
        "low_iterations": int(low_iterations),
        "reference_iterations": int(reference_iterations),
        "solver_backend": solver_backend,
        "baseline_update": baseline_update,
        "candidate_update": candidate_update,
        "n_cases": int(len(records)),
        **summary,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a low-budget solver update rule to low-budget CFR+ against a higher-budget CFR+ teacher."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument(
        "--baseline-update",
        choices=("cfr_plus", "dcfr_plus", "pdcfr_plus"),
        default="cfr_plus",
    )
    parser.add_argument(
        "--candidate-update",
        choices=("cfr_plus", "dcfr_plus", "pdcfr_plus"),
        default="dcfr_plus",
    )
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--max-candidate-latency-ratio", type=float, default=2.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_solver_update_gate(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        start_index=args.start_index,
        limit=args.limit,
        low_iterations=args.low_iterations,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
        baseline_update=args.baseline_update,
        candidate_update=args.candidate_update,
        min_evaluated=args.min_evaluated,
        max_candidate_latency_ratio=args.max_candidate_latency_ratio,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
