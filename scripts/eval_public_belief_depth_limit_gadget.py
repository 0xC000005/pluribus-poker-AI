#!/usr/bin/env python3
"""Evaluate an exact public-belief depth-limit cut teacher.

This is a diagnostic, not a gameplay path. It traces exact cut-node
counterfactual values from a full resolver solve, then replays those values
through the existing cut-node interface. If this exact teacher cannot preserve
root decisions, the boundary is not a good neural target.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_value_probe import (  # noqa: E402
    load_public_belief_cfv_dataset_cache,
    save_metrics,
)
from poker_ai.research.resolver_benchmark import (  # noqa: E402
    ResolverBenchmarkCase,
    load_cases_json,
)

from eval_callback_leaf_diverse_range_probe import load_high_margin_flip_records  # noqa: E402
from eval_joint_pbs_resolver_cut_ab import (  # noqa: E402
    _select_successor_cut_node_records,
    _summarize_cut_risk_records,
)
from eval_joint_pbs_resolver_leaf_ab import (  # noqa: E402
    _local_ranges_from_belief,
    _pre_street_prefix,
    _strategy_vector,
)
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    card_str_to_index,
    parse_action,
)
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402


@dataclass
class ExactCutTrace:
    node_indices: np.ndarray
    hero_values_by_iter: np.ndarray
    villain_values_by_iter: np.ndarray


class ExactCutValueCallback:
    """Replay traced cut values through ``solve_cfr``'s cut callback."""

    def __init__(
        self,
        trace: ExactCutTrace,
        *,
        mode: str = "replay",
        fixed_iteration: int | None = None,
    ):
        if mode not in ("replay", "final", "fixed"):
            raise ValueError(f"unknown exact cut callback mode: {mode}")
        if mode == "fixed":
            if fixed_iteration is None:
                raise ValueError("fixed_iteration is required for fixed mode")
            if fixed_iteration < 0 or fixed_iteration >= trace.hero_values_by_iter.shape[0]:
                raise ValueError(
                    "fixed_iteration must be within traced iteration range"
                )
        self.trace = trace
        self.mode = mode
        self.fixed_iteration = fixed_iteration
        self.calls = 0
        self.node_to_pos = {
            int(node_idx): pos for pos, node_idx in enumerate(trace.node_indices.tolist())
        }

    def __call__(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(kwargs["cut_indices"], dtype=np.int32)
        if self.mode == "final":
            iter_idx = self.trace.hero_values_by_iter.shape[0] - 1
        elif self.mode == "fixed":
            iter_idx = int(self.fixed_iteration)
        else:
            iter_idx = min(self.calls, self.trace.hero_values_by_iter.shape[0] - 1)
        self.calls += 1
        positions = [self.node_to_pos[int(node_idx)] for node_idx in requested.tolist()]
        return (
            self.trace.hero_values_by_iter[iter_idx, positions].astype(np.float32, copy=True),
            self.trace.villain_values_by_iter[iter_idx, positions].astype(np.float32, copy=True),
        )


def summarize_exact_cut_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for mode, result in record.get("mode_results", {}).items():
            if result.get("passed"):
                by_mode.setdefault(mode, []).append(result)
    return {
        mode: {
            "n_roots": int(len(items)),
            "mean_l1": float(np.mean([float(item["action_l1"]) for item in items])),
            "max_l1": float(np.max([float(item["action_l1"]) for item in items])),
            "action_agreement": float(
                np.mean([int(item["action_agreement"]) for item in items])
            ),
            "mean_solve_ms": float(np.mean([float(item["solve_ms"]) for item in items])),
        }
        for mode, items in sorted(by_mode.items())
    }


def _case_indices_from_high_margin_records(
    leaf_ab_jsons: list[str],
    *,
    min_margin: float,
    limit: int,
) -> list[int]:
    selected = load_high_margin_flip_records(
        leaf_ab_jsons,
        min_margin=min_margin,
        limit=limit,
    )
    return [int(record["case_index"]) for record in selected]


def _trace_exact_cut_values(
    solver: StreetSolver,
    *,
    n_iterations: int,
    hero_range: np.ndarray,
    villain_range: np.ndarray,
    backend: str,
    device: Any,
    cut_indices: list[int],
) -> ExactCutTrace:
    hero_values: list[np.ndarray] = []
    villain_values: list[np.ndarray] = []

    def trace_node_fn(**kwargs: Any) -> None:
        hero_values.append(np.asarray(kwargs["hero_values"], dtype=np.float32).copy())
        villain_values.append(np.asarray(kwargs["villain_values"], dtype=np.float32).copy())

    solver.solve(
        n_iterations=n_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=device,
        trace_node_indices=cut_indices,
        trace_node_fn=trace_node_fn,
    )
    if len(hero_values) != int(n_iterations):
        raise RuntimeError(
            f"expected {n_iterations} trace records, got {len(hero_values)}"
        )
    return ExactCutTrace(
        node_indices=np.asarray(cut_indices, dtype=np.int32),
        hero_values_by_iter=np.stack(hero_values).astype(np.float32),
        villain_values_by_iter=np.stack(villain_values).astype(np.float32),
    )


def _solve_case(
    case: ResolverBenchmarkCase,
    *,
    belief_row: np.ndarray | None,
    solver_iterations: int,
    solver_backend: str,
    modes: tuple[str, ...],
    fixed_iterations: tuple[int, ...],
) -> dict[str, Any]:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        return {"label": case.label, "passed": False, "error": parsed["error"]}
    street = int(parsed.get("st", -1))
    if street != 2:
        return {"label": case.label, "passed": False, "skipped": f"street:{street}"}

    board_idx = [card_str_to_index(card) for card in case.board[:4]]
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=street,
    )
    pot = our_bet_pre + opp_bet_pre
    hero_stack = 20000 - our_bet_pre
    villain_stack = 20000 - opp_bet_pre
    hero_first = case.client_pos == 0
    street_action = case.action_str.split("/")[street] if len(case.action_str.split("/")) > street else ""
    action_prefix = _pre_street_prefix(case.action_str, street)
    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    backend, backend_device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("exact cut teacher diagnostic requires CPU backend")

    teacher = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    active_node = teacher.navigate(_parse_nav(street_action, teacher))
    if active_node is None or active_node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "terminal_or_missing_node"}
    candidate_cut_records, selected_cut_records = _select_successor_cut_node_records(
        teacher,
        active_node,
        action_prefix=action_prefix,
        client_pos=case.client_pos,
        min_bet_count=0,
        target_action_shapes=(),
        risk_predictor=None,
    )
    cut_indices = [int(record["node_idx"]) for record in selected_cut_records]
    if not cut_indices:
        return {
            "label": case.label,
            "passed": False,
            "skipped": "no_successor_cut_nodes",
            "cut_risk_records": _summarize_cut_risk_records(candidate_cut_records),
        }

    trace_started = time.perf_counter()
    trace = _trace_exact_cut_values(
        teacher,
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
        cut_indices=cut_indices,
    )
    trace_ms = (time.perf_counter() - trace_started) * 1000.0
    teacher_node = teacher.navigate(_parse_nav(street_action, teacher))
    if teacher_node is None or teacher_node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "teacher_missing_node"}
    hand = tuple(sorted(our_cards_idx))
    teacher_strategy = _strategy_vector(teacher.get_strategy(hand, teacher_node))
    teacher_action = int(np.argmax(teacher_strategy))

    mode_results: dict[str, Any] = {}
    mode_specs: list[tuple[str, str, int | None]] = [
        (mode, mode, None) for mode in modes
    ]
    mode_specs.extend(
        (f"fixed_{int(iteration)}", "fixed", int(iteration))
        for iteration in fixed_iterations
    )
    for result_key, mode, fixed_iteration in mode_specs:
        cut_solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
        callback = ExactCutValueCallback(
            trace,
            mode=mode,
            fixed_iteration=fixed_iteration,
        )
        started = time.perf_counter()
        cut_solver.solve(
            n_iterations=solver_iterations,
            hero_range=hero_range,
            villain_range=villain_range,
            backend=backend,
            device=backend_device,
            cut_node_indices=cut_indices,
            cut_node_fn=callback,
        )
        solve_ms = (time.perf_counter() - started) * 1000.0
        cut_node = cut_solver.navigate(_parse_nav(street_action, cut_solver))
        if cut_node is None or cut_node.is_terminal:
            mode_results[result_key] = {"passed": False, "skipped": "cut_missing_node"}
            continue
        cut_strategy = _strategy_vector(cut_solver.get_strategy(hand, cut_node))
        cut_action = int(np.argmax(cut_strategy))
        mode_results[result_key] = {
            "passed": True,
            "source_mode": mode,
            "fixed_iteration": fixed_iteration,
            "action_agreement": bool(cut_action == teacher_action),
            "teacher_action": teacher_action,
            "cut_action": cut_action,
            "action_l1": float(np.abs(cut_strategy - teacher_strategy).sum()),
            "teacher_strategy": teacher_strategy.round(6).tolist(),
            "cut_strategy": cut_strategy.round(6).tolist(),
            "solve_ms": round(float(solve_ms), 3),
            "callback_calls": int(callback.calls),
        }

    return {
        "label": case.label,
        "passed": any(result.get("passed") for result in mode_results.values()),
        "street": street,
        "n_cut_nodes": int(len(cut_indices)),
        "trace_ms": round(float(trace_ms), 3),
        "teacher_solve_ms": round(float(getattr(teacher, "last_solve_ms", 0.0)), 3),
        "cut_risk_records": _summarize_cut_risk_records(candidate_cut_records),
        "mode_results": mode_results,
    }


def eval_public_belief_depth_limit_gadget(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    leaf_ab_jsons: list[str] | None = None,
    start_index: int = 0,
    limit: int = 4,
    min_margin: float = 0.5,
    solver_iterations: int = 10,
    solver_backend: str = "cpu",
    modes: tuple[str, ...] = ("replay", "final"),
    fixed_iterations: tuple[int, ...] = (),
    min_action_agreement: float = 0.75,
    max_mean_action_l1: float = 0.25,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    if leaf_ab_jsons:
        case_indices = _case_indices_from_high_margin_records(
            leaf_ab_jsons,
            min_margin=min_margin,
            limit=limit,
        )
    else:
        case_indices = list(range(int(start_index), min(int(start_index) + int(limit), len(cases))))
    records = [
        _solve_case(
            cases[case_idx],
            belief_row=dataset.belief[case_idx],
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            modes=modes,
            fixed_iterations=fixed_iterations,
        )
        | {"case_index": int(case_idx)}
        for case_idx in case_indices
    ]
    summaries = summarize_exact_cut_records(records)
    replay_summary = summaries.get("replay")
    behavior_passed = bool(
        replay_summary
        and replay_summary["action_agreement"] >= float(min_action_agreement)
        and replay_summary["mean_l1"] <= float(max_mean_action_l1)
    )
    return {
        "mode": "public_belief_depth_limit_exact_cut_teacher",
        "passed": behavior_passed,
        "promotion": False,
        "promotion_blockers": [
            "exact_cut_teacher_is_diagnostic_not_live_policy",
            "requires_neural_approximation_and_root_disjoint_resolver_gate",
        ],
        "cases": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "leaf_ab_jsons": list(leaf_ab_jsons or []),
        "case_indices": case_indices,
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "modes": list(modes),
        "fixed_iterations": [int(iteration) for iteration in fixed_iterations],
        "min_action_agreement": float(min_action_agreement),
        "max_mean_action_l1": float(max_mean_action_l1),
        "summaries": summaries,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--leaf-ab-json", action="append", default=[])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--min-margin", type=float, default=0.5)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--mode", action="append", choices=("replay", "final"), dest="modes")
    parser.add_argument(
        "--fixed-iteration",
        action="append",
        type=int,
        default=[],
        help="Also replay one traced iteration's cut CFVs for all callback calls.",
    )
    parser.add_argument("--min-action-agreement", type=float, default=0.75)
    parser.add_argument("--max-mean-action-l1", type=float, default=0.25)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = eval_public_belief_depth_limit_gadget(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        leaf_ab_jsons=args.leaf_ab_json,
        start_index=args.start_index,
        limit=args.limit,
        min_margin=args.min_margin,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        modes=tuple(args.modes or ("replay", "final")),
        fixed_iterations=tuple(args.fixed_iteration or ()),
        min_action_agreement=args.min_action_agreement,
        max_mean_action_l1=args.max_mean_action_l1,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
