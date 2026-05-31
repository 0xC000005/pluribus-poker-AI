#!/usr/bin/env python3
"""Evaluate a bounded residual regret/policy update through the CFR hook."""

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

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import SolverDecision, load_cases_json

from eval_joint_pbs_policy_warm_start import (  # noqa: E402
    _case_slice,
    _illegal_mass,
    _kl_to_reference,
    _root_disjoint_audit,
    _solve_with_belief,
    _solver_context,
    _strategy_decision,
)
from eval_joint_pbs_resolver_leaf_ab import _local_ranges_from_belief  # noqa: E402
from eval_regret_policy_iteration_hook import _current_node_fields  # noqa: E402
from eval_regret_policy_warm_start import (  # noqa: E402
    _predict_fields_for_solver_node,
    _resolve_payload_path,
    load_regret_policy_warm_start_checkpoint,
)
from play_slumbot import parse_action  # noqa: E402
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402
from train_cfr_trace_policy_residual import _mean, _rate  # noqa: E402


def _clip_row_delta(delta: np.ndarray, current: np.ndarray, legal: np.ndarray, *, fraction: float, floor: float) -> np.ndarray:
    masked = np.asarray(delta, dtype=np.float32) * legal
    current_mass = float(np.abs(np.asarray(current, dtype=np.float32) * legal).sum())
    cap = max(float(floor), max(0.0, float(fraction)) * current_mass)
    norm = float(np.abs(masked).sum())
    if cap > 0.0 and norm > cap:
        masked = masked * (cap / max(norm, 1e-12))
    return masked.astype(np.float32, copy=False)


def apply_bounded_residual_field_update(
    *,
    regret_sum: np.ndarray,
    strategy_sum: np.ndarray,
    node_idx: int,
    legal_actions: list[int],
    predicted_regret: np.ndarray,
    predicted_strategy: np.ndarray,
    residual_l1_fraction: float = 0.25,
    residual_l1_floor: float = 1.0,
) -> dict[str, np.ndarray]:
    """Add a capped legal residual to one selected public node."""
    regret = np.asarray(regret_sum, dtype=np.float32).copy()
    strategy = np.asarray(strategy_sum, dtype=np.float32).copy()
    if regret.shape != strategy.shape:
        raise ValueError("regret_sum and strategy_sum must have matching shapes")
    if regret.ndim != 3:
        raise ValueError("CFR state tensors must have shape (nodes, actions, hands)")
    node = int(node_idx)
    if node < 0 or node >= regret.shape[0]:
        raise ValueError("node_idx is outside the CFR state tensor")
    action_dim = int(regret.shape[1])
    expected = (regret.shape[2], action_dim)
    pred_regret = np.asarray(predicted_regret, dtype=np.float32)
    pred_strategy = np.asarray(predicted_strategy, dtype=np.float32)
    if pred_regret.shape != expected:
        raise ValueError(f"predicted_regret shape {pred_regret.shape} != {expected}")
    if pred_strategy.shape != expected:
        raise ValueError(f"predicted_strategy shape {pred_strategy.shape} != {expected}")
    legal = np.zeros(action_dim, dtype=np.float32)
    for action in legal_actions:
        action_idx = int(action)
        if 0 <= action_idx < action_dim:
            legal[action_idx] = 1.0
    if float(legal.sum()) <= 0.0:
        raise ValueError("legal_actions must contain at least one supported action")

    current_regret = np.maximum(regret[node, :action_dim, :], 0.0).T
    current_strategy = np.maximum(strategy[node, :action_dim, :], 0.0).T
    updated_regret_rows = current_regret.copy()
    updated_strategy_rows = current_strategy.copy()
    for hand_idx in range(expected[0]):
        regret_delta = pred_regret[hand_idx] - current_regret[hand_idx]
        strategy_delta = pred_strategy[hand_idx] - current_strategy[hand_idx]
        updated_regret_rows[hand_idx] = np.maximum(
            current_regret[hand_idx]
            + _clip_row_delta(
                regret_delta,
                current_regret[hand_idx],
                legal,
                fraction=residual_l1_fraction,
                floor=residual_l1_floor,
            ),
            0.0,
        ) * legal
        updated_strategy_rows[hand_idx] = np.maximum(
            current_strategy[hand_idx]
            + _clip_row_delta(
                strategy_delta,
                current_strategy[hand_idx],
                legal,
                fraction=residual_l1_fraction,
                floor=residual_l1_floor,
            ),
            0.0,
        ) * legal
    regret[node, :action_dim, :] = updated_regret_rows.T
    strategy[node, :action_dim, :] = updated_strategy_rows.T
    return {"regret_sum": regret, "strategy_sum": strategy}


def _solve_with_bounded_residual_hook(
    *,
    model: Any,
    payload: dict[str, Any],
    device: Any,
    case: Any,
    parsed: dict[str, Any],
    belief_row: np.ndarray,
    solver_iterations: int,
    solver_backend: str,
    batch_size: int,
    residual_l1_fraction: float,
    residual_l1_floor: float,
) -> tuple[StreetSolver, Any, SolverDecision, dict[str, Any]] | None:
    backend, _device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("bounded residual hook diagnostics require solver_backend='cpu'")
    if int(parsed["st"]) not in (2, 3):
        return None
    board_idx, _our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action = _solver_context(
        case,
        parsed,
    )
    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None or node.is_terminal:
        return None
    node_idx = int(solver._tree["all_nodes"].index(node))
    legal_actions = sorted(int(action) for action in node.children.keys())
    hook_calls: list[int] = []
    started = time.perf_counter()

    def iteration_update_fn(**kwargs: Any) -> dict[str, np.ndarray]:
        hook_calls.append(int(kwargs["iteration"]))
        regret = np.asarray(kwargs["regret_sum"], dtype=np.float32)
        strategy = np.asarray(kwargs["strategy_sum"], dtype=np.float32)
        current_regret, current_strategy = _current_node_fields(regret, strategy, node_idx=node_idx)
        predicted_regret, predicted_strategy = _predict_fields_for_solver_node(
            model=model,
            payload=payload,
            device=device,
            case=case,
            parsed=parsed,
            belief_row=np.asarray(belief_row, dtype=np.float32),
            solver=solver,
            node=node,
            low_regret=current_regret,
            low_strategy=current_strategy,
            batch_size=batch_size,
        )
        return apply_bounded_residual_field_update(
            regret_sum=regret,
            strategy_sum=strategy,
            node_idx=node_idx,
            legal_actions=legal_actions,
            predicted_regret=predicted_regret,
            predicted_strategy=predicted_strategy,
            residual_l1_fraction=residual_l1_fraction,
            residual_l1_floor=residual_l1_floor,
        )

    solver.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend="cpu",
        iteration_update_fn=iteration_update_fn,
    )
    solved_node = solver.navigate(_parse_nav(street_action, solver))
    if solved_node is None or solved_node.is_terminal:
        return None
    decision = _strategy_decision(case, parsed, solver, solved_node, latency_ms=(time.perf_counter() - started) * 1000.0)
    return solver, solved_node, decision, {"hook_calls": hook_calls}


def eval_bounded_regret_residual_hook(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    device: str = "auto",
    start_index: int = 128,
    limit: int = 16,
    low_iterations: int = 5,
    baseline_iterations: int = 10,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    train_labels_npz: str | Path | None = None,
    min_evaluated: int = 1,
    batch_size: int = 8192,
    residual_l1_fraction: float = 0.25,
    residual_l1_floor: float = 1.0,
) -> dict[str, Any]:
    model, payload, torch_device = load_regret_policy_warm_start_checkpoint(checkpoint, device=device)
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)
    train_path = train_labels_npz
    if train_path is None and payload.get("train_npz"):
        train_path = _resolve_payload_path(payload["train_npz"], checkpoint=checkpoint)
    root_audit = _root_disjoint_audit(
        checkpoint=checkpoint,
        payload=payload,
        selected_cases=selected_cases,
        train_labels_npz=train_path,
        require_root_disjoint=True,
    )
    low_l1: list[float] = []
    baseline_l1: list[float] = []
    hook_l1: list[float] = []
    low_kl: list[float] = []
    baseline_kl: list[float] = []
    hook_kl: list[float] = []
    low_match: list[bool] = []
    baseline_match: list[bool] = []
    hook_match: list[bool] = []
    illegal_mass: list[float] = []
    hook_latency: list[float] = []
    low_latency: list[float] = []
    baseline_latency: list[float] = []
    reference_latency: list[float] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for local_idx, case in enumerate(selected_cases):
        case_idx = int(start_index) + int(local_idx)
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            skipped.append({"label": case.label, "reason": parsed["error"]})
            continue
        belief_row = np.asarray(base_dataset.belief[case_idx], dtype=np.float32)
        low = _solve_with_belief(case, parsed, belief_row=belief_row, solver_iterations=low_iterations, solver_backend=solver_backend)
        baseline = _solve_with_belief(case, parsed, belief_row=belief_row, solver_iterations=baseline_iterations, solver_backend=solver_backend)
        reference = _solve_with_belief(case, parsed, belief_row=belief_row, solver_iterations=reference_iterations, solver_backend=solver_backend)
        hook = _solve_with_bounded_residual_hook(
            model=model,
            payload=payload,
            device=torch_device,
            case=case,
            parsed=parsed,
            belief_row=belief_row,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
            batch_size=batch_size,
            residual_l1_fraction=residual_l1_fraction,
            residual_l1_floor=residual_l1_floor,
        )
        if low is None or baseline is None or reference is None or hook is None:
            skipped.append({"label": case.label, "reason": "solver_skipped"})
            continue
        _low_solver, low_node, low_decision = low
        _baseline_solver, baseline_node, baseline_decision = baseline
        _reference_solver, reference_node, reference_decision = reference
        _hook_solver, hook_node, hook_decision, hook_info = hook
        low_l1_value = float(np.abs(low_decision.strategy - reference_decision.strategy).sum())
        baseline_l1_value = float(np.abs(baseline_decision.strategy - reference_decision.strategy).sum())
        hook_l1_value = float(np.abs(hook_decision.strategy - reference_decision.strategy).sum())
        low_l1.append(low_l1_value)
        baseline_l1.append(baseline_l1_value)
        hook_l1.append(hook_l1_value)
        low_kl.append(_kl_to_reference(low_decision.strategy, reference_decision.strategy))
        baseline_kl.append(_kl_to_reference(baseline_decision.strategy, reference_decision.strategy))
        hook_kl.append(_kl_to_reference(hook_decision.strategy, reference_decision.strategy))
        low_match.append(int(np.argmax(low_decision.strategy)) == int(np.argmax(reference_decision.strategy)))
        baseline_match.append(int(np.argmax(baseline_decision.strategy)) == int(np.argmax(reference_decision.strategy)))
        hook_match.append(int(np.argmax(hook_decision.strategy)) == int(np.argmax(reference_decision.strategy)))
        illegal = _illegal_mass(hook_decision.strategy, hook_node)
        illegal_mass.append(float(illegal))
        low_latency.append(float(low_decision.latency_ms))
        baseline_latency.append(float(baseline_decision.latency_ms))
        reference_latency.append(float(reference_decision.latency_ms))
        hook_latency.append(float(hook_decision.latency_ms))
        records.append(
            {
                "label": case.label,
                "case_index": int(case_idx),
                "low_l1_to_reference": round(low_l1_value, 8),
                "baseline_l1_to_reference": round(baseline_l1_value, 8),
                "hook_l1_to_reference": round(hook_l1_value, 8),
                "low_action": int(np.argmax(low_decision.strategy)),
                "baseline_action": int(np.argmax(baseline_decision.strategy)),
                "hook_action": int(np.argmax(hook_decision.strategy)),
                "reference_action": int(np.argmax(reference_decision.strategy)),
                "hook_illegal_mass": illegal,
                **hook_info,
            }
        )
    passed = (
        len(records) >= int(min_evaluated)
        and root_audit["passed"]
        and _mean(hook_l1) < _mean(low_l1)
        and _mean(hook_l1) < _mean(baseline_l1)
        and _mean(hook_kl) < _mean(baseline_kl)
        and _rate(hook_match) >= _rate(baseline_match)
        and max(illegal_mass or [0.0]) <= 1e-7
    )
    return {
        "mode": "bounded_regret_residual_hook",
        "passed": bool(passed),
        "promotion": False,
        "checkpoint": str(checkpoint),
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "device": str(torch_device),
        "start_index": int(start_index),
        "limit": int(limit),
        "n_evaluated": int(len(records)),
        "low_iterations": int(low_iterations),
        "baseline_iterations": int(baseline_iterations),
        "reference_iterations": int(reference_iterations),
        "solver_backend": solver_backend,
        "root_disjoint_audit": root_audit,
        "residual_l1_fraction": float(residual_l1_fraction),
        "residual_l1_floor": float(residual_l1_floor),
        "mean_hook_l1_to_reference": _mean(hook_l1),
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_baseline_l1_to_reference": _mean(baseline_l1),
        "mean_hook_kl_to_reference": _mean(hook_kl),
        "mean_low_kl_to_reference": _mean(low_kl),
        "mean_baseline_kl_to_reference": _mean(baseline_kl),
        "hook_action_agreement": _rate(hook_match),
        "low_action_agreement": _rate(low_match),
        "baseline_action_agreement": _rate(baseline_match),
        "max_hook_illegal_mass": round(float(max(illegal_mass or [0.0])), 8),
        "mean_hook_latency_ms": _mean(hook_latency),
        "mean_low_latency_ms": _mean(low_latency),
        "mean_baseline_latency_ms": _mean(baseline_latency),
        "mean_reference_latency_ms": _mean(reference_latency),
        "records": records,
        "skipped": skipped[:100],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=("cpu",), default="cpu")
    parser.add_argument("--train-labels-npz")
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--residual-l1-fraction", type=float, default=0.25)
    parser.add_argument("--residual-l1-floor", type=float, default=1.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_bounded_regret_residual_hook(
        checkpoint=args.checkpoint,
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        device=args.device,
        start_index=args.start_index,
        limit=args.limit,
        low_iterations=args.low_iterations,
        baseline_iterations=args.baseline_iterations,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
        train_labels_npz=args.train_labels_npz,
        min_evaluated=args.min_evaluated,
        batch_size=args.batch_size,
        residual_l1_fraction=args.residual_l1_fraction,
        residual_l1_floor=args.residual_l1_floor,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
