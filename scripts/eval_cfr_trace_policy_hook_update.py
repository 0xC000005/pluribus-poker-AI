#!/usr/bin/env python3
"""Evaluate a learned trace-policy update consumed through the CFR hook."""

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

from analyze_cfr_trace_predictor import _load_payload  # noqa: E402
from eval_joint_pbs_policy_warm_start import (  # noqa: E402
    _case_slice,
    _illegal_mass,
    _kl_to_reference,
    _solve_with_belief,
    _solver_context,
    _strategy_decision,
)
from eval_joint_pbs_resolver_leaf_ab import _local_ranges_from_belief  # noqa: E402
from play_slumbot import card_str_to_index, parse_action  # noqa: E402
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402
from train_cfr_trace_delta_mlp import (  # noqa: E402
    _build_dataset,
    _common_labels,
    _predict_delta,
    _standardize,
    _train_mlp,
)
from train_cfr_trace_policy_residual import _mean, _normalize_prediction, _rate, _record_map, _top_match  # noqa: E402


def _normalise_target_policy(
    target_policy: np.ndarray,
    legal_actions: list[int],
) -> np.ndarray:
    arr = np.asarray(target_policy, dtype=np.float32).reshape(-1)
    out = np.zeros_like(arr, dtype=np.float32)
    legal = [int(action) for action in legal_actions if 0 <= int(action) < arr.shape[0]]
    if not legal:
        raise ValueError("legal_actions must contain at least one valid action")
    clean = np.where(np.isfinite(arr), arr, 0.0)
    clean = np.maximum(clean, 0.0)
    total = float(clean[legal].sum())
    if total <= 1e-8:
        out[legal] = 1.0 / float(len(legal))
    else:
        out[legal] = clean[legal] / total
    return out


def apply_policy_update_to_cfr_state(
    *,
    regret_sum: np.ndarray,
    strategy_sum: np.ndarray,
    node_idx: int,
    legal_actions: list[int],
    target_policy: np.ndarray,
    regret_mass: float,
    strategy_mass: float,
) -> dict[str, np.ndarray]:
    """Convert an aggregate legal action policy into selected-node CFR state."""
    regret = np.asarray(regret_sum, dtype=np.float32).copy()
    strategy = np.asarray(strategy_sum, dtype=np.float32).copy()
    if regret.shape != strategy.shape:
        raise ValueError("regret_sum and strategy_sum must have the same shape")
    if regret.ndim != 3:
        raise ValueError("CFR state tensors must have shape (nodes, actions, hands)")
    node = int(node_idx)
    if node < 0 or node >= regret.shape[0]:
        raise ValueError("node_idx is outside the CFR state tensor")
    policy = _normalise_target_policy(np.asarray(target_policy, dtype=np.float32), legal_actions)
    if policy.shape[0] != regret.shape[1]:
        raise ValueError("target_policy action dimension must match CFR state")
    regret[node, :, :] = 0.0
    strategy[node, :, :] = 0.0
    r_mass = max(float(regret_mass), 0.0)
    s_mass = max(float(strategy_mass), 0.0)
    for action in legal_actions:
        action_idx = int(action)
        if 0 <= action_idx < regret.shape[1]:
            regret[node, action_idx, :] = r_mass * float(policy[action_idx])
            strategy[node, action_idx, :] = s_mass * float(policy[action_idx])
    return {"regret_sum": regret, "strategy_sum": strategy}


def _fit_trace_policy_delta(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    low_trace_iteration: int,
    target_trace_iteration: int,
    hidden_dim: int,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str,
    include_advantage_features: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_low = _record_map(train_payload, low_trace_iteration)
    train_target = _record_map(train_payload, target_trace_iteration)
    train_labels = _common_labels(train_low, train_target)
    x_train, _low_train, y_train_delta, train_mask = _build_dataset(
        train_low,
        train_target,
        train_labels,
        include_advantage_features=include_advantage_features,
    )
    x_train, feature_mean, feature_std = _standardize(x_train)
    model, train_metrics = _train_mlp(
        x_train,
        y_train_delta,
        train_mask,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    holdout_low = _record_map(holdout_payload, low_trace_iteration)
    holdout_target = _record_map(holdout_payload, target_trace_iteration)
    holdout_labels = _common_labels(holdout_low, holdout_target)
    x_holdout, low_policies, _holdout_delta, _holdout_mask = _build_dataset(
        holdout_low,
        holdout_target,
        holdout_labels,
        include_advantage_features=include_advantage_features,
    )
    x_holdout, _, _ = _standardize(x_holdout, mean=feature_mean, std=feature_std)
    pred_delta = _predict_delta(model, x_holdout, train_metrics["device"])
    predictions: dict[str, np.ndarray] = {}
    for row_idx, label in enumerate(holdout_labels):
        record = holdout_low[label]
        predictions[label] = _normalize_prediction(
            low_policies[row_idx] + pred_delta[row_idx],
            [int(action) for action in record.get("legal_actions", ())],
            fallback=low_policies[row_idx],
        ).astype(np.float32, copy=False)
    return predictions, {
        "device": train_metrics["device"],
        "n_train": int(len(train_labels)),
        "n_holdout_predictions": int(len(predictions)),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "train_seconds": train_metrics["train_seconds"],
        "include_advantage_features": bool(include_advantage_features),
    }


def _node_state_mass(state: np.ndarray, node_idx: int, legal_actions: list[int]) -> float:
    legal = [int(action) for action in legal_actions if 0 <= int(action) < state.shape[1]]
    if not legal:
        return 1.0
    value = float(np.maximum(state[int(node_idx), legal, :], 0.0).sum())
    per_hand = value / max(1, int(state.shape[2]))
    return max(per_hand, 1.0)


def _solve_with_policy_update_hook(
    case: Any,
    parsed: dict[str, Any],
    *,
    belief_row: np.ndarray,
    solver_iterations: int,
    target_policy: np.ndarray,
    solver_backend: str,
) -> tuple[StreetSolver, Any, SolverDecision, dict[str, Any]] | None:
    backend, _device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("policy update hook diagnostics require solver_backend='cpu'")
    street = int(parsed["st"])
    if street not in (2, 3):
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
    normalised_policy = _normalise_target_policy(target_policy, legal_actions)
    hook_calls: list[int] = []
    started = time.perf_counter()

    def iteration_update_fn(**kwargs: Any) -> dict[str, np.ndarray]:
        hook_calls.append(int(kwargs["iteration"]))
        regret = np.asarray(kwargs["regret_sum"], dtype=np.float32)
        strategy = np.asarray(kwargs["strategy_sum"], dtype=np.float32)
        return apply_policy_update_to_cfr_state(
            regret_sum=regret,
            strategy_sum=strategy,
            node_idx=node_idx,
            legal_actions=legal_actions,
            target_policy=normalised_policy,
            regret_mass=_node_state_mass(regret, node_idx, legal_actions),
            strategy_mass=_node_state_mass(strategy, node_idx, legal_actions),
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
    total_ms = (time.perf_counter() - started) * 1000.0
    decision = _strategy_decision(case, parsed, solver, solved_node, latency_ms=total_ms)
    return solver, solved_node, decision, {
        "hook_calls": hook_calls,
        "target_policy": normalised_policy.round(8).tolist(),
    }


def eval_trace_policy_hook_update(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    start_index: int = 128,
    limit: int = 16,
    low_iterations: int = 5,
    uniform_iterations: int = 10,
    reference_iterations: int = 25,
    target_trace_iteration: int = 24,
    solver_backend: str = "cpu",
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    seed: int = 20260522,
    device: str = "auto",
    include_advantage_features: bool = False,
) -> dict[str, Any]:
    train_payload = _load_payload(train_trace_json)
    holdout_payload = _load_payload(holdout_trace_json)
    predictions, model_metrics = _fit_trace_policy_delta(
        train_payload,
        holdout_payload,
        low_trace_iteration=low_iterations,
        target_trace_iteration=target_trace_iteration,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
        include_advantage_features=include_advantage_features,
    )

    cases = load_cases_json(cases_json)
    dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")

    hook_l1: list[float] = []
    low_l1: list[float] = []
    uniform_l1: list[float] = []
    hook_kl: list[float] = []
    low_kl: list[float] = []
    uniform_kl: list[float] = []
    hook_match: list[bool] = []
    low_match: list[bool] = []
    uniform_match: list[bool] = []
    low_latency: list[float] = []
    uniform_latency: list[float] = []
    hook_latency: list[float] = []
    reference_latency: list[float] = []
    illegal_mass: list[float] = []
    records_out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for local_idx, case in enumerate(_case_slice(cases, start_index=start_index, limit=limit)):
        case_idx = int(start_index) + int(local_idx)
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            skipped.append({"label": case.label, "reason": parsed["error"]})
            continue
        predicted_policy = predictions.get(str(case.label))
        if predicted_policy is None:
            skipped.append({"label": case.label, "reason": "missing_trace_prediction"})
            continue
        belief_row = np.asarray(dataset.belief[case_idx], dtype=np.float32)
        low = _solve_with_belief(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
        )
        uniform = _solve_with_belief(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=uniform_iterations,
            solver_backend=solver_backend,
        )
        reference = _solve_with_belief(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=reference_iterations,
            solver_backend=solver_backend,
        )
        hook = _solve_with_policy_update_hook(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=low_iterations,
            target_policy=predicted_policy,
            solver_backend=solver_backend,
        )
        if low is None or uniform is None or reference is None or hook is None:
            skipped.append({"label": case.label, "reason": "solver_skipped"})
            continue
        low_solver, low_node, low_decision = low
        uniform_solver, uniform_node, uniform_decision = uniform
        reference_solver, reference_node, reference_decision = reference
        hook_solver, hook_node, hook_decision, hook_info = hook
        hook_l1_value = float(np.abs(hook_decision.strategy - reference_decision.strategy).sum())
        low_l1_value = float(np.abs(low_decision.strategy - reference_decision.strategy).sum())
        uniform_l1_value = float(np.abs(uniform_decision.strategy - reference_decision.strategy).sum())
        hook_l1.append(hook_l1_value)
        low_l1.append(low_l1_value)
        uniform_l1.append(uniform_l1_value)
        hook_kl_value = _kl_to_reference(hook_decision.strategy, reference_decision.strategy)
        low_kl_value = _kl_to_reference(low_decision.strategy, reference_decision.strategy)
        uniform_kl_value = _kl_to_reference(uniform_decision.strategy, reference_decision.strategy)
        hook_kl.append(hook_kl_value)
        low_kl.append(low_kl_value)
        uniform_kl.append(uniform_kl_value)
        hook_match.append(_top_match(hook_decision.strategy, reference_decision.strategy))
        low_match.append(_top_match(low_decision.strategy, reference_decision.strategy))
        uniform_match.append(_top_match(uniform_decision.strategy, reference_decision.strategy))
        low_latency.append(float(low_decision.latency_ms))
        uniform_latency.append(float(uniform_decision.latency_ms))
        hook_latency.append(float(hook_decision.latency_ms))
        reference_latency.append(float(reference_decision.latency_ms))
        illegal = _illegal_mass(hook_decision.strategy, hook_node)
        illegal_mass.append(float(illegal))
        records_out.append(
            {
                "label": case.label,
                "case_index": int(case_idx),
                "low_l1_to_reference": round(low_l1_value, 8),
                "uniform_l1_to_reference": round(uniform_l1_value, 8),
                "hook_l1_to_reference": round(hook_l1_value, 8),
                "low_kl_to_reference": low_kl_value,
                "uniform_kl_to_reference": uniform_kl_value,
                "hook_kl_to_reference": hook_kl_value,
                "low_action": int(low_decision.action),
                "uniform_action": int(uniform_decision.action),
                "hook_action": int(hook_decision.action),
                "reference_action": int(reference_decision.action),
                "low_agrees_with_reference": bool(low_match[-1]),
                "uniform_agrees_with_reference": bool(uniform_match[-1]),
                "hook_agrees_with_reference": bool(hook_match[-1]),
                "hook_illegal_mass": illegal,
                "low_latency_ms": round(float(low_decision.latency_ms), 3),
                "uniform_latency_ms": round(float(uniform_decision.latency_ms), 3),
                "hook_latency_ms": round(float(hook_decision.latency_ms), 3),
                "reference_latency_ms": round(float(reference_decision.latency_ms), 3),
                **hook_info,
            }
        )

    if not records_out:
        raise RuntimeError("no cases were evaluated")
    mean_hook_l1 = _mean(hook_l1)
    mean_low_l1 = _mean(low_l1)
    mean_uniform_l1 = _mean(uniform_l1)
    hook_action_rate = _rate(hook_match)
    uniform_action_rate = _rate(uniform_match)
    passed = (
        mean_hook_l1 < mean_low_l1
        and mean_hook_l1 < mean_uniform_l1
        and hook_action_rate >= uniform_action_rate
        and max(illegal_mass) <= 1e-7
    )
    return {
        "mode": "cfr_trace_policy_hook_update",
        "passed": bool(passed),
        "promotion": False,
        "train_trace_json": str(train_trace_json),
        "holdout_trace_json": str(holdout_trace_json),
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "start_index": int(start_index),
        "limit": int(limit),
        "n_evaluated": int(len(records_out)),
        "low_iterations": int(low_iterations),
        "uniform_iterations": int(uniform_iterations),
        "reference_iterations": int(reference_iterations),
        "target_trace_iteration": int(target_trace_iteration),
        "solver_backend": solver_backend,
        "model": model_metrics,
        "mean_hook_l1_to_reference": mean_hook_l1,
        "mean_low_l1_to_reference": mean_low_l1,
        "mean_uniform_l1_to_reference": mean_uniform_l1,
        "mean_hook_kl_to_reference": _mean(hook_kl),
        "mean_low_kl_to_reference": _mean(low_kl),
        "mean_uniform_kl_to_reference": _mean(uniform_kl),
        "hook_action_agreement": hook_action_rate,
        "low_action_agreement": _rate(low_match),
        "uniform_action_agreement": uniform_action_rate,
        "max_hook_illegal_mass": round(float(max(illegal_mass)), 8),
        "mean_hook_latency_ms": _mean(hook_latency),
        "mean_low_latency_ms": _mean(low_latency),
        "mean_uniform_latency_ms": _mean(uniform_latency),
        "mean_reference_latency_ms": _mean(reference_latency),
        "records": records_out,
        "skipped": skipped[:100],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-json", required=True)
    parser.add_argument("--holdout-trace-json", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--uniform-iterations", type=int, default=10)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--target-trace-iteration", type=int, default=24)
    parser.add_argument("--solver-backend", choices=("cpu",), default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-advantage-features", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_trace_policy_hook_update(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        cases_json=args.cases_json,
        cfv_cache=args.cfv_cache,
        start_index=args.start_index,
        limit=args.limit,
        low_iterations=args.low_iterations,
        uniform_iterations=args.uniform_iterations,
        reference_iterations=args.reference_iterations,
        target_trace_iteration=args.target_trace_iteration,
        solver_backend=args.solver_backend,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        include_advantage_features=args.include_advantage_features,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
