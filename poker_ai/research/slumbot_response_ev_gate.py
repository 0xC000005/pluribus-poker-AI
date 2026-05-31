"""Counterfactual-EV replay gate for Slumbot response-range diagnostics.

This module deliberately keeps explicit opponent ranges in a diagnostic role.
It asks whether a learned response range improves the value of the resolver's
chosen mixed strategy under a replay evaluator that knows the revealed Slumbot
private hand. That is still not live strength evidence; it is a falsifier for
whether better range likelihood translates into better decisions.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase
from poker_ai.research.slumbot_response_solver_ab import (
    _hand_index_from_case,
    _range_summary,
    _solve_case_strategy,
    baseline_solver_ranges_for_case,
    load_trace_bot_hands,
    response_villain_range_for_case,
    true_hand_log_lift,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    card_str_to_index,
    parse_action,
)
from solver import StreetSolver, _parse_nav  # noqa: E402


def one_hot_truth_range(
    solver_hands: Iterable[tuple[int, int]],
    *,
    bot_hand: tuple[int, int],
) -> np.ndarray:
    """Return a normalized one-hot range for the revealed Slumbot hand."""
    hands = [tuple(sorted(hand)) for hand in solver_hands]
    true_hand = tuple(sorted(int(card) for card in bot_hand))
    values = np.zeros(len(hands), dtype=np.float64)
    try:
        values[hands.index(true_hand)] = 1.0
    except ValueError:
        return values
    return values


def strategy_ev(strategy: np.ndarray, action_values: np.ndarray) -> float:
    """Score a mixed strategy on a fixed per-action value vector."""
    probs = np.asarray(strategy, dtype=np.float64).reshape(-1)
    values = np.asarray(action_values, dtype=np.float64).reshape(-1)
    n = min(probs.shape[0], values.shape[0])
    if n == 0:
        return 0.0
    probs = np.maximum(probs[:n], 0.0)
    total = float(probs.sum())
    if total <= 1e-12:
        return 0.0
    probs = probs / total
    return float(np.dot(probs, values[:n]))


def summarize_counterfactual_ev_records(
    records: list[dict[str, Any]],
    *,
    min_evaluated: int = 1,
    min_mean_strategy_ev_delta: float = 0.0,
    min_response_beats_rate: float = 0.5,
) -> dict[str, Any]:
    """Summarize replay records without treating range likelihood as promotion."""
    evaluated = [record for record in records if record.get("passed")]
    strategy_deltas = [
        float(record["response_strategy_ev"]) - float(record["baseline_strategy_ev"])
        for record in evaluated
    ]
    selected_deltas = [
        float(record["response_selected_action_ev"])
        - float(record["baseline_selected_action_ev"])
        for record in evaluated
    ]
    range_lifts = [
        float(record["delta_true_hand_log_lift"])
        for record in evaluated
        if "delta_true_hand_log_lift" in record
    ]
    response_beats = [delta > 0.0 for delta in strategy_deltas]
    selected_beats = [delta > 0.0 for delta in selected_deltas]
    mean_strategy_delta = (
        float(np.mean(strategy_deltas)) if strategy_deltas else None
    )
    response_beats_rate = (
        float(np.mean(response_beats)) if response_beats else 0.0
    )
    ev_gate_passed = bool(
        len(evaluated) >= int(min_evaluated)
        and mean_strategy_delta is not None
        and mean_strategy_delta > float(min_mean_strategy_ev_delta)
        and response_beats_rate >= float(min_response_beats_rate)
    )
    blockers = [
        "counterfactual_ev_replay_is_diagnostic_not_slumbot_confidence",
    ]
    if not ev_gate_passed:
        blockers.append("positive_counterfactual_ev_not_demonstrated")
    return {
        "mode": "slumbot_response_range_counterfactual_ev_gate",
        "passed": bool(evaluated and all(record.get("passed") or record.get("skipped") for record in records)),
        "ev_gate_passed": ev_gate_passed,
        "promotable": False,
        "promotion_blockers": blockers,
        "n_cases": len(records),
        "n_evaluated": len(evaluated),
        "min_evaluated": int(min_evaluated),
        "min_mean_strategy_ev_delta": float(min_mean_strategy_ev_delta),
        "min_response_beats_rate": float(min_response_beats_rate),
        "mean_strategy_ev_delta": mean_strategy_delta,
        "median_strategy_ev_delta": (
            float(np.median(strategy_deltas)) if strategy_deltas else None
        ),
        "response_strategy_ev_beats_baseline_rate": response_beats_rate,
        "mean_selected_action_ev_delta": (
            float(np.mean(selected_deltas)) if selected_deltas else None
        ),
        "response_selected_action_ev_beats_baseline_rate": (
            float(np.mean(selected_beats)) if selected_beats else 0.0
        ),
        "mean_delta_true_hand_log_lift": (
            float(np.mean(range_lifts)) if range_lifts else None
        ),
        "true_hand_response_beats_baseline_rate": (
            float(np.mean([delta > 0.0 for delta in range_lifts]))
            if range_lifts
            else None
        ),
    }


def counterfactual_ev_gate_succeeded(metrics: dict[str, Any]) -> bool:
    """Return true only when the mechanical run and EV decision gate pass."""
    return bool(metrics.get("passed") and metrics.get("ev_gate_passed"))


def _case_solver_context(case: ResolverBenchmarkCase, parsed: dict) -> dict[str, Any]:
    street = int(parsed.get("st", -1))
    n_board = 5 if street == 3 else 4
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        int(case.client_pos),
        target_street=street,
    )
    pot = our_bet_pre + opp_bet_pre
    hero_stack = 20000 - our_bet_pre
    villain_stack = 20000 - opp_bet_pre
    street_parts = case.action_str.split("/")
    street_action = street_parts[street] if len(street_parts) > street else ""
    return {
        "street": street,
        "board_idx": board_idx,
        "our_cards_idx": our_cards_idx,
        "pot": pot,
        "hero_stack": hero_stack,
        "villain_stack": villain_stack,
        "hero_first": int(case.client_pos) == 0,
        "street_action": street_action,
    }


def _counterfactual_action_values_for_case(
    case: ResolverBenchmarkCase,
    *,
    hero_range: np.ndarray,
    truth_villain_range: np.ndarray,
    evaluator_iterations: int,
) -> dict[str, Any]:
    parsed = parse_action(case.action_str)
    context = _case_solver_context(case, parsed)
    solver = StreetSolver(
        context["board_idx"],
        context["pot"],
        context["hero_stack"],
        context["villain_stack"],
        context["hero_first"],
    )
    active_node = solver.navigate(_parse_nav(context["street_action"], solver))
    if active_node is None or active_node.is_terminal:
        return {"passed": False, "skipped": "terminal_or_missing_node"}
    if int(active_node.player) != 0:
        return {"passed": False, "skipped": f"not_hero_node:{active_node.player}"}
    hand = tuple(sorted(context["our_cards_idx"]))
    hand_idx = solver.hand_to_idx.get(hand)
    if hand_idx is None:
        return {"passed": False, "skipped": "hero_hand_not_in_solver"}
    if float(np.asarray(truth_villain_range).sum()) <= 1e-12:
        return {"passed": False, "skipped": "truth_hand_not_in_solver"}

    node_idx = solver._tree["all_nodes"].index(active_node)
    legal_actions = tuple(sorted(int(action) for action in active_node.children.keys()))
    latest: dict[str, Any] = {}

    def collect_trace(**kwargs: Any) -> None:
        hero_action_values = np.asarray(kwargs["hero_action_values"], dtype=np.float64)[0]
        action_values = np.zeros(N_ACTIONS, dtype=np.float64)
        for action in legal_actions:
            action_values[int(action)] = float(hero_action_values[int(action), hand_idx])
        latest["action_values"] = action_values
        latest["iteration"] = int(kwargs["iteration"])

    solver.solve(
        n_iterations=int(evaluator_iterations),
        hero_range=hero_range,
        villain_range=truth_villain_range,
        backend="cpu",
        trace_node_indices=[node_idx],
        trace_node_fn=collect_trace,
    )
    if "action_values" not in latest:
        return {"passed": False, "skipped": "no_trace_values"}
    return {
        "passed": True,
        "action_values": latest["action_values"],
        "legal_actions": legal_actions,
        "evaluator_iteration": latest["iteration"],
        "evaluator_solve_ms": float(getattr(solver, "last_solve_ms", 0.0)),
    }


def evaluate_response_range_counterfactual_ev(
    value_net: ValueNetwork,
    response_model: torch.nn.Module,
    response_mean: np.ndarray,
    response_std: np.ndarray,
    response_temperature: float,
    device: torch.device,
    *,
    cases: Iterable[ResolverBenchmarkCase],
    strategy_source: str,
    solver_iterations: int = 5,
    policy_solver_backend: str = "auto",
    evaluator_iterations: int = 10,
    range_prune_threshold: float = 1e-4,
    bot_hands_by_hand_index: dict[int, tuple[int, int]],
    min_evaluated: int = 1,
    min_mean_strategy_ev_delta: float = 0.0,
    min_response_beats_rate: float = 0.5,
) -> dict[str, Any]:
    """Evaluate whether response ranges improve replay counterfactual EV."""
    records: list[dict[str, Any]] = []
    for case in cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed or int(parsed.get("st", -1)) not in (2, 3):
            records.append({**asdict(case), "passed": False, "skipped": "invalid_case"})
            continue
        hand_index = _hand_index_from_case(case)
        bot_hand = (
            bot_hands_by_hand_index.get(hand_index)
            if hand_index is not None
            else None
        )
        if bot_hand is None:
            records.append({**asdict(case), "passed": False, "skipped": "missing_revealed_bot_hand"})
            continue
        baseline_ranges = baseline_solver_ranges_for_case(
            case,
            value_net=value_net,
            device=device,
            strategy_source=strategy_source,
        )
        response_range = response_villain_range_for_case(
            case,
            model=response_model,
            mean=response_mean,
            std=response_std,
            temperature=response_temperature,
            device=device,
        )
        if baseline_ranges.solver_hands != response_range.solver_hands:
            records.append({**asdict(case), "passed": False, "skipped": "hand_index_mismatch"})
            continue
        truth_range = one_hot_truth_range(response_range.solver_hands, bot_hand=bot_hand)
        evaluator = _counterfactual_action_values_for_case(
            case,
            hero_range=baseline_ranges.hero_range,
            truth_villain_range=truth_range,
            evaluator_iterations=evaluator_iterations,
        )
        if not evaluator.get("passed"):
            records.append({**asdict(case), **evaluator})
            continue
        baseline = _solve_case_strategy(
            case,
            hero_range=baseline_ranges.hero_range,
            villain_range=baseline_ranges.villain_range,
            solver_iterations=solver_iterations,
            solver_backend=policy_solver_backend,
            range_prune_threshold=range_prune_threshold,
        )
        response = _solve_case_strategy(
            case,
            hero_range=baseline_ranges.hero_range,
            villain_range=response_range.villain_range,
            solver_iterations=solver_iterations,
            solver_backend=policy_solver_backend,
            range_prune_threshold=range_prune_threshold,
        )
        action_values = np.asarray(evaluator["action_values"], dtype=np.float64)
        baseline_strategy_ev = strategy_ev(baseline["strategy"], action_values)
        response_strategy_ev = strategy_ev(response["strategy"], action_values)
        baseline_action = int(baseline["action"])
        response_action = int(response["action"])
        baseline_truth = true_hand_log_lift(
            baseline_ranges.solver_hands,
            baseline_ranges.villain_range,
            bot_hand=bot_hand,
        )
        response_truth = true_hand_log_lift(
            response_range.solver_hands,
            response_range.villain_range,
            bot_hand=bot_hand,
        )
        record = {
            **asdict(case),
            "street": int(parsed.get("st", -1)),
            "passed": True,
            "baseline_action": baseline_action,
            "response_action": response_action,
            "action_agreement": bool(baseline_action == response_action),
            "action_l1_drift": float(np.abs(baseline["strategy"] - response["strategy"]).sum()),
            "baseline_strategy_ev": baseline_strategy_ev,
            "response_strategy_ev": response_strategy_ev,
            "strategy_ev_delta": float(response_strategy_ev - baseline_strategy_ev),
            "baseline_selected_action_ev": float(action_values[baseline_action]),
            "response_selected_action_ev": float(action_values[response_action]),
            "selected_action_ev_delta": float(
                action_values[response_action] - action_values[baseline_action]
            ),
            "counterfactual_action_values": action_values.round(6).tolist(),
            "legal_actions": list(evaluator["legal_actions"]),
            "evaluator_iterations": int(evaluator_iterations),
            "evaluator_solve_ms": round(float(evaluator["evaluator_solve_ms"]), 3),
            "policy_solver_backend": policy_solver_backend,
            "baseline_strategy": baseline["strategy"].round(6).tolist(),
            "response_strategy": response["strategy"].round(6).tolist(),
            "baseline_latency_ms": round(float(baseline["latency_ms"]), 3),
            "response_latency_ms": round(float(response["latency_ms"]), 3),
            "n_opponent_updates": int(response_range.n_opponent_updates),
            **_range_summary("baseline_villain_range", baseline_ranges.villain_range),
            **_range_summary("response_villain_range", response_range.villain_range),
        }
        if baseline_truth is not None and response_truth is not None:
            for key, value in baseline_truth.items():
                record[f"baseline_{key}"] = value
            for key, value in response_truth.items():
                record[f"response_{key}"] = value
            record["delta_true_hand_log_lift"] = float(
                response_truth["log_lift_vs_uniform"]
                - baseline_truth["log_lift_vs_uniform"]
            )
        records.append(record)

    summary = summarize_counterfactual_ev_records(
        records,
        min_evaluated=min_evaluated,
        min_mean_strategy_ev_delta=min_mean_strategy_ev_delta,
        min_response_beats_rate=min_response_beats_rate,
    )
    summary.update(
        {
            "solver_iterations": int(solver_iterations),
            "policy_solver_backend": policy_solver_backend,
            "evaluator_iterations": int(evaluator_iterations),
            "range_prune_threshold": float(range_prune_threshold),
            "strategy_source": strategy_source,
            "records": records,
        }
    )
    return summary
