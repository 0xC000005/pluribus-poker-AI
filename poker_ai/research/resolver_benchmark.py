"""Fixed public-state benchmarks for turn/river resolver quality.

This module deliberately measures mechanics that are cheap and repeatable:
legal-mask agreement, blueprint-vs-resolver action drift, learned-advantage
proxies, and resolver latency. It is not a replacement for Slumbot confidence
intervals, but it gives the autoresearch loop a harder local gate than random
opponent play.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.deep_cfr.networks import ValueNetwork


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    action_to_slumbot,
    build_features,
    card_str_to_index,
    get_legal_mask_from_parsed,
    parse_action,
    regret_match,
)
from range_tracker import map_slumbot_action_to_idx  # noqa: E402
from solver import StreetSolver, _parse_nav, solver_action_to_slumbot  # noqa: E402


@dataclass(frozen=True)
class ResolverBenchmarkCase:
    label: str
    hole_cards: tuple[str, str]
    board: tuple[str, ...]
    action_str: str
    client_pos: int
    source: str = "fixed"


@dataclass(frozen=True)
class PolicyDecision:
    action: int
    increment: str
    strategy: np.ndarray
    advantages: np.ndarray
    legal_mask: np.ndarray


@dataclass(frozen=True)
class SolverDecision:
    action: int
    increment: str
    strategy: np.ndarray
    latency_ms: float
    node_terminal: bool


def default_benchmark_cases() -> list[ResolverBenchmarkCase]:
    """Return deterministic turn/river spots that stress common resolver paths."""
    return [
        ResolverBenchmarkCase(
            label="turn-open-check",
            hole_cards=("Ac", "Kd"),
            board=("2c", "7d", "Jh", "4s"),
            action_str="ck/kk/",
            client_pos=0,
        ),
        ResolverBenchmarkCase(
            label="turn-facing-bet",
            hole_cards=("Qs", "Qd"),
            board=("2h", "8c", "Td", "3s"),
            action_str="ck/kk/b300",
            client_pos=1,
        ),
        ResolverBenchmarkCase(
            label="river-open-check",
            hole_cards=("Ac", "Kd"),
            board=("2c", "7d", "Jh", "4s", "9c"),
            action_str="ck/kk/kk/",
            client_pos=0,
        ),
        ResolverBenchmarkCase(
            label="river-facing-bet",
            hole_cards=("As", "Qc"),
            board=("3c", "6d", "Jd", "4h", "Ts"),
            action_str="ck/kk/kk/b600",
            client_pos=1,
        ),
    ]


def load_cases_json(path: str | Path) -> list[ResolverBenchmarkCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    return [
        ResolverBenchmarkCase(
            label=str(item["label"]),
            hole_cards=tuple(item["hole_cards"]),
            board=tuple(item["board"]),
            action_str=str(item["action_str"]),
            client_pos=int(item["client_pos"]),
            source=str(item.get("source", path)),
        )
        for item in cases
    ]


def _increment_char_and_bet_to(increment: str) -> tuple[str, int]:
    if not increment:
        return "", 0
    if increment[0] == "b" and increment[1:].isdigit():
        return "b", int(increment[1:])
    return increment[0], 0


def _mapped_increment_legal(increment: str, case: ResolverBenchmarkCase, parsed: dict, mask: np.ndarray) -> bool:
    action_char, bet_to = _increment_char_and_bet_to(increment)
    if not action_char:
        return False
    mapped = map_slumbot_action_to_idx(
        action_char,
        bet_to,
        case.action_str,
        case.client_pos,
        parsed,
    )
    return any(mask[idx] > 0 and weight > 0 for idx, weight in mapped)


def _policy_decision(
    value_net: ValueNetwork,
    device: torch.device,
    case: ResolverBenchmarkCase,
    parsed: dict,
) -> PolicyDecision:
    features = build_features(
        list(case.hole_cards),
        list(case.board),
        case.action_str,
        case.client_pos,
        parsed,
    )
    legal_mask = get_legal_mask_from_parsed(parsed, case.action_str, case.client_pos)
    with torch.no_grad():
        advantages = (
            value_net(torch.from_numpy(features).unsqueeze(0).to(device))
            .cpu()
            .numpy()[0]
            .astype(np.float64)
        )
    strategy = regret_match(advantages, legal_mask).astype(np.float64)
    action = int(np.argmax(strategy))
    increment = action_to_slumbot(action, parsed, case.action_str, case.client_pos)
    return PolicyDecision(action, increment, strategy, advantages, legal_mask)


def _solver_decision(
    case: ResolverBenchmarkCase,
    parsed: dict,
    *,
    solver_iterations: int,
) -> SolverDecision | None:
    street = int(parsed["st"])
    if street not in (2, 3):
        return None

    n_board = 4 if street == 2 else 5
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
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
    street_parts = case.action_str.split("/")
    street_action = street_parts[street] if len(street_parts) > street else ""

    started = time.perf_counter()
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    solver.solve(n_iterations=solver_iterations)
    nav = _parse_nav(street_action, solver)
    node = solver.navigate(nav)

    strategy_vec = np.zeros(N_ACTIONS, dtype=np.float64)
    if node is None or node.is_terminal:
        action = 1
        strategy_vec[1] = 1.0
        increment = "k" if parsed["last_bet_size"] == 0 else "c"
        node_terminal = True
    else:
        strategy = solver.get_strategy(tuple(sorted(our_cards_idx)), node)
        if strategy:
            for action_idx, prob in strategy.items():
                if 0 <= action_idx < N_ACTIONS:
                    strategy_vec[action_idx] = float(prob)
            total = strategy_vec.sum()
            if total > 0:
                strategy_vec /= total
            else:
                strategy_vec[1] = 1.0
        else:
            strategy_vec[1] = 1.0
        action = int(np.argmax(strategy_vec))
        increment = solver_action_to_slumbot(action, node, solver, parsed)
        node_terminal = False

    latency_ms = (time.perf_counter() - started) * 1000.0
    return SolverDecision(action, increment, strategy_vec, latency_ms, node_terminal)


def _validate_case(case: ResolverBenchmarkCase, parsed: dict) -> list[str]:
    errors: list[str] = []
    if "error" in parsed:
        errors.append(f"parse_error:{parsed['error']}")
        return errors
    if parsed.get("pos") != case.client_pos:
        errors.append(f"not_hero_turn:pos={parsed.get('pos')}:client_pos={case.client_pos}")
    street = int(parsed.get("st", -1))
    required_board = 4 if street == 2 else 5 if street == 3 else None
    if required_board is None:
        errors.append(f"unsupported_street:{street}")
    elif len(case.board) < required_board:
        errors.append(f"board_too_short:{len(case.board)}<{required_board}")
    cards = [*case.hole_cards, *case.board]
    if len(cards) != len(set(cards)):
        errors.append("duplicate_cards")
    return errors


def _case_metrics(
    value_net: ValueNetwork,
    device: torch.device,
    case: ResolverBenchmarkCase,
    *,
    solver_iterations: int,
) -> dict[str, Any]:
    parsed = parse_action(case.action_str)
    validation_errors = _validate_case(case, parsed)
    base = {
        **asdict(case),
        "street": parsed.get("st"),
        "validation_errors": validation_errors,
    }
    if validation_errors:
        return {
            **base,
            "passed": False,
            "blueprint_action_legal": False,
            "solver_action_legal": False,
            "solver_increment_legal": False,
        }

    policy = _policy_decision(value_net, device, case, parsed)
    solver = _solver_decision(case, parsed, solver_iterations=solver_iterations)
    if solver is None:
        return {
            **base,
            "passed": False,
            "validation_errors": ["solver_skipped"],
            "blueprint_action": policy.action,
            "blueprint_action_legal": bool(policy.legal_mask[policy.action] > 0),
            "solver_action_legal": False,
            "solver_increment_legal": False,
        }

    blueprint_legal = bool(policy.legal_mask[policy.action] > 0)
    solver_legal = bool(policy.legal_mask[solver.action] > 0)
    solver_increment_legal = _mapped_increment_legal(
        solver.increment,
        case,
        parsed,
        policy.legal_mask,
    )
    finite = (
        np.isfinite(policy.strategy).all()
        and np.isfinite(policy.advantages).all()
        and np.isfinite(solver.strategy).all()
        and math.isfinite(solver.latency_ms)
    )

    action_l1_drift = float(np.abs(policy.strategy - solver.strategy).sum())
    advantage_delta = float(policy.advantages[solver.action] - policy.advantages[policy.action])
    return {
        **base,
        "passed": bool(finite and blueprint_legal and solver_legal and solver_increment_legal),
        "blueprint_action": policy.action,
        "blueprint_increment": policy.increment,
        "blueprint_action_legal": blueprint_legal,
        "solver_action": solver.action,
        "solver_increment": solver.increment,
        "solver_action_legal": solver_legal,
        "solver_increment_legal": solver_increment_legal,
        "solver_latency_ms": round(float(solver.latency_ms), 3),
        "solver_node_terminal": solver.node_terminal,
        "action_l1_drift": action_l1_drift,
        "advantage_delta_proxy": advantage_delta,
        "blueprint_advantage_proxy": float(policy.advantages[policy.action]),
        "solver_advantage_proxy": float(policy.advantages[solver.action]),
        "blueprint_strategy": policy.strategy.round(6).tolist(),
        "solver_strategy": solver.strategy.round(6).tolist(),
    }


def run_resolver_benchmark(
    value_net: ValueNetwork,
    device: torch.device,
    *,
    cases: Iterable[ResolverBenchmarkCase] | None = None,
    solver_iterations: int = 25,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value_net.eval()
    selected_cases = list(cases) if cases is not None else default_benchmark_cases()
    results = [
        _case_metrics(
            value_net,
            device,
            case,
            solver_iterations=solver_iterations,
        )
        for case in selected_cases
    ]
    solver_results = [item for item in results if "solver_latency_ms" in item]
    latency_values = [float(item["solver_latency_ms"]) for item in solver_results]
    drift_values = [float(item["action_l1_drift"]) for item in solver_results]
    advantage_deltas = [float(item["advantage_delta_proxy"]) for item in solver_results]
    metadata = checkpoint_metadata or {}
    return {
        "passed": all(bool(item["passed"]) for item in results),
        "mode": "fixed_public_state_resolver_benchmark",
        "n_cases": len(results),
        "n_solver_cases": len(solver_results),
        "solver_iterations": int(solver_iterations),
        "checkpoint": metadata.get("checkpoint"),
        "checkpoint_iteration": metadata.get("checkpoint_iteration", metadata.get("iteration")),
        "hidden_dim": metadata.get("hidden_dim"),
        "n_layers": metadata.get("n_layers"),
        "avg_solver_latency_ms": float(np.mean(latency_values)) if latency_values else 0.0,
        "max_solver_latency_ms": float(np.max(latency_values)) if latency_values else 0.0,
        "mean_action_l1_drift": float(np.mean(drift_values)) if drift_values else 0.0,
        "mean_advantage_delta_proxy": (
            float(np.mean(advantage_deltas)) if advantage_deltas else 0.0
        ),
        "illegal_case_count": sum(
            1
            for item in results
            if not item.get("blueprint_action_legal")
            or not item.get("solver_action_legal")
            or not item.get("solver_increment_legal")
        ),
        "promotion_blockers": [
            "fixed_state_resolver_benchmark_is_not_slumbot_confidence",
        ],
        "cases": results,
    }
