#!/usr/bin/env python3
"""Evaluate a policy-head CFR+ warm start against a stronger solver reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.research.belief_value_probe import save_metrics
from poker_ai.research.evaluation import load_value_network_checkpoint
from poker_ai.research.resolver_benchmark import (
    SolverDecision,
    _solver_decision,
    load_cases_json,
)
from play_slumbot import (
    _compute_bets_before_street,
    build_features,
    card_str_to_index,
    parse_action,
)
from solver import StreetSolver, _parse_nav, solver_action_to_slumbot


_SUITS = ("c", "d", "h", "s")
_RANKS = tuple("23456789TJQKA")


def _card_to_str(card: int) -> str:
    return _RANKS[int(card) // 4] + _SUITS[int(card) % 4]


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _rate(values: list[bool]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def build_policy_warm_start(
    *,
    solver: StreetSolver,
    node_idx: int,
    policy_by_action_hand: np.ndarray,
    regret_mass: float,
    strategy_mass: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build CFR+ initializer tensors from a policy prior at one public node."""
    shape = (
        solver._tree["n_nodes"],
        solver._tree["n_actions"],
        solver.n,
    )
    policy = np.asarray(policy_by_action_hand, dtype=np.float32)
    if policy.shape != (shape[1], shape[2]):
        raise ValueError(f"policy_by_action_hand shape {policy.shape} != {(shape[1], shape[2])}")
    if not np.isfinite(policy).all() or (policy < 0).any():
        raise ValueError("policy_by_action_hand must be finite and non-negative")
    regret = np.zeros(shape, dtype=np.float32)
    strategy = np.zeros(shape, dtype=np.float32)
    regret[node_idx] = float(regret_mass) * policy
    if strategy_mass > 0:
        strategy[node_idx] = float(strategy_mass) * policy
    return regret, strategy


def _solver_context(case, parsed: dict) -> tuple[list[int], int, int, int, bool, str]:
    street = int(parsed["st"])
    n_board = 4 if street == 2 else 5
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
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
    return board_idx, pot, hero_stack, villain_stack, hero_first, street_action


def _policy_prior_for_solver_hands(
    *,
    value_net,
    device: torch.device,
    case,
    parsed: dict,
    solver: StreetSolver,
    node,
    batch_size: int = 256,
) -> np.ndarray:
    actions = sorted(node.children.keys())
    legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for action in actions:
        if 0 <= action < N_ACTIONS:
            legal_mask[action] = 1.0
    if legal_mask.sum() <= 0:
        raise ValueError("warm-start node has no legal action mask")

    priors = np.zeros((solver._tree["n_actions"], solver.n), dtype=np.float32)
    features = []
    hand_indices = []
    for hand_idx, hand in enumerate(solver.hands):
        hole_cards = [_card_to_str(hand[0]), _card_to_str(hand[1])]
        features.append(
            build_features(
                hole_cards,
                list(case.board),
                case.action_str,
                case.client_pos,
                parsed,
            )
        )
        hand_indices.append(hand_idx)
        if len(features) >= batch_size:
            _fill_policy_batch(value_net, device, features, hand_indices, legal_mask, priors)
            features = []
            hand_indices = []
    if features:
        _fill_policy_batch(value_net, device, features, hand_indices, legal_mask, priors)
    return priors


def _fill_policy_batch(
    value_net,
    device: torch.device,
    features: list[np.ndarray],
    hand_indices: list[int],
    legal_mask: np.ndarray,
    priors: np.ndarray,
) -> None:
    feat_t = torch.from_numpy(np.stack(features).astype(np.float32)).to(device)
    with torch.no_grad():
        _, logits_t = value_net.forward_with_policy(feat_t)
    logits = logits_t.detach().cpu().numpy().astype(np.float64)
    masked = np.where(legal_mask[np.newaxis, :] > 0, logits, -1e9)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    probs = np.exp(shifted) * legal_mask[np.newaxis, :]
    totals = probs.sum(axis=1, keepdims=True)
    probs = np.where(totals > 0, probs / totals, legal_mask[np.newaxis, :] / legal_mask.sum())
    for row, hand_idx in enumerate(hand_indices):
        priors[:N_ACTIONS, hand_idx] = probs[row].astype(np.float32)


def _warm_start_decision(
    *,
    value_net,
    device: torch.device,
    case,
    parsed: dict,
    solver_iterations: int,
    solver_backend: str,
    regret_mass_scale: float,
    strategy_mass: float,
    policy_batch_size: int,
) -> SolverDecision | None:
    street = int(parsed["st"])
    if street not in (2, 3):
        return None
    board_idx, pot, hero_stack, villain_stack, hero_first, street_action = _solver_context(case, parsed)
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    nav = _parse_nav(street_action, solver)
    node = solver.navigate(nav)
    if node is None or node.is_terminal:
        return None
    node_idx = solver._tree["all_nodes"].index(node)
    policy = _policy_prior_for_solver_hands(
        value_net=value_net,
        device=device,
        case=case,
        parsed=parsed,
        solver=solver,
        node=node,
        batch_size=policy_batch_size,
    )
    regret_mass = float(regret_mass_scale) * float(max(hero_stack, villain_stack, pot, 1))
    initial_regret, initial_strategy = build_policy_warm_start(
        solver=solver,
        node_idx=node_idx,
        policy_by_action_hand=policy,
        regret_mass=regret_mass,
        strategy_mass=strategy_mass,
    )
    solver.solve(
        n_iterations=solver_iterations,
        backend=solver_backend,
        initial_regret_sum=initial_regret,
        initial_strategy_sum=initial_strategy,
    )

    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    strategy = solver.get_strategy(tuple(sorted(our_cards_idx)), node)
    strategy_vec = np.zeros(N_ACTIONS, dtype=np.float64)
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
    return SolverDecision(action, increment, strategy_vec, 0.0, False)


def eval_policy_warm_start_solver_budget(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    device: str = "auto",
    low_iterations: int = 5,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    regret_mass_scale: float = 1.0,
    strategy_mass: float = 0.0,
    policy_batch_size: int = 256,
    max_warm_allin_rate: float = 0.05,
) -> dict[str, Any]:
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    loaded = load_value_network_checkpoint(checkpoint, torch_device)
    cases = load_cases_json(cases_json)
    records = []
    for case in cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append({"label": case.label, "passed": False, "skipped": parsed["error"]})
            continue
        low = _solver_decision(
            case,
            parsed,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
        )
        reference = _solver_decision(
            case,
            parsed,
            solver_iterations=reference_iterations,
            solver_backend=solver_backend,
        )
        warm = _warm_start_decision(
            value_net=loaded.value_net,
            device=torch_device,
            case=case,
            parsed=parsed,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
            regret_mass_scale=regret_mass_scale,
            strategy_mass=strategy_mass,
            policy_batch_size=policy_batch_size,
        )
        if low is None or reference is None or warm is None:
            records.append({"label": case.label, "passed": False, "skipped": "solver_skipped"})
            continue
        records.append(
            {
                "label": case.label,
                "passed": True,
                "low_action": int(np.argmax(low.strategy)),
                "reference_action": int(np.argmax(reference.strategy)),
                "warm_action": int(np.argmax(warm.strategy)),
                "low_l1_to_reference": round(float(np.abs(low.strategy - reference.strategy).sum()), 8),
                "warm_l1_to_reference": round(float(np.abs(warm.strategy - reference.strategy).sum()), 8),
                "low_allin_selected": bool(int(np.argmax(low.strategy)) == 8),
                "reference_allin_selected": bool(int(np.argmax(reference.strategy)) == 8),
                "warm_allin_selected": bool(int(np.argmax(warm.strategy)) == 8),
                "low_agrees_with_reference": bool(
                    int(np.argmax(low.strategy)) == int(np.argmax(reference.strategy))
                ),
                "warm_agrees_with_reference": bool(
                    int(np.argmax(warm.strategy)) == int(np.argmax(reference.strategy))
                ),
            }
        )
    evaluated = [record for record in records if record.get("passed")]
    low_l1 = [float(record["low_l1_to_reference"]) for record in evaluated]
    warm_l1 = [float(record["warm_l1_to_reference"]) for record in evaluated]
    warm_allin_rate = _rate([bool(record["warm_allin_selected"]) for record in evaluated])
    return {
        "mode": "policy_warm_start_solver_budget",
        "checkpoint": str(checkpoint),
        "cases_json": str(cases_json),
        "device": str(torch_device),
        "solver_backend": solver_backend,
        "low_iterations": int(low_iterations),
        "reference_iterations": int(reference_iterations),
        "regret_mass_scale": float(regret_mass_scale),
        "strategy_mass": float(strategy_mass),
        "max_warm_allin_rate": float(max_warm_allin_rate),
        "n_cases": int(len(records)),
        "n_evaluated": int(len(evaluated)),
        "passed": bool(
            evaluated
            and _mean(warm_l1) < _mean(low_l1)
            and warm_allin_rate <= float(max_warm_allin_rate)
        ),
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_warm_l1_to_reference": _mean(warm_l1),
        "low_action_agreement": _rate([bool(record["low_agrees_with_reference"]) for record in evaluated]),
        "warm_action_agreement": _rate(
            [bool(record["warm_agrees_with_reference"]) for record in evaluated]
        ),
        "low_allin_rate": _rate([bool(record["low_allin_selected"]) for record in evaluated]),
        "reference_allin_rate": _rate(
            [bool(record["reference_allin_selected"]) for record in evaluated]
        ),
        "warm_allin_rate": warm_allin_rate,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare low-iteration solver and policy-head CFR+ warm start against a stronger solver."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", default="cpu")
    parser.add_argument("--regret-mass-scale", type=float, default=1.0)
    parser.add_argument("--strategy-mass", type=float, default=0.0)
    parser.add_argument("--policy-batch-size", type=int, default=256)
    parser.add_argument("--max-warm-allin-rate", type=float, default=0.05)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_policy_warm_start_solver_budget(
        checkpoint=args.checkpoint,
        cases_json=args.cases_json,
        device=args.device,
        low_iterations=args.low_iterations,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
        regret_mass_scale=args.regret_mass_scale,
        strategy_mass=args.strategy_mass,
        policy_batch_size=args.policy_batch_size,
        max_warm_allin_rate=args.max_warm_allin_rate,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
