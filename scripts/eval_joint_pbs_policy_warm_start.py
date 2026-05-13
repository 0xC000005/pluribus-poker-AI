#!/usr/bin/env python3
"""Evaluate joint-PBS policy priors as CFR+ warm starts."""

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

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import SolverDecision, load_cases_json

from build_dynamic_successor_cut_pbs_targets import _card_to_str  # noqa: E402
from eval_joint_pbs_continuation_probe import (  # noqa: E402
    _DEFAULT_MAX_ACTION_TOKENS,
    _encode_action_sequence,
    load_joint_pbs_continuation_checkpoint,
    predict_joint_pbs_policy_model,
)
from eval_joint_pbs_resolver_leaf_ab import _local_ranges_from_belief, _strategy_vector  # noqa: E402
from eval_joint_pbs_root_policy import _case_slice  # noqa: E402
from eval_policy_warm_start_solver_budget import build_policy_warm_start  # noqa: E402
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    build_features,
    card_str_to_index,
    parse_action,
)
from solver import StreetSolver, _parse_nav, resolve_solver_backend, solver_action_to_slumbot  # noqa: E402


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _rate(values: list[bool]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _solver_context(case, parsed: dict) -> tuple[list[int], list[int], int, int, int, bool, str]:
    street = int(parsed["st"])
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
    street_action = case.action_str.split("/")[street] if len(case.action_str.split("/")) > street else ""
    return board_idx, our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action


def _strategy_decision(case, parsed: dict, solver: StreetSolver, node: Any) -> SolverDecision:
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    strategy_vec = _strategy_vector(solver.get_strategy(tuple(sorted(our_cards_idx)), node))
    action = int(np.argmax(strategy_vec))
    increment = solver_action_to_slumbot(action, node, solver, parsed)
    return SolverDecision(action, increment, strategy_vec, 0.0, False)


def _solve_with_belief(
    case,
    parsed: dict,
    *,
    belief_row: np.ndarray,
    solver_iterations: int,
    solver_backend: str,
    initial_regret_sum: np.ndarray | None = None,
    initial_strategy_sum: np.ndarray | None = None,
) -> tuple[StreetSolver, Any, SolverDecision] | None:
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
        initial_regret_sum=initial_regret_sum,
        initial_strategy_sum=initial_strategy_sum,
    )
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None or node.is_terminal:
        return None
    return solver, node, _strategy_decision(case, parsed, solver, node)


def _joint_policy_prior_for_solver_hands(
    *,
    model: Any,
    payload: dict[str, Any],
    device: Any,
    case: Any,
    parsed: dict,
    belief_row: np.ndarray,
    solver: StreetSolver,
    node: Any,
) -> np.ndarray:
    legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for action in sorted(node.children.keys()):
        if 0 <= int(action) < N_ACTIONS:
            legal_mask[int(action)] = 1.0
    if legal_mask.sum() <= 0:
        raise ValueError("warm-start node has no legal actions")
    public_feature = build_features(
        [],
        list(case.board),
        case.action_str,
        int(case.client_pos),
        parsed,
    ).astype(np.float32, copy=False)
    max_tokens = int(payload.get("max_action_tokens", _DEFAULT_MAX_ACTION_TOKENS))
    action_tokens, action_amounts = _encode_action_sequence(case.action_str, max_tokens=max_tokens)
    policy_features = []
    for hand in solver.hands:
        hole_cards = [_card_to_str(hand[0]), _card_to_str(hand[1])]
        policy_features.append(
            build_features(
                hole_cards,
                list(case.board),
                case.action_str,
                int(case.client_pos),
                parsed,
            ).astype(np.float32, copy=False)
        )
    n = len(policy_features)
    probs = predict_joint_pbs_policy_model(
        model,
        payload,
        np.repeat(public_feature[np.newaxis, :], n, axis=0),
        np.stack(policy_features).astype(np.float32),
        np.repeat(np.asarray(belief_row, dtype=np.float32)[np.newaxis, :], n, axis=0),
        np.repeat(legal_mask[np.newaxis, :], n, axis=0),
        action_tokens=np.repeat(action_tokens[np.newaxis, :], n, axis=0),
        action_amounts=np.repeat(action_amounts[np.newaxis, :], n, axis=0),
        device=device,
    )
    priors = np.zeros((solver._tree["n_actions"], solver.n), dtype=np.float32)
    priors[:N_ACTIONS, :] = probs.T.astype(np.float32, copy=False)
    return priors


def _warm_start_decision(
    *,
    model: Any,
    payload: dict[str, Any],
    device: Any,
    case: Any,
    parsed: dict,
    belief_row: np.ndarray,
    solver_iterations: int,
    solver_backend: str,
    regret_mass_scale: float,
    strategy_mass: float,
) -> SolverDecision | None:
    board_idx, _our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action = _solver_context(
        case,
        parsed,
    )
    probe_solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    node = probe_solver.navigate(_parse_nav(street_action, probe_solver))
    if node is None or node.is_terminal:
        return None
    node_idx = probe_solver._tree["all_nodes"].index(node)
    policy = _joint_policy_prior_for_solver_hands(
        model=model,
        payload=payload,
        device=device,
        case=case,
        parsed=parsed,
        belief_row=belief_row,
        solver=probe_solver,
        node=node,
    )
    regret_mass = float(regret_mass_scale) * float(max(hero_stack, villain_stack, pot, 1))
    initial_regret, initial_strategy = build_policy_warm_start(
        solver=probe_solver,
        node_idx=node_idx,
        policy_by_action_hand=policy,
        regret_mass=regret_mass,
        strategy_mass=strategy_mass,
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
    return solved[2] if solved is not None else None


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [record for record in records if record.get("passed")]
    low_l1 = [float(record["low_l1_to_reference"]) for record in evaluated]
    warm_l1 = [float(record["warm_l1_to_reference"]) for record in evaluated]
    low_allin = _rate([bool(record["low_allin_selected"]) for record in evaluated])
    warm_allin = _rate([bool(record["warm_allin_selected"]) for record in evaluated])
    reference_allin = _rate([bool(record["reference_allin_selected"]) for record in evaluated])
    low_agreement = _rate([bool(record["low_agrees_with_reference"]) for record in evaluated])
    warm_agreement = _rate([bool(record["warm_agrees_with_reference"]) for record in evaluated])
    return {
        "n_evaluated": int(len(evaluated)),
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_warm_l1_to_reference": _mean(warm_l1),
        "low_action_agreement": low_agreement,
        "warm_action_agreement": warm_agreement,
        "low_allin_rate": low_allin,
        "warm_allin_rate": warm_allin,
        "reference_allin_rate": reference_allin,
        "low_allin_gap": round(abs(low_allin - reference_allin), 8),
        "warm_allin_gap": round(abs(warm_allin - reference_allin), 8),
        "passed": bool(
            evaluated
            and _mean(warm_l1) < _mean(low_l1)
            and warm_agreement >= low_agreement
            and abs(warm_allin - reference_allin) <= abs(low_allin - reference_allin)
        ),
    }


def eval_joint_pbs_policy_warm_start(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    device: str = "auto",
    start_index: int = 128,
    limit: int = 64,
    low_iterations: int = 5,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    regret_mass_scale: float = 1.0,
    strategy_mass: float = 0.0,
) -> dict[str, Any]:
    import torch

    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    model, payload = load_joint_pbs_continuation_checkpoint(checkpoint, device=torch_device)
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)
    records = []
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
        warm = _warm_start_decision(
            model=model,
            payload=payload,
            device=torch_device,
            case=case,
            parsed=parsed,
            belief_row=belief_row,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
            regret_mass_scale=regret_mass_scale,
            strategy_mass=strategy_mass,
        )
        if low_solved is None or reference_solved is None or warm is None:
            records.append({"label": case.label, "passed": False, "skipped": "solver_skipped"})
            continue
        low = low_solved[2]
        reference = reference_solved[2]
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
    summary = _summarize_records(records)
    return {
        "mode": "joint_pbs_policy_warm_start_solver_budget",
        "checkpoint": str(checkpoint),
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "device": str(torch_device),
        "start_index": int(start_index),
        "limit": int(limit),
        "low_iterations": int(low_iterations),
        "reference_iterations": int(reference_iterations),
        "solver_backend": solver_backend,
        "regret_mass_scale": float(regret_mass_scale),
        "strategy_mass": float(strategy_mass),
        "n_cases": int(len(records)),
        **summary,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a joint-PBS policy head as a CFR+ warm start against a stronger resolver."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--regret-mass-scale", type=float, default=1.0)
    parser.add_argument("--strategy-mass", type=float, default=0.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_joint_pbs_policy_warm_start(
        checkpoint=args.checkpoint,
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        device=args.device,
        start_index=args.start_index,
        limit=args.limit,
        low_iterations=args.low_iterations,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
        regret_mass_scale=args.regret_mass_scale,
        strategy_mass=args.strategy_mass,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
