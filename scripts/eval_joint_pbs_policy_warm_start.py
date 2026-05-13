#!/usr/bin/env python3
"""Evaluate joint-PBS policy priors as CFR+ warm starts."""

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


def _kl_to_reference(candidate: np.ndarray, reference: np.ndarray, *, eps: float = 1e-9) -> float:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.clip(cand, eps, None)
    ref = np.clip(ref, eps, None)
    cand /= cand.sum()
    ref /= ref.sum()
    return round(float(np.sum(ref * np.log(ref / cand))), 8)


def _illegal_mass(strategy: np.ndarray, node: Any) -> float:
    legal = np.zeros_like(np.asarray(strategy, dtype=np.float64))
    for action in node.children:
        if 0 <= int(action) < legal.shape[0]:
            legal[int(action)] = 1.0
    return round(float(np.asarray(strategy, dtype=np.float64)[legal <= 0].sum()), 8)


def _root_key(label: str) -> str:
    root = str(label)
    for marker in ("-root-policy-hand", "-successor", "-cut", "-hand"):
        if marker in root:
            root = root.split(marker, 1)[0]
    if "-street" in root:
        root = root.rsplit("-street", 1)[0]
    return root


def _roots_from_cases(cases: list[Any]) -> set[str]:
    return {_root_key(case.label) for case in cases}


def _resolve_payload_path(path: str | Path, *, checkpoint: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    checkpoint_relative = Path(checkpoint).parent / candidate
    return checkpoint_relative


def _roots_from_label_npz(path: str | Path) -> set[str]:
    with np.load(path, allow_pickle=True) as data:
        if "labels" not in data:
            raise ValueError(f"{path} does not contain labels")
        return {_root_key(str(label)) for label in data["labels"]}


def _root_disjoint_audit(
    *,
    checkpoint: str | Path,
    payload: dict[str, Any],
    selected_cases: list[Any],
    train_labels_npz: str | Path | None,
    require_root_disjoint: bool,
) -> dict[str, Any]:
    eval_roots = _roots_from_cases(selected_cases)
    train_path: Path | None = None
    if train_labels_npz is not None:
        train_path = Path(train_labels_npz)
    elif payload.get("train_joint_npz"):
        train_path = _resolve_payload_path(payload["train_joint_npz"], checkpoint=checkpoint)
    if train_path is None:
        return {
            "required": bool(require_root_disjoint),
            "passed": not require_root_disjoint,
            "root_disjoint": False,
            "train_labels_npz": "",
            "train_roots": 0,
            "eval_roots": int(len(eval_roots)),
            "root_overlap_count": 0,
            "root_overlap": [],
            "reason": "missing_train_labels",
        }
    train_roots = _roots_from_label_npz(train_path)
    overlap = sorted(train_roots & eval_roots)
    root_disjoint = len(overlap) == 0
    return {
        "required": bool(require_root_disjoint),
        "passed": bool(root_disjoint or not require_root_disjoint),
        "root_disjoint": bool(root_disjoint),
        "train_labels_npz": str(train_path),
        "train_roots": int(len(train_roots)),
        "eval_roots": int(len(eval_roots)),
        "root_overlap_count": int(len(overlap)),
        "root_overlap": overlap[:20],
    }


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


def _strategy_decision(
    case,
    parsed: dict,
    solver: StreetSolver,
    node: Any,
    *,
    latency_ms: float | None = None,
) -> SolverDecision:
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    strategy_vec = _strategy_vector(solver.get_strategy(tuple(sorted(our_cards_idx)), node))
    action = int(np.argmax(strategy_vec))
    increment = solver_action_to_slumbot(action, node, solver, parsed)
    if latency_ms is None:
        latency_ms = float(getattr(solver, "last_solve_ms", 0.0))
    return SolverDecision(action, increment, strategy_vec, float(latency_ms), False)


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
    started = time.perf_counter()
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
    if solved is None:
        return None
    decision = solved[2]
    total_ms = (time.perf_counter() - started) * 1000.0
    return SolverDecision(
        decision.action,
        decision.increment,
        decision.strategy,
        total_ms,
        decision.node_terminal,
    )


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    root_audit: dict[str, Any] | None = None,
    min_evaluated: int = 1,
    max_warm_latency_ratio: float = 2.0,
) -> dict[str, Any]:
    evaluated = [record for record in records if record.get("passed")]
    low_l1 = [float(record["low_l1_to_reference"]) for record in evaluated]
    warm_l1 = [float(record["warm_l1_to_reference"]) for record in evaluated]
    low_kl = [float(record["low_kl_to_reference"]) for record in evaluated if "low_kl_to_reference" in record]
    warm_kl = [float(record["warm_kl_to_reference"]) for record in evaluated if "warm_kl_to_reference" in record]
    low_allin = _rate([bool(record["low_allin_selected"]) for record in evaluated])
    warm_allin = _rate([bool(record["warm_allin_selected"]) for record in evaluated])
    reference_allin = _rate([bool(record["reference_allin_selected"]) for record in evaluated])
    low_allin_prob = [float(record.get("low_allin_prob", 0.0)) for record in evaluated]
    warm_allin_prob = [float(record.get("warm_allin_prob", 0.0)) for record in evaluated]
    reference_allin_prob = [float(record.get("reference_allin_prob", 0.0)) for record in evaluated]
    low_agreement = _rate([bool(record["low_agrees_with_reference"]) for record in evaluated])
    warm_agreement = _rate([bool(record["warm_agrees_with_reference"]) for record in evaluated])
    low_latency = [float(record.get("low_latency_ms", 0.0)) for record in evaluated]
    warm_latency = [float(record.get("warm_latency_ms", 0.0)) for record in evaluated]
    reference_latency = [float(record.get("reference_latency_ms", 0.0)) for record in evaluated]
    illegal_masses = [
        float(record.get(key, 0.0))
        for record in evaluated
        for key in ("low_illegal_mass", "warm_illegal_mass", "reference_illegal_mass")
    ]
    low_latency_mean = _mean(low_latency)
    warm_latency_mean = _mean(warm_latency)
    latency_ratio = round(float(warm_latency_mean / max(low_latency_mean, 1e-9)), 8) if evaluated else 0.0
    kl_gate = not low_kl or _mean(warm_kl) < _mean(low_kl)
    illegal_gate = not illegal_masses or max(illegal_masses) <= 1e-6
    root_gate = root_audit is None or bool(root_audit.get("passed"))
    latency_gate = not evaluated or latency_ratio <= float(max_warm_latency_ratio)
    enough_cases = len(evaluated) >= int(min_evaluated)
    return {
        "n_evaluated": int(len(evaluated)),
        "min_evaluated": int(min_evaluated),
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_warm_l1_to_reference": _mean(warm_l1),
        "mean_low_kl_to_reference": _mean(low_kl),
        "mean_warm_kl_to_reference": _mean(warm_kl),
        "low_action_agreement": low_agreement,
        "warm_action_agreement": warm_agreement,
        "low_allin_rate": low_allin,
        "warm_allin_rate": warm_allin,
        "reference_allin_rate": reference_allin,
        "mean_low_allin_prob": _mean(low_allin_prob),
        "mean_warm_allin_prob": _mean(warm_allin_prob),
        "mean_reference_allin_prob": _mean(reference_allin_prob),
        "low_allin_gap": round(abs(low_allin - reference_allin), 8),
        "warm_allin_gap": round(abs(warm_allin - reference_allin), 8),
        "low_allin_prob_gap": round(abs(_mean(low_allin_prob) - _mean(reference_allin_prob)), 8),
        "warm_allin_prob_gap": round(abs(_mean(warm_allin_prob) - _mean(reference_allin_prob)), 8),
        "mean_low_latency_ms": low_latency_mean,
        "mean_warm_latency_ms": warm_latency_mean,
        "mean_reference_latency_ms": _mean(reference_latency),
        "warm_to_low_latency_ratio": latency_ratio,
        "max_warm_latency_ratio": float(max_warm_latency_ratio),
        "max_illegal_mass": round(float(max(illegal_masses)), 8) if illegal_masses else 0.0,
        "root_disjoint_passed": bool(root_gate),
        "passed": bool(
            enough_cases
            and root_gate
            and _mean(warm_l1) < _mean(low_l1)
            and kl_gate
            and warm_agreement >= low_agreement
            and abs(warm_allin - reference_allin) <= abs(low_allin - reference_allin)
            and abs(_mean(warm_allin_prob) - _mean(reference_allin_prob))
            <= abs(_mean(low_allin_prob) - _mean(reference_allin_prob))
            and illegal_gate
            and latency_gate
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
    train_labels_npz: str | Path | None = None,
    require_root_disjoint: bool = True,
    min_evaluated: int = 1,
    max_warm_latency_ratio: float = 2.0,
) -> dict[str, Any]:
    import torch

    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    model, payload = load_joint_pbs_continuation_checkpoint(checkpoint, device=torch_device)
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)
    root_audit = _root_disjoint_audit(
        checkpoint=checkpoint,
        payload=payload,
        selected_cases=selected_cases,
        train_labels_npz=train_labels_npz,
        require_root_disjoint=require_root_disjoint,
    )
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
        low_node = low_solved[1]
        reference = reference_solved[2]
        reference_node = reference_solved[1]
        records.append(
            {
                "label": case.label,
                "passed": True,
                "low_action": int(np.argmax(low.strategy)),
                "reference_action": int(np.argmax(reference.strategy)),
                "warm_action": int(np.argmax(warm.strategy)),
                "low_l1_to_reference": round(float(np.abs(low.strategy - reference.strategy).sum()), 8),
                "warm_l1_to_reference": round(float(np.abs(warm.strategy - reference.strategy).sum()), 8),
                "low_kl_to_reference": _kl_to_reference(low.strategy, reference.strategy),
                "warm_kl_to_reference": _kl_to_reference(warm.strategy, reference.strategy),
                "low_allin_prob": round(float(low.strategy[8]), 8),
                "reference_allin_prob": round(float(reference.strategy[8]), 8),
                "warm_allin_prob": round(float(warm.strategy[8]), 8),
                "low_allin_selected": bool(int(np.argmax(low.strategy)) == 8),
                "reference_allin_selected": bool(int(np.argmax(reference.strategy)) == 8),
                "warm_allin_selected": bool(int(np.argmax(warm.strategy)) == 8),
                "low_illegal_mass": _illegal_mass(low.strategy, low_node),
                "reference_illegal_mass": _illegal_mass(reference.strategy, reference_node),
                "warm_illegal_mass": _illegal_mass(warm.strategy, low_node),
                "low_latency_ms": round(float(low.latency_ms), 3),
                "reference_latency_ms": round(float(reference.latency_ms), 3),
                "warm_latency_ms": round(float(warm.latency_ms), 3),
                "low_agrees_with_reference": bool(
                    int(np.argmax(low.strategy)) == int(np.argmax(reference.strategy))
                ),
                "warm_agrees_with_reference": bool(
                    int(np.argmax(warm.strategy)) == int(np.argmax(reference.strategy))
                ),
            }
        )
    summary = _summarize_records(
        records,
        root_audit=root_audit,
        min_evaluated=min_evaluated,
        max_warm_latency_ratio=max_warm_latency_ratio,
    )
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
        "require_root_disjoint": bool(require_root_disjoint),
        "train_labels_npz": str(train_labels_npz or ""),
        "root_disjoint_audit": root_audit,
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
    parser.add_argument("--train-labels-npz")
    parser.add_argument("--no-require-root-disjoint", action="store_true")
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--max-warm-latency-ratio", type=float, default=2.0)
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
        train_labels_npz=args.train_labels_npz,
        require_root_disjoint=not args.no_require_root_disjoint,
        min_evaluated=args.min_evaluated,
        max_warm_latency_ratio=args.max_warm_latency_ratio,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
