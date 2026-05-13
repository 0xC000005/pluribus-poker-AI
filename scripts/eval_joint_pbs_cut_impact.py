#!/usr/bin/env python3
"""Measure per-cut root policy impact for joint-PBS successor values."""

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
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json

from eval_joint_pbs_continuation_probe import (  # noqa: E402
    load_joint_pbs_continuation_checkpoint,
)
from eval_joint_pbs_resolver_cut_ab import (  # noqa: E402
    JointPBSSuccessorCutCallback,
    StructuralCutRiskPredictor,
    _frontier_action_shape,
    _select_successor_cut_node_records,
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
from solver import StreetSolver, _parse_nav, resolve_solver_backend, solver_action_to_slumbot  # noqa: E402


def _pearson_or_none(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 8)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 8)


def _case_window(
    cases: list[ResolverBenchmarkCase],
    belief: np.ndarray,
    *,
    start_index: int,
    limit: int,
) -> tuple[int, list[ResolverBenchmarkCase], np.ndarray]:
    start = max(0, min(int(start_index), len(cases)))
    if int(limit) <= 0:
        stop = len(cases)
    else:
        stop = min(start + int(limit), len(cases))
    selected = cases[start:stop]
    if not selected:
        raise ValueError("selected cut-impact case slice is empty")
    return start, selected, belief[start:stop]


def _case_cut_impacts(
    case: ResolverBenchmarkCase,
    *,
    belief_row: np.ndarray | None,
    model: Any,
    payload: dict[str, Any],
    device: Any,
    solver_iterations: int,
    solver_backend: str,
    value_scale: float,
    batch_size: int,
    min_bet_count: int,
    target_action_shapes: tuple[str, ...],
    risk_predictor: StructuralCutRiskPredictor | None,
) -> list[dict[str, Any]]:
    parsed = parse_action(case.action_str)
    if "error" in parsed or int(parsed.get("st", -1)) != 2:
        return []

    board_idx = [card_str_to_index(card) for card in case.board[:4]]
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=2,
    )
    pot = our_bet_pre + opp_bet_pre
    hero_stack = 20000 - our_bet_pre
    villain_stack = 20000 - opp_bet_pre
    hero_first = case.client_pos == 0
    street_action = case.action_str.split("/")[2] if len(case.action_str.split("/")) > 2 else ""
    action_prefix = _pre_street_prefix(case.action_str, 2)
    root_context = {
        "root_action_shape": _frontier_action_shape(case.action_str),
        "root_bet_count": int(case.action_str.count("b")),
        "pot": int(pot),
        "hero_stack": int(hero_stack),
        "villain_stack": int(villain_stack),
        "client_pos": int(case.client_pos),
        "hero_first": bool(hero_first),
    }

    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    backend, backend_device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("joint PBS cut-impact diagnostic requires CPU backend")

    template = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    template_node = template.navigate(_parse_nav(street_action, template))
    if template_node is None or template_node.is_terminal:
        return []
    _candidates, selected_records = _select_successor_cut_node_records(
        template,
        template_node,
        action_prefix=action_prefix,
        client_pos=case.client_pos,
        min_bet_count=min_bet_count,
        target_action_shapes=target_action_shapes,
        risk_predictor=None,
    )
    if risk_predictor is not None:
        annotated_candidates, _unused = _select_successor_cut_node_records(
            template,
            template_node,
            action_prefix=action_prefix,
            client_pos=case.client_pos,
            min_bet_count=min_bet_count,
            target_action_shapes=target_action_shapes,
            risk_predictor=risk_predictor,
        )
        annotations = {int(record["node_idx"]): record for record in annotated_candidates}
        selected_records = [
            {**record, **annotations.get(int(record["node_idx"]), {})}
            for record in selected_records
        ]

    baseline = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    baseline.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
    )
    baseline_node = baseline.navigate(_parse_nav(street_action, baseline))
    if baseline_node is None or baseline_node.is_terminal:
        return []
    hand = tuple(sorted(our_cards_idx))
    baseline_strategy = _strategy_vector(baseline.get_strategy(hand, baseline_node))
    baseline_action = int(np.argmax(baseline_strategy))

    impacts: list[dict[str, Any]] = []
    for record in selected_records:
        learned = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
        learned_node = learned.navigate(_parse_nav(street_action, learned))
        if learned_node is None or learned_node.is_terminal:
            continue
        callback = JointPBSSuccessorCutCallback(
            case=case,
            board4=board_idx,
            action_prefix=action_prefix,
            model=model,
            payload=payload,
            device=device,
            value_scale=value_scale,
            batch_size=batch_size,
        )
        callback.solver_hands = list(learned.hands)
        learned.solve(
            n_iterations=solver_iterations,
            hero_range=hero_range,
            villain_range=villain_range,
            backend=backend,
            device=backend_device,
            cut_node_indices=[int(record["node_idx"])],
            cut_node_fn=callback,
        )
        learned_strategy = _strategy_vector(learned.get_strategy(hand, learned_node))
        learned_action = int(np.argmax(learned_strategy))
        l1 = float(np.abs(baseline_strategy - learned_strategy).sum())
        item = {
            "label": case.label,
            **root_context,
            "cut_pos": int(record["cut_pos"]),
            "action_shape": str(record["action_shape"]),
            "bet_count": int(record["bet_count"]),
            "legal_action_count": int(record.get("legal_action_count", 0)),
            "risk_decision": str(record.get("risk_decision", "unscored")),
            "baseline_action": baseline_action,
            "baseline_increment": solver_action_to_slumbot(
                baseline_action,
                baseline_node,
                baseline,
                parsed,
            ),
            "learned_action": learned_action,
            "learned_increment": solver_action_to_slumbot(
                learned_action,
                learned_node,
                learned,
                parsed,
            ),
            "action_agreement": bool(baseline_action == learned_action),
            "action_l1_drift": round(l1, 8),
            "replaced_cut_nodes": int(callback.stats.replaced_cut_nodes),
            "cut_prediction_ms": round(float(callback.stats.prediction_ms), 3),
        }
        if "predicted_mae" in record:
            item["predicted_mae"] = float(record["predicted_mae"])
        impacts.append(item)
    return impacts


def evaluate_cut_impacts(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    device: str,
    limit: int,
    solver_iterations: int,
    solver_backend: str,
    value_scale: float,
    batch_size: int,
    min_bet_count: int,
    target_action_shapes: tuple[str, ...],
    error_predictor_json: str | Path | None,
    max_cut_evals: int,
    start_index: int = 0,
) -> dict[str, Any]:
    from poker_ai.research.belief_probe import _resolve_device

    resolved_device = _resolve_device(device)
    cases = load_cases_json(cases_json)
    dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    start, selected_cases, selected_beliefs = _case_window(
        cases,
        dataset.belief,
        start_index=start_index,
        limit=limit,
    )
    risk_predictor = (
        StructuralCutRiskPredictor.from_json(error_predictor_json)
        if error_predictor_json
        else None
    )
    model, payload = load_joint_pbs_continuation_checkpoint(checkpoint, device=resolved_device)
    impacts: list[dict[str, Any]] = []
    for idx, case in enumerate(selected_cases):
        impacts.extend(
            _case_cut_impacts(
                case,
                belief_row=selected_beliefs[idx],
                model=model,
                payload=payload,
                device=resolved_device,
                solver_iterations=solver_iterations,
                solver_backend=solver_backend,
                value_scale=value_scale,
                batch_size=batch_size,
                min_bet_count=min_bet_count,
                target_action_shapes=target_action_shapes,
                risk_predictor=risk_predictor,
            )
        )
        if len(impacts) >= max_cut_evals:
            impacts = impacts[:max_cut_evals]
            break

    drifts = [float(record["action_l1_drift"]) for record in impacts]
    predicted = [float(record["predicted_mae"]) for record in impacts if "predicted_mae" in record]
    predicted_drifts = [
        float(record["action_l1_drift"]) for record in impacts if "predicted_mae" in record
    ]
    selected_drifts = [
        float(record["action_l1_drift"])
        for record in impacts
        if record.get("risk_decision") == "selected_for_learned_value"
    ]
    rejected_drifts = [
        float(record["action_l1_drift"])
        for record in impacts
        if record.get("risk_decision") == "risk_rejected_exact_fallback"
    ]
    return {
        "mode": "joint_pbs_single_cut_impact",
        "passed": bool(impacts),
        "promotion_blockers": ["cut_impact_is_diagnostic_not_slumbot_confidence"],
        "checkpoint": str(checkpoint),
        "cases": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "error_predictor_json": str(error_predictor_json) if error_predictor_json else None,
        "device": str(resolved_device),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "start_index": int(start),
        "case_scan_limit": int(len(selected_cases)),
        "max_cut_evals": int(max_cut_evals),
        "min_bet_count": int(min_bet_count),
        "target_action_shapes": list(target_action_shapes),
        "n_cut_evaluated": len(impacts),
        "mean_single_cut_l1_drift": _mean_or_none(drifts),
        "max_single_cut_l1_drift": round(float(np.max(drifts)), 8) if drifts else None,
        "action_agreement_rate": (
            round(float(np.mean([record["action_agreement"] for record in impacts])), 8)
            if impacts
            else None
        ),
        "predicted_mae_to_action_l1_pearson": _pearson_or_none(predicted, predicted_drifts),
        "selected_mean_l1_drift": _mean_or_none(selected_drifts),
        "risk_rejected_mean_l1_drift": _mean_or_none(rejected_drifts),
        "worst_cuts": sorted(
            impacts,
            key=lambda record: float(record["action_l1_drift"]),
            reverse=True,
        )[:10],
        "records": impacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure one-cut-at-a-time root action drift for joint-PBS values."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-cut-evals", type=int, default=32)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--min-bet-count", type=int, default=0)
    parser.add_argument("--target-action-shapes", nargs="*", default=())
    parser.add_argument("--error-predictor-json")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = evaluate_cut_impacts(
        checkpoint=args.checkpoint,
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        device=args.device,
        limit=args.limit,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        batch_size=args.batch_size,
        min_bet_count=args.min_bet_count,
        target_action_shapes=tuple(args.target_action_shapes),
        error_predictor_json=args.error_predictor_json,
        max_cut_evals=args.max_cut_evals,
        start_index=args.start_index,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
