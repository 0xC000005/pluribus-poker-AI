#!/usr/bin/env python3
"""Run resolver A/B with a public-belief DCVN-style learned leaf evaluator."""

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

from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX, _resolve_device
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json

from eval_joint_pbs_resolver_leaf_ab import (  # noqa: E402
    _action_path_to_street_string,
    _card_to_str,
    _global_board_mask,
    _local_ranges_from_belief,
    _normalize,
    _pre_street_prefix,
    _strategy_vector,
)
from eval_public_belief_dual_hand_cfv_probe import (  # noqa: E402
    load_public_belief_dual_hand_cfv_checkpoint,
    predict_public_belief_dual_hand_cfv_model_vectorized,
    project_dual_cfv_zero_sum,
)
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    build_features,
    card_str_to_index,
    parse_action,
)
from solver import StreetSolver, _parse_nav, resolve_solver_backend, solver_action_to_slumbot  # noqa: E402


@dataclass
class DCVNLeafStats:
    callback_calls: int = 0
    replaced_showdowns: int = 0
    fallback_showdowns: int = 0
    prediction_states: int = 0
    prediction_ms: float = 0.0


def _apply_leaf_predictions(
    *,
    out_h: np.ndarray,
    out_v: np.ndarray,
    pred: np.ndarray,
    tasks: list[int],
    local_to_global: np.ndarray,
    valid_m: np.ndarray,
    hero_reach: np.ndarray,
    villain_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_h = np.asarray(out_h, dtype=np.float32).copy()
    out_v = np.asarray(out_v, dtype=np.float32).copy()
    valid_m = np.asarray(valid_m, dtype=np.float32)
    valid_m_t = valid_m.T.copy()
    for pred_idx, row in enumerate(tasks):
        hero_den = valid_m @ villain_reach[row]
        villain_den = valid_m_t @ hero_reach[row]
        out_h[row] = pred[0, pred_idx, local_to_global] * hero_den
        out_v[row] = pred[1, pred_idx, local_to_global] * villain_den
    return out_h, out_v


class PublicBeliefDCVNLeafCallback:
    """Use a dual-player public-belief CFV network as a depth-limited leaf."""

    def __init__(
        self,
        *,
        case: ResolverBenchmarkCase,
        board4: list[int],
        action_prefix: str,
        model: Any,
        payload: dict[str, Any],
        device: Any,
        value_scale: float,
        state_batch_size: int,
        hand_batch_size: int,
        project_zero_sum: bool,
    ):
        self.case = case
        self.board4 = [int(card) for card in board4]
        self.board_str = [_card_to_str(card) for card in self.board4]
        self.board_mask = _global_board_mask(self.board4)
        self.action_prefix = action_prefix
        self.model = model
        self.payload = payload
        self.device = device
        self.value_scale = float(value_scale)
        self.state_batch_size = int(state_batch_size)
        self.hand_batch_size = int(hand_batch_size)
        self.project_zero_sum = bool(project_zero_sum)
        self.stats = DCVNLeafStats()

    def __call__(self, **kwargs):
        tree = kwargs["tree"]
        showdown_indices = np.asarray(kwargs["showdown_indices"], dtype=np.int32)
        hero_reach = np.asarray(kwargs["hero_reach"], dtype=np.float32)
        villain_reach = np.asarray(kwargs["villain_reach"], dtype=np.float32)
        valid_m = np.asarray(kwargs["valid_m"], dtype=np.float32)
        out_h = np.asarray(kwargs["default_hero_values"], dtype=np.float32).copy()
        out_v = np.asarray(kwargs["default_villain_values"], dtype=np.float32).copy()
        self.stats.callback_calls += 1

        features: list[np.ndarray] = []
        beliefs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        tasks: list[int] = []
        for row, node_idx in enumerate(showdown_indices.tolist()):
            action_str = self._terminal_action_str(tree, int(node_idx))
            parsed = parse_action(action_str)
            if "error" in parsed:
                self.stats.fallback_showdowns += 1
                continue
            hero_belief = np.zeros(N_HANDS, dtype=np.float32)
            villain_belief = np.zeros(N_HANDS, dtype=np.float32)
            hero_belief[self._local_to_global] = hero_reach[row]
            villain_belief[self._local_to_global] = villain_reach[row]
            hero_belief = _normalize(hero_belief * self.board_mask)
            villain_belief = _normalize(villain_belief * self.board_mask)
            features.append(
                build_features(
                    [],
                    self.board_str,
                    action_str,
                    self.case.client_pos,
                    parsed,
                )
            )
            beliefs.append(np.concatenate([hero_belief, villain_belief]).astype(np.float32))
            masks.append(self.board_mask)
            tasks.append(row)

        if not tasks:
            return out_h, out_v

        features_np = np.stack(features).astype(np.float32)
        beliefs_np = np.stack(beliefs).astype(np.float32)
        masks_np = np.stack(masks).astype(np.float32)
        started = time.perf_counter()
        pred = predict_public_belief_dual_hand_cfv_model_vectorized(
            self.model,
            self.payload,
            features_np,
            beliefs_np,
            masks_np,
            masks_np,
            device=self.device,
            state_batch_size=self.state_batch_size,
            hand_batch_size=self.hand_batch_size,
        )
        if self.project_zero_sum:
            pred = project_dual_cfv_zero_sum(pred, beliefs_np, masks_np, masks_np)
        pred = pred * self.value_scale
        self.stats.prediction_ms += (time.perf_counter() - started) * 1000.0
        self.stats.prediction_states += int(len(tasks))

        out_h, out_v = _apply_leaf_predictions(
            out_h=out_h,
            out_v=out_v,
            pred=pred,
            tasks=tasks,
            local_to_global=self._local_to_global,
            valid_m=valid_m,
            hero_reach=hero_reach,
            villain_reach=villain_reach,
        )
        self.stats.replaced_showdowns += len(tasks)
        return out_h, out_v

    def _terminal_action_str(self, tree: dict[str, Any], node_idx: int) -> str:
        street_action = _action_path_to_street_string(
            tree,
            node_idx,
            hero_stack_start=int(tree["stacks_h"][0]),
            villain_stack_start=int(tree["stacks_v"][0]),
        )
        return f"{self.action_prefix}/{street_action}" if self.action_prefix else street_action

    @property
    def solver_hands(self) -> list[tuple[int, int]]:
        return self._solver_hands

    @solver_hands.setter
    def solver_hands(self, hands: list[tuple[int, int]]) -> None:
        self._solver_hands = hands
        self._local_to_global = np.array(
            [_HAND_TO_INDEX[tuple(sorted(hand))] for hand in hands],
            dtype=np.int32,
        )


def _solve_case(
    case: ResolverBenchmarkCase,
    *,
    belief_row: np.ndarray | None,
    model: Any,
    payload: dict[str, Any],
    device: Any,
    solver_iterations: int,
    solver_backend: str,
    value_scale: float,
    state_batch_size: int,
    hand_batch_size: int,
    project_zero_sum: bool,
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
        raise ValueError("DCVN leaf A/B requires CPU backend for leaf callbacks")

    baseline = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    baseline.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
    )
    nav = _parse_nav(street_action, baseline)
    baseline_node = baseline.navigate(nav)
    if baseline_node is None or baseline_node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "terminal_or_missing_node"}
    hand = tuple(sorted(our_cards_idx))
    baseline_strategy = _strategy_vector(baseline.get_strategy(hand, baseline_node))

    learned = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    callback = PublicBeliefDCVNLeafCallback(
        case=case,
        board4=board_idx,
        action_prefix=action_prefix,
        model=model,
        payload=payload,
        device=device,
        value_scale=value_scale,
        state_batch_size=state_batch_size,
        hand_batch_size=hand_batch_size,
        project_zero_sum=project_zero_sum,
    )
    callback.solver_hands = list(learned.hands)
    learned.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
        showdown_leaf_fn=callback,
    )
    learned_node = learned.navigate(nav)
    if learned_node is None or learned_node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "learned_terminal_or_missing_node"}
    learned_strategy = _strategy_vector(learned.get_strategy(hand, learned_node))

    baseline_action = int(np.argmax(baseline_strategy))
    learned_action = int(np.argmax(learned_strategy))
    l1 = float(np.abs(baseline_strategy - learned_strategy).sum())
    return {
        "label": case.label,
        "street": street,
        "passed": bool(np.isfinite(l1)),
        "leaf_applied": bool(callback.stats.replaced_showdowns > 0),
        "baseline_action": baseline_action,
        "baseline_increment": solver_action_to_slumbot(baseline_action, baseline_node, baseline, parsed),
        "learned_action": learned_action,
        "learned_increment": solver_action_to_slumbot(learned_action, learned_node, learned, parsed),
        "action_agreement": bool(baseline_action == learned_action),
        "action_l1_drift": round(l1, 8),
        "baseline_strategy": baseline_strategy.round(6).tolist(),
        "learned_strategy": learned_strategy.round(6).tolist(),
        "baseline_solve_ms": round(float(getattr(baseline, "last_solve_ms", 0.0)), 3),
        "learned_solve_ms": round(float(getattr(learned, "last_solve_ms", 0.0)), 3),
        "leaf_prediction_ms": round(float(callback.stats.prediction_ms), 3),
        "leaf_prediction_states": int(callback.stats.prediction_states),
        "leaf_callback_calls": int(callback.stats.callback_calls),
        "replaced_showdowns": int(callback.stats.replaced_showdowns),
        "fallback_showdowns": int(callback.stats.fallback_showdowns),
        "solver_n_hands": int(learned.n),
    }


def eval_public_belief_dcvn_leaf_ab(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path | None = None,
    device: str = "auto",
    start_index: int = 0,
    limit: int = 8,
    solver_iterations: int = 5,
    solver_backend: str = "cpu",
    value_scale: float | None = None,
    state_batch_size: int = 16,
    hand_batch_size: int = N_HANDS,
    project_zero_sum: bool = True,
    min_action_agreement: float = 0.5,
    max_mean_l1_drift: float = 0.75,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    cases = load_cases_json(cases_json)
    base_dataset = None
    if cfv_cache:
        base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
        if base_dataset.features.shape[0] != len(cases):
            raise ValueError("case count does not match CFV cache rows")
    model, payload = load_public_belief_dual_hand_cfv_checkpoint(checkpoint, device=resolved_device)
    scale = float(value_scale) if value_scale is not None else float(payload.get("value_scale", 20000.0))
    start = max(0, min(int(start_index), len(cases)))
    max_cases = max(0, min(int(limit), len(cases) - start))

    records = []
    for offset, case in enumerate(cases[start : start + max_cases]):
        idx = start + offset
        belief_row = base_dataset.belief[idx] if base_dataset is not None else None
        records.append(
            _solve_case(
                case,
                belief_row=belief_row,
                model=model,
                payload=payload,
                device=resolved_device,
                solver_iterations=solver_iterations,
                solver_backend=solver_backend,
                value_scale=scale,
                state_batch_size=state_batch_size,
                hand_batch_size=hand_batch_size,
                project_zero_sum=project_zero_sum,
            )
        )

    evaluated = [record for record in records if record.get("passed") and "action_l1_drift" in record]
    leaf_evaluated = [record for record in evaluated if record.get("leaf_applied")]
    drift = [float(record["action_l1_drift"]) for record in evaluated]
    leaf_drift = [float(record["action_l1_drift"]) for record in leaf_evaluated]
    agreement = (
        float(np.mean([record["action_agreement"] for record in leaf_evaluated]))
        if leaf_evaluated
        else 0.0
    )
    mean_l1 = float(np.mean(leaf_drift)) if leaf_drift else float("inf")
    return {
        "mode": "public_belief_dcvn_leaf_ab",
        "passed": bool(
            leaf_evaluated
            and agreement >= float(min_action_agreement)
            and mean_l1 <= float(max_mean_l1_drift)
        ),
        "pass_criteria": (
            "diagnostic DCVN leaf must apply to held-out turn roots and keep "
            "action agreement/L1 drift within predeclared bounds"
        ),
        "promotion_blockers": [
            "dcvn_leaf_ab_is_diagnostic_not_slumbot_confidence",
            "requires_root_disjoint_training_and_holdout_slices",
        ],
        "checkpoint": str(checkpoint),
        "cases": str(cases_json),
        "cfv_cache": str(cfv_cache) if cfv_cache else None,
        "device": str(resolved_device),
        "start_index": int(start),
        "case_scan_limit": int(max_cases),
        "solver_iterations": int(solver_iterations),
        "solver_backend": str(solver_backend),
        "value_scale": float(scale),
        "state_batch_size": int(state_batch_size),
        "hand_batch_size": int(hand_batch_size),
        "project_zero_sum": bool(project_zero_sum),
        "min_action_agreement": float(min_action_agreement),
        "max_mean_l1_drift": float(max_mean_l1_drift),
        "n_cases": len(records),
        "n_evaluated": len(evaluated),
        "n_leaf_applied": len(leaf_evaluated),
        "action_agreement_rate": (
            round(float(np.mean([record["action_agreement"] for record in evaluated])), 6)
            if evaluated
            else 0.0
        ),
        "leaf_action_agreement_rate": round(float(agreement), 6) if leaf_evaluated else 0.0,
        "mean_action_l1_drift": round(float(np.mean(drift)), 8) if drift else None,
        "max_action_l1_drift": round(float(np.max(drift)), 8) if drift else None,
        "leaf_mean_action_l1_drift": round(float(mean_l1), 8) if leaf_drift else None,
        "leaf_max_action_l1_drift": round(float(np.max(leaf_drift)), 8) if leaf_drift else None,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run turn resolver A/B with a public-belief DCVN leaf evaluator."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--value-scale", type=float)
    parser.add_argument("--state-batch-size", type=int, default=16)
    parser.add_argument("--hand-batch-size", type=int, default=N_HANDS)
    parser.add_argument("--no-project-zero-sum", action="store_true")
    parser.add_argument("--min-action-agreement", type=float, default=0.5)
    parser.add_argument("--max-mean-l1-drift", type=float, default=0.75)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = eval_public_belief_dcvn_leaf_ab(
        checkpoint=args.checkpoint,
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        device=args.device,
        start_index=args.start_index,
        limit=args.limit,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        state_batch_size=args.state_batch_size,
        hand_batch_size=args.hand_batch_size,
        project_zero_sum=not args.no_project_zero_sum,
        min_action_agreement=args.min_action_agreement,
        max_mean_l1_drift=args.max_mean_l1_drift,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
