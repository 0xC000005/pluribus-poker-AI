#!/usr/bin/env python3
"""Run fixed turn resolver A/B with joint-PBS successor cut values.

This is a diagnostic gate, not a gameplay path. The learned variant performs
one local resolver decision and treats immediate non-terminal successor nodes as
depth-limit frontier states evaluated by the joint public-belief continuation
model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
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

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX, _resolve_device
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json

from eval_joint_pbs_continuation_probe import (  # noqa: E402
    load_joint_pbs_continuation_checkpoint,
    predict_joint_pbs_cfv_model,
)
from eval_joint_pbs_resolver_leaf_ab import (  # noqa: E402
    _action_path_to_street_string,
    _global_board_mask,
    _local_ranges_from_belief,
    _normalize,
    _pre_street_prefix,
    _strategy_vector,
)
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    build_features,
    card_str_to_index,
    parse_action,
)
from solver import StreetSolver, _parse_nav, resolve_solver_backend, solver_action_to_slumbot  # noqa: E402

_BET_TOKEN_RE = re.compile(r"b\d+")


@dataclass
class JointCutStats:
    callback_calls: int = 0
    replaced_cut_nodes: int = 0
    fallback_cut_nodes: int = 0
    prediction_states: int = 0
    prediction_ms: float = 0.0


def _node_index(solver: StreetSolver, node: Any) -> int:
    return int(solver._tree["all_nodes"].index(node))


def _frontier_action_shape(action_str: str) -> str:
    return _BET_TOKEN_RE.sub("b", action_str)


def _frontier_matches_filter(
    *,
    action_str: str,
    min_bet_count: int = 0,
    target_action_shapes: tuple[str, ...] = (),
) -> bool:
    action_shape = _frontier_action_shape(action_str)
    bet_count = int(len(_BET_TOKEN_RE.findall(action_str)))
    if target_action_shapes and action_shape not in target_action_shapes:
        return False
    return bet_count >= int(min_bet_count)


def _successor_cut_node_indices(
    solver: StreetSolver,
    active_node: Any,
    *,
    action_prefix: str = "",
    min_bet_count: int = 0,
    target_action_shapes: tuple[str, ...] = (),
) -> list[int]:
    """Return immediate non-terminal successors to use as a depth-limit frontier."""
    cut_indices: list[int] = []
    for action in sorted(active_node.children):
        child = active_node.children[action]
        if child.is_terminal:
            continue
        if min_bet_count > 0 or target_action_shapes:
            node_idx = _node_index(solver, child)
            street_action = _action_path_to_street_string(
                solver._tree,
                node_idx,
                hero_stack_start=int(solver._tree["stacks_h"][0]),
                villain_stack_start=int(solver._tree["stacks_v"][0]),
            )
            action_str = f"{action_prefix}/{street_action}" if action_prefix else street_action
            if not _frontier_matches_filter(
                action_str=action_str,
                min_bet_count=min_bet_count,
                target_action_shapes=target_action_shapes,
            ):
                continue
        cut_indices.append(_node_index(solver, child))
    return cut_indices


class JointPBSSuccessorCutCallback:
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
        batch_size: int,
    ):
        self.case = case
        self.board4 = [int(card) for card in board4]
        self.board_str = [
            "23456789TJQKA"[int(card) // 4] + ("c", "d", "h", "s")[int(card) % 4]
            for card in self.board4
        ]
        self.board_mask = _global_board_mask(self.board4)
        self.action_prefix = action_prefix
        self.model = model
        self.payload = payload
        self.device = device
        self.value_scale = float(value_scale)
        self.batch_size = int(batch_size)
        self.stats = JointCutStats()

    def __call__(self, **kwargs):
        tree = kwargs["tree"]
        cut_indices = np.asarray(kwargs["cut_indices"], dtype=np.int32)
        hero_reach = np.asarray(kwargs["hero_reach"], dtype=np.float32)
        villain_reach = np.asarray(kwargs["villain_reach"], dtype=np.float32)
        valid_m = np.asarray(kwargs["valid_m"], dtype=np.float32)
        out_h = np.zeros((len(cut_indices), len(self.solver_hands)), dtype=np.float32)
        out_v = np.zeros_like(out_h)
        self.stats.callback_calls += 1

        features: list[np.ndarray] = []
        beliefs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        tasks: list[int] = []
        for row, node_idx in enumerate(cut_indices.tolist()):
            action_str = self._cut_action_str(tree, int(node_idx))
            parsed = parse_action(action_str)
            if "error" in parsed:
                self.stats.fallback_cut_nodes += 1
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

        started = time.perf_counter()
        pred = predict_joint_pbs_cfv_model(
            self.model,
            self.payload,
            np.stack(features).astype(np.float32),
            np.stack(beliefs).astype(np.float32),
            np.stack(masks).astype(np.float32),
            np.stack(masks).astype(np.float32),
            device=self.device,
            batch_size=self.batch_size,
        )
        pred = pred * self.value_scale
        self.stats.prediction_ms += (time.perf_counter() - started) * 1000.0
        self.stats.prediction_states += int(len(tasks))

        valid_m_t = valid_m.T.copy()
        for pred_idx, row in enumerate(tasks):
            hero_den = valid_m @ villain_reach[row]
            villain_den = valid_m_t @ hero_reach[row]
            out_h[row] = pred[0, pred_idx, self._local_to_global] * hero_den
            out_v[row] = pred[1, pred_idx, self._local_to_global] * villain_den
        self.stats.replaced_cut_nodes += len(tasks)
        return out_h, out_v

    def _cut_action_str(self, tree: dict[str, Any], node_idx: int) -> str:
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
    batch_size: int,
    min_bet_count: int,
    target_action_shapes: tuple[str, ...],
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
        raise ValueError("joint PBS cut A/B requires CPU backend for cut callbacks")

    learned = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    learned_node = learned.navigate(_parse_nav(street_action, learned))
    if learned_node is None or learned_node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "learned_terminal_or_missing_node"}
    candidate_cut_indices = _successor_cut_node_indices(learned, learned_node)
    cut_indices = _successor_cut_node_indices(
        learned,
        learned_node,
        action_prefix=action_prefix,
        min_bet_count=min_bet_count,
        target_action_shapes=target_action_shapes,
    )
    if not cut_indices:
        return {
            "label": case.label,
            "passed": False,
            "skipped": "no_matching_successor_cut_nodes",
            "candidate_cut_nodes": int(len(candidate_cut_indices)),
        }

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
        cut_node_indices=cut_indices,
        cut_node_fn=callback,
    )
    learned_strategy = _strategy_vector(learned.get_strategy(hand, learned_node))

    baseline_action = int(np.argmax(baseline_strategy))
    learned_action = int(np.argmax(learned_strategy))
    l1 = float(np.abs(baseline_strategy - learned_strategy).sum())
    return {
        "label": case.label,
        "street": street,
        "passed": bool(np.isfinite(l1)),
        "cut_applied": bool(callback.stats.replaced_cut_nodes > 0),
        "candidate_cut_nodes": int(len(candidate_cut_indices)),
        "n_cut_nodes": int(len(cut_indices)),
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
        "cut_prediction_ms": round(float(callback.stats.prediction_ms), 3),
        "cut_prediction_states": int(callback.stats.prediction_states),
        "cut_callback_calls": int(callback.stats.callback_calls),
        "replaced_cut_nodes": int(callback.stats.replaced_cut_nodes),
        "fallback_cut_nodes": int(callback.stats.fallback_cut_nodes),
        "solver_n_hands": int(learned.n),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed turn resolver A/B using joint-PBS successor cut values."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--min-bet-count",
        type=int,
        default=0,
        help="Only replace successor cuts with at least this many bet tokens; other successors are solved exactly.",
    )
    parser.add_argument(
        "--target-action-shapes",
        nargs="*",
        default=(),
        help="Optional action-shape allowlist for diagnostic cut replacement, e.g. bbc/bbc/kb.",
    )
    parser.add_argument("--min-action-agreement", type=float, default=0.75)
    parser.add_argument("--max-mean-action-l1-drift", type=float, default=0.25)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    cases = load_cases_json(args.cases)
    base_dataset = None
    if args.cfv_cache:
        base_dataset, _records = load_public_belief_cfv_dataset_cache(args.cfv_cache)
        if base_dataset.features.shape[0] != len(cases):
            raise ValueError("case count does not match CFV cache rows")
    model, payload = load_joint_pbs_continuation_checkpoint(args.checkpoint, device=device)
    max_cases = max(1, min(int(args.limit), len(cases)))

    records = []
    for idx, case in enumerate(cases[:max_cases]):
        belief_row = base_dataset.belief[idx] if base_dataset is not None else None
        records.append(
            _solve_case(
                case,
                belief_row=belief_row,
                model=model,
                payload=payload,
                device=device,
                solver_iterations=args.solver_iterations,
                solver_backend=args.solver_backend,
                value_scale=args.value_scale,
                batch_size=args.batch_size,
                min_bet_count=args.min_bet_count,
                target_action_shapes=tuple(args.target_action_shapes),
            )
        )

    evaluated = [record for record in records if record.get("passed") and "action_l1_drift" in record]
    cut_evaluated = [record for record in evaluated if record.get("cut_applied")]
    drift = [float(record["action_l1_drift"]) for record in evaluated]
    cut_drift = [float(record["action_l1_drift"]) for record in cut_evaluated]
    action_agreement_rate = (
        round(float(np.mean([record["action_agreement"] for record in evaluated])), 6)
        if evaluated
        else 0.0
    )
    cut_action_agreement_rate = (
        round(float(np.mean([record["action_agreement"] for record in cut_evaluated])), 6)
        if cut_evaluated
        else 0.0
    )
    mean_action_l1_drift = round(float(np.mean(drift)), 8) if drift else None
    max_action_l1_drift = round(float(np.max(drift)), 8) if drift else None
    cut_mean_action_l1_drift = round(float(np.mean(cut_drift)), 8) if cut_drift else None
    cut_max_action_l1_drift = round(float(np.max(cut_drift)), 8) if cut_drift else None
    mechanical_passed = bool(cut_evaluated and all(record.get("passed") for record in evaluated))
    behavior_passed = bool(
        mechanical_passed
        and action_agreement_rate >= float(args.min_action_agreement)
        and mean_action_l1_drift is not None
        and mean_action_l1_drift <= float(args.max_mean_action_l1_drift)
    )
    metrics = {
        "mode": "joint_pbs_resolver_successor_cut_ab",
        "passed": behavior_passed,
        "mechanical_passed": mechanical_passed,
        "behavior_passed": behavior_passed,
        "pass_criteria": (
            "must execute cut replacement, keep action agreement above "
            f"{float(args.min_action_agreement):.3f}, and keep mean action L1 drift at or below "
            f"{float(args.max_mean_action_l1_drift):.3f}"
        ),
        "promotion_blockers": [
            "joint_pbs_resolver_successor_cut_ab_is_diagnostic_not_slumbot_confidence",
        ],
        "checkpoint": str(args.checkpoint),
        "cases": str(args.cases),
        "cfv_cache": str(args.cfv_cache) if args.cfv_cache else None,
        "device": str(device),
        "solver_iterations": int(args.solver_iterations),
        "solver_backend": args.solver_backend,
        "min_bet_count": int(args.min_bet_count),
        "target_action_shapes": list(args.target_action_shapes),
        "min_action_agreement": float(args.min_action_agreement),
        "max_mean_action_l1_drift": float(args.max_mean_action_l1_drift),
        "case_scan_limit": int(max_cases),
        "n_cases": len(records),
        "n_evaluated": len(evaluated),
        "n_cut_applied": len(cut_evaluated),
        "action_agreement_rate": action_agreement_rate,
        "cut_action_agreement_rate": cut_action_agreement_rate,
        "mean_action_l1_drift": mean_action_l1_drift,
        "max_action_l1_drift": max_action_l1_drift,
        "cut_mean_action_l1_drift": cut_mean_action_l1_drift,
        "cut_max_action_l1_drift": cut_max_action_l1_drift,
        "records": records,
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
