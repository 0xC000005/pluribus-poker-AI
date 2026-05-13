#!/usr/bin/env python3
"""Compare turn resolver behavior with learned river continuation leaves.

This is a diagnostic gate, not a gameplay path.  The baseline turn resolver
uses equity-at-showdown leaves.  The learned-leaf variant replaces normal
end-of-turn showdown leaves with a saved river dual-CFV ensemble averaged over
legal river cards, then reports root/action drift at fixed turn states.
"""

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

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX, _resolve_device
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json

from eval_public_belief_dual_hand_cfv_probe import (  # noqa: E402
    DualCFVDataset,
    load_public_belief_dual_hand_cfv_ensemble,
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


_SUITS = ("c", "d", "h", "s")
_RANKS = tuple("23456789TJQKA")


def _card_to_str(card: int) -> str:
    return _RANKS[int(card) // 4] + _SUITS[int(card) % 4]


def _normalize(values: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    total = float(out.sum())
    if total <= 1e-12:
        out = np.ones_like(out, dtype=np.float32)
        total = float(out.sum())
    return out / max(total, 1e-12)


def _global_board_mask(board: list[int]) -> np.ndarray:
    blocked = set(int(card) for card in board)
    mask = np.zeros(N_HANDS, dtype=np.float32)
    for hand, index in _HAND_TO_INDEX.items():
        if hand[0] not in blocked and hand[1] not in blocked:
            mask[index] = 1.0
    return mask


def _local_ranges_from_belief(
    belief_row: np.ndarray | None,
    solver_hands: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if belief_row is None:
        return (
            np.ones(len(solver_hands), dtype=np.float32) / max(len(solver_hands), 1),
            np.ones(len(solver_hands), dtype=np.float32) / max(len(solver_hands), 1),
        )
    belief_row = np.asarray(belief_row, dtype=np.float32)
    hero = np.zeros(len(solver_hands), dtype=np.float32)
    villain = np.zeros(len(solver_hands), dtype=np.float32)
    for local_idx, hand in enumerate(solver_hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        hero[local_idx] = float(belief_row[global_idx])
        villain[local_idx] = float(belief_row[N_HANDS + global_idx])
    return _normalize(hero), _normalize(villain)


def _pre_street_prefix(action_str: str, street: int) -> str:
    parts = action_str.split("/")
    return "/".join(parts[:street])


def _action_path(tree: dict[str, Any], node_idx: int) -> list[int]:
    actions: list[int] = []
    current = int(node_idx)
    while int(tree["parent_idx"][current]) >= 0:
        actions.append(int(tree["parent_action"][current]))
        current = int(tree["parent_idx"][current])
    return list(reversed(actions))


def _action_path_to_street_string(
    tree: dict[str, Any],
    node_idx: int,
    *,
    hero_stack_start: int,
    villain_stack_start: int,
) -> str:
    parts: list[str] = []
    current = 0
    for action in _action_path(tree, node_idx):
        child = int(tree["children"][current, action])
        actor = int(tree["player"][current])
        if action == 0:
            parts.append("f")
        elif action == 1:
            parts.append("c" if int(tree["to_call"][current]) > 0 else "k")
        else:
            start_stack = hero_stack_start if actor == 0 else villain_stack_start
            actor_spent = int(start_stack - int(
                tree["stacks_h"][child] if actor == 0 else tree["stacks_v"][child]
            ))
            parts.append(f"b{actor_spent}")
        current = child
    return "".join(parts)


def _strategy_vector(strategy: dict[int, float]) -> np.ndarray:
    vec = np.zeros(N_ACTIONS, dtype=np.float64)
    for action, prob in strategy.items():
        if 0 <= int(action) < N_ACTIONS:
            vec[int(action)] = float(prob)
    total = float(vec.sum())
    if total > 0:
        vec /= total
    else:
        vec[1] = 1.0
    return vec


def _is_descendant(tree: dict[str, Any], node_idx: int, ancestor_idx: int) -> bool:
    current = int(node_idx)
    ancestor_idx = int(ancestor_idx)
    while current >= 0:
        if current == ancestor_idx:
            return True
        current = int(tree["parent_idx"][current])
    return False


@dataclass
class RiverLeafStats:
    callback_calls: int = 0
    replaced_showdowns: int = 0
    fallback_showdowns: int = 0
    skipped_showdowns: int = 0
    prediction_states: int = 0
    prediction_ms: float = 0.0
    prediction_std_sum: float = 0.0
    prediction_std_count: int = 0
    prediction_std_max: float = 0.0


class LearnedRiverLeafCallback:
    def __init__(
        self,
        *,
        case: ResolverBenchmarkCase,
        board4: list[int],
        action_prefix: str,
        loaded_ensemble: list[tuple[Any, dict[str, Any]]],
        device: Any,
        value_scale: float,
        batch_size: int,
        project_zero_sum: bool = False,
        active_subtree_node_idx: int | None = None,
    ):
        self.case = case
        self.board4 = [int(card) for card in board4]
        self.action_prefix = action_prefix
        self.loaded_ensemble = loaded_ensemble
        self.device = device
        self.value_scale = float(value_scale)
        self.batch_size = int(batch_size)
        self.project_zero_sum = bool(project_zero_sum)
        self.active_subtree_node_idx = active_subtree_node_idx
        self.stats = RiverLeafStats()
        self._board4_set = set(self.board4)
        self._river_cards = sorted(set(range(52)) - self._board4_set)
        self._river_count_per_pair = float(52 - len(self.board4) - 4)

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
        tasks: list[tuple[int, np.ndarray]] = []
        replaced_rows: set[int] = set()
        for row, node_idx in enumerate(showdown_indices.tolist()):
            if (
                self.active_subtree_node_idx is not None
                and not _is_descendant(tree, int(node_idx), int(self.active_subtree_node_idx))
            ):
                self.stats.skipped_showdowns += 1
                continue
            action_str = self._terminal_action_str(tree, int(node_idx))
            parsed = parse_action(action_str)
            if (
                "error" in parsed
                or int(parsed.get("st", -1)) != 3
                or int(parsed.get("pos", -1)) < 0
            ):
                self.stats.fallback_showdowns += 1
                continue
            replaced_rows.add(row)
            for river_card in self._river_cards:
                board_mask, local_legal, board_str = self._river_structure(river_card)
                hero_belief = np.zeros(N_HANDS, dtype=np.float32)
                villain_belief = np.zeros(N_HANDS, dtype=np.float32)
                hero_belief[self._local_to_global] = hero_reach[row] * local_legal
                villain_belief[self._local_to_global] = villain_reach[row] * local_legal
                hero_belief = _normalize(hero_belief * board_mask)
                villain_belief = _normalize(villain_belief * board_mask)
                features.append(
                    build_features(
                        [],
                        board_str,
                        action_str,
                        self.case.client_pos,
                        parsed,
                    )
                )
                beliefs.append(
                    np.concatenate([hero_belief, villain_belief]).astype(
                        np.float32,
                        copy=False,
                    )
                )
                masks.append(board_mask)
                tasks.append((row, local_legal))

        if not tasks:
            return out_h, out_v

        dataset = DualCFVDataset(
            features=np.stack(features).astype(np.float32, copy=False),
            belief=np.stack(beliefs).astype(np.float32, copy=False),
            hero_values=np.zeros((len(features), N_HANDS), dtype=np.float32),
            villain_values=np.zeros((len(features), N_HANDS), dtype=np.float32),
            hero_masks=np.stack(masks).astype(np.float32, copy=False),
            villain_masks=np.stack(masks).astype(np.float32, copy=False),
            labels=tuple(f"leaf-{idx}" for idx in range(len(features))),
        )
        started = time.perf_counter()
        preds = [
            predict_public_belief_dual_hand_cfv_model_vectorized(
                model,
                payload,
                dataset.features,
                dataset.belief,
                dataset.hero_masks,
                dataset.villain_masks,
                device=self.device,
                state_batch_size=32,
                hand_batch_size=self.batch_size,
            )
            for model, payload in self.loaded_ensemble
        ]
        pred_stack = np.stack(preds, axis=0)
        if pred_stack.shape[0] > 1:
            pred_std = np.std(pred_stack, axis=0, dtype=np.float32)
            legal_std = pred_std[
                np.stack([dataset.hero_masks, dataset.villain_masks], axis=0) > 0
            ]
            if legal_std.size:
                self.stats.prediction_std_sum += float(np.sum(legal_std))
                self.stats.prediction_std_count += int(legal_std.size)
                self.stats.prediction_std_max = max(
                    self.stats.prediction_std_max,
                    float(np.max(legal_std)),
                )
        pred = np.mean(pred_stack, axis=0, dtype=np.float32)
        if self.project_zero_sum:
            pred = project_dual_cfv_zero_sum(
                pred,
                dataset.belief,
                dataset.hero_masks,
                dataset.villain_masks,
            )
        pred = pred * self.value_scale
        self.stats.prediction_ms += (time.perf_counter() - started) * 1000.0
        self.stats.prediction_states += int(len(features))

        leaf_h = np.zeros_like(out_h)
        leaf_v = np.zeros_like(out_v)
        local_idx = np.arange(len(self.solver_hands), dtype=np.int32)
        global_idx = self._local_to_global
        valid_m_t = valid_m.T.copy()
        for pred_idx, (row, local_legal) in enumerate(tasks):
            hero_den = valid_m @ (villain_reach[row] * local_legal)
            villain_den = valid_m_t @ (hero_reach[row] * local_legal)
            legal_idx = local_idx[local_legal > 0]
            legal_global = global_idx[legal_idx]
            leaf_h[row, legal_idx] += pred[0, pred_idx, legal_global] * hero_den[legal_idx]
            leaf_v[row, legal_idx] += pred[1, pred_idx, legal_global] * villain_den[legal_idx]

        for row in sorted(replaced_rows):
            out_h[row] = leaf_h[row] / max(self._river_count_per_pair, 1.0)
            out_v[row] = leaf_v[row] / max(self._river_count_per_pair, 1.0)
        self.stats.replaced_showdowns += len(replaced_rows)
        return out_h, out_v

    def _terminal_action_str(self, tree: dict[str, Any], node_idx: int) -> str:
        street_action = _action_path_to_street_string(
            tree,
            node_idx,
            hero_stack_start=int(tree["stacks_h"][0]),
            villain_stack_start=int(tree["stacks_v"][0]),
        )
        return f"{self.action_prefix}/{street_action}/" if self.action_prefix else f"{street_action}/"

    def _river_structure(self, river_card: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        river_card = int(river_card)
        cached = self._river_cache.get(river_card)
        if cached is not None:
            return cached
        board5 = [*self.board4, river_card]
        board_mask = _global_board_mask(board5)
        local_legal = np.array(
            [0.0 if river_card in hand else 1.0 for hand in self.solver_hands],
            dtype=np.float32,
        )
        board_str = [_card_to_str(card) for card in board5]
        cached = (board_mask, local_legal, board_str)
        self._river_cache[river_card] = cached
        return cached

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
        self._river_cache: dict[int, tuple[np.ndarray, np.ndarray, list[str]]] = {}


def _solve_case(
    case: ResolverBenchmarkCase,
    *,
    belief_row: np.ndarray | None,
    loaded_ensemble: list[tuple[Any, dict[str, Any]]],
    device: Any,
    solver_iterations: int,
    solver_backend: str,
    value_scale: float,
    batch_size: int,
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
    street_parts = case.action_str.split("/")
    street_action = street_parts[street] if len(street_parts) > street else ""
    action_prefix = _pre_street_prefix(case.action_str, street)

    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    backend, backend_device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("learned river leaf A/B requires the CPU solver callback backend")

    baseline = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    started = time.perf_counter()
    baseline.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
    )
    baseline_ms = (time.perf_counter() - started) * 1000.0
    nav = _parse_nav(street_action, baseline)
    baseline_node = baseline.navigate(nav)
    if baseline_node is None or baseline_node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "terminal_or_missing_node"}
    hand = tuple(sorted(our_cards_idx))
    baseline_strategy = _strategy_vector(baseline.get_strategy(hand, baseline_node))

    learned = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    learned_pre_node = learned.navigate(nav)
    if learned_pre_node is None or learned_pre_node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "learned_pre_terminal_or_missing_node"}
    active_subtree_node_idx = int(learned._tree["all_nodes"].index(learned_pre_node))
    callback = LearnedRiverLeafCallback(
        case=case,
        board4=board_idx,
        action_prefix=action_prefix,
        loaded_ensemble=loaded_ensemble,
        device=device,
        value_scale=value_scale,
        batch_size=batch_size,
        project_zero_sum=project_zero_sum,
        active_subtree_node_idx=active_subtree_node_idx,
    )
    callback.solver_hands = list(learned.hands)
    started = time.perf_counter()
    learned.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
        showdown_leaf_fn=callback,
    )
    learned_ms = (time.perf_counter() - started) * 1000.0
    learned_node = learned.navigate(nav)
    if learned_node is None or learned_node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "learned_terminal_or_missing_node"}
    learned_strategy = _strategy_vector(learned.get_strategy(hand, learned_node))

    baseline_action = int(np.argmax(baseline_strategy))
    learned_action = int(np.argmax(learned_strategy))
    l1 = float(np.abs(baseline_strategy - learned_strategy).sum())
    leaf_applied = bool(callback.stats.replaced_showdowns > 0)
    return {
        "label": case.label,
        "street": street,
        "passed": bool(np.isfinite(l1)),
        "leaf_applied": leaf_applied,
        "baseline_action": baseline_action,
        "baseline_increment": solver_action_to_slumbot(baseline_action, baseline_node, baseline, parsed),
        "learned_action": learned_action,
        "learned_increment": solver_action_to_slumbot(learned_action, learned_node, learned, parsed),
        "action_agreement": bool(baseline_action == learned_action),
        "action_l1_drift": round(l1, 8),
        "baseline_strategy": baseline_strategy.round(6).tolist(),
        "learned_strategy": learned_strategy.round(6).tolist(),
        "baseline_solve_ms": round(float(baseline_ms), 3),
        "learned_solve_ms": round(float(learned_ms), 3),
        "leaf_prediction_ms": round(float(callback.stats.prediction_ms), 3),
        "leaf_prediction_states": int(callback.stats.prediction_states),
        "leaf_prediction_std_mean": (
            round(
                float(
                    callback.stats.prediction_std_sum
                    / max(callback.stats.prediction_std_count, 1)
                ),
                8,
            )
            if callback.stats.prediction_std_count
            else 0.0
        ),
        "leaf_prediction_std_max": round(float(callback.stats.prediction_std_max), 8),
        "project_zero_sum": bool(project_zero_sum),
        "leaf_callback_calls": int(callback.stats.callback_calls),
        "replaced_showdowns": int(callback.stats.replaced_showdowns),
        "fallback_showdowns": int(callback.stats.fallback_showdowns),
        "skipped_showdowns": int(callback.stats.skipped_showdowns),
        "solver_n_hands": int(learned.n),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed turn resolver A/B with learned river continuation leaves."
    )
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum number of cases to scan.",
    )
    parser.add_argument(
        "--target-leaf-applied",
        type=int,
        default=0,
        help="Stop early after this many cases with learned leaves applied; 0 disables.",
    )
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--project-zero-sum", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    cases = load_cases_json(args.cases)
    base_dataset = None
    if args.cfv_cache:
        base_dataset, _ = load_public_belief_cfv_dataset_cache(args.cfv_cache)
        if base_dataset.features.shape[0] != len(cases):
            raise ValueError("case count does not match CFV cache rows")
    loaded = load_public_belief_dual_hand_cfv_ensemble(args.checkpoint, device=device)
    max_cases = max(1, min(int(args.limit), len(cases)))
    target_leaf_applied = max(0, int(args.target_leaf_applied))

    records = []
    for idx, case in enumerate(cases[:max_cases]):
        belief_row = base_dataset.belief[idx] if base_dataset is not None else None
        record = _solve_case(
            case,
            belief_row=belief_row,
            loaded_ensemble=loaded,
            device=device,
            solver_iterations=args.solver_iterations,
            solver_backend=args.solver_backend,
            value_scale=args.value_scale,
            batch_size=args.batch_size,
            project_zero_sum=args.project_zero_sum,
        )
        records.append(record)
        if target_leaf_applied and sum(1 for item in records if item.get("leaf_applied")) >= target_leaf_applied:
            break
    evaluated = [record for record in records if record.get("passed") and "action_l1_drift" in record]
    leaf_evaluated = [record for record in evaluated if record.get("leaf_applied")]
    drift = [float(record["action_l1_drift"]) for record in evaluated]
    leaf_drift = [float(record["action_l1_drift"]) for record in leaf_evaluated]
    learned_ms = [float(record["learned_solve_ms"]) for record in evaluated]
    baseline_ms = [float(record["baseline_solve_ms"]) for record in evaluated]
    metrics = {
        "mode": "learned_river_leaf_resolver_ab",
        "passed": bool(leaf_evaluated and all(record.get("passed") for record in evaluated)),
        "promotion_blockers": [
            "learned_leaf_resolver_ab_is_diagnostic_not_slumbot_confidence",
        ],
        "checkpoint_count": len(args.checkpoint),
        "checkpoints": [str(path) for path in args.checkpoint],
        "cases": str(args.cases),
        "cfv_cache": str(args.cfv_cache) if args.cfv_cache else None,
        "device": str(device),
        "project_zero_sum": bool(args.project_zero_sum),
        "solver_iterations": int(args.solver_iterations),
        "solver_backend": args.solver_backend,
        "case_scan_limit": int(max_cases),
        "target_leaf_applied": int(target_leaf_applied),
        "n_cases": len(records),
        "n_evaluated": len(evaluated),
        "n_leaf_applied": len(leaf_evaluated),
        "action_agreement_rate": (
            round(float(np.mean([record["action_agreement"] for record in evaluated])), 6)
            if evaluated
            else 0.0
        ),
        "leaf_action_agreement_rate": (
            round(float(np.mean([record["action_agreement"] for record in leaf_evaluated])), 6)
            if leaf_evaluated
            else 0.0
        ),
        "mean_action_l1_drift": round(float(np.mean(drift)), 8) if drift else None,
        "max_action_l1_drift": round(float(np.max(drift)), 8) if drift else None,
        "leaf_mean_action_l1_drift": (
            round(float(np.mean(leaf_drift)), 8) if leaf_drift else None
        ),
        "leaf_max_action_l1_drift": (
            round(float(np.max(leaf_drift)), 8) if leaf_drift else None
        ),
        "mean_baseline_solve_ms": round(float(np.mean(baseline_ms)), 3) if baseline_ms else None,
        "mean_learned_solve_ms": round(float(np.mean(learned_ms)), 3) if learned_ms else None,
        "records": records,
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
