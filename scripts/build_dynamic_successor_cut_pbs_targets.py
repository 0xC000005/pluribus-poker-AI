#!/usr/bin/env python3
"""Export dynamic joint-PBS successor-cut targets from exact CFR traces."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from dataclasses import dataclass, field
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
from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json

from build_successor_cut_pbs_targets import _legal_uniform, _reach_summary  # noqa: E402
from eval_joint_pbs_resolver_cut_ab import (  # noqa: E402
    _frontier_action_shape,
    _select_successor_cut_node_records,
)
from eval_joint_pbs_resolver_leaf_ab import (  # noqa: E402
    _global_board_mask,
    _local_ranges_from_belief,
    _pre_street_prefix,
    _strategy_vector,
)
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    build_features,
    card_str_to_index,
    get_legal_mask_from_parsed,
    parse_action,
)
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402

_BET_TOKEN_RE = re.compile(r"b\d+")
_SUITS = ("c", "d", "h", "s")
_RANKS = tuple("23456789TJQKA")


def _card_to_str(card: int) -> str:
    return _RANKS[int(card) // 4] + _SUITS[int(card) % 4]


def _root_policy_target_hands(
    solver_hands: list[tuple[int, int]],
    observed_cards: list[int],
    *,
    include_all_hands: bool,
) -> list[tuple[int, int]]:
    if include_all_hands:
        return [tuple(sorted(hand)) for hand in solver_hands]
    return [tuple(sorted(observed_cards))]


def _count_value_targets(policy_weights: list[float]) -> int:
    return int(sum(float(weight) <= 0.0 for weight in policy_weights))


def _case_window(
    cases: list[ResolverBenchmarkCase],
    *,
    start_index: int,
    limit: int,
) -> tuple[int, list[ResolverBenchmarkCase]]:
    start = max(0, min(int(start_index), len(cases)))
    if int(limit) <= 0:
        stop = len(cases)
    else:
        stop = min(start + int(limit), len(cases))
    selected = cases[start:stop]
    if not selected:
        raise ValueError("selected dynamic successor target slice is empty")
    return start, selected


def _normalized_strategy_target(strategy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    legal = (np.asarray(legal_mask, dtype=np.float32) > 0).astype(np.float32)
    target = np.asarray(strategy, dtype=np.float32) * legal
    total = float(target.sum())
    if total > 1e-8:
        return (target / total).astype(np.float32, copy=False)
    return _legal_uniform(legal)


@dataclass
class DynamicTargetCollector:
    case: ResolverBenchmarkCase
    board: list[str]
    board_idx: list[int]
    board_mask: np.ndarray
    solver_hands: list[tuple[int, int]]
    records_by_node: dict[int, dict[str, Any]]
    value_scale: float
    max_targets: int
    value_weight_mode: str = "mask"
    features: list[np.ndarray] = field(default_factory=list)
    policy_features: list[np.ndarray] = field(default_factory=list)
    beliefs: list[np.ndarray] = field(default_factory=list)
    legal_masks: list[np.ndarray] = field(default_factory=list)
    target_probs: list[np.ndarray] = field(default_factory=list)
    policy_weights: list[float] = field(default_factory=list)
    hero_values: list[np.ndarray] = field(default_factory=list)
    villain_values: list[np.ndarray] = field(default_factory=list)
    hero_masks: list[np.ndarray] = field(default_factory=list)
    villain_masks: list[np.ndarray] = field(default_factory=list)
    hero_value_weights: list[np.ndarray] = field(default_factory=list)
    villain_value_weights: list[np.ndarray] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    cut_records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._local_to_global = np.asarray(
            [_HAND_TO_INDEX[tuple(sorted(hand))] for hand in self.solver_hands],
            dtype=np.int32,
        )

    def __call__(self, **kwargs) -> None:
        if 0 < self.max_targets <= len(self.features):
            return
        node_indices = np.asarray(kwargs["node_indices"], dtype=np.int32)
        hero_reach = np.asarray(kwargs["hero_reach"], dtype=np.float32)
        villain_reach = np.asarray(kwargs["villain_reach"], dtype=np.float32)
        hero_values = np.asarray(kwargs["hero_values"], dtype=np.float32)
        villain_values = np.asarray(kwargs["villain_values"], dtype=np.float32)
        valid_m = np.asarray(kwargs["valid_m"], dtype=np.float32)
        valid_m_t = valid_m.T.copy()
        iteration = int(kwargs["iteration"])
        for row, node_idx in enumerate(node_indices.tolist()):
            if 0 < self.max_targets <= len(self.features):
                return
            record = self.records_by_node.get(int(node_idx))
            if record is None:
                continue
            parsed = parse_action(str(record["action_str"]))
            if "error" in parsed:
                continue
            legal_mask = get_legal_mask_from_parsed(
                parsed,
                str(record["action_str"]),
                int(self.case.client_pos),
            ).astype(np.float32)
            hero_den = valid_m @ villain_reach[row]
            villain_den = valid_m_t @ hero_reach[row]
            hero_mask_local = (hero_den > 1e-8).astype(np.float32)
            villain_mask_local = (villain_den > 1e-8).astype(np.float32)
            hero_local = np.zeros_like(hero_values[row], dtype=np.float32)
            villain_local = np.zeros_like(villain_values[row], dtype=np.float32)
            np.divide(
                hero_values[row],
                np.maximum(hero_den, 1e-8),
                out=hero_local,
                where=hero_mask_local > 0,
            )
            np.divide(
                villain_values[row],
                np.maximum(villain_den, 1e-8),
                out=villain_local,
                where=villain_mask_local > 0,
            )
            hero_global = np.zeros(N_HANDS, dtype=np.float32)
            villain_global = np.zeros(N_HANDS, dtype=np.float32)
            hero_mask = np.zeros(N_HANDS, dtype=np.float32)
            villain_mask = np.zeros(N_HANDS, dtype=np.float32)
            hero_weight = np.zeros(N_HANDS, dtype=np.float32)
            villain_weight = np.zeros(N_HANDS, dtype=np.float32)
            hero_global[self._local_to_global] = hero_local / float(self.value_scale)
            villain_global[self._local_to_global] = villain_local / float(self.value_scale)
            hero_mask[self._local_to_global] = hero_mask_local
            villain_mask[self._local_to_global] = villain_mask_local
            if self.value_weight_mode == "denominator_squared":
                # The runtime callback multiplies conditional CFVs by these denominators
                # before regret updates. Squared weights train on the consumed tensor.
                hero_weight[self._local_to_global] = (hero_den * hero_mask_local) ** 2
                villain_weight[self._local_to_global] = (villain_den * villain_mask_local) ** 2
            else:
                hero_weight[self._local_to_global] = hero_mask_local
                villain_weight[self._local_to_global] = villain_mask_local

            hero_belief = np.zeros(N_HANDS, dtype=np.float32)
            villain_belief = np.zeros(N_HANDS, dtype=np.float32)
            hero_belief[self._local_to_global] = hero_reach[row]
            villain_belief[self._local_to_global] = villain_reach[row]
            hero_belief = _normalize(hero_belief * self.board_mask)
            villain_belief = _normalize(villain_belief * self.board_mask)
            public_feature = build_features(
                [],
                self.board,
                str(record["action_str"]),
                int(self.case.client_pos),
                parsed,
            ).astype(np.float32, copy=False)
            self.features.append(public_feature)
            self.policy_features.append(public_feature)
            self.beliefs.append(np.concatenate([hero_belief, villain_belief]).astype(np.float32))
            self.legal_masks.append(legal_mask)
            self.target_probs.append(_legal_uniform(legal_mask))
            self.policy_weights.append(0.0)
            self.hero_values.append(hero_global)
            self.villain_values.append(villain_global)
            self.hero_masks.append(hero_mask)
            self.villain_masks.append(villain_mask)
            self.hero_value_weights.append(hero_weight)
            self.villain_value_weights.append(villain_weight)
            label = f"{self.case.label}-cut{int(record['cut_pos'])}-iter{iteration}"
            self.labels.append(label)
            self.cut_records.append(
                {
                    "label": label,
                    "root_label": self.case.label,
                    "cut_pos": int(record["cut_pos"]),
                    "iteration": iteration,
                    "action_str": str(record["action_str"]),
                    "action_shape": str(record["action_shape"]),
                    "bet_count": int(record["bet_count"]),
                    "actor_to_act": int(parsed.get("pos", -1)),
                    "client_pos": int(self.case.client_pos),
                    "legal_action_count": int(np.count_nonzero(legal_mask > 0)),
                    "hero_mask_count": int(hero_mask.sum()),
                    "villain_mask_count": int(villain_mask.sum()),
                    **_reach_summary("hero_reach", hero_reach[row]),
                    **_reach_summary("villain_reach", villain_reach[row]),
                }
            )


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    total = float(values.sum())
    if total <= 0:
        return np.full_like(values, 1.0 / max(len(values), 1), dtype=np.float32)
    return values / total


def export_dynamic_successor_cut_targets(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    output: str | Path,
    solver_iterations: int,
    solver_backend: str,
    limit: int,
    target_cuts: int,
    min_bet_count: int,
    target_action_shapes: tuple[str, ...],
    value_scale: float,
    start_index: int = 0,
    value_weight_mode: str = "mask",
    include_root_policy_targets: bool = False,
    include_all_hand_root_policy_targets: bool = False,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    backend, backend_device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("dynamic successor target export requires CPU backend")
    if value_weight_mode not in ("mask", "denominator_squared"):
        raise ValueError("value_weight_mode must be 'mask' or 'denominator_squared'")
    start, selected_cases = _case_window(cases, start_index=start_index, limit=limit)

    all_features: list[np.ndarray] = []
    all_policy_features: list[np.ndarray] = []
    all_beliefs: list[np.ndarray] = []
    all_legal_masks: list[np.ndarray] = []
    all_target_probs: list[np.ndarray] = []
    all_policy_weights: list[float] = []
    all_hero_values: list[np.ndarray] = []
    all_villain_values: list[np.ndarray] = []
    all_hero_masks: list[np.ndarray] = []
    all_villain_masks: list[np.ndarray] = []
    all_hero_value_weights: list[np.ndarray] = []
    all_villain_value_weights: list[np.ndarray] = []
    all_labels: list[str] = []
    cut_records: list[dict[str, Any]] = []
    root_records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    root_policy_requested = bool(include_root_policy_targets or include_all_hand_root_policy_targets)
    for local_idx, case in enumerate(selected_cases):
        idx = start + local_idx
        value_target_count = _count_value_targets(all_policy_weights)
        if 0 < target_cuts <= value_target_count and not root_policy_requested:
            break
        started = time.perf_counter()
        parsed = parse_action(case.action_str)
        if "error" in parsed or int(parsed.get("st", -1)) != 2:
            skipped.append({"label": case.label, "reason": "not_turn_or_parse_error"})
            continue
        board_idx = [card_str_to_index(card) for card in case.board[:4]]
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
        full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
        hero_range, villain_range = _local_ranges_from_belief(dataset.belief[idx], full_hands)
        solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
        active_node = solver.navigate(_parse_nav(street_action, solver))
        if active_node is None or active_node.is_terminal:
            skipped.append({"label": case.label, "reason": "terminal_or_missing_node"})
            continue
        _candidates, selected = _select_successor_cut_node_records(
            solver,
            active_node,
            action_prefix=action_prefix,
            client_pos=case.client_pos,
            min_bet_count=min_bet_count,
            target_action_shapes=target_action_shapes,
            risk_predictor=None,
        )
        if not selected and not root_policy_requested:
            skipped.append({"label": case.label, "reason": "no_matching_successor_cuts"})
            continue
        remaining = int(target_cuts - value_target_count) if target_cuts > 0 else 0
        max_for_root = remaining if remaining > 0 else 0
        records_by_node = (
            {int(record["node_idx"]): record for record in selected}
            if max_for_root > 0
            else {}
        )
        collector = DynamicTargetCollector(
            case=case,
            board=list(case.board),
            board_idx=board_idx,
            board_mask=_global_board_mask(board_idx),
            solver_hands=list(solver.hands),
            records_by_node=records_by_node,
            value_scale=value_scale,
            max_targets=max_for_root,
            value_weight_mode=value_weight_mode,
        )
        solve_kwargs: dict[str, Any] = {}
        if records_by_node:
            solve_kwargs["trace_node_indices"] = list(records_by_node)
            solve_kwargs["trace_node_fn"] = collector
        solver.solve(
            n_iterations=solver_iterations,
            hero_range=hero_range,
            villain_range=villain_range,
            backend=backend,
            device=backend_device,
            **solve_kwargs,
        )
        if (
            (include_root_policy_targets or include_all_hand_root_policy_targets)
            and active_node is not None
            and not active_node.is_terminal
        ):
            our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
            legal_mask = get_legal_mask_from_parsed(
                parsed,
                case.action_str,
                int(case.client_pos),
            ).astype(np.float32)
            public_feature = build_features(
                [],
                list(case.board),
                case.action_str,
                int(case.client_pos),
                parsed,
            ).astype(np.float32, copy=False)
            policy_hands = _root_policy_target_hands(
                list(solver.hands),
                our_cards_idx,
                include_all_hands=include_all_hand_root_policy_targets,
            )
            for policy_hand in policy_hands:
                root_strategy = _strategy_vector(solver.get_strategy(policy_hand, active_node))
                hole_cards = [_card_to_str(policy_hand[0]), _card_to_str(policy_hand[1])]
                policy_feature = build_features(
                    hole_cards,
                    list(case.board),
                    case.action_str,
                    int(case.client_pos),
                    parsed,
                ).astype(np.float32, copy=False)
                hand_index = int(_HAND_TO_INDEX[tuple(sorted(policy_hand))])
                root_label = (
                    f"{case.label}-root-policy-hand{hand_index:04d}"
                    if include_all_hand_root_policy_targets
                    else f"{case.label}-root-policy"
                )
                collector.features.append(public_feature)
                collector.policy_features.append(policy_feature)
                collector.beliefs.append(np.asarray(dataset.belief[idx], dtype=np.float32))
                collector.legal_masks.append(legal_mask)
                collector.target_probs.append(_normalized_strategy_target(root_strategy, legal_mask))
                collector.policy_weights.append(1.0)
                collector.hero_values.append(np.zeros(N_HANDS, dtype=np.float32))
                collector.villain_values.append(np.zeros(N_HANDS, dtype=np.float32))
                collector.hero_masks.append(np.zeros(N_HANDS, dtype=np.float32))
                collector.villain_masks.append(np.zeros(N_HANDS, dtype=np.float32))
                collector.hero_value_weights.append(np.zeros(N_HANDS, dtype=np.float32))
                collector.villain_value_weights.append(np.zeros(N_HANDS, dtype=np.float32))
                collector.labels.append(root_label)
                collector.cut_records.append(
                    {
                        "label": root_label,
                        "root_label": case.label,
                        "cut_pos": -1,
                        "iteration": -1,
                        "action_str": str(case.action_str),
                        "action_shape": _frontier_action_shape(str(case.action_str)),
                        "bet_count": int(len(_BET_TOKEN_RE.findall(case.action_str))),
                        "actor_to_act": int(parsed.get("pos", -1)),
                        "client_pos": int(case.client_pos),
                        "legal_action_count": int(np.count_nonzero(legal_mask > 0)),
                        "hero_mask_count": 0,
                        "villain_mask_count": 0,
                        "policy_weight": 1.0,
                        "target_kind": "root_policy",
                        "root_policy_target_mode": (
                            "all_hands"
                            if include_all_hand_root_policy_targets
                            else "observed_hand"
                        ),
                        "hand_index": hand_index,
                    }
                )
        all_features.extend(collector.features)
        all_policy_features.extend(collector.policy_features)
        all_beliefs.extend(collector.beliefs)
        all_legal_masks.extend(collector.legal_masks)
        all_target_probs.extend(collector.target_probs)
        all_policy_weights.extend(collector.policy_weights)
        all_hero_values.extend(collector.hero_values)
        all_villain_values.extend(collector.villain_values)
        all_hero_masks.extend(collector.hero_masks)
        all_villain_masks.extend(collector.villain_masks)
        all_hero_value_weights.extend(collector.hero_value_weights)
        all_villain_value_weights.extend(collector.villain_value_weights)
        all_labels.extend(collector.labels)
        cut_records.extend(collector.cut_records)
        root_records.append(
            {
                "label": case.label,
                "dynamic_targets": int(len(collector.labels)),
                "trace_nodes": int(len(records_by_node)),
                "solver_latency_ms": round(float(getattr(solver, "last_solve_ms", 0.0)), 3),
                "export_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )

    if not all_features:
        raise RuntimeError("no dynamic successor cut targets were exported")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(all_features).astype(np.float32, copy=False),
        policy_features=np.stack(all_policy_features).astype(np.float32, copy=False),
        belief=np.stack(all_beliefs).astype(np.float32, copy=False),
        legal_masks=np.stack(all_legal_masks).astype(np.float32, copy=False),
        target_probs=np.stack(all_target_probs).astype(np.float32, copy=False),
        policy_weights=np.asarray(all_policy_weights, dtype=np.float32),
        hero_values=np.stack(all_hero_values).astype(np.float32, copy=False),
        villain_values=np.stack(all_villain_values).astype(np.float32, copy=False),
        hero_masks=np.stack(all_hero_masks).astype(np.float32, copy=False),
        villain_masks=np.stack(all_villain_masks).astype(np.float32, copy=False),
        hero_value_weights=np.stack(all_hero_value_weights).astype(np.float32, copy=False),
        villain_value_weights=np.stack(all_villain_value_weights).astype(np.float32, copy=False),
        labels=np.asarray(all_labels),
    )
    metadata = {
        "mode": "dynamic_successor_cut_joint_pbs_targets",
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "output": str(output),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "start_index": int(start),
        "limit": int(limit),
        "target_cuts": int(target_cuts),
        "min_bet_count": int(min_bet_count),
        "target_action_shapes": list(target_action_shapes),
        "value_scale": float(value_scale),
        "value_weight_mode": value_weight_mode,
        "include_root_policy_targets": bool(include_root_policy_targets),
        "include_all_hand_root_policy_targets": bool(include_all_hand_root_policy_targets),
        "n_targets": int(len(all_features)),
        "policy_target_count": int(sum(float(weight) > 0.0 for weight in all_policy_weights)),
        "n_roots": int(len(root_records)),
        "cut_records": cut_records,
        "records": root_records,
        "skipped": skipped[:200],
    }
    save_metrics(metadata, output.with_suffix(".json"))
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export dynamic joint-PBS successor targets from exact CFR traces."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--target-cuts", type=int, default=0)
    parser.add_argument("--min-bet-count", type=int, default=0)
    parser.add_argument("--target-action-shapes", default="")
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument(
        "--value-weight-mode",
        choices=("mask", "denominator_squared"),
        default="mask",
    )
    parser.add_argument(
        "--include-root-policy-targets",
        action="store_true",
        help="Also add exact root resolver policy rows for the actual private hand.",
    )
    parser.add_argument(
        "--include-all-hand-root-policy-targets",
        action="store_true",
        help="Add exact root resolver policy rows for every legal private hand at each public root.",
    )
    args = parser.parse_args(argv)
    target_action_shapes = tuple(
        part.strip() for part in args.target_action_shapes.split(",") if part.strip()
    )
    metrics = export_dynamic_successor_cut_targets(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        output=args.output,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        start_index=args.start_index,
        limit=args.limit,
        target_cuts=args.target_cuts,
        min_bet_count=args.min_bet_count,
        target_action_shapes=target_action_shapes,
        value_scale=args.value_scale,
        value_weight_mode=args.value_weight_mode,
        include_root_policy_targets=args.include_root_policy_targets,
        include_all_hand_root_policy_targets=args.include_all_hand_root_policy_targets,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
