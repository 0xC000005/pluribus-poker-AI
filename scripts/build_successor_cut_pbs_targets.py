#!/usr/bin/env python3
"""Export joint PBS targets for immediate successor cut nodes."""

from __future__ import annotations

import argparse
import itertools
import json
import re
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
from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX
from poker_ai.research.belief_value_probe import (
    compute_hero_cfv_vector,
    compute_villain_cfv_vector,
    load_public_belief_cfv_dataset_cache,
    save_metrics,
)
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json

from eval_joint_pbs_resolver_cut_ab import (  # noqa: E402
    _action_path_to_street_string,
    _global_board_mask,
    _local_ranges_from_belief,
    _normalize,
    _pre_street_prefix,
    _successor_cut_node_indices,
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


def _average_strategy(
    strategy_sum: np.ndarray,
    node_idx: int,
    actions: list[int],
) -> np.ndarray:
    sums = np.stack([strategy_sum[node_idx, action] for action in actions], axis=0)
    totals = sums.sum(axis=0, keepdims=True)
    uniform = np.full_like(sums, 1.0 / max(len(actions), 1), dtype=np.float64)
    return np.where(totals > 0, sums / np.maximum(totals, 1e-12), uniform)


def _average_reaches(
    solver: StreetSolver,
    hero_range: np.ndarray,
    villain_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = solver._tree
    n_nodes = int(tree["n_nodes"])
    hero = np.zeros((n_nodes, solver.n), dtype=np.float32)
    villain = np.zeros_like(hero)
    hero[0] = np.asarray(hero_range, dtype=np.float32)
    villain[0] = np.asarray(villain_range, dtype=np.float32)
    for node_idx in range(n_nodes):
        player = int(tree["player"][node_idx])
        if player == -1:
            continue
        actions = list(tree["decision_actions"][node_idx])
        if not actions:
            continue
        strategy = _average_strategy(solver._strategy_sum, node_idx, actions)
        for action_pos, action in enumerate(actions):
            child_idx = int(tree["children"][node_idx, action])
            if player == 0:
                hero[child_idx] = hero[node_idx] * strategy[action_pos]
                villain[child_idx] = villain[node_idx]
            else:
                hero[child_idx] = hero[node_idx]
                villain[child_idx] = villain[node_idx] * strategy[action_pos]
    return hero, villain


def _global_belief_from_reaches(
    *,
    solver_hands: list[tuple[int, int]],
    hero_reach: np.ndarray,
    villain_reach: np.ndarray,
    board_mask: np.ndarray,
) -> np.ndarray:
    hero = np.zeros(N_HANDS, dtype=np.float32)
    villain = np.zeros(N_HANDS, dtype=np.float32)
    for local_idx, hand in enumerate(solver_hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        hero[global_idx] = float(hero_reach[local_idx])
        villain[global_idx] = float(villain_reach[local_idx])
    hero = _normalize(hero * board_mask)
    villain = _normalize(villain * board_mask)
    return np.concatenate([hero, villain]).astype(np.float32)


def _global_cfv_values(
    *,
    solver: StreetSolver,
    node: Any,
    hero_reach: np.ndarray,
    villain_reach: np.ndarray,
    value_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hero_local, hero_mask_local = compute_hero_cfv_vector(solver, node, villain_reach)
    villain_local, villain_mask_local = compute_villain_cfv_vector(solver, node, hero_reach)
    hero_values = np.zeros(N_HANDS, dtype=np.float32)
    villain_values = np.zeros(N_HANDS, dtype=np.float32)
    hero_masks = np.zeros(N_HANDS, dtype=np.float32)
    villain_masks = np.zeros(N_HANDS, dtype=np.float32)
    for local_idx, hand in enumerate(solver.hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        hero_values[global_idx] = float(hero_local[local_idx]) / float(value_scale)
        villain_values[global_idx] = float(villain_local[local_idx]) / float(value_scale)
        hero_masks[global_idx] = float(hero_mask_local[local_idx])
        villain_masks[global_idx] = float(villain_mask_local[local_idx])
    return hero_values, villain_values, hero_masks, villain_masks


def _legal_uniform(mask: np.ndarray) -> np.ndarray:
    legal = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
    total = float(legal.sum())
    if total <= 0:
        legal = np.zeros(N_ACTIONS, dtype=np.float32)
        legal[1] = 1.0
        return legal
    return legal / total


def _action_shape(action_str: str) -> str:
    return _BET_TOKEN_RE.sub("b", action_str)


def _reach_summary(prefix: str, reach: np.ndarray) -> dict[str, float | int]:
    probs = _normalize(reach)
    positive = probs[probs > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    denom = float(np.log(max(positive.size, 2))) if positive.size else 1.0
    sorted_probs = np.sort(probs)[::-1]
    return {
        f"{prefix}_support": int(positive.size),
        f"{prefix}_top1_mass": round(float(sorted_probs[0]) if sorted_probs.size else 0.0, 6),
        f"{prefix}_top10_mass": round(float(sorted_probs[:10].sum()) if sorted_probs.size else 0.0, 6),
        f"{prefix}_entropy": round(entropy, 6),
        f"{prefix}_normalized_entropy": round(entropy / denom if denom > 0 else 0.0, 6),
    }


def _matches_frontier_filter(
    *,
    action_shape: str,
    bet_count: int,
    target_action_shapes: tuple[str, ...],
    min_bet_count: int,
) -> bool:
    if target_action_shapes and action_shape not in target_action_shapes:
        return False
    return int(bet_count) >= int(min_bet_count)


def _policy_target_for_cut(
    solver: StreetSolver,
    node: Any,
    case: ResolverBenchmarkCase,
    parsed: dict[str, Any],
    legal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    public_features = build_features([], list(case.board), _safe_action_str(case), case.client_pos, parsed)
    if int(parsed.get("pos", -1)) != int(case.client_pos):
        return public_features, _legal_uniform(legal_mask), 0.0
    policy_features = build_features(
        list(case.hole_cards),
        list(case.board),
        _safe_action_str(case),
        case.client_pos,
        parsed,
    )
    hand = tuple(sorted(card_str_to_index(card) for card in case.hole_cards))
    strategy = solver.get_strategy(hand, node)
    target = np.zeros(N_ACTIONS, dtype=np.float32)
    for action, prob in strategy.items():
        if 0 <= int(action) < N_ACTIONS:
            target[int(action)] = float(prob)
    target = target * (legal_mask > 0)
    total = float(target.sum())
    return policy_features, target / total if total > 0 else _legal_uniform(legal_mask), 1.0


def _safe_action_str(case: ResolverBenchmarkCase) -> str:
    return str(case.action_str)


def _cut_action_str(
    solver: StreetSolver,
    node_idx: int,
    action_prefix: str,
) -> str:
    street_action = _action_path_to_street_string(
        solver._tree,
        node_idx,
        hero_stack_start=int(solver._tree["stacks_h"][0]),
        villain_stack_start=int(solver._tree["stacks_v"][0]),
    )
    return f"{action_prefix}/{street_action}" if action_prefix else street_action


def export_successor_cut_targets(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    output: str | Path,
    solver_iterations: int = 25,
    solver_backend: str = "cpu",
    value_scale: float = 20000.0,
    limit: int = 0,
    min_bet_count: int = 0,
    target_action_shapes: tuple[str, ...] = (),
    target_cuts: int = 0,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    if limit > 0:
        cases = cases[: int(limit)]
    backend, backend_device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("successor cut target export requires CPU solver backend")

    features: list[np.ndarray] = []
    policy_features: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    target_probs: list[np.ndarray] = []
    policy_weights: list[float] = []
    hero_values: list[np.ndarray] = []
    villain_values: list[np.ndarray] = []
    hero_masks: list[np.ndarray] = []
    villain_masks: list[np.ndarray] = []
    labels: list[str] = []
    records: list[dict[str, Any]] = []
    cut_records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    roots_scanned = 0
    filtered_cut_count = 0

    for root_idx, case in enumerate(cases):
        roots_scanned += 1
        started = time.perf_counter()
        parsed = parse_action(case.action_str)
        if "error" in parsed or int(parsed.get("st", -1)) != 2:
            skipped.append({"label": case.label, "reason": "parse_or_non_turn"})
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
        solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
        full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
        hero_range, villain_range = _local_ranges_from_belief(
            base_dataset.belief[root_idx],
            full_hands,
        )
        solver.solve(
            n_iterations=solver_iterations,
            hero_range=hero_range,
            villain_range=villain_range,
            backend=backend,
            device=backend_device,
        )
        nav = _parse_nav(street_action, solver)
        active_node = solver.navigate(nav)
        if active_node is None or active_node.is_terminal:
            skipped.append({"label": case.label, "reason": "terminal_or_missing_node"})
            continue
        cut_indices = _successor_cut_node_indices(solver, active_node)
        if not cut_indices:
            skipped.append({"label": case.label, "reason": "no_successor_cut_nodes"})
            continue
        reach_h, reach_v = _average_reaches(solver, hero_range, villain_range)
        board_mask = _global_board_mask(board_idx)
        emitted = 0
        root_filtered = 0
        for cut_pos, cut_idx in enumerate(cut_indices):
            node = solver._tree["all_nodes"][cut_idx]
            action_str = _cut_action_str(solver, cut_idx, action_prefix)
            parsed_cut = parse_action(action_str)
            if "error" in parsed_cut:
                skipped.append({"label": f"{case.label}:cut{cut_pos}", "reason": "cut_parse_error"})
                continue
            action_shape = _action_shape(action_str)
            bet_count = int(len(_BET_TOKEN_RE.findall(action_str)))
            if not _matches_frontier_filter(
                action_shape=action_shape,
                bet_count=bet_count,
                target_action_shapes=target_action_shapes,
                min_bet_count=min_bet_count,
            ):
                filtered_cut_count += 1
                root_filtered += 1
                continue
            public_feature = build_features([], list(case.board), action_str, case.client_pos, parsed_cut)
            legal_mask = get_legal_mask_from_parsed(parsed_cut, action_str, case.client_pos).astype(np.float32)
            cut_case = ResolverBenchmarkCase(
                label=f"{case.label}-cut{cut_pos}",
                hole_cards=case.hole_cards,
                board=case.board,
                action_str=action_str,
                client_pos=case.client_pos,
                source="successor_cut_frontier",
            )
            pol_feature, target, pol_weight = _policy_target_for_cut(
                solver,
                node,
                cut_case,
                parsed_cut,
                legal_mask,
            )
            hv, vv, hm, vm = _global_cfv_values(
                solver=solver,
                node=node,
                hero_reach=reach_h[cut_idx],
                villain_reach=reach_v[cut_idx],
                value_scale=value_scale,
            )
            features.append(public_feature.astype(np.float32, copy=False))
            policy_features.append(pol_feature.astype(np.float32, copy=False))
            beliefs.append(
                _global_belief_from_reaches(
                    solver_hands=list(solver.hands),
                    hero_reach=reach_h[cut_idx],
                    villain_reach=reach_v[cut_idx],
                    board_mask=board_mask,
                )
            )
            legal_masks.append(legal_mask)
            target_probs.append(target.astype(np.float32, copy=False))
            policy_weights.append(float(pol_weight))
            hero_values.append(hv)
            villain_values.append(vv)
            hero_masks.append(hm)
            villain_masks.append(vm)
            labels.append(cut_case.label)
            cut_records.append(
                {
                    "label": cut_case.label,
                    "root_label": case.label,
                    "cut_pos": int(cut_pos),
                    "action_str": action_str,
                    "action_shape": action_shape,
                    "bet_count": bet_count,
                    "actor_to_act": int(parsed_cut.get("pos", -1)),
                    "client_pos": int(case.client_pos),
                    "policy_weight": float(pol_weight),
                    "legal_action_count": int(np.count_nonzero(legal_mask > 0)),
                    "hero_mask_count": int(hm.sum()),
                    "villain_mask_count": int(vm.sum()),
                    **_reach_summary("hero_reach", reach_h[cut_idx]),
                    **_reach_summary("villain_reach", reach_v[cut_idx]),
                }
            )
            emitted += 1
            if target_cuts > 0 and len(features) >= int(target_cuts):
                break
        records.append(
            {
                "label": case.label,
                "cut_targets": int(emitted),
                "filtered_cuts": int(root_filtered),
                "solver_latency_ms": round(float(getattr(solver, "last_solve_ms", 0.0)), 3),
                "export_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        if target_cuts > 0 and len(features) >= int(target_cuts):
            break

    if not features:
        raise RuntimeError("no successor cut targets were exported")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(features).astype(np.float32, copy=False),
        policy_features=np.stack(policy_features).astype(np.float32, copy=False),
        belief=np.stack(beliefs).astype(np.float32, copy=False),
        legal_masks=np.stack(legal_masks).astype(np.float32, copy=False),
        target_probs=np.stack(target_probs).astype(np.float32, copy=False),
        policy_weights=np.asarray(policy_weights, dtype=np.float32),
        hero_values=np.stack(hero_values).astype(np.float32, copy=False),
        villain_values=np.stack(villain_values).astype(np.float32, copy=False),
        hero_masks=np.stack(hero_masks).astype(np.float32, copy=False),
        villain_masks=np.stack(villain_masks).astype(np.float32, copy=False),
        labels=np.asarray(labels),
    )
    metadata = {
        "mode": "successor_cut_joint_pbs_targets",
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "output": str(output),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "value_scale": float(value_scale),
        "limit": int(limit),
        "min_bet_count": int(min_bet_count),
        "target_action_shapes": list(target_action_shapes),
        "target_cuts": int(target_cuts),
        "root_cases_scanned": int(roots_scanned),
        "n_targets": int(len(features)),
        "filtered_cut_count": int(filtered_cut_count),
        "policy_target_count": int(np.count_nonzero(np.asarray(policy_weights) > 0)),
        "value_label_count": int(
            sum(float(mask.sum()) for mask in hero_masks)
            + sum(float(mask.sum()) for mask in villain_masks)
        ),
        "records": records,
        "cut_records": cut_records,
        "skipped": skipped,
    }
    metadata_path = output.with_suffix(".json")
    save_metrics(metadata, metadata_path)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export joint PBS value targets for successor cut frontier states."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-bet-count", type=int, default=0)
    parser.add_argument(
        "--target-action-shapes",
        default="",
        help="Comma-separated action-shape allowlist, e.g. bbc/bbc/kb,bbbc/bbc/b.",
    )
    parser.add_argument("--target-cuts", type=int, default=0)
    args = parser.parse_args(argv)
    target_action_shapes = tuple(
        part.strip() for part in args.target_action_shapes.split(",") if part.strip()
    )
    metrics = export_successor_cut_targets(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        output=args.output,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        limit=args.limit,
        min_bet_count=args.min_bet_count,
        target_action_shapes=target_action_shapes,
        target_cuts=args.target_cuts,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
