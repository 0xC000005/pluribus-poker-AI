#!/usr/bin/env python3
"""Collect dynamic CFR root trace diagnostics for public-belief cases."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.deep_cfr.fast_state import N_ACTIONS  # noqa: E402
from poker_ai.research.belief_value_probe import (  # noqa: E402
    load_public_belief_cfv_dataset_cache,
    save_metrics,
)
from poker_ai.research.resolver_benchmark import load_cases_json  # noqa: E402

from eval_joint_pbs_policy_warm_start import _local_ranges_from_belief, _solver_context  # noqa: E402
from play_slumbot import card_str_to_index, parse_action  # noqa: E402
from solver import StreetSolver, _parse_nav  # noqa: E402


def action_policy_from_field(values: np.ndarray, legal_actions: tuple[int, ...]) -> np.ndarray:
    """Normalize one action field over legal actions with a legal-uniform fallback."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros_like(arr, dtype=np.float32)
    legal = [int(action) for action in legal_actions if 0 <= int(action) < arr.shape[0]]
    if not legal:
        raise ValueError("legal_actions must contain at least one valid action")
    legal_values = np.maximum(arr[legal], 0.0)
    total = float(legal_values.sum())
    if total > 1e-8:
        out[legal] = legal_values / total
    else:
        out[legal] = 1.0 / float(len(legal))
    return out


def _normalized_range(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=np.float32).reshape(-1), 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / total


def _range_entropy(probs: np.ndarray) -> float:
    nz = probs[probs > 1e-12]
    if nz.size == 0:
        return 0.0
    return float(-(nz * np.log(nz)).sum())


def _topk_mass(probs: np.ndarray, k: int = 5) -> float:
    if probs.size == 0:
        return 0.0
    k = max(1, min(int(k), int(probs.size)))
    return float(np.sort(probs)[-k:].sum())


def belief_summary_features(hero_range: np.ndarray, villain_range: np.ndarray) -> list[float]:
    """Return compact public-belief range-shape features for trace diagnostics."""
    hero = _normalized_range(hero_range)
    villain = _normalized_range(villain_range)
    hero_entropy = _range_entropy(hero)
    villain_entropy = _range_entropy(villain)
    hero_top = float(hero.max()) if hero.size else 0.0
    villain_top = float(villain.max()) if villain.size else 0.0
    return [
        round(hero_entropy, 8),
        round(villain_entropy, 8),
        round(hero_top, 8),
        round(_topk_mass(hero), 8),
        round(float(np.exp(hero_entropy)) if hero_entropy > 0.0 else 0.0, 8),
        round(float(np.exp(villain_entropy)) if villain_entropy > 0.0 else 0.0, 8),
        round(villain_top, 8),
        round(_topk_mass(villain), 8),
    ]


def summarize_trace_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_iteration: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if "l1_to_final_strategy" not in record:
            continue
        by_iteration[int(record["iteration"])].append(float(record["l1_to_final_strategy"]))
    return {
        "n_trace_records": int(len(records)),
        "mean_l1_to_final_by_iteration": {
            str(iteration): round(float(np.mean(values)), 8)
            for iteration, values in sorted(by_iteration.items())
            if values
        },
    }


def _trace_case(
    case: Any,
    *,
    belief_row: np.ndarray,
    public_features: np.ndarray | None = None,
    solver_iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    parsed = parse_action(case.action_str)
    if "error" in parsed or int(parsed.get("st", -1)) not in (2, 3):
        return [], {"label": case.label, "reason": "parse_error_or_unsupported_street"}
    (
        board_idx,
        _our_cards_idx,
        pot,
        hero_stack,
        villain_stack,
        hero_first,
        street_action,
    ) = _solver_context(case, parsed)
    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    context_features = belief_summary_features(hero_range, villain_range)
    if public_features is not None:
        public_context = (
            np.asarray(public_features, dtype=np.float32)
            .reshape(-1)
            .round(8)
            .astype(float)
            .tolist()
        )
        context_features = (
            public_context + context_features
        )
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    active_node = solver.navigate(_parse_nav(street_action, solver))
    if active_node is None or active_node.is_terminal:
        return [], {"label": case.label, "reason": "terminal_or_missing_node"}
    node_idx = solver._tree["all_nodes"].index(active_node)
    hand = tuple(sorted(card_str_to_index(card) for card in case.hole_cards))
    hand_idx = solver.hand_to_idx.get(hand)
    if hand_idx is None:
        return [], {"label": case.label, "reason": "hole_cards_not_in_solver_hands"}
    legal_actions = tuple(sorted(int(action) for action in active_node.children.keys()))
    records: list[dict[str, Any]] = []

    def collect_trace(**kwargs: Any) -> None:
        regret_field = np.asarray(kwargs["regret_sum"], dtype=np.float32)[0, :, hand_idx]
        strategy_field = np.asarray(kwargs["strategy_sum"], dtype=np.float32)[0, :, hand_idx]
        regret_policy = action_policy_from_field(regret_field, legal_actions)
        strategy_policy = action_policy_from_field(strategy_field, legal_actions)
        records.append(
            {
                "label": str(case.label),
                "iteration": int(kwargs["iteration"]),
                "street": int(parsed["st"]),
                "legal_actions": list(legal_actions),
                "regret_mass": round(float(np.maximum(regret_field[list(legal_actions)], 0.0).sum()), 8),
                "strategy_mass": round(float(np.maximum(strategy_field[list(legal_actions)], 0.0).sum()), 8),
                "hero_reach_mass": round(float(np.asarray(kwargs["hero_reach"], dtype=np.float32)[0].sum()), 8),
                "villain_reach_mass": round(float(np.asarray(kwargs["villain_reach"], dtype=np.float32)[0].sum()), 8),
                "public_belief_features": context_features,
                "regret_top_action": int(np.argmax(regret_policy)),
                "strategy_top_action": int(np.argmax(strategy_policy)),
                "regret_policy": regret_policy.astype(float).round(8).tolist(),
                "strategy_policy": strategy_policy.astype(float).round(8).tolist(),
            }
        )

    solver.solve(
        n_iterations=int(solver_iterations),
        hero_range=hero_range,
        villain_range=villain_range,
        backend="cpu",
        trace_node_indices=[node_idx],
        trace_node_fn=collect_trace,
    )
    final_strategy = action_policy_from_field(
        solver._strategy_sum[node_idx, :, hand_idx],
        legal_actions,
    )
    final_top = int(np.argmax(final_strategy))
    for record in records:
        strategy_policy = np.asarray(record["strategy_policy"], dtype=np.float32)
        record["final_top_action"] = final_top
        record["top_matches_final"] = bool(int(record["strategy_top_action"]) == final_top)
        record["l1_to_final_strategy"] = round(float(np.abs(strategy_policy - final_strategy).sum()), 8)
    return records, None


def diagnose_cfr_trace_state(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    start_index: int = 0,
    limit: int = 8,
    solver_iterations: int = 25,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    start = max(0, int(start_index))
    end = min(len(cases), start + max(0, int(limit)))
    trace_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for case_idx in range(start, end):
        case_records, skip = _trace_case(
            cases[case_idx],
            belief_row=np.asarray(dataset.belief[case_idx], dtype=np.float32),
            public_features=np.asarray(dataset.features[case_idx], dtype=np.float32),
            solver_iterations=solver_iterations,
        )
        trace_records.extend(case_records)
        if skip is not None:
            skipped.append(skip)
    summary = summarize_trace_records(trace_records)
    top_matches = [bool(record["top_matches_final"]) for record in trace_records]
    return {
        "mode": "cfr_dynamic_root_trace_diagnostic",
        "passed": bool(trace_records),
        "cases": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "start_index": int(start),
        "limit": int(limit),
        "solver_iterations": int(solver_iterations),
        "n_cases_scanned": int(end - start),
        "n_skipped": int(len(skipped)),
        "skipped": skipped,
        **summary,
        "top_match_final_rate": round(float(np.mean(top_matches)), 8) if top_matches else None,
        "records": trace_records,
        "promotion": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = diagnose_cfr_trace_state(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        start_index=args.start_index,
        limit=args.limit,
        solver_iterations=args.solver_iterations,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
