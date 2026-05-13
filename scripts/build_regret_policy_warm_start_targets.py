#!/usr/bin/env python3
"""Export public-node CFR+ regret/policy warm-start targets."""

from __future__ import annotations

import argparse
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
from poker_ai.research.belief_probe import _HAND_TO_INDEX
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import load_cases_json

from eval_joint_pbs_policy_warm_start import (  # noqa: E402
    _case_slice,
    _solve_with_belief,
)
from play_slumbot import build_features, parse_action  # noqa: E402


_SUITS = ("c", "d", "h", "s")
_RANKS = tuple("23456789TJQKA")


def _card_to_str(card: int) -> str:
    return _RANKS[int(card) // 4] + _SUITS[int(card) % 4]


def normalize_action_rows(
    values: np.ndarray,
    legal_mask: np.ndarray,
) -> np.ndarray:
    """Normalize nonnegative action rows with a legal-uniform fallback."""
    arr = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    legal = (np.asarray(legal_mask, dtype=np.float32) > 0.0).astype(np.float32)
    if arr.shape[-1] != legal.shape[-1]:
        raise ValueError("action value rows and legal mask use different action dimensions")
    arr = arr * legal.reshape((1, -1))
    totals = arr.sum(axis=1, keepdims=True)
    legal_total = float(max(legal.sum(), 1.0))
    fallback = legal.reshape((1, -1)) / legal_total
    out = np.repeat(fallback, arr.shape[0], axis=0).astype(np.float32, copy=False)
    valid = totals[:, 0] > 1e-8
    if np.any(valid):
        out[valid] = arr[valid] / totals[valid]
    return out.astype(np.float32, copy=False)


def _node_legal_mask(node: Any) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for action in node.children:
        action_idx = int(action)
        if 0 <= action_idx < N_ACTIONS:
            mask[action_idx] = 1.0
    if mask.sum() <= 0:
        raise ValueError("selected solver node has no legal actions")
    return mask


def _node_fields(solver: Any, node: Any) -> tuple[np.ndarray, np.ndarray]:
    if solver._regret_sum is None or solver._strategy_sum is None:
        raise ValueError("solver must be solved before extracting node fields")
    node_idx = solver._tree["all_nodes"].index(node)
    regret = np.maximum(solver._regret_sum[node_idx, :N_ACTIONS], 0.0).T.astype(
        np.float32,
        copy=False,
    )
    strategy = np.maximum(solver._strategy_sum[node_idx, :N_ACTIONS], 0.0).T.astype(
        np.float32,
        copy=False,
    )
    return regret, strategy


def build_regret_policy_warm_start_targets(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    output: str | Path,
    start_index: int = 0,
    limit: int = 128,
    low_iterations: int = 5,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")

    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)
    features: list[np.ndarray] = []
    policy_features: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    target_regrets: list[np.ndarray] = []
    target_strategy_sums: list[np.ndarray] = []
    target_probs: list[np.ndarray] = []
    low_regrets: list[np.ndarray] = []
    low_strategy_sums: list[np.ndarray] = []
    hand_indices: list[int] = []
    labels: list[str] = []
    root_labels: list[str] = []
    root_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for local_idx, case in enumerate(selected_cases):
        case_idx = int(start_index) + int(local_idx)
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            skipped.append({"label": case.label, "reason": parsed["error"]})
            continue
        belief_row = np.asarray(dataset.belief[case_idx], dtype=np.float32)
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
        if low_solved is None or reference_solved is None:
            skipped.append({"label": case.label, "reason": "solver_skipped"})
            continue
        low_solver, low_node, _low_decision = low_solved
        reference_solver, reference_node, _reference_decision = reference_solved
        if low_solver.n != reference_solver.n:
            raise ValueError("low and reference solvers use different hand counts")
        legal_mask = _node_legal_mask(reference_node)
        reference_regret, reference_strategy = _node_fields(reference_solver, reference_node)
        low_regret, low_strategy = _node_fields(low_solver, low_node)
        strategy_probs = normalize_action_rows(reference_strategy, legal_mask)
        public_feature = build_features(
            [],
            list(case.board),
            case.action_str,
            int(case.client_pos),
            parsed,
        ).astype(np.float32, copy=False)

        start_row = len(features)
        for hand_idx, hand in enumerate(reference_solver.hands):
            global_hand = tuple(sorted((int(hand[0]), int(hand[1]))))
            global_hand_idx = int(_HAND_TO_INDEX[global_hand])
            hole_cards = [_card_to_str(global_hand[0]), _card_to_str(global_hand[1])]
            features.append(public_feature)
            policy_features.append(
                build_features(
                    hole_cards,
                    list(case.board),
                    case.action_str,
                    int(case.client_pos),
                    parsed,
                ).astype(np.float32, copy=False)
            )
            beliefs.append(belief_row)
            legal_masks.append(legal_mask)
            target_regrets.append(reference_regret[hand_idx])
            target_strategy_sums.append(reference_strategy[hand_idx])
            target_probs.append(strategy_probs[hand_idx])
            low_regrets.append(low_regret[hand_idx])
            low_strategy_sums.append(low_strategy[hand_idx])
            hand_indices.append(global_hand_idx)
            labels.append(f"{case.label}-regret-policy-hand{global_hand_idx:04d}")
            root_labels.append(case.label)
        root_records.append(
            {
                "label": case.label,
                "case_index": int(case_idx),
                "n_hand_rows": int(len(features) - start_row),
                "legal_action_count": int(np.count_nonzero(legal_mask > 0.0)),
                "low_latency_ms": round(float(low_solver.last_solve_ms), 3),
                "reference_latency_ms": round(float(reference_solver.last_solve_ms), 3),
            }
        )

    if not features:
        raise RuntimeError("no regret/policy warm-start targets were exported")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_regret_arr = np.stack(target_regrets).astype(np.float32, copy=False)
    target_strategy_arr = np.stack(target_strategy_sums).astype(np.float32, copy=False)
    low_regret_arr = np.stack(low_regrets).astype(np.float32, copy=False)
    low_strategy_arr = np.stack(low_strategy_sums).astype(np.float32, copy=False)
    np.savez_compressed(
        output_path,
        features=np.stack(features).astype(np.float32, copy=False),
        policy_features=np.stack(policy_features).astype(np.float32, copy=False),
        belief=np.stack(beliefs).astype(np.float32, copy=False),
        legal_masks=np.stack(legal_masks).astype(np.float32, copy=False),
        target_regret_sum=target_regret_arr,
        target_strategy_sum=target_strategy_arr,
        target_probs=np.stack(target_probs).astype(np.float32, copy=False),
        low_regret_sum=low_regret_arr,
        low_strategy_sum=low_strategy_arr,
        hand_indices=np.asarray(hand_indices, dtype=np.int32),
        labels=np.asarray(labels),
        root_labels=np.asarray(root_labels),
    )
    legal_arr = np.stack(legal_masks).astype(np.float32, copy=False)
    metadata = {
        "mode": "regret_policy_warm_start_targets",
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "output": str(output_path),
        "start_index": int(start_index),
        "limit": int(limit),
        "low_iterations": int(low_iterations),
        "reference_iterations": int(reference_iterations),
        "solver_backend": solver_backend,
        "n_roots": int(len(root_records)),
        "n_targets": int(len(features)),
        "feature_dim": int(np.asarray(features[0]).shape[0]),
        "belief_dim": int(np.asarray(beliefs[0]).shape[0]),
        "target_regret_mean": round(float(target_regret_arr[legal_arr > 0].mean()), 8),
        "target_strategy_mean": round(float(target_strategy_arr[legal_arr > 0].mean()), 8),
        "low_regret_mean": round(float(low_regret_arr[legal_arr > 0].mean()), 8),
        "low_strategy_mean": round(float(low_strategy_arr[legal_arr > 0].mean()), 8),
        "records": root_records,
        "skipped": skipped[:200],
    }
    save_metrics(metadata, output_path.with_suffix(".json"))
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export teacher regret/policy solver-state warm-start targets."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    args = parser.parse_args(argv)
    metrics = build_regret_policy_warm_start_targets(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        output=args.output,
        start_index=args.start_index,
        limit=args.limit,
        low_iterations=args.low_iterations,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
