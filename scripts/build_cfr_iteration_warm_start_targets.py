#!/usr/bin/env python3
"""Export per-iteration solver-state targets for in-resolver update learning."""

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
from poker_ai.research.belief_probe import _HAND_TO_INDEX
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import load_cases_json

from build_regret_policy_warm_start_targets import (  # noqa: E402
    _card_to_str,
    _node_fields,
    _node_legal_mask,
    normalize_action_rows,
)
from eval_joint_pbs_policy_warm_start import _case_slice, _local_ranges_from_belief, _solver_context  # noqa: E402
from play_slumbot import build_features, parse_action  # noqa: E402
from solver import StreetSolver, _parse_nav  # noqa: E402


def _as_action_rows(values: np.ndarray, *, action_dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != action_dim:
        raise ValueError(f"expected hand/action rows with shape (*, {action_dim})")
    if not np.isfinite(arr).all():
        raise ValueError("solver-state rows must be finite")
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def materialize_iteration_target_rows(
    *,
    root_label: str,
    public_feature: np.ndarray,
    policy_features: np.ndarray,
    belief: np.ndarray,
    legal_mask: np.ndarray,
    current_regret_by_iteration: dict[int, np.ndarray],
    current_strategy_by_iteration: dict[int, np.ndarray],
    target_regret: np.ndarray,
    target_strategy: np.ndarray,
    hand_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return NPZ-compatible rows for one root across trace iterations."""
    hand_indices = np.asarray(hand_indices, dtype=np.int32).reshape(-1)
    n_hands = int(hand_indices.shape[0])
    legal = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    if legal.shape[0] != N_ACTIONS:
        raise ValueError(f"legal_mask must have {N_ACTIONS} actions")
    if float(legal.sum()) <= 0.0:
        raise ValueError("legal_mask must contain at least one legal action")
    public = np.asarray(public_feature, dtype=np.float32).reshape(-1)
    policy = np.asarray(policy_features, dtype=np.float32)
    if policy.ndim != 2 or policy.shape[0] != n_hands:
        raise ValueError("policy_features must have one row per hand")
    belief_row = np.asarray(belief, dtype=np.float32).reshape(-1)
    target_regret = _as_action_rows(target_regret, action_dim=N_ACTIONS)
    target_strategy = _as_action_rows(target_strategy, action_dim=N_ACTIONS)
    if target_regret.shape[0] != n_hands or target_strategy.shape[0] != n_hands:
        raise ValueError("target fields must have one row per hand")
    rows: dict[str, list[Any]] = {
        "features": [],
        "policy_features": [],
        "belief": [],
        "legal_masks": [],
        "target_regret_sum": [],
        "target_strategy_sum": [],
        "target_probs": [],
        "low_regret_sum": [],
        "low_strategy_sum": [],
        "hand_indices": [],
        "labels": [],
        "root_labels": [],
    }
    target_probs = normalize_action_rows(target_strategy, legal)
    for iteration in sorted(int(key) for key in current_regret_by_iteration):
        if iteration not in current_strategy_by_iteration:
            raise ValueError(f"missing strategy field for iteration {iteration}")
        current_regret = _as_action_rows(
            current_regret_by_iteration[iteration],
            action_dim=N_ACTIONS,
        )
        current_strategy = _as_action_rows(
            current_strategy_by_iteration[iteration],
            action_dim=N_ACTIONS,
        )
        if current_regret.shape[0] != n_hands or current_strategy.shape[0] != n_hands:
            raise ValueError("current fields must have one row per hand")
        for hand_row, hand_idx in enumerate(hand_indices):
            rows["features"].append(public)
            rows["policy_features"].append(policy[hand_row])
            rows["belief"].append(belief_row)
            rows["legal_masks"].append(legal)
            rows["target_regret_sum"].append(target_regret[hand_row])
            rows["target_strategy_sum"].append(target_strategy[hand_row])
            rows["target_probs"].append(target_probs[hand_row])
            rows["low_regret_sum"].append(current_regret[hand_row])
            rows["low_strategy_sum"].append(current_strategy[hand_row])
            rows["hand_indices"].append(int(hand_idx))
            rows["labels"].append(f"{root_label}-iter{iteration:03d}-hand{int(hand_idx):04d}")
            rows["root_labels"].append(str(root_label))
    return {
        "features": np.stack(rows["features"]).astype(np.float32, copy=False),
        "policy_features": np.stack(rows["policy_features"]).astype(np.float32, copy=False),
        "belief": np.stack(rows["belief"]).astype(np.float32, copy=False),
        "legal_masks": np.stack(rows["legal_masks"]).astype(np.float32, copy=False),
        "target_regret_sum": np.stack(rows["target_regret_sum"]).astype(np.float32, copy=False),
        "target_strategy_sum": np.stack(rows["target_strategy_sum"]).astype(np.float32, copy=False),
        "target_probs": np.stack(rows["target_probs"]).astype(np.float32, copy=False),
        "low_regret_sum": np.stack(rows["low_regret_sum"]).astype(np.float32, copy=False),
        "low_strategy_sum": np.stack(rows["low_strategy_sum"]).astype(np.float32, copy=False),
        "hand_indices": np.asarray(rows["hand_indices"], dtype=np.int32),
        "labels": np.asarray(rows["labels"]),
        "root_labels": np.asarray(rows["root_labels"]),
    }


def _merge_row_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not batches:
        raise RuntimeError("no iteration warm-start targets were exported")
    out: dict[str, np.ndarray] = {}
    for key in batches[0]:
        out[key] = np.concatenate([batch[key] for batch in batches], axis=0)
    return out


def _trace_current_fields(
    solver: StreetSolver,
    node_idx: int,
    *,
    solver_iterations: int,
    hero_range: np.ndarray,
    villain_range: np.ndarray,
    trace_iterations: set[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    current_regret: dict[int, np.ndarray] = {}
    current_strategy: dict[int, np.ndarray] = {}

    def collect_trace(**kwargs: Any) -> None:
        iteration = int(kwargs["iteration"])
        if iteration not in trace_iterations:
            return
        regret = np.asarray(kwargs["regret_sum"], dtype=np.float32)[0, :N_ACTIONS].T
        strategy = np.asarray(kwargs["strategy_sum"], dtype=np.float32)[0, :N_ACTIONS].T
        current_regret[iteration] = regret.astype(np.float32, copy=True)
        current_strategy[iteration] = strategy.astype(np.float32, copy=True)

    solver.solve(
        n_iterations=int(solver_iterations),
        hero_range=hero_range,
        villain_range=villain_range,
        backend="cpu",
        trace_node_indices=[int(node_idx)],
        trace_node_fn=collect_trace,
    )
    return current_regret, current_strategy


def build_cfr_iteration_warm_start_targets(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    output: str | Path,
    start_index: int = 0,
    limit: int = 64,
    trace_iterations: tuple[int, ...] = (0, 1, 2, 3, 4),
    reference_iterations: int = 25,
    trace_solver_iterations: int | None = None,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    trace_iterations = tuple(sorted(set(int(item) for item in trace_iterations)))
    if not trace_iterations:
        raise ValueError("trace_iterations must not be empty")
    max_trace_iteration = max(trace_iterations)
    trace_solver_iterations = (
        int(max_trace_iteration) + 1
        if trace_solver_iterations is None
        else int(trace_solver_iterations)
    )
    if trace_solver_iterations <= max_trace_iteration:
        raise ValueError("trace_solver_iterations must exceed the largest trace iteration")

    batches: list[dict[str, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for local_idx, case in enumerate(_case_slice(cases, start_index=start_index, limit=limit)):
        case_idx = int(start_index) + int(local_idx)
        parsed = parse_action(case.action_str)
        if "error" in parsed or int(parsed.get("st", -1)) not in (2, 3):
            skipped.append({"label": case.label, "reason": "parse_error_or_unsupported_street"})
            continue
        board_idx, _our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action = _solver_context(
            case,
            parsed,
        )
        full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
        belief_row = np.asarray(dataset.belief[case_idx], dtype=np.float32)
        hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)

        trace_solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
        trace_node = trace_solver.navigate(_parse_nav(street_action, trace_solver))
        if trace_node is None or trace_node.is_terminal:
            skipped.append({"label": case.label, "reason": "terminal_or_missing_node"})
            continue
        node_idx = int(trace_solver._tree["all_nodes"].index(trace_node))
        current_regret, current_strategy = _trace_current_fields(
            trace_solver,
            node_idx,
            solver_iterations=trace_solver_iterations,
            hero_range=hero_range,
            villain_range=villain_range,
            trace_iterations=set(trace_iterations),
        )
        missing = [iteration for iteration in trace_iterations if iteration not in current_regret]
        if missing:
            skipped.append({"label": case.label, "reason": f"missing_trace_iterations:{missing}"})
            continue

        reference_solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
        reference_node = reference_solver.navigate(_parse_nav(street_action, reference_solver))
        if reference_node is None or reference_node.is_terminal:
            skipped.append({"label": case.label, "reason": "reference_terminal_or_missing_node"})
            continue
        reference_solver.solve(
            n_iterations=int(reference_iterations),
            hero_range=hero_range,
            villain_range=villain_range,
            backend="cpu",
        )
        solved_reference_node = reference_solver.navigate(_parse_nav(street_action, reference_solver))
        if solved_reference_node is None or solved_reference_node.is_terminal:
            skipped.append({"label": case.label, "reason": "reference_solve_missing_node"})
            continue
        legal_mask = _node_legal_mask(solved_reference_node)
        target_regret, target_strategy = _node_fields(reference_solver, solved_reference_node)
        public_feature = build_features(
            [],
            list(case.board),
            case.action_str,
            int(case.client_pos),
            parsed,
        ).astype(np.float32, copy=False)
        policy_features: list[np.ndarray] = []
        hand_indices: list[int] = []
        for hand in reference_solver.hands:
            global_hand = tuple(sorted((int(hand[0]), int(hand[1]))))
            global_hand_idx = int(_HAND_TO_INDEX[global_hand])
            hole_cards = [_card_to_str(global_hand[0]), _card_to_str(global_hand[1])]
            policy_features.append(
                build_features(
                    hole_cards,
                    list(case.board),
                    case.action_str,
                    int(case.client_pos),
                    parsed,
                ).astype(np.float32, copy=False)
            )
            hand_indices.append(global_hand_idx)
        batches.append(
            materialize_iteration_target_rows(
                root_label=str(case.label),
                public_feature=public_feature,
                policy_features=np.stack(policy_features).astype(np.float32, copy=False),
                belief=belief_row,
                legal_mask=legal_mask,
                current_regret_by_iteration=current_regret,
                current_strategy_by_iteration=current_strategy,
                target_regret=target_regret,
                target_strategy=target_strategy,
                hand_indices=np.asarray(hand_indices, dtype=np.int32),
            )
        )
        records.append(
            {
                "label": str(case.label),
                "case_index": int(case_idx),
                "n_hand_rows_per_iteration": int(len(reference_solver.hands)),
                "n_iterations_exported": int(len(trace_iterations)),
                "trace_solver_ms": round(float(trace_solver.last_solve_ms), 3),
                "reference_solver_ms": round(float(reference_solver.last_solve_ms), 3),
            }
        )

    rows = _merge_row_batches(batches)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **rows)
    legal = rows["legal_masks"] > 0
    metadata = {
        "mode": "cfr_iteration_warm_start_targets",
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "output": str(output_path),
        "start_index": int(start_index),
        "limit": int(limit),
        "trace_iterations": [int(item) for item in trace_iterations],
        "trace_solver_iterations": int(trace_solver_iterations),
        "reference_iterations": int(reference_iterations),
        "n_roots": int(len(records)),
        "n_targets": int(rows["features"].shape[0]),
        "feature_dim": int(rows["features"].shape[1]),
        "belief_dim": int(rows["belief"].shape[1]),
        "target_regret_mean": round(float(rows["target_regret_sum"][legal].mean()), 8),
        "target_strategy_mean": round(float(rows["target_strategy_sum"][legal].mean()), 8),
        "low_regret_mean": round(float(rows["low_regret_sum"][legal].mean()), 8),
        "low_strategy_mean": round(float(rows["low_strategy_sum"][legal].mean()), 8),
        "records": records,
        "skipped": skipped[:200],
    }
    save_metrics(metadata, output_path.with_suffix(".json"))
    return metadata


def _parse_iterations(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--trace-iterations", default="0,1,2,3,4")
    parser.add_argument("--trace-solver-iterations", type=int)
    parser.add_argument("--reference-iterations", type=int, default=25)
    args = parser.parse_args(argv)
    metrics = build_cfr_iteration_warm_start_targets(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        output=args.output,
        start_index=args.start_index,
        limit=args.limit,
        trace_iterations=_parse_iterations(args.trace_iterations),
        trace_solver_iterations=args.trace_solver_iterations,
        reference_iterations=args.reference_iterations,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
