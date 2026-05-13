#!/usr/bin/env python3
"""Evaluate a joint-PBS policy head against exact root resolver policies."""

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

from poker_ai.research.belief_probe import _normalize_targets, _policy_metrics, _resolve_device
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json

from eval_joint_pbs_continuation_probe import (  # noqa: E402
    _DEFAULT_MAX_ACTION_TOKENS,
    _encode_action_sequence,
    load_joint_pbs_continuation_checkpoint,
    predict_joint_pbs_policy_model,
)
from eval_joint_pbs_resolver_leaf_ab import (  # noqa: E402
    _local_ranges_from_belief,
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


def _case_slice(cases: list[ResolverBenchmarkCase], *, start_index: int, limit: int) -> list[ResolverBenchmarkCase]:
    if int(start_index) < 0:
        raise ValueError("start_index must be non-negative")
    start = min(int(start_index), len(cases))
    stop = len(cases) if int(limit) <= 0 else min(start + int(limit), len(cases))
    return list(cases[start:stop])


def _legal_uniform(legal_masks: np.ndarray) -> np.ndarray:
    legal = (np.asarray(legal_masks, dtype=np.float32) > 0).astype(np.float32)
    totals = legal.sum(axis=1, keepdims=True)
    return legal / np.maximum(totals, 1.0)


def _policy_eval_metrics(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    legal_masks: np.ndarray,
    min_top1_match: float,
    max_mean_l1: float,
) -> dict[str, Any]:
    model = _policy_metrics(predictions, targets, legal_masks)
    uniform = _policy_metrics(_legal_uniform(legal_masks), targets, legal_masks)
    passed = bool(
        model["mean_l1"] < uniform["mean_l1"]
        and model["top1_match_rate"] >= float(min_top1_match)
        and model["mean_l1"] <= float(max_mean_l1)
    )
    return {
        "passed": passed,
        "policy": model,
        "legal_uniform": uniform,
        "mean_l1_improvement_vs_uniform": round(
            float(uniform["mean_l1"] - model["mean_l1"]),
            6,
        ),
    }


def _root_policy_target_row(
    case: ResolverBenchmarkCase,
    *,
    belief_row: np.ndarray,
    solver_iterations: int,
    solver_backend: str,
) -> dict[str, Any] | None:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        return {"label": case.label, "passed": False, "skipped": parsed["error"]}
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
    )
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None or node.is_terminal:
        return {"label": case.label, "passed": False, "skipped": "terminal_or_missing_node"}

    hand = tuple(sorted(our_cards_idx))
    target = _strategy_vector(solver.get_strategy(hand, node)).astype(np.float32)
    legal = get_legal_mask_from_parsed(parsed, case.action_str, case.client_pos).astype(np.float32)
    target = _normalize_targets(target[np.newaxis, :], legal[np.newaxis, :])[0]
    public_feature = build_features(
        [],
        list(case.board),
        case.action_str,
        int(case.client_pos),
        parsed,
    ).astype(np.float32, copy=False)
    policy_feature = build_features(
        list(case.hole_cards),
        list(case.board),
        case.action_str,
        int(case.client_pos),
        parsed,
    ).astype(np.float32, copy=False)
    return {
        "label": case.label,
        "passed": True,
        "feature": public_feature,
        "policy_feature": policy_feature,
        "belief": np.asarray(belief_row, dtype=np.float32),
        "legal_mask": legal,
        "target": target.astype(np.float32, copy=False),
        "target_action": int(np.argmax(target)),
        "solver_ms": round(float(getattr(solver, "last_solve_ms", 0.0)), 3),
    }


def eval_joint_pbs_root_policy(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    device: str = "auto",
    start_index: int = 0,
    limit: int = 64,
    solver_iterations: int = 5,
    solver_backend: str = "cpu",
    min_top1_match: float = 0.75,
    max_mean_l1: float = 0.55,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    model, payload = load_joint_pbs_continuation_checkpoint(checkpoint, device=resolved_device)
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")

    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for local_idx, case in enumerate(selected_cases):
        case_idx = int(start_index) + int(local_idx)
        row = _root_policy_target_row(
            case,
            belief_row=base_dataset.belief[case_idx],
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
        )
        if row is None:
            continue
        records.append(row)
        if row.get("passed"):
            rows.append(row)

    if rows:
        max_tokens = int(payload.get("max_action_tokens", _DEFAULT_MAX_ACTION_TOKENS))
        tokens = []
        amounts = []
        for local_idx, record in enumerate(records):
            if not record.get("passed"):
                continue
            case = selected_cases[local_idx]
            encoded_tokens, encoded_amounts = _encode_action_sequence(case.action_str, max_tokens=max_tokens)
            tokens.append(encoded_tokens)
            amounts.append(encoded_amounts)
        predictions = predict_joint_pbs_policy_model(
            model,
            payload,
            np.stack([row["feature"] for row in rows]).astype(np.float32),
            np.stack([row["policy_feature"] for row in rows]).astype(np.float32),
            np.stack([row["belief"] for row in rows]).astype(np.float32),
            np.stack([row["legal_mask"] for row in rows]).astype(np.float32),
            action_tokens=np.stack(tokens).astype(np.int64),
            action_amounts=np.stack(amounts).astype(np.float32),
            device=resolved_device,
        )
        targets = np.stack([row["target"] for row in rows]).astype(np.float32)
        legal_masks = np.stack([row["legal_mask"] for row in rows]).astype(np.float32)
        metrics = _policy_eval_metrics(
            predictions=predictions,
            targets=targets,
            legal_masks=legal_masks,
            min_top1_match=min_top1_match,
            max_mean_l1=max_mean_l1,
        )
        pred_actions = np.argmax(predictions, axis=1)
        for row, pred_action, pred in zip(rows, pred_actions.tolist(), predictions, strict=True):
            row["pred_action"] = int(pred_action)
            row["action_agreement"] = bool(int(pred_action) == int(row["target_action"]))
            row["action_l1"] = round(float(np.abs(pred - row["target"]).sum()), 8)
    else:
        metrics = {
            "passed": False,
            "policy": {},
            "legal_uniform": {},
            "mean_l1_improvement_vs_uniform": 0.0,
        }

    serializable_records = []
    for record in records:
        serializable = {
            key: value
            for key, value in record.items()
            if key not in {"feature", "policy_feature", "belief", "legal_mask", "target"}
        }
        serializable_records.append(serializable)

    return {
        "mode": "joint_pbs_root_policy_eval",
        "checkpoint": str(checkpoint),
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "device": str(resolved_device),
        "start_index": int(start_index),
        "limit": int(limit),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "min_top1_match": float(min_top1_match),
        "max_mean_l1": float(max_mean_l1),
        "n_cases": int(len(records)),
        "n_evaluated": int(len(rows)),
        **metrics,
        "records": serializable_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a joint-PBS root policy head against exact resolver policies."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--min-top1-match", type=float, default=0.75)
    parser.add_argument("--max-mean-l1", type=float, default=0.55)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_joint_pbs_root_policy(
        checkpoint=args.checkpoint,
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        device=args.device,
        start_index=args.start_index,
        limit=args.limit,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        min_top1_match=args.min_top1_match,
        max_mean_l1=args.max_mean_l1,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
