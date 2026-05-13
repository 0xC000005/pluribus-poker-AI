#!/usr/bin/env python3
"""Evaluate hand-CFV checkpoint robustness under perturbed public beliefs."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX, _resolve_device
from poker_ai.research.belief_value_probe import (
    compute_hero_cfv_vector,
    load_public_belief_cfv_dataset_cache,
    load_public_belief_hand_cfv_checkpoint,
    predict_public_belief_hand_cfv_model,
    save_metrics,
)
from poker_ai.research.resolver_benchmark import load_cases_json

from play_slumbot import _compute_bets_before_street, card_str_to_index, parse_action
from solver import StreetSolver, _parse_nav, resolve_solver_backend


ALL_HANDS = tuple(itertools.combinations(range(52), 2))


def _legal_global_mask(board_idx: list[int]) -> np.ndarray:
    board = set(int(card) for card in board_idx)
    mask = np.zeros(N_HANDS, dtype=np.float32)
    for idx, hand in enumerate(ALL_HANDS):
        if hand[0] not in board and hand[1] not in board:
            mask[idx] = 1.0
    return mask


def _normalize_on_mask(values: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(values, dtype=np.float32), 0.0) * legal_mask
    total = float(out.sum())
    if total <= 1e-12:
        out = legal_mask.astype(np.float32, copy=True)
        total = float(out.sum())
    return out / max(total, 1e-12)


def _mix_with_uniform(
    values: np.ndarray,
    legal_mask: np.ndarray,
    mix_uniform: float,
) -> np.ndarray:
    base = _normalize_on_mask(values, legal_mask)
    uniform = _normalize_on_mask(legal_mask, legal_mask)
    alpha = min(max(float(mix_uniform), 0.0), 1.0)
    return _normalize_on_mask((1.0 - alpha) * base + alpha * uniform, legal_mask)


def _solve_case_with_global_ranges(
    case,
    hero_full: np.ndarray,
    villain_full: np.ndarray,
    *,
    solver_iterations: int,
    solver_backend: str,
    value_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        raise ValueError(f"{case.label}: parse error: {parsed['error']}")
    street = int(parsed.get("st", -1))
    if street != 3:
        raise ValueError(f"{case.label}: expected river state, got street {street}")

    board_idx = [card_str_to_index(card) for card in case.board[:5]]
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

    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range = np.zeros(len(full_hands), dtype=np.float32)
    villain_range = np.zeros(len(full_hands), dtype=np.float32)
    for local_idx, hand in enumerate(full_hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        hero_range[local_idx] = float(hero_full[global_idx])
        villain_range[local_idx] = float(villain_full[global_idx])
    hero_range = _normalize_on_mask(hero_range, np.ones_like(hero_range))
    villain_range = _normalize_on_mask(villain_range, np.ones_like(villain_range))

    backend, backend_device = resolve_solver_backend(solver_backend)
    started = time.perf_counter()
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    solver.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
    )
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None:
        raise ValueError(f"{case.label}: solver could not navigate street action")
    local_values, local_mask = compute_hero_cfv_vector(solver, node, villain_range)
    values = np.zeros(N_HANDS, dtype=np.float32)
    mask = np.zeros(N_HANDS, dtype=np.float32)
    for local_idx, hand in enumerate(solver.hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        values[global_idx] = float(local_values[local_idx]) / float(value_scale)
        mask[global_idx] = float(local_mask[local_idx])
    latency_ms = (time.perf_counter() - started) * 1000.0
    return values, mask, {
        "label": case.label,
        "solver_latency_ms": round(float(latency_ms), 3),
        "solver_n_hands": int(solver.n),
        "value_mask_count": int(mask.sum()),
    }


def _masked_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    selected = mask > 0
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    err = err[selected]
    if err.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0}
    return {
        "mae": round(float(np.mean(np.abs(err))), 8),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 8),
        "bias": round(float(np.mean(err)), 8),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Perturb river public beliefs and compare model CFVs to re-solved CFVs."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--mix-uniform", type=float, default=0.5)
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    cases = load_cases_json(args.cases)
    dataset, _records = load_public_belief_cfv_dataset_cache(args.cfv_cache)
    if len(cases) != dataset.features.shape[0]:
        raise ValueError("case count does not match CFV cache rows")
    model, payload = load_public_belief_hand_cfv_checkpoint(args.checkpoint, device=device)

    limit = min(max(1, int(args.limit)), len(cases))
    features = []
    beliefs = []
    masks = []
    targets = []
    records = []
    for idx, case in enumerate(cases[:limit]):
        parsed = parse_action(case.action_str)
        if int(parsed.get("st", -1)) != 3:
            continue
        board_idx = [card_str_to_index(card) for card in case.board[:5]]
        legal = _legal_global_mask(board_idx)
        base_belief = dataset.belief[idx]
        hero = _mix_with_uniform(base_belief[:N_HANDS], legal, args.mix_uniform)
        villain = _mix_with_uniform(base_belief[N_HANDS:], legal, args.mix_uniform)
        target, mask, record = _solve_case_with_global_ranges(
            case,
            hero,
            villain,
            solver_iterations=args.solver_iterations,
            solver_backend=args.solver_backend,
            value_scale=args.value_scale,
        )
        features.append(dataset.features[idx])
        beliefs.append(np.concatenate([hero, villain]).astype(np.float32, copy=False))
        masks.append(mask)
        targets.append(target)
        records.append(record)

    if not records:
        raise ValueError("no river cases were evaluated")
    pred = predict_public_belief_hand_cfv_model(
        model,
        payload,
        np.stack(features).astype(np.float32, copy=False),
        np.stack(beliefs).astype(np.float32, copy=False),
        np.stack(masks).astype(np.float32, copy=False),
        device=device,
        batch_size=args.batch_size,
    )
    target_arr = np.stack(targets).astype(np.float32, copy=False)
    mask_arr = np.stack(masks).astype(np.float32, copy=False)
    metrics = _masked_metrics(pred, target_arr, mask_arr)
    metrics = {
        "mode": "public_belief_hand_cfv_range_robustness",
        "passed": bool(np.isfinite(metrics["mae"])),
        "checkpoint": str(args.checkpoint),
        "cases": str(args.cases),
        "cfv_cache": str(args.cfv_cache),
        "device": str(device),
        "limit": int(args.limit),
        "n_evaluated": len(records),
        "mix_uniform": float(args.mix_uniform),
        "solver_iterations": int(args.solver_iterations),
        "solver_backend": args.solver_backend,
        "value_scale": float(args.value_scale),
        "n_labels": int(mask_arr.sum()),
        **metrics,
        "solver_mean_ms": round(float(np.mean([r["solver_latency_ms"] for r in records])), 3),
        "records": records,
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
