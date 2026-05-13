#!/usr/bin/env python3
"""Attribute joint-PBS root-policy errors on held-out public roots."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
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

from poker_ai.research.belief_probe import _policy_metrics, _resolve_device
from poker_ai.research.belief_value_probe import (
    load_public_belief_cfv_dataset_cache,
    save_metrics,
)
from poker_ai.research.resolver_benchmark import load_cases_json

from eval_joint_pbs_continuation_probe import (  # noqa: E402
    _DEFAULT_MAX_ACTION_TOKENS,
    _encode_action_sequence,
    load_joint_pbs_continuation_checkpoint,
    predict_joint_pbs_policy_model,
)
from eval_joint_pbs_root_policy import (  # noqa: E402
    _case_slice,
    _legal_uniform,
    _root_policy_target_row,
)
from play_slumbot import _compute_bets_before_street, parse_action  # noqa: E402


def _safe_corr(xs: list[float], ys: list[float]) -> float | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 8)


def _entropy(probs: np.ndarray, legal: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64) * (np.asarray(legal) > 0)
    total = float(p.sum())
    if total <= 1e-12:
        return 0.0
    p = p / total
    p = p[p > 1e-12]
    return round(float(-(p * np.log(p)).sum()), 8)


def _margin(probs: np.ndarray, legal: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).copy()
    p[np.asarray(legal) <= 0] = -np.inf
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return 0.0
    top = np.sort(finite)[-2:]
    if top.size == 1:
        return round(float(top[-1]), 8)
    return round(float(top[-1] - top[-2]), 8)


def _belief_entropy(belief: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(belief, dtype=np.float64)
    mid = arr.shape[0] // 2
    return _entropy(arr[:mid], np.ones(mid)), _entropy(arr[mid:], np.ones(arr.shape[0] - mid))


def _action_shape(action_str: str) -> str:
    return re.sub(r"b\d+", "b", str(action_str))


def _case_context(case: Any) -> dict[str, Any]:
    parsed = parse_action(case.action_str)
    street = int(parsed.get("st", -1)) if "error" not in parsed else -1
    our_bet, opp_bet = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=max(street, 0),
    )
    pot = float(our_bet + opp_bet)
    hero_stack = float(20000 - our_bet)
    villain_stack = float(20000 - opp_bet)
    return {
        "street": int(street),
        "pot": round(pot, 6),
        "hero_stack": round(hero_stack, 6),
        "villain_stack": round(villain_stack, 6),
        "effective_stack": round(float(min(hero_stack, villain_stack)), 6),
        "client_pos": int(case.client_pos),
        "bet_count": int(str(case.action_str).count("b")),
        "action_shape": _action_shape(case.action_str),
        "board": "".join(case.board),
    }


def _summarize_numeric_correlations(
    records: list[dict[str, Any]],
    *,
    error_field: str = "model_high_l1",
) -> list[dict[str, Any]]:
    fields = [
        "pot",
        "hero_stack",
        "villain_stack",
        "effective_stack",
        "bet_count",
        "legal_count",
        "target_entropy",
        "target_margin",
        "pred_entropy",
        "pred_margin",
        "pred_confidence",
        "target_confidence",
        "target_allin_prob",
        "pred_allin_prob",
        "low_high_l1",
        "hero_belief_entropy",
        "villain_belief_entropy",
    ]
    rows = []
    for field in fields:
        corr = _safe_corr(
            [float(record.get(field, float("nan"))) for record in records],
            [float(record.get(error_field, float("nan"))) for record in records],
        )
        if corr is not None:
            rows.append({"field": field, "pearson_to_error": corr})
    return sorted(rows, key=lambda row: abs(float(row["pearson_to_error"])), reverse=True)


def _group_summary(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(field, "missing"))].append(record)
    rows = []
    for key, items in groups.items():
        errors = np.asarray([float(item["model_high_l1"]) for item in items], dtype=np.float64)
        matches = np.asarray([bool(item["top1_match"]) for item in items], dtype=np.float64)
        low_high = np.asarray(
            [float(item.get("low_high_l1", 0.0)) for item in items],
            dtype=np.float64,
        )
        rows.append(
            {
                "field": field,
                "key": key,
                "n": int(len(items)),
                "mean_model_high_l1": round(float(errors.mean()), 8),
                "top1_match_rate": round(float(matches.mean()), 8),
                "mean_low_high_l1": round(float(low_high.mean()), 8),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["n"]), -float(row["mean_model_high_l1"])))


def _policy_features_for_rows(
    rows: list[dict[str, Any]],
    cases: list[Any],
    *,
    max_action_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = []
    amounts = []
    for case in cases:
        encoded_tokens, encoded_amounts = _encode_action_sequence(
            case.action_str,
            max_tokens=max_action_tokens,
        )
        tokens.append(encoded_tokens)
        amounts.append(encoded_amounts)
    return np.stack(tokens).astype(np.int64), np.stack(amounts).astype(np.float32)


def analyze_joint_pbs_policy_alignment(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    device: str = "auto",
    start_index: int = 128,
    limit: int = 64,
    low_solver_iterations: int = 5,
    high_solver_iterations: int = 25,
    solver_backend: str = "cpu",
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    model, payload = load_joint_pbs_continuation_checkpoint(checkpoint, device=resolved_device)
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)

    high_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any] | None] = []
    row_cases: list[Any] = []
    skipped: list[dict[str, Any]] = []
    for local_idx, case in enumerate(selected_cases):
        case_idx = int(start_index) + int(local_idx)
        belief = base_dataset.belief[case_idx]
        high = _root_policy_target_row(
            case,
            belief_row=belief,
            solver_iterations=high_solver_iterations,
            solver_backend=solver_backend,
        )
        if high is None or not high.get("passed"):
            skipped.append({"label": case.label, "reason": None if high is None else high.get("skipped")})
            continue
        low = _root_policy_target_row(
            case,
            belief_row=belief,
            solver_iterations=low_solver_iterations,
            solver_backend=solver_backend,
        )
        high_rows.append(high)
        low_rows.append(low if low is not None and low.get("passed") else None)
        row_cases.append(case)

    if not high_rows:
        return {
            "mode": "joint_pbs_policy_alignment",
            "passed": False,
            "n_evaluated": 0,
            "skipped": skipped,
        }

    max_tokens = int(payload.get("max_action_tokens", _DEFAULT_MAX_ACTION_TOKENS))
    action_tokens, action_amounts = _policy_features_for_rows(
        high_rows,
        row_cases,
        max_action_tokens=max_tokens,
    )
    predictions = predict_joint_pbs_policy_model(
        model,
        payload,
        np.stack([row["feature"] for row in high_rows]).astype(np.float32),
        np.stack([row["policy_feature"] for row in high_rows]).astype(np.float32),
        np.stack([row["belief"] for row in high_rows]).astype(np.float32),
        np.stack([row["legal_mask"] for row in high_rows]).astype(np.float32),
        action_tokens=action_tokens,
        action_amounts=action_amounts,
        device=resolved_device,
    )
    targets = np.stack([row["target"] for row in high_rows]).astype(np.float32)
    legal_masks = np.stack([row["legal_mask"] for row in high_rows]).astype(np.float32)
    uniform = _legal_uniform(legal_masks)
    policy_metrics = _policy_metrics(predictions, targets, legal_masks)
    uniform_metrics = _policy_metrics(uniform, targets, legal_masks)

    records: list[dict[str, Any]] = []
    for case, high, low, pred, target, legal, uni in zip(
        row_cases,
        high_rows,
        low_rows,
        predictions,
        targets,
        legal_masks,
        uniform,
        strict=True,
    ):
        context = _case_context(case)
        pred_action = int(np.argmax(pred))
        target_action = int(np.argmax(target))
        low_target = low["target"] if low is not None else None
        low_action = int(np.argmax(low_target)) if low_target is not None else None
        hero_ent, villain_ent = _belief_entropy(high["belief"])
        record = {
            "label": str(high["label"]),
            **context,
            "legal_count": int(np.count_nonzero(legal > 0)),
            "model_high_l1": round(float(np.abs(pred - target).sum()), 8),
            "uniform_high_l1": round(float(np.abs(uni - target).sum()), 8),
            "top1_match": bool(pred_action == target_action),
            "pred_action": pred_action,
            "target_action": target_action,
            "low_action": low_action,
            "low_high_l1": (
                round(float(np.abs(low_target - target).sum()), 8)
                if low_target is not None
                else None
            ),
            "low_high_top1_match": (
                bool(low_action == target_action) if low_action is not None else None
            ),
            "pred_allin_prob": round(float(pred[8]), 8),
            "target_allin_prob": round(float(target[8]), 8),
            "uniform_allin_prob": round(float(uni[8]), 8),
            "pred_confidence": round(float(pred[pred_action]), 8),
            "target_confidence": round(float(target[target_action]), 8),
            "pred_entropy": _entropy(pred, legal),
            "target_entropy": _entropy(target, legal),
            "pred_margin": _margin(pred, legal),
            "target_margin": _margin(target, legal),
            "hero_belief_entropy": hero_ent,
            "villain_belief_entropy": villain_ent,
            "high_solver_ms": high.get("solver_ms", 0.0),
            "low_solver_ms": low.get("solver_ms", 0.0) if low is not None else None,
        }
        records.append(record)

    low_high_l1 = [float(r["low_high_l1"]) for r in records if r["low_high_l1"] is not None]
    low_high_match = [
        bool(r["low_high_top1_match"])
        for r in records
        if r["low_high_top1_match"] is not None
    ]
    error_corrs = _summarize_numeric_correlations(records)
    worst = sorted(records, key=lambda row: float(row["model_high_l1"]), reverse=True)[:10]
    return {
        "mode": "joint_pbs_policy_alignment",
        "checkpoint": str(checkpoint),
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "device": str(resolved_device),
        "start_index": int(start_index),
        "limit": int(limit),
        "low_solver_iterations": int(low_solver_iterations),
        "high_solver_iterations": int(high_solver_iterations),
        "solver_backend": solver_backend,
        "n_cases": int(len(selected_cases)),
        "n_evaluated": int(len(records)),
        "n_skipped": int(len(skipped)),
        "policy": policy_metrics,
        "legal_uniform": uniform_metrics,
        "mean_l1_improvement_vs_uniform": round(
            float(uniform_metrics["mean_l1"] - policy_metrics["mean_l1"]),
            8,
        ),
        "low_vs_high_solver": {
            "mean_l1": round(float(np.mean(low_high_l1)), 8) if low_high_l1 else None,
            "top1_match_rate": (
                round(float(np.mean(low_high_match)), 8) if low_high_match else None
            ),
            "corr_model_error_to_low_high_l1": _safe_corr(
                [float(record["low_high_l1"]) for record in records if record["low_high_l1"] is not None],
                [float(record["model_high_l1"]) for record in records if record["low_high_l1"] is not None],
            ),
        },
        "numeric_correlations_to_model_error": error_corrs,
        "groups": {
            "action_shape": _group_summary(records, "action_shape")[:12],
            "bet_count": _group_summary(records, "bet_count"),
            "target_action": _group_summary(records, "target_action"),
            "pred_action": _group_summary(records, "pred_action"),
        },
        "worst_records": worst,
        "records": records,
        "skipped": skipped,
        "passed": False,
        "pass_criteria": (
            "diagnostic only; use to decide whether failure tracks public-root "
            "distribution shift, solver instability, or model confidence"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attribute joint-PBS policy error on held-out public roots."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--low-solver-iterations", type=int, default=5)
    parser.add_argument("--high-solver-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = analyze_joint_pbs_policy_alignment(
        checkpoint=args.checkpoint,
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        device=args.device,
        start_index=args.start_index,
        limit=args.limit,
        low_solver_iterations=args.low_solver_iterations,
        high_solver_iterations=args.high_solver_iterations,
        solver_backend=args.solver_backend,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
