#!/usr/bin/env python3
"""Evaluate profile-drift selective escalation against a CFR teacher frontier."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _case_index(label: str) -> int:
    match = re.search(r"-(\d+)-", str(label))
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", str(label))
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse case index from label: {label}")


def _profile_l1(record: dict[str, Any], reference_profile: str, candidate_profile: str) -> float:
    reference = np.asarray(record["profiles"][reference_profile]["strategy"], dtype=np.float64)
    candidate = np.asarray(record["profiles"][candidate_profile]["strategy"], dtype=np.float64)
    return float(np.abs(candidate - reference).sum())


def _frontier_by_label(frontier_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["label"]): record
        for record in frontier_metrics.get("records", [])
        if record.get("passed")
    }


def _rows(
    frontier_metrics: dict[str, Any],
    profile_metrics: dict[str, Any],
    *,
    reference_profile: str,
    candidate_profile: str,
    escalation_budget: int,
) -> list[dict[str, Any]]:
    frontier = _frontier_by_label(frontier_metrics)
    rows: list[dict[str, Any]] = []
    for record in profile_metrics.get("records", []):
        if not record.get("passed"):
            continue
        label = str(record["label"])
        if label not in frontier:
            continue
        live_iterations = str(int(record["profiles"][reference_profile]["iterations"]))
        frontier_record = frontier[label]
        candidate_profile_data = record["profiles"][candidate_profile]
        live_budget = frontier_record["budgets"][live_iterations]
        escalation = frontier_record["budgets"][str(int(escalation_budget))]
        rows.append(
            {
                "label": label,
                "case_index": _case_index(label),
                "profile_l1": _profile_l1(record, reference_profile, candidate_profile),
                "live_l1": float(live_budget["l1_to_reference"]),
                "live_kl": float(live_budget["kl_to_reference"]),
                "live_latency_ms": float(live_budget["latency_ms"]),
                "candidate_profile_latency_ms": float(candidate_profile_data.get("latency_ms", 0.0)),
                "escalation_l1": float(escalation["l1_to_reference"]),
                "escalation_kl": float(escalation["kl_to_reference"]),
                "escalation_latency_ms": float(escalation["latency_ms"]),
            }
        )
    return rows


def _select_threshold(train_rows: list[dict[str, Any]], select_train_top_k: int) -> float:
    if not train_rows:
        raise ValueError("train split is empty")
    if select_train_top_k <= 0:
        raise ValueError("select_train_top_k must be positive")
    if select_train_top_k > len(train_rows):
        raise ValueError("select_train_top_k exceeds train split size")
    ranked = sorted(train_rows, key=lambda row: row["profile_l1"], reverse=True)
    return float(ranked[int(select_train_top_k) - 1]["profile_l1"])


def evaluate_selective_escalation(
    frontier_metrics: dict[str, Any],
    profile_metrics: dict[str, Any],
    *,
    train_start_index: int = 128,
    train_limit: int = 32,
    holdout_start_index: int = 160,
    holdout_limit: int = 32,
    select_train_top_k: int = 4,
    escalation_budget: int = 350,
    reference_profile: str = "live",
    candidate_profile: str = "fast-live",
) -> dict[str, Any]:
    rows = _rows(
        frontier_metrics,
        profile_metrics,
        reference_profile=reference_profile,
        candidate_profile=candidate_profile,
        escalation_budget=escalation_budget,
    )
    train_end = int(train_start_index) + int(train_limit)
    holdout_end = int(holdout_start_index) + int(holdout_limit)
    train_rows = [
        row for row in rows
        if int(train_start_index) <= row["case_index"] < train_end
    ]
    holdout_rows = [
        row for row in rows
        if int(holdout_start_index) <= row["case_index"] < holdout_end
    ]
    threshold = _select_threshold(train_rows, int(select_train_top_k))
    selected = [row for row in holdout_rows if row["profile_l1"] >= threshold]

    selective_l1 = [
        row["escalation_l1"] if row["profile_l1"] >= threshold else row["live_l1"]
        for row in holdout_rows
    ]
    selective_kl = [
        row["escalation_kl"] if row["profile_l1"] >= threshold else row["live_kl"]
        for row in holdout_rows
    ]
    selective_latency = [
        row["escalation_latency_ms"] if row["profile_l1"] >= threshold else row["live_latency_ms"]
        for row in holdout_rows
    ]
    online_decision_latency = [
        row["candidate_profile_latency_ms"]
        + row["live_latency_ms"]
        + (row["escalation_latency_ms"] if row["profile_l1"] >= threshold else 0.0)
        for row in holdout_rows
    ]
    live_l1 = _mean([row["live_l1"] for row in holdout_rows])
    selective_mean_l1 = _mean(selective_l1)
    live_kl = _mean([row["live_kl"] for row in holdout_rows])
    selective_mean_kl = _mean(selective_kl)
    live_latency = _mean([row["live_latency_ms"] for row in holdout_rows])
    selective_mean_latency = _mean(selective_latency)
    online_decision_mean_latency = _mean(online_decision_latency)
    return {
        "mode": "solver_budget_selective_escalation",
        "reference_profile": reference_profile,
        "candidate_profile": candidate_profile,
        "escalation_budget": int(escalation_budget),
        "train_start_index": int(train_start_index),
        "train_limit": int(train_limit),
        "holdout_start_index": int(holdout_start_index),
        "holdout_limit": int(holdout_limit),
        "select_train_top_k": int(select_train_top_k),
        "threshold_profile_l1": round(float(threshold), 8),
        "n_train": int(len(train_rows)),
        "n_holdout": int(len(holdout_rows)),
        "holdout_selected": int(len(selected)),
        "selected_labels": [row["label"] for row in selected],
        "live_mean_l1": live_l1,
        "selective_mean_l1": selective_mean_l1,
        "uniform_escalation_mean_l1": _mean([row["escalation_l1"] for row in holdout_rows]),
        "live_mean_kl": live_kl,
        "selective_mean_kl": selective_mean_kl,
        "uniform_escalation_mean_kl": _mean([row["escalation_kl"] for row in holdout_rows]),
        "live_mean_latency_ms": live_latency,
        "selective_mean_latency_ms": selective_mean_latency,
        "online_decision_mean_latency_ms": online_decision_mean_latency,
        "online_decision_latency_ratio_to_live": round(
            float(online_decision_mean_latency / max(live_latency, 1e-9)),
            8,
        ) if holdout_rows else 0.0,
        "uniform_escalation_mean_latency_ms": _mean(
            [row["escalation_latency_ms"] for row in holdout_rows]
        ),
        "latency_ratio_to_live": round(
            float(selective_mean_latency / max(live_latency, 1e-9)),
            8,
        ) if holdout_rows else 0.0,
        "l1_improvement": round(float(live_l1 - selective_mean_l1), 8),
        "kl_improvement": round(float(live_kl - selective_mean_kl), 8),
        "passed": bool(
            len(train_rows) == int(train_limit)
            and len(holdout_rows) == int(holdout_limit)
            and selective_mean_l1 <= live_l1
            and selective_mean_kl <= live_kl
        ),
        "promotion": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate train-thresholded profile-drift escalation on a holdout split."
    )
    parser.add_argument("--frontier-json", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--train-start-index", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--holdout-start-index", type=int, default=160)
    parser.add_argument("--holdout-limit", type=int, default=32)
    parser.add_argument("--select-train-top-k", type=int, default=4)
    parser.add_argument("--escalation-budget", type=int, default=350)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    frontier_metrics = json.loads(Path(args.frontier_json).read_text(encoding="utf-8"))
    profile_metrics = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    metrics = evaluate_selective_escalation(
        frontier_metrics,
        profile_metrics,
        train_start_index=args.train_start_index,
        train_limit=args.train_limit,
        holdout_start_index=args.holdout_start_index,
        holdout_limit=args.holdout_limit,
        select_train_top_k=args.select_train_top_k,
        escalation_budget=args.escalation_budget,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
