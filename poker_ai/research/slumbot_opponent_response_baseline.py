"""Leave-one-hand-out opponent-response baselines for Slumbot trace actions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from range_tracker import N_ACTIONS, _get_legal_mask, _parse_action  # noqa: E402


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _target_vector(record: dict[str, Any]) -> np.ndarray:
    target = np.zeros(N_ACTIONS, dtype=np.float64)
    for item in record.get("mapped_actions") or []:
        try:
            action_idx = int(item["action_idx"])
            weight = float(item.get("weight", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= action_idx < N_ACTIONS and weight > 0.0:
            target[action_idx] += weight
    total = float(target.sum())
    if total > 0.0:
        target /= total
    return target


def _legal_mask(record: dict[str, Any]) -> np.ndarray | None:
    action_str = str(record.get("action_str_before") or "")
    try:
        client_pos = int(record["client_pos"])
    except (KeyError, TypeError, ValueError):
        return None
    parsed = _parse_action(action_str)
    if "error" in parsed:
        return None
    return _get_legal_mask(parsed, action_str, 1 - client_pos).astype(np.float64)


def _score_prior(
    target: np.ndarray,
    legal_mask: np.ndarray,
    counts: np.ndarray,
    *,
    alpha: float,
) -> float:
    legal = (legal_mask > 0).astype(np.float64)
    prior = (counts.astype(np.float64) + float(alpha)) * legal
    total = float(prior.sum())
    if total <= 0.0:
        legal_total = float(legal.sum())
        if legal_total <= 0.0:
            return 0.0
        prior = legal / legal_total
    else:
        prior /= total
    return float(np.dot(target, prior))


def _aggregate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "mean_prob": 0.0,
            "mean_log_lift_vs_uniform": 0.0,
        }
    prob_key = f"{key}_prob" if key != "model" else "model_prob"
    lift_key = (
        f"{key}_log_lift_vs_uniform"
        if key != "model"
        else "model_log_lift_vs_uniform"
    )
    return {
        "n": len(records),
        "mean_prob": float(np.mean([float(record[prob_key]) for record in records])),
        "mean_log_lift_vs_uniform": float(
            np.mean([float(record[lift_key]) for record in records])
        ),
    }


def _group_aggregates(records: list[dict[str, Any]], key: str, group_key: str) -> dict[str, Any]:
    values = sorted({str(record[group_key]) for record in records})
    return {
        value: _aggregate([record for record in records if str(record[group_key]) == value], key)
        for value in values
    }


def analyze_opponent_response_baseline(
    action_likelihood_json: str | Path,
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Compare model likelihood with leave-one-hand-out empirical priors."""
    payload = _read_json(action_likelihood_json)
    rows: list[dict[str, Any]] = []
    for record in payload.get("records") or []:
        target = _target_vector(record)
        legal = _legal_mask(record)
        if legal is None or float(target.sum()) <= 0.0 or float(legal.sum()) <= 0.0:
            continue
        try:
            hand_index = int(record["hand_index"])
        except (KeyError, TypeError, ValueError):
            continue
        street = str(record.get("street") or "unknown")
        action_char = str(record.get("action_char") or "?")
        uniform_prob = float(np.dot(target, legal / max(float(legal.sum()), 1.0)))
        model_prob = float(record.get("action_prob", 0.0))
        rows.append(
            {
                "source": record,
                "target": target,
                "legal": legal,
                "hand_index": hand_index,
                "street": street,
                "action_char": action_char,
                "model_prob": model_prob,
                "uniform_prob": uniform_prob,
            }
        )

    targets = [row["target"] for row in rows]
    total_counts = np.sum(targets, axis=0) if targets else np.zeros(N_ACTIONS, dtype=np.float64)
    records: list[dict[str, Any]] = []
    eps = 1e-12
    for row in rows:
        same_hand = [other for other in rows if other["hand_index"] == row["hand_index"]]
        same_hand_counts = (
            np.sum([other["target"] for other in same_hand], axis=0)
            if same_hand
            else np.zeros(N_ACTIONS, dtype=np.float64)
        )
        global_counts = total_counts - same_hand_counts
        street_rows = [
            other
            for other in rows
            if other["street"] == row["street"] and other["hand_index"] != row["hand_index"]
        ]
        street_counts = (
            np.sum([other["target"] for other in street_rows], axis=0)
            if street_rows
            else global_counts
        )
        global_prob = _score_prior(row["target"], row["legal"], global_counts, alpha=alpha)
        street_prob = _score_prior(row["target"], row["legal"], street_counts, alpha=alpha)
        uniform_prob = float(row["uniform_prob"])
        model_prob = float(row["model_prob"])
        records.append(
            {
                "hand_index": row["hand_index"],
                "street": row["street"],
                "action_char": row["action_char"],
                "model_prob": model_prob,
                "uniform_prob": uniform_prob,
                "global_loo_prob": global_prob,
                "street_loo_prob": street_prob,
                "model_log_lift_vs_uniform": float(
                    math.log(max(model_prob, eps)) - math.log(max(uniform_prob, eps))
                ),
                "global_loo_log_lift_vs_uniform": float(
                    math.log(max(global_prob, eps)) - math.log(max(uniform_prob, eps))
                ),
                "street_loo_log_lift_vs_uniform": float(
                    math.log(max(street_prob, eps)) - math.log(max(uniform_prob, eps))
                ),
            }
        )

    return {
        "mode": "slumbot_opponent_response_baseline",
        "source": str(action_likelihood_json),
        "alpha": float(alpha),
        "n_records": len(records),
        "n_hands": len({record["hand_index"] for record in records}),
        "aggregates": {
            "model": _aggregate(records, "model"),
            "global_loo": _aggregate(records, "global_loo"),
            "street_loo": _aggregate(records, "street_loo"),
        },
        "by_street": {
            "model": _group_aggregates(records, "model", "street"),
            "global_loo": _group_aggregates(records, "global_loo", "street"),
            "street_loo": _group_aggregates(records, "street_loo", "street"),
        },
        "by_action_char": {
            "model": _group_aggregates(records, "model", "action_char"),
            "global_loo": _group_aggregates(records, "global_loo", "action_char"),
            "street_loo": _group_aggregates(records, "street_loo", "action_char"),
        },
        "records": records,
        "passed": bool(records),
    }
