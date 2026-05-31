"""Decision-aware gate for Slumbot response-conditioned resolving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


_N_ACTIONS = 9


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _strategy(record: dict[str, Any], key: str) -> np.ndarray:
    raw = record.get(key, [])
    arr = np.zeros(_N_ACTIONS, dtype=np.float64)
    for idx, value in enumerate(raw[:_N_ACTIONS]):
        arr[idx] = max(0.0, _finite_float(value))
    total = float(arr.sum())
    if total > 0.0:
        arr /= total
    return arr


def _entropy(strategy: np.ndarray) -> float:
    positive = strategy[strategy > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


def _one_hot(index: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _feature_names() -> list[str]:
    return [
        "street_0",
        "street_1",
        "street_2",
        "street_3",
        "client_pos",
        "legal_count",
        "n_opponent_updates",
        "action_l1_drift",
        "action_agreement",
        "baseline_action_norm",
        "response_action_norm",
        "action_changed",
        "baseline_strategy_entropy",
        "response_strategy_entropy",
        "strategy_l1",
        "baseline_top_prob",
        "response_top_prob",
        "baseline_range_entropy",
        "response_range_entropy",
        "range_entropy_delta",
        "baseline_range_top10_mass",
        "response_range_top10_mass",
        "range_top10_delta",
        "baseline_range_top1_mass",
        "response_range_top1_mass",
        "range_top1_delta",
        "latency_delta_ms",
        "latency_ratio",
    ]


def _record_features(record: dict[str, Any]) -> list[float]:
    street = int(_finite_float(record.get("street"), 0.0))
    baseline_strategy = _strategy(record, "baseline_strategy")
    response_strategy = _strategy(record, "response_strategy")
    baseline_action = int(_finite_float(record.get("baseline_action"), -1.0))
    response_action = int(_finite_float(record.get("response_action"), -1.0))
    baseline_latency = _finite_float(record.get("baseline_latency_ms"), 0.0)
    response_latency = _finite_float(record.get("response_latency_ms"), baseline_latency)
    baseline_range_entropy = _finite_float(
        record.get("baseline_villain_range_normalized_entropy")
    )
    response_range_entropy = _finite_float(
        record.get("response_villain_range_normalized_entropy")
    )
    baseline_top10 = _finite_float(record.get("baseline_villain_range_top10_mass"))
    response_top10 = _finite_float(record.get("response_villain_range_top10_mass"))
    baseline_top1 = _finite_float(record.get("baseline_villain_range_top1_mass"))
    response_top1 = _finite_float(record.get("response_villain_range_top1_mass"))
    latency_ratio = response_latency / max(1.0, baseline_latency)
    return [
        *_one_hot(street, 4),
        _finite_float(record.get("client_pos")),
        float(len(record.get("legal_actions") or [])),
        _finite_float(record.get("n_opponent_updates")),
        _finite_float(record.get("action_l1_drift")),
        1.0 if bool(record.get("action_agreement")) else 0.0,
        baseline_action / float(_N_ACTIONS - 1) if baseline_action >= 0 else 0.0,
        response_action / float(_N_ACTIONS - 1) if response_action >= 0 else 0.0,
        1.0 if baseline_action != response_action else 0.0,
        _entropy(baseline_strategy),
        _entropy(response_strategy),
        float(np.abs(response_strategy - baseline_strategy).sum()),
        float(baseline_strategy.max(initial=0.0)),
        float(response_strategy.max(initial=0.0)),
        baseline_range_entropy,
        response_range_entropy,
        response_range_entropy - baseline_range_entropy,
        baseline_top10,
        response_top10,
        response_top10 - baseline_top10,
        baseline_top1,
        response_top1,
        response_top1 - baseline_top1,
        response_latency - baseline_latency,
        latency_ratio,
    ]


def _load_ev_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    usable: list[dict[str, Any]] = []
    for record in records:
        if not record.get("passed", True):
            continue
        if "strategy_ev_delta" not in record:
            continue
        delta = _finite_float(record.get("strategy_ev_delta"), default=np.nan)
        selected_delta = _finite_float(
            record.get("selected_action_ev_delta"),
            default=np.nan,
        )
        if not np.isfinite(delta) or not np.isfinite(selected_delta):
            continue
        usable.append(record)
    return usable


def _matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("no usable EV gate records found")
    x = np.asarray([_record_features(record) for record in records], dtype=np.float64)
    y_strategy = np.asarray(
        [_finite_float(record.get("strategy_ev_delta")) for record in records],
        dtype=np.float64,
    )
    y_selected = np.asarray(
        [_finite_float(record.get("selected_action_ev_delta")) for record in records],
        dtype=np.float64,
    )
    return x, y_strategy, y_selected


def _standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (eval_x - mean) / std


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge_l2: float) -> np.ndarray:
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    reg = np.eye(design.shape[1], dtype=x.dtype) * max(0.0, float(ridge_l2))
    reg[-1, -1] = 0.0
    lhs = design.T @ design + reg
    rhs = design.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return design @ weights


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _selection_metrics(
    *,
    strategy_delta: np.ndarray,
    selected_delta: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    choose_response = predictions > threshold
    selected_count = int(choose_response.sum())
    selective_strategy = np.where(choose_response, strategy_delta, 0.0)
    selective_selected = np.where(choose_response, selected_delta, 0.0)
    chosen_positive = strategy_delta[choose_response] > 0.0
    return {
        "threshold": float(threshold),
        "selected_count": selected_count,
        "selection_rate": float(selected_count / max(1, strategy_delta.size)),
        "direct_mean_strategy_ev_delta": float(strategy_delta.mean()),
        "direct_mean_selected_action_ev_delta": float(selected_delta.mean()),
        "direct_response_beats_baseline_rate": float((strategy_delta > 0.0).mean()),
        "selective_mean_strategy_ev_delta": float(selective_strategy.mean()),
        "selective_mean_selected_action_ev_delta": float(selective_selected.mean()),
        "selective_response_beats_baseline_rate": (
            float(chosen_positive.mean()) if selected_count else 0.0
        ),
        "oracle_mean_strategy_ev_delta": float(np.maximum(strategy_delta, 0.0).mean()),
        "oracle_selected_count": int((strategy_delta > 0.0).sum()),
    }


def evaluate_response_decision_gate(
    train_ev_gate_json: str | Path,
    eval_ev_gate_json: str | Path,
    *,
    ridge_l2: float = 1.0,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Train a cross-trace selector for response-conditioned resolving.

    The feature set deliberately excludes revealed-hand likelihood, true-hand
    ranks, counterfactual action values, and realized winnings. It only uses
    quantities available from the baseline/response solvers and their ranges.
    """

    train_records = _load_ev_records(train_ev_gate_json)
    eval_records = _load_ev_records(eval_ev_gate_json)
    train_x, train_y, train_selected_y = _matrix(train_records)
    eval_x, eval_y, eval_selected_y = _matrix(eval_records)
    train_x_std, eval_x_std = _standardize(train_x, eval_x)
    weights = _fit_ridge(train_x_std, train_y, ridge_l2)
    selected_weights = _fit_ridge(train_x_std, train_selected_y, ridge_l2)
    train_predictions = _predict(train_x_std, weights)
    eval_predictions = _predict(eval_x_std, weights)
    train_selected_predictions = _predict(train_x_std, selected_weights)
    eval_selected_predictions = _predict(eval_x_std, selected_weights)
    train_metrics = _selection_metrics(
        strategy_delta=train_y,
        selected_delta=train_selected_y,
        predictions=train_predictions,
        threshold=threshold,
    )
    eval_metrics = _selection_metrics(
        strategy_delta=eval_y,
        selected_delta=eval_selected_y,
        predictions=eval_predictions,
        threshold=threshold,
    )
    decision_passed = bool(
        eval_metrics["selected_count"] > 0
        and eval_metrics["selective_mean_strategy_ev_delta"]
        > max(0.0, eval_metrics["direct_mean_strategy_ev_delta"])
        and eval_metrics["selective_mean_selected_action_ev_delta"]
        > max(0.0, eval_metrics["direct_mean_selected_action_ev_delta"])
        and eval_metrics["selective_response_beats_baseline_rate"]
        >= eval_metrics["direct_response_beats_baseline_rate"]
    )
    return {
        "mode": "slumbot_response_decision_gate",
        "train_source": str(train_ev_gate_json),
        "eval_source": str(eval_ev_gate_json),
        "ridge_l2": float(ridge_l2),
        "feature_names": _feature_names(),
        "train_n": int(train_y.size),
        "eval_n": int(eval_y.size),
        "train_prediction_mae": float(np.abs(train_predictions - train_y).mean()),
        "eval_prediction_mae": float(np.abs(eval_predictions - eval_y).mean()),
        "train_prediction_pearson": _pearson(train_predictions, train_y),
        "eval_prediction_pearson": _pearson(eval_predictions, eval_y),
        "train_selected_prediction_mae": float(
            np.abs(train_selected_predictions - train_selected_y).mean()
        ),
        "eval_selected_prediction_mae": float(
            np.abs(eval_selected_predictions - eval_selected_y).mean()
        ),
        "train_selected_prediction_pearson": _pearson(
            train_selected_predictions,
            train_selected_y,
        ),
        "eval_selected_prediction_pearson": _pearson(
            eval_selected_predictions,
            eval_selected_y,
        ),
        "train": train_metrics,
        "eval": eval_metrics,
        **eval_metrics,
        "decision_gate_passed": decision_passed,
        "passed": True,
    }


def write_metrics(metrics: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
