"""Score revealed Slumbot hands under trace-conditioned opponent ranges."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import card_str_to_index, parse_action  # noqa: E402
from range_tracker import RangeTracker, update_tracker_from_actions  # noqa: E402


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _last_decisions_by_hand(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    decisions: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("event") != "decision":
            continue
        if record.get("source") == "solver" and not record.get("full_action_str"):
            continue
        try:
            hand_index = int(record["hand_index"])
        except (KeyError, TypeError, ValueError):
            continue
        decisions[hand_index] = record
    return decisions


def _decision_action_str(decision: dict[str, Any]) -> str:
    return str(decision.get("full_action_str") or decision.get("action_str") or "")


def _street_name(record: dict[str, Any]) -> str:
    street = record.get("street")
    if street:
        return str(street)
    try:
        street_index = int(record.get("street_index"))
    except (TypeError, ValueError):
        return "unknown"
    return {0: "preflop", 1: "flop", 2: "turn", 3: "river"}.get(
        street_index,
        "unknown",
    )


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "mean_log_lift_vs_uniform": 0.0,
            "mean_true_hand_percentile": 0.0,
            "mean_winnings": 0.0,
        }
    return {
        "n": len(records),
        "mean_log_lift_vs_uniform": float(
            np.mean([float(record["log_lift_vs_uniform"]) for record in records])
        ),
        "mean_true_hand_percentile": float(
            np.mean([float(record["true_hand_percentile"]) for record in records])
        ),
        "mean_winnings": float(np.mean([float(record.get("winnings", 0)) for record in records])),
    }


def _score_true_hand(
    *,
    value_net: Any,
    device: torch.device,
    strategy_source: str,
    decision: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    bot_cards = list(result.get("bot_hole_cards") or [])
    hero_cards = list(decision.get("hole_cards") or [])
    board = list(decision.get("board") or [])
    if len(bot_cards) != 2 or len(hero_cards) != 2:
        return None
    visible = [*bot_cards, *hero_cards, *board]
    if len(visible) != len(set(visible)):
        return None
    action_str = _decision_action_str(decision)
    parsed = parse_action(action_str)
    if "error" in parsed:
        return None
    try:
        client_pos = int(decision["client_pos"])
        hand_index = int(decision["hand_index"])
    except (KeyError, TypeError, ValueError):
        return None

    board_idx = [card_str_to_index(card) for card in board]
    hero_idx = [card_str_to_index(card) for card in hero_cards]
    bot_hand = tuple(sorted(card_str_to_index(card) for card in bot_cards))
    tracker = RangeTracker(
        hero_idx,
        value_net,
        device,
        strategy_source=strategy_source,
    )
    update_tracker_from_actions(tracker, action_str, client_pos, board_idx)

    remaining = sorted(set(range(52)) - set(board_idx))
    solver_hands = list(itertools.combinations(remaining, 2))
    hand_to_idx = {tuple(hand): idx for idx, hand in enumerate(solver_hands)}
    true_idx = hand_to_idx.get(bot_hand)
    if true_idx is None:
        return None
    _, villain_range = tracker.get_solver_ranges(solver_hands, hand_to_idx)
    probs = np.asarray(villain_range, dtype=np.float64)
    total = float(probs.sum())
    if total <= 0.0:
        return None
    probs = probs / total
    positive = probs[probs > 0]
    if positive.size <= 0:
        return None
    true_prob = float(probs[true_idx])
    uniform_prob = 1.0 / float(positive.size)
    eps = 1e-12
    log_lift = math.log(max(true_prob, eps)) - math.log(max(uniform_prob, eps))
    percentile = float(np.mean(positive <= true_prob))
    rank = int(1 + np.sum(positive > true_prob))
    return {
        "hand_index": hand_index,
        "street": _street_name(decision),
        "street_index": decision.get("street_index"),
        "action_str": action_str,
        "client_pos": client_pos,
        "hole_cards": hero_cards,
        "bot_hole_cards": bot_cards,
        "board": board,
        "winnings": int(result.get("winnings", 0)),
        "true_hand_prob": true_prob,
        "uniform_prob": uniform_prob,
        "log_lift_vs_uniform": float(log_lift),
        "true_hand_percentile": percentile,
        "true_hand_rank": rank,
        "range_support": int(positive.size),
        "range_top1_mass": float(np.max(positive)),
        "range_top10_mass": float(np.sort(positive)[::-1][:10].sum()),
    }


def diagnose_trace_true_range(
    checkpoint: str | Path,
    trace_path: str | Path,
    *,
    strategy_source: str = "regret",
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Score revealed Slumbot hands under the model's opponent range."""
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, strategy_source)
    records = _read_trace(trace_path)
    last_decisions = _last_decisions_by_hand(records)
    scored: list[dict[str, Any]] = []
    skipped_results = 0
    for record in records:
        if record.get("event") != "hand_result":
            continue
        try:
            hand_index = int(record["hand_index"])
        except (KeyError, TypeError, ValueError):
            skipped_results += 1
            continue
        decision = last_decisions.get(hand_index)
        if decision is None:
            skipped_results += 1
            continue
        scored_record = _score_true_hand(
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=strategy_source,
            decision=decision,
            result=record,
        )
        if scored_record is None:
            skipped_results += 1
            continue
        scored.append(scored_record)

    log_lifts = [float(record["log_lift_vs_uniform"]) for record in scored]
    percentiles = [float(record["true_hand_percentile"]) for record in scored]
    by_street = {
        street: _aggregate([record for record in scored if record["street"] == street])
        for street in ("preflop", "flop", "turn", "river", "unknown")
        if any(record["street"] == street for record in scored)
    }
    outcome_buckets = {
        "loss": _aggregate([record for record in scored if int(record.get("winnings", 0)) < 0]),
        "big_loss": _aggregate(
            [record for record in scored if int(record.get("winnings", 0)) <= -1000]
        ),
        "big_win": _aggregate(
            [record for record in scored if int(record.get("winnings", 0)) >= 1000]
        ),
        "large_pot_abs": _aggregate(
            [record for record in scored if abs(int(record.get("winnings", 0))) >= 1000]
        ),
    }
    return {
        "mode": "slumbot_trace_true_range_likelihood",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": loaded.metadata.get("checkpoint_iteration"),
        "trace": str(trace_path),
        "strategy_source": strategy_source,
        "device": str(resolved_device),
        "n_scored": len(scored),
        "n_skipped_results": skipped_results,
        "mean_log_lift_vs_uniform": float(np.mean(log_lifts)) if log_lifts else 0.0,
        "mean_true_hand_percentile": float(np.mean(percentiles)) if percentiles else 0.0,
        "by_street": by_street,
        "outcome_buckets": outcome_buckets,
        "records": scored,
        "passed": bool(scored),
    }
