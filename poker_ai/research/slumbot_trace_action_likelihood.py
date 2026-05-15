"""Score observed Slumbot actions under the model likelihood for revealed hands."""

from __future__ import annotations

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

from play_slumbot import card_str_to_index  # noqa: E402
from range_tracker import (  # noqa: E402
    RangeTracker,
    _get_legal_mask,
    _parse_action,
    map_slumbot_action_to_idx,
    walk_actions,
)


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
        if record.get("event") != "decision" or record.get("source") != "policy":
            continue
        try:
            hand_index = int(record["hand_index"])
        except (KeyError, TypeError, ValueError):
            continue
        decisions[hand_index] = record
    return decisions


def _street_name(street_index: int) -> str:
    return {0: "preflop", 1: "flop", 2: "turn", 3: "river"}.get(
        int(street_index),
        "unknown",
    )


def _visible_board_for(prefix: str, board_idx: list[int]) -> list[int]:
    n_slashes = prefix.count("/")
    if n_slashes >= 3 and len(board_idx) >= 5:
        return list(board_idx[:5])
    if n_slashes >= 2 and len(board_idx) >= 4:
        return list(board_idx[:4])
    if n_slashes >= 1 and len(board_idx) >= 3:
        return list(board_idx[:3])
    return []


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "mean_log_lift_vs_uniform": 0.0,
            "mean_action_prob": 0.0,
            "mean_uniform_prob": 0.0,
        }
    return {
        "n": len(records),
        "mean_log_lift_vs_uniform": float(
            np.mean([float(record["log_lift_vs_uniform"]) for record in records])
        ),
        "mean_action_prob": float(np.mean([float(record["action_prob"]) for record in records])),
        "mean_uniform_prob": float(np.mean([float(record["uniform_prob"]) for record in records])),
    }


def _score_hand_actions(
    *,
    value_net: Any,
    device: torch.device,
    strategy_source: str,
    decision: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    bot_cards = list(result.get("bot_hole_cards") or [])
    hero_cards = list(decision.get("hole_cards") or [])
    board = list(decision.get("board") or [])
    if len(bot_cards) != 2 or len(hero_cards) != 2:
        return []
    visible = [*bot_cards, *hero_cards, *board]
    if len(visible) != len(set(visible)):
        return []
    try:
        client_pos = int(decision["client_pos"])
        hand_index = int(decision["hand_index"])
    except (KeyError, TypeError, ValueError):
        return []

    board_idx = [card_str_to_index(card) for card in board]
    hero_idx = [card_str_to_index(card) for card in hero_cards]
    bot_hand = tuple(sorted(card_str_to_index(card) for card in bot_cards))
    opp_pos = 1 - client_pos
    tracker = RangeTracker(
        hero_idx,
        value_net,
        device,
        strategy_source=strategy_source,
    )
    action_str = str(decision.get("action_str") or "")
    records: list[dict[str, Any]] = []

    for before, acting_pos, action_char, bet_to in walk_actions(action_str):
        if int(acting_pos) != opp_pos:
            continue
        parsed_before = _parse_action(before)
        if "error" in parsed_before:
            continue
        current_board = _visible_board_for(before, board_idx)
        mapped = map_slumbot_action_to_idx(
            action_char,
            bet_to,
            before,
            acting_pos,
            parsed_before,
        )
        strategies = tracker._batch_blueprint(  # noqa: SLF001 - diagnostic mirrors RangeTracker.
            [bot_hand],
            current_board,
            before,
            opp_pos,
            parsed_before,
        )
        legal_mask = _get_legal_mask(parsed_before, before, opp_pos)
        legal_count = float(max(1.0, legal_mask.sum()))
        action_prob = float(
            sum(float(strategies[0, idx]) * float(weight) for idx, weight in mapped)
        )
        uniform_prob = float(
            sum(float(legal_mask[idx]) * float(weight) / legal_count for idx, weight in mapped)
        )
        eps = 1e-12
        log_lift = math.log(max(action_prob, eps)) - math.log(max(uniform_prob, eps))
        street_index = int(parsed_before.get("st", -1))
        records.append(
            {
                "hand_index": hand_index,
                "client_pos": client_pos,
                "street": _street_name(street_index),
                "street_index": street_index,
                "action_str_before": before,
                "action_char": action_char,
                "bet_to": int(bet_to),
                "mapped_actions": [
                    {"action_idx": int(idx), "weight": float(weight)}
                    for idx, weight in mapped
                ],
                "hole_cards": hero_cards,
                "bot_hole_cards": bot_cards,
                "board": board,
                "visible_board": [
                    board[board_idx.index(card_idx)]
                    for card_idx in current_board
                    if card_idx in board_idx
                ],
                "winnings": int(result.get("winnings", 0)),
                "action_prob": action_prob,
                "uniform_prob": uniform_prob,
                "log_lift_vs_uniform": float(log_lift),
            }
        )
    return records


def diagnose_trace_opponent_action_likelihood(
    checkpoint: str | Path,
    trace_path: str | Path,
    *,
    strategy_source: str = "regret",
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Score observed Slumbot actions for hands with revealed bot cards."""
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, strategy_source)
    trace_records = _read_trace(trace_path)
    last_decisions = _last_decisions_by_hand(trace_records)
    scored: list[dict[str, Any]] = []
    skipped_results = 0
    for record in trace_records:
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
        hand_records = _score_hand_actions(
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=strategy_source,
            decision=decision,
            result=record,
        )
        if not hand_records:
            skipped_results += 1
            continue
        scored.extend(hand_records)

    by_street = {
        street: _aggregate([record for record in scored if record["street"] == street])
        for street in ("preflop", "flop", "turn", "river", "unknown")
        if any(record["street"] == street for record in scored)
    }
    by_action_char = {
        action_char: _aggregate(
            [record for record in scored if record["action_char"] == action_char]
        )
        for action_char in ("b", "c", "k", "f")
        if any(record["action_char"] == action_char for record in scored)
    }
    log_lifts = [float(record["log_lift_vs_uniform"]) for record in scored]
    scored_hands = sorted({int(record["hand_index"]) for record in scored})
    return {
        "mode": "slumbot_trace_opponent_action_likelihood",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": loaded.metadata.get("checkpoint_iteration"),
        "trace": str(trace_path),
        "strategy_source": strategy_source,
        "device": str(resolved_device),
        "n_scored_actions": len(scored),
        "n_scored_hands": len(scored_hands),
        "n_skipped_results": skipped_results,
        "mean_log_lift_vs_uniform": float(np.mean(log_lifts)) if log_lifts else 0.0,
        "by_street": by_street,
        "by_action_char": by_action_char,
        "records": scored,
        "passed": bool(scored),
    }
