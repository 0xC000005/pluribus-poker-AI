"""Offline analysis helpers for Slumbot JSONL decision traces."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _bet_amount(increment: str | None) -> int:
    if increment and increment.startswith("b") and increment[1:].isdigit():
        return int(increment[1:])
    return 0


def _selected_margin(decision: dict[str, Any]) -> float | None:
    advantages = decision.get("advantages") or []
    legal_mask = decision.get("legal_mask") or []
    action_idx = decision.get("action_idx")
    if action_idx is None or not advantages or not legal_mask:
        return None
    if action_idx >= len(advantages):
        return None
    legal = [
        (idx, float(advantage))
        for idx, advantage in enumerate(advantages)
        if idx < len(legal_mask) and int(legal_mask[idx]) > 0
    ]
    if not legal:
        return None
    selected = float(advantages[action_idx])
    alternatives = [advantage for idx, advantage in legal if idx != action_idx]
    second_best = max(alternatives) if alternatives else selected
    return selected - second_best


def _is_risk_decision(
    decision: dict[str, Any],
    *,
    risk_bet_threshold: int,
    risk_call_threshold: int,
) -> bool:
    action_idx = decision.get("action_idx")
    bet_amount = _bet_amount(decision.get("increment"))
    to_call = int(decision.get("last_bet_size") or 0)
    return (
        action_idx in (6, 7, 8)
        or bet_amount >= risk_bet_threshold
        or (action_idx == 1 and to_call >= risk_call_threshold)
    )


def _hand_bucket(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "avg_chips": 0.0,
            "total_chips": 0,
            "stack_losses": 0,
            "big_losses": 0,
            "wins": 0,
        }
    total = int(sum(values))
    return {
        "n": len(values),
        "avg_chips": round(float(total / len(values)), 3),
        "total_chips": total,
        "stack_losses": sum(1 for value in values if value <= -19000),
        "big_losses": sum(1 for value in values if value <= -1000),
        "wins": sum(1 for value in values if value > 0),
    }


def analyze_trace(
    trace_path: str | Path,
    *,
    risk_bet_threshold: int = 5000,
    risk_call_threshold: int = 5000,
    low_margin_threshold: float = 0.01,
    worst_limit: int = 12,
) -> dict[str, Any]:
    """Summarize policy risk/calibration evidence from one Slumbot JSONL trace."""
    path = Path(trace_path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    event_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        event_counts[str(record.get("event"))] += 1
        if record.get("source"):
            source_counts[str(record.get("source"))] += 1
        hand_index = record.get("hand_index")
        if hand_index is not None:
            grouped[int(hand_index)].append(record)

    hand_summaries = []
    for hand_index, records in grouped.items():
        result = next((r for r in records if r.get("event") == "hand_result"), None)
        if result is None:
            continue
        winnings = int(result.get("winnings", 0))
        policy_decisions = [
            r for r in records
            if r.get("event") == "decision" and r.get("source") == "policy"
        ]
        risk_decisions = [
            r for r in policy_decisions
            if _is_risk_decision(
                r,
                risk_bet_threshold=risk_bet_threshold,
                risk_call_threshold=risk_call_threshold,
            )
        ]
        low_margin_risk = [
            r for r in risk_decisions
            if (_selected_margin(r) is not None and _selected_margin(r) <= low_margin_threshold)
        ]
        big_calls = [
            r for r in policy_decisions
            if r.get("action_idx") == 1 and int(r.get("last_bet_size") or 0) >= risk_call_threshold
        ]
        allins = [r for r in policy_decisions if r.get("action_idx") == 8]
        max_bet = max((_bet_amount(r.get("increment")) for r in policy_decisions), default=0)
        hand_summaries.append({
            "hand_index": hand_index,
            "winnings": winnings,
            "hole_cards": result.get("hole_cards"),
            "bot_hole_cards": result.get("bot_hole_cards"),
            "board": result.get("board"),
            "policy_decisions": len(policy_decisions),
            "has_risk_decision": bool(risk_decisions),
            "has_low_margin_risk_decision": bool(low_margin_risk),
            "has_big_call": bool(big_calls),
            "has_allin": bool(allins),
            "max_bet": max_bet,
        })

    winnings = [int(hand["winnings"]) for hand in hand_summaries]
    risk_winnings = [int(hand["winnings"]) for hand in hand_summaries if hand["has_risk_decision"]]
    no_risk_winnings = [int(hand["winnings"]) for hand in hand_summaries if not hand["has_risk_decision"]]
    low_margin_risk_winnings = [
        int(hand["winnings"]) for hand in hand_summaries
        if hand["has_low_margin_risk_decision"]
    ]
    allin_winnings = [int(hand["winnings"]) for hand in hand_summaries if hand["has_allin"]]
    big_call_winnings = [int(hand["winnings"]) for hand in hand_summaries if hand["has_big_call"]]
    max_bet_winnings = [
        int(hand["winnings"]) for hand in hand_summaries
        if int(hand["max_bet"]) >= risk_bet_threshold
    ]

    worst_hands = sorted(hand_summaries, key=lambda hand: int(hand["winnings"]))[:worst_limit]
    action_mix = Counter()
    margin_values = []
    risk_margin_values = []
    for records in grouped.values():
        for record in records:
            if record.get("event") != "decision" or record.get("source") != "policy":
                continue
            action_mix[str(record.get("action_name"))] += 1
            margin = _selected_margin(record)
            if margin is not None:
                margin_values.append(float(margin))
                if _is_risk_decision(
                    record,
                    risk_bet_threshold=risk_bet_threshold,
                    risk_call_threshold=risk_call_threshold,
                ):
                    risk_margin_values.append(float(margin))

    return {
        "trace_path": str(path),
        "hands": len(hand_summaries),
        "total_chips": int(sum(winnings)),
        "avg_chips_per_hand": round(float(sum(winnings) / len(winnings)), 3) if winnings else 0.0,
        "stack_losses": sum(1 for value in winnings if value <= -19000),
        "big_losses": sum(1 for value in winnings if value <= -1000),
        "win_rate": round(float(sum(1 for value in winnings if value > 0) / len(winnings)), 6)
        if winnings else 0.0,
        "events": dict(event_counts),
        "sources": dict(source_counts),
        "policy_decisions": int(source_counts.get("policy", 0)),
        "solver_decisions": int(source_counts.get("solver", 0)),
        "action_mix": dict(action_mix),
        "risk_hands": _hand_bucket(risk_winnings),
        "no_risk_hands": _hand_bucket(no_risk_winnings),
        "low_margin_risk_hands": _hand_bucket(low_margin_risk_winnings),
        "allin_hands": _hand_bucket(allin_winnings),
        "big_call_hands": _hand_bucket(big_call_winnings),
        "max_bet_hands": _hand_bucket(max_bet_winnings),
        "mean_margin": round(float(sum(margin_values) / len(margin_values)), 6)
        if margin_values else None,
        "mean_risk_margin": round(float(sum(risk_margin_values) / len(risk_margin_values)), 6)
        if risk_margin_values else None,
        "risk_bet_threshold": risk_bet_threshold,
        "risk_call_threshold": risk_call_threshold,
        "low_margin_threshold": low_margin_threshold,
        "worst_hands": worst_hands,
    }
