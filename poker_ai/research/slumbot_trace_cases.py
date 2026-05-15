"""Build fixed resolver cases from Slumbot decision traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import parse_action  # noqa: E402


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _valid_cards(cards: list[str]) -> bool:
    return len(cards) == len(set(cards)) and all(isinstance(card, str) for card in cards)


def _case_from_record(record: dict[str, Any], *, decision_index: int) -> ResolverBenchmarkCase | None:
    if record.get("event") != "decision" or record.get("source") not in {"policy", "solver"}:
        return None
    try:
        street = int(record.get("street_index"))
        client_pos = int(record.get("client_pos"))
    except (TypeError, ValueError):
        return None
    if street not in (2, 3):
        return None

    hole_cards = list(record.get("hole_cards") or [])
    board = list(record.get("board") or [])
    required_board = 4 if street == 2 else 5
    if len(hole_cards) != 2 or len(board) < required_board:
        return None
    board = board[:required_board]
    if not _valid_cards([*hole_cards, *board]):
        return None

    action_str = str(record.get("full_action_str") or record.get("action_str") or "")
    parsed = parse_action(action_str)
    if "error" in parsed:
        return None
    if int(parsed.get("st", -1)) != street or int(parsed.get("pos", -1)) != client_pos:
        return None

    hand_index = record.get("hand_index")
    hand_label = f"hand{int(hand_index)}" if isinstance(hand_index, int) else "handNA"
    return ResolverBenchmarkCase(
        label=f"trace-{hand_label}-decision{decision_index}-street{street}",
        hole_cards=(str(hole_cards[0]), str(hole_cards[1])),
        board=tuple(str(card) for card in board),
        action_str=action_str,
        client_pos=client_pos,
        source="slumbot_trace",
    )


def extract_resolver_cases_from_trace(
    trace_path: str | Path,
    *,
    limit: int | None = None,
) -> list[ResolverBenchmarkCase]:
    """Extract valid turn/river policy-decision cases from a Slumbot trace."""
    cases: list[ResolverBenchmarkCase] = []
    decision_index = 0
    for record in _read_jsonl(trace_path):
        case = _case_from_record(record, decision_index=decision_index + 1)
        if case is None:
            continue
        decision_index += 1
        cases.append(case)
        if limit is not None and len(cases) >= int(limit):
            break
    return cases
