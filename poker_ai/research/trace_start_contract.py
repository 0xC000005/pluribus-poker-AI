"""Observation contract for Slumbot trace-derived hard states.

This module deliberately does not reconstruct a playable poker environment.
It preserves the exact Slumbot feature and legality surface used by live play,
plus explicit provenance context for future trace-start training experiments.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import build_features, get_legal_mask_from_parsed, parse_action  # noqa: E402
from range_tracker import map_slumbot_action_to_idx, walk_actions  # noqa: E402


@dataclass(frozen=True)
class TraceStartObservation:
    """Trace-derived observation, legal mask, and provenance for hard-state training."""

    label: str
    source: str
    contract_kind: str
    street: int
    client_pos: int
    action_str: str
    hole_cards: tuple[str, str]
    board: tuple[str, ...]
    features: np.ndarray
    legal_mask: np.ndarray
    source_flag: float

    @property
    def features_with_context(self) -> np.ndarray:
        """Return base Slumbot features with one explicit source/provenance scalar."""
        return np.concatenate(
            [
                np.asarray(self.features, dtype=np.float32),
                np.asarray([self.source_flag], dtype=np.float32),
            ]
        )


def _has_valid_slumbot_action_tokens(action_str: str) -> bool:
    idx = 0
    while idx < len(action_str):
        char = action_str[idx]
        if char in {"k", "c", "f", "/"}:
            idx += 1
            continue
        if char != "b":
            return False
        idx += 1
        start = idx
        while idx < len(action_str) and action_str[idx].isdigit():
            idx += 1
        if idx == start:
            return False
    return True


def build_trace_start_observation(
    case: ResolverBenchmarkCase,
    *,
    source_flag: float = 1.0,
) -> TraceStartObservation:
    """Build a parity-checked observation contract from one trace resolver case."""
    if not _has_valid_slumbot_action_tokens(case.action_str):
        raise ValueError(f"invalid Slumbot action for {case.label}: malformed token")
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        raise ValueError(f"invalid Slumbot action for {case.label}: {parsed['error']}")
    street = int(parsed.get("st", -1))
    client_pos = int(case.client_pos)
    if int(parsed.get("pos", -2)) != client_pos:
        raise ValueError(
            f"case {case.label} acts at client_pos={client_pos}, "
            f"but parsed action expects pos={parsed.get('pos')}"
        )

    required_board = 4 if street == 2 else 5 if street == 3 else 0
    if required_board and len(case.board) < required_board:
        raise ValueError(
            f"case {case.label} needs at least {required_board} board cards for street {street}"
        )
    board = tuple(case.board[:required_board] if required_board else case.board)
    features = build_features(
        list(case.hole_cards),
        list(board),
        case.action_str,
        client_pos,
        parsed,
    )
    legal_mask = get_legal_mask_from_parsed(parsed, case.action_str, client_pos)
    if not np.any(legal_mask > 0):
        raise ValueError(f"case {case.label} has no legal actions")

    return TraceStartObservation(
        label=str(case.label),
        source=str(case.source),
        contract_kind="observation_only",
        street=street,
        client_pos=client_pos,
        action_str=str(case.action_str),
        hole_cards=tuple(case.hole_cards),
        board=tuple(board),
        features=np.asarray(features, dtype=np.float32),
        legal_mask=np.asarray(legal_mask, dtype=np.float32),
        source_flag=float(source_flag),
    )


def load_trace_start_observations(
    cases_path: str | Path,
    *,
    source_flag: float = 1.0,
) -> list[TraceStartObservation]:
    """Load trace-start observations from a resolver-case JSON artifact."""
    return [
        build_trace_start_observation(case, source_flag=source_flag)
        for case in load_cases_json(cases_path)
    ]


def summarize_trace_start_observations(
    observations: list[TraceStartObservation],
    *,
    source_flag: float,
) -> dict[str, Any]:
    """Summarize trace-start contract coverage for autoresearch gates."""
    street_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    legal_action_counts: list[int] = []
    for obs in observations:
        street_counts[str(obs.street)] = street_counts.get(str(obs.street), 0) + 1
        source_counts[obs.source] = source_counts.get(obs.source, 0) + 1
        legal_action_counts.append(int(np.sum(obs.legal_mask > 0)))

    feature_dim = int(observations[0].features.shape[0]) if observations else 0
    feature_context_dim = (
        int(observations[0].features_with_context.shape[0]) if observations else 0
    )
    return {
        "mode": "trace_start_observation_contract",
        "passed": bool(observations),
        "n_observations": len(observations),
        "contract_kind": "observation_only",
        "feature_dim": feature_dim,
        "feature_context_dim": feature_context_dim,
        "source_flag": float(source_flag),
        "street_counts": dict(sorted(street_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "min_legal_actions": int(min(legal_action_counts)) if legal_action_counts else 0,
        "max_legal_actions": int(max(legal_action_counts)) if legal_action_counts else 0,
        "mean_legal_actions": (
            float(np.mean(legal_action_counts)) if legal_action_counts else 0.0
        ),
        "action_mapping": summarize_trace_action_mapping(observations),
        "promotion_blockers": [
            "trace_start_contract_is_not_a_playable_environment"
        ],
    }


def summarize_trace_action_mapping(
    observations: list[TraceStartObservation],
    *,
    hard_weight_threshold: float = 0.95,
) -> dict[str, Any]:
    """Measure how cleanly Slumbot bet histories map into the 9-action space."""
    n_actions = 0
    n_bet_actions = 0
    ambiguous_bet_actions = 0
    top_weights: list[float] = []
    for obs in observations:
        for before, acting_pos, action_char, bet_to in walk_actions(obs.action_str):
            n_actions += 1
            if action_char != "b":
                continue
            n_bet_actions += 1
            parsed_before = parse_action(before)
            if "error" in parsed_before:
                ambiguous_bet_actions += 1
                top_weights.append(0.0)
                continue
            mapping = map_slumbot_action_to_idx(
                action_char,
                bet_to,
                before,
                acting_pos,
                parsed_before,
            )
            top_weight = max((float(weight) for _, weight in mapping), default=0.0)
            top_weights.append(top_weight)
            if top_weight < float(hard_weight_threshold):
                ambiguous_bet_actions += 1
    hard_mappable = n_bet_actions - ambiguous_bet_actions
    return {
        "n_actions": int(n_actions),
        "n_bet_actions": int(n_bet_actions),
        "ambiguous_bet_actions": int(ambiguous_bet_actions),
        "hard_mappable_bet_actions": int(hard_mappable),
        "hard_mappable_bet_rate": (
            float(hard_mappable / n_bet_actions) if n_bet_actions else 1.0
        ),
        "mean_top_mapping_weight": float(np.mean(top_weights)) if top_weights else 1.0,
        "min_top_mapping_weight": float(min(top_weights)) if top_weights else 1.0,
    }
