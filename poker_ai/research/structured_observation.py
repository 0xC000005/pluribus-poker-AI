"""Structured observation features for native self-play RL pilots.

The legacy full-deck feature vector includes aggregate per-street action counts.
This module keeps the same raw card one-hots and continuous game scalars, but
represents betting history as an ordered sequence of action tokens plus amounts.
It is intentionally an opt-in research feature surface, not a replacement for
the Deep CFR parity contract.
"""

from __future__ import annotations

import numpy as np

from poker_ai.games.full_deck.state import ACTION_TO_INDEX, N_ACTIONS, card_to_index


RAW_SEQUENCE_MAX_ACTIONS = 32
RAW_SEQUENCE_SCALAR_FEATURES = 8
RAW_SEQUENCE_BASE_FEATURES = 52 + 52 + 4 + RAW_SEQUENCE_SCALAR_FEATURES

RAW_SEQUENCE_ACTION_PLAYER_OFFSET = 0
RAW_SEQUENCE_ACTION_STREET_OFFSET = 1
RAW_SEQUENCE_ACTION_TOKEN_OFFSET = 5
RAW_SEQUENCE_ACTION_AMOUNT_OFFSET = RAW_SEQUENCE_ACTION_TOKEN_OFFSET + N_ACTIONS
RAW_SEQUENCE_ACTION_TO_CALL_OFFSET = RAW_SEQUENCE_ACTION_AMOUNT_OFFSET + 1
RAW_SEQUENCE_ACTION_POT_OFFSET = RAW_SEQUENCE_ACTION_TO_CALL_OFFSET + 1
RAW_SEQUENCE_ACTION_BET_TO_OFFSET = RAW_SEQUENCE_ACTION_POT_OFFSET + 1
RAW_SEQUENCE_ACTION_STACK_OFFSET = RAW_SEQUENCE_ACTION_BET_TO_OFFSET + 1
RAW_SEQUENCE_ACTION_RECORD_WIDTH = RAW_SEQUENCE_ACTION_STACK_OFFSET + 1

RAW_SEQUENCE_ACTION_OFFSET = RAW_SEQUENCE_BASE_FEATURES
RAW_SEQUENCE_N_FEATURES = (
    RAW_SEQUENCE_ACTION_OFFSET
    + RAW_SEQUENCE_MAX_ACTIONS * RAW_SEQUENCE_ACTION_RECORD_WIDTH
)


def encode_raw_sequence_observation(state, *, max_actions: int = RAW_SEQUENCE_MAX_ACTIONS) -> np.ndarray:
    """Encode an observation with ordered betting history instead of count features."""
    if int(max_actions) != RAW_SEQUENCE_MAX_ACTIONS:
        raise ValueError(
            f"max_actions must be {RAW_SEQUENCE_MAX_ACTIONS}; variable lengths would break checkpoint loading"
        )

    features = np.zeros(RAW_SEQUENCE_N_FEATURES, dtype=np.float32)
    for card in state.current_player.cards:
        features[card_to_index(card)] = 1.0
    for card in state.community_cards:
        features[52 + card_to_index(card)] = 1.0

    round_idx = int(getattr(state, "betting_round", 4))
    if 0 <= round_idx < 4:
        features[104 + round_idx] = 1.0

    n_players = max(len(state.players), 1)
    initial_chips = max(float(getattr(state, "_initial_n_chips", 1)), 1.0)
    total_chips = initial_chips * float(n_players)
    biggest_bet = max((float(p.n_bet_chips) for p in state.players), default=0.0)
    to_call = biggest_bet - float(state.current_player.n_bet_chips)
    other_stacks = [
        float(player.n_chips)
        for idx, player in enumerate(state.players)
        if int(idx) != int(state.player_i)
    ]
    scalar_offset = 108
    features[scalar_offset] = float(state._table.pot.total) / total_chips
    features[scalar_offset + 1] = float(state.current_player.n_chips) / initial_chips
    features[scalar_offset + 2] = float(state.current_player.n_bet_chips) / initial_chips
    features[scalar_offset + 3] = biggest_bet / initial_chips
    features[scalar_offset + 4] = max(to_call, 0.0) / initial_chips
    features[scalar_offset + 5] = sum(1 for p in state.players if p.is_active) / float(n_players)
    features[scalar_offset + 6] = float(state.player_i) / float(max(n_players - 1, 1))
    features[scalar_offset + 7] = (
        float(np.mean(other_stacks)) / initial_chips if other_stacks else 0.0
    )

    records = list(getattr(state, "_action_records", []))[-RAW_SEQUENCE_MAX_ACTIONS:]
    for slot_idx, record in enumerate(records):
        offset = RAW_SEQUENCE_ACTION_OFFSET + slot_idx * RAW_SEQUENCE_ACTION_RECORD_WIDTH
        player_i = int(record.get("player_i", 0))
        features[offset + RAW_SEQUENCE_ACTION_PLAYER_OFFSET] = float(player_i) / float(max(n_players - 1, 1))
        street_idx = int(record.get("round_idx", 0))
        if 0 <= street_idx < 4:
            features[offset + RAW_SEQUENCE_ACTION_STREET_OFFSET + street_idx] = 1.0
        action_idx = record.get("action_index")
        if action_idx is None:
            action_idx = ACTION_TO_INDEX.get(str(record.get("action", "")), -1)
        action_idx = int(action_idx)
        if 0 <= action_idx < N_ACTIONS:
            features[offset + RAW_SEQUENCE_ACTION_TOKEN_OFFSET + action_idx] = 1.0
        features[offset + RAW_SEQUENCE_ACTION_AMOUNT_OFFSET] = (
            float(record.get("amount_added", 0.0)) / initial_chips
        )
        features[offset + RAW_SEQUENCE_ACTION_TO_CALL_OFFSET] = (
            float(record.get("to_call", 0.0)) / initial_chips
        )
        features[offset + RAW_SEQUENCE_ACTION_POT_OFFSET] = (
            float(record.get("pot_before", 0.0)) / total_chips
        )
        features[offset + RAW_SEQUENCE_ACTION_BET_TO_OFFSET] = (
            float(record.get("player_bet_after", 0.0)) / initial_chips
        )
        features[offset + RAW_SEQUENCE_ACTION_STACK_OFFSET] = (
            float(record.get("stack_before", 0.0)) / initial_chips
        )
    return features
