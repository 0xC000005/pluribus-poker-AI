"""Compiled batch transition primitives for the native fast poker state.

This module is deliberately narrower than a training backend. It moves the
state-update hot path into array-oriented Numba kernels and exposes parity and
throughput evidence before the learner is allowed to consume it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from poker_ai.deep_cfr.fast_state import FastPokerState, N_ACTIONS, N_FEATURES, RAISE_FRACTIONS, new_fast_game


_PREFLOP = FastPokerState.PREFLOP
_FLOP = FastPokerState.FLOP
_TURN = FastPokerState.TURN
_RIVER = FastPokerState.RIVER
_SHOWDOWN = FastPokerState.SHOWDOWN
_TERMINAL = FastPokerState.TERMINAL
_RAISE_FRACTIONS = np.array(RAISE_FRACTIONS, dtype=np.float32)


@dataclass
class CompiledFastStateBatch:
    """Array representation of a batch of 2-player ``FastPokerState`` objects."""

    chips: np.ndarray
    bets: np.ndarray
    active: np.ndarray
    hole_cards: np.ndarray
    community: np.ndarray
    deck_order: np.ndarray
    deck_cursor: np.ndarray
    stage: np.ndarray
    n_raises: np.ndarray
    player_i_index: np.ndarray
    n_actions: np.ndarray
    n_players_started_round: np.ndarray
    pot_total: np.ndarray
    history: np.ndarray
    small_blind: int
    big_blind: int
    initial_chips: int
    needs_python_showdown: np.ndarray

    @classmethod
    def from_fast_states(cls, states: list[FastPokerState]) -> "CompiledFastStateBatch":
        if not states:
            raise ValueError("states must be non-empty")
        n = len(states)
        first = states[0]
        for state in states:
            if int(state.n_players) != 2:
                raise ValueError("compiled fast batch currently supports heads-up states only")
            if int(state.small_blind) != int(first.small_blind):
                raise ValueError("all states must share small_blind")
            if int(state.big_blind) != int(first.big_blind):
                raise ValueError("all states must share big_blind")
            if int(state.initial_chips) != int(first.initial_chips):
                raise ValueError("all states must share initial_chips")
        return cls(
            chips=np.stack([s.chips for s in states]).astype(np.int32, copy=True),
            bets=np.stack([s.bets for s in states]).astype(np.int32, copy=True),
            active=np.stack([s.active for s in states]).astype(np.bool_, copy=True),
            hole_cards=np.stack([s.hole_cards for s in states]).astype(np.int8, copy=True),
            community=np.stack([s.community for s in states]).astype(np.int8, copy=True),
            deck_order=np.stack([s.deck_order for s in states]).astype(np.int8, copy=True),
            deck_cursor=np.array([int(s.deck_cursor) for s in states], dtype=np.int16),
            stage=np.array([int(s.stage) for s in states], dtype=np.int8),
            n_raises=np.array([int(s.n_raises) for s in states], dtype=np.int8),
            player_i_index=np.array([int(s._player_i_index) for s in states], dtype=np.int8),
            n_actions=np.array([int(s.n_actions) for s in states], dtype=np.int16),
            n_players_started_round=np.array(
                [int(s.n_players_started_round) for s in states],
                dtype=np.int8,
            ),
            pot_total=np.array([int(s.pot_total) for s in states], dtype=np.int32),
            history=np.stack([s.history for s in states]).astype(np.int16, copy=True),
            small_blind=int(first.small_blind),
            big_blind=int(first.big_blind),
            initial_chips=int(first.initial_chips),
            needs_python_showdown=np.zeros(n, dtype=np.bool_),
        )

    def current_players(self) -> np.ndarray:
        return _compiled_current_players(self.stage, self.player_i_index)


@njit(cache=True)
def _current_player_hu(stage: int, player_i_index: int) -> int:
    idx = int(player_i_index) % 2
    if int(stage) == _PREFLOP:
        return idx
    return 1 - idx


@njit(cache=True)
def _n_active_players(active: np.ndarray, row: int) -> int:
    total = 0
    for player_i in range(2):
        if active[row, player_i]:
            total += 1
    return total


@njit(cache=True)
def _n_players_with_moves(active: np.ndarray, chips: np.ndarray, row: int) -> int:
    total = 0
    for player_i in range(2):
        if active[row, player_i] and chips[row, player_i] > 0:
            total += 1
    return total


@njit(cache=True)
def _is_betting_finished(active: np.ndarray, chips: np.ndarray, bets: np.ndarray, row: int) -> bool:
    active_bets_count = 0
    biggest = 0
    for player_i in range(2):
        if active[row, player_i]:
            active_bets_count += 1
            if bets[row, player_i] > biggest:
                biggest = int(bets[row, player_i])
    if active_bets_count <= 1:
        return True
    for player_i in range(2):
        if active[row, player_i] and chips[row, player_i] > 0 and bets[row, player_i] < biggest:
            return False
    return True


@njit(cache=True)
def _min_raise_contribution(to_call: int, big_blind: int) -> int:
    if to_call <= 0:
        return int(big_blind)
    if to_call > big_blind:
        return int(to_call + to_call)
    return int(to_call + big_blind)


@njit(cache=True)
def _deal_community(
    community: np.ndarray,
    deck_order: np.ndarray,
    deck_cursor: np.ndarray,
    row: int,
    n_cards: int,
) -> None:
    for _ in range(n_cards):
        slot = 0
        while slot < 5 and community[row, slot] >= 0:
            slot += 1
        if slot >= 5:
            return
        community[row, slot] = deck_order[row, deck_cursor[row]]
        deck_cursor[row] += 1


@njit(cache=True)
def _increment_stage(
    stage: np.ndarray,
    community: np.ndarray,
    deck_order: np.ndarray,
    deck_cursor: np.ndarray,
    row: int,
) -> None:
    if stage[row] == _PREFLOP:
        stage[row] = _FLOP
        _deal_community(community, deck_order, deck_cursor, row, 3)
    elif stage[row] == _FLOP:
        stage[row] = _TURN
        _deal_community(community, deck_order, deck_cursor, row, 1)
    elif stage[row] == _TURN:
        stage[row] = _RIVER
        _deal_community(community, deck_order, deck_cursor, row, 1)
    elif stage[row] == _RIVER:
        stage[row] = _SHOWDOWN


@njit(cache=True)
def _deal_remaining_to_showdown(
    stage: np.ndarray,
    community: np.ndarray,
    deck_order: np.ndarray,
    deck_cursor: np.ndarray,
    row: int,
) -> None:
    while stage[row] < _SHOWDOWN:
        _increment_stage(stage, community, deck_order, deck_cursor, row)


@njit(cache=True)
def _skip_to_first_active(
    chips: np.ndarray,
    active: np.ndarray,
    stage: np.ndarray,
    player_i_index: np.ndarray,
    row: int,
) -> None:
    for _ in range(2):
        player_i = _current_player_hu(stage[row], player_i_index[row])
        if active[row, player_i] and chips[row, player_i] > 0:
            return
        player_i_index[row] = (player_i_index[row] + 1) % 2


@njit(cache=True)
def _reset_round(
    chips: np.ndarray,
    bets: np.ndarray,
    active: np.ndarray,
    stage: np.ndarray,
    n_raises: np.ndarray,
    player_i_index: np.ndarray,
    n_actions: np.ndarray,
    n_players_started_round: np.ndarray,
    row: int,
) -> None:
    n_actions[row] = 0
    n_raises[row] = 0
    player_i_index[row] = 0
    n_players_started_round[row] = _n_players_with_moves(active, chips, row)
    if n_players_started_round[row] > 0:
        _skip_to_first_active(chips, active, stage, player_i_index, row)


@njit(cache=True)
def _settle_fold_terminal(
    chips: np.ndarray,
    bets: np.ndarray,
    active: np.ndarray,
    pot_total: np.ndarray,
    row: int,
) -> None:
    winner = 0
    for player_i in range(2):
        if active[row, player_i]:
            winner = player_i
            break
    chips[row, winner] += pot_total[row]
    bets[row, 0] = 0
    bets[row, 1] = 0
    pot_total[row] = 0


@njit(cache=True)
def _score5(category: int, r1: int, r2: int, r3: int, r4: int, r5: int) -> int:
    score = int(category)
    score = score * 15 + int(r1)
    score = score * 15 + int(r2)
    score = score * 15 + int(r3)
    score = score * 15 + int(r4)
    score = score * 15 + int(r5)
    return score


@njit(cache=True)
def _straight_high(present: np.ndarray) -> int:
    for high in range(14, 5, -1):
        ok = True
        for offset in range(5):
            if not present[high - offset]:
                ok = False
                break
        if ok:
            return high
    if present[14] and present[5] and present[4] and present[3] and present[2]:
        return 5
    return 0


@njit(cache=True)
def _hand_strength_7(cards: np.ndarray) -> int:
    rank_counts = np.zeros(15, dtype=np.int16)
    suit_counts = np.zeros(4, dtype=np.int16)
    rank_present = np.zeros(15, dtype=np.bool_)
    suit_rank_present = np.zeros((4, 15), dtype=np.bool_)

    for i in range(cards.shape[0]):
        card = int(cards[i])
        if card < 0:
            continue
        rank = card // 4 + 2
        suit = card % 4
        rank_counts[rank] += 1
        suit_counts[suit] += 1
        rank_present[rank] = True
        suit_rank_present[suit, rank] = True

    best_straight_flush = 0
    flush_suit = -1
    for suit in range(4):
        if suit_counts[suit] >= 5:
            high = _straight_high(suit_rank_present[suit])
            if high > best_straight_flush:
                best_straight_flush = high
            if flush_suit < 0:
                flush_suit = suit
    if best_straight_flush > 0:
        return _score5(8, best_straight_flush, 0, 0, 0, 0)

    quad = 0
    for rank in range(14, 1, -1):
        if rank_counts[rank] == 4:
            quad = rank
            break
    if quad > 0:
        kicker = 0
        for rank in range(14, 1, -1):
            if rank != quad and rank_counts[rank] > 0:
                kicker = rank
                break
        return _score5(7, quad, kicker, 0, 0, 0)

    trip1 = 0
    trip2 = 0
    for rank in range(14, 1, -1):
        if rank_counts[rank] >= 3:
            if trip1 == 0:
                trip1 = rank
            elif trip2 == 0:
                trip2 = rank
    pair = 0
    for rank in range(14, 1, -1):
        if rank != trip1 and rank_counts[rank] >= 2:
            pair = rank
            break
    if pair == 0 and trip2 > 0:
        pair = trip2
    if trip1 > 0 and pair > 0:
        return _score5(6, trip1, pair, 0, 0, 0)

    if flush_suit >= 0:
        top = np.zeros(5, dtype=np.int16)
        count = 0
        for rank in range(14, 1, -1):
            if suit_rank_present[flush_suit, rank]:
                top[count] = rank
                count += 1
                if count == 5:
                    break
        return _score5(5, top[0], top[1], top[2], top[3], top[4])

    straight = _straight_high(rank_present)
    if straight > 0:
        return _score5(4, straight, 0, 0, 0, 0)

    if trip1 > 0:
        kickers = np.zeros(2, dtype=np.int16)
        count = 0
        for rank in range(14, 1, -1):
            if rank != trip1 and rank_counts[rank] > 0:
                kickers[count] = rank
                count += 1
                if count == 2:
                    break
        return _score5(3, trip1, kickers[0], kickers[1], 0, 0)

    pair1 = 0
    pair2 = 0
    for rank in range(14, 1, -1):
        if rank_counts[rank] >= 2:
            if pair1 == 0:
                pair1 = rank
            elif pair2 == 0:
                pair2 = rank
                break
    if pair1 > 0 and pair2 > 0:
        kicker = 0
        for rank in range(14, 1, -1):
            if rank != pair1 and rank != pair2 and rank_counts[rank] > 0:
                kicker = rank
                break
        return _score5(2, pair1, pair2, kicker, 0, 0)

    if pair1 > 0:
        kickers3 = np.zeros(3, dtype=np.int16)
        count = 0
        for rank in range(14, 1, -1):
            if rank != pair1 and rank_counts[rank] > 0:
                kickers3[count] = rank
                count += 1
                if count == 3:
                    break
        return _score5(1, pair1, kickers3[0], kickers3[1], kickers3[2], 0)

    high_cards = np.zeros(5, dtype=np.int16)
    count = 0
    for rank in range(14, 1, -1):
        if rank_counts[rank] > 0:
            high_cards[count] = rank
            count += 1
            if count == 5:
                break
    return _score5(0, high_cards[0], high_cards[1], high_cards[2], high_cards[3], high_cards[4])


@njit(cache=True)
def _showdown_strength(
    hole_cards: np.ndarray,
    community: np.ndarray,
    row: int,
    player_i: int,
) -> int:
    cards = np.empty(7, dtype=np.int16)
    cards[0] = hole_cards[row, player_i, 0]
    cards[1] = hole_cards[row, player_i, 1]
    for i in range(5):
        cards[2 + i] = community[row, i]
    return _hand_strength_7(cards)


@njit(cache=True)
def _settle_showdown_terminal(
    chips: np.ndarray,
    bets: np.ndarray,
    active: np.ndarray,
    hole_cards: np.ndarray,
    community: np.ndarray,
    pot_total: np.ndarray,
    row: int,
) -> None:
    strengths = np.empty(2, dtype=np.int64)
    for player_i in range(2):
        if active[row, player_i]:
            strengths[player_i] = _showdown_strength(hole_cards, community, row, player_i)
        else:
            strengths[player_i] = -1

    remaining0 = int(bets[row, 0])
    remaining1 = int(bets[row, 1])
    while remaining0 + remaining1 > 0:
        min_bet = 0
        if remaining0 > 0 and remaining1 > 0:
            min_bet = remaining0 if remaining0 < remaining1 else remaining1
        elif remaining0 > 0:
            min_bet = remaining0
        else:
            min_bet = remaining1

        contribution0 = 0
        contribution1 = 0
        if remaining0 > 0:
            contribution0 = min_bet if remaining0 >= min_bet else remaining0
            remaining0 -= contribution0
        if remaining1 > 0:
            contribution1 = min_bet if remaining1 >= min_bet else remaining1
            remaining1 -= contribution1

        total = contribution0 + contribution1
        eligible0 = contribution0 > 0 and active[row, 0]
        eligible1 = contribution1 > 0 and active[row, 1]
        if eligible0 and eligible1:
            if strengths[0] > strengths[1]:
                chips[row, 0] += total
            elif strengths[1] > strengths[0]:
                chips[row, 1] += total
            else:
                half = total // 2
                chips[row, 0] += half + (total - half * 2)
                chips[row, 1] += half
        elif eligible0:
            chips[row, 0] += total
        elif eligible1:
            chips[row, 1] += total

    bets[row, 0] = 0
    bets[row, 1] = 0
    pot_total[row] = 0


@njit(cache=True)
def _compiled_current_players(stage: np.ndarray, player_i_index: np.ndarray) -> np.ndarray:
    out = np.empty(stage.shape[0], dtype=np.int8)
    for row in range(stage.shape[0]):
        out[row] = _current_player_hu(stage[row], player_i_index[row])
    return out


@njit(cache=True)
def _compiled_legal_masks(
    chips: np.ndarray,
    bets: np.ndarray,
    active: np.ndarray,
    stage: np.ndarray,
    n_raises: np.ndarray,
    player_i_index: np.ndarray,
    pot_total: np.ndarray,
    big_blind: int,
) -> np.ndarray:
    masks = np.zeros((chips.shape[0], N_ACTIONS), dtype=np.float32)
    for row in range(chips.shape[0]):
        if stage[row] >= _SHOWDOWN:
            continue
        player_i = _current_player_hu(stage[row], player_i_index[row])
        if not active[row, player_i] or chips[row, player_i] <= 0:
            continue
        masks[row, 0] = 1.0
        masks[row, 1] = 1.0
        if n_raises[row] < 3:
            biggest = bets[row, 0]
            if bets[row, 1] > biggest:
                biggest = bets[row, 1]
            to_call = int(biggest - bets[row, player_i])
            player_chips = int(chips[row, player_i])
            min_raise = _min_raise_contribution(to_call, big_blind)
            for raise_i in range(6):
                raise_amount = int(_RAISE_FRACTIONS[raise_i] * pot_total[row]) + to_call
                if raise_amount >= min_raise and raise_amount <= player_chips:
                    masks[row, 2 + raise_i] = 1.0
            if player_chips > 0:
                masks[row, 8] = 1.0
    return masks


@njit(cache=True)
def _compiled_advance_rows(
    chips: np.ndarray,
    bets: np.ndarray,
    active: np.ndarray,
    hole_cards: np.ndarray,
    community: np.ndarray,
    deck_order: np.ndarray,
    deck_cursor: np.ndarray,
    stage: np.ndarray,
    n_raises: np.ndarray,
    player_i_index: np.ndarray,
    n_actions: np.ndarray,
    n_players_started_round: np.ndarray,
    pot_total: np.ndarray,
    needs_python_showdown: np.ndarray,
    row: int,
) -> None:
    while True:
        player_i_index[row] = (player_i_index[row] + 1) % 2

        if _n_active_players(active, row) == 1:
            stage[row] = _TERMINAL
            _settle_fold_terminal(chips, bets, active, pot_total, row)
            break

        betting_done = _is_betting_finished(active, chips, bets, row)
        if betting_done and _n_players_with_moves(active, chips, row) <= 1:
            _deal_remaining_to_showdown(stage, community, deck_order, deck_cursor, row)
            _settle_showdown_terminal(chips, bets, active, hole_cards, community, pot_total, row)
            break

        if betting_done and n_actions[row] >= n_players_started_round[row]:
            _increment_stage(stage, community, deck_order, deck_cursor, row)
            if stage[row] >= _SHOWDOWN:
                _settle_showdown_terminal(chips, bets, active, hole_cards, community, pot_total, row)
                break
            if _n_players_with_moves(active, chips, row) <= 1:
                _deal_remaining_to_showdown(stage, community, deck_order, deck_cursor, row)
                _settle_showdown_terminal(chips, bets, active, hole_cards, community, pot_total, row)
                break
            _reset_round(
                chips,
                bets,
                active,
                stage,
                n_raises,
                player_i_index,
                n_actions,
                n_players_started_round,
                row,
            )

        player_i = _current_player_hu(stage[row], player_i_index[row])
        if not active[row, player_i] or chips[row, player_i] <= 0:
            continue
        if stage[row] >= _SHOWDOWN:
            _settle_showdown_terminal(chips, bets, active, hole_cards, community, pot_total, row)
            break
        break


@njit(cache=True)
def _compiled_apply_actions(
    chips: np.ndarray,
    bets: np.ndarray,
    active: np.ndarray,
    hole_cards: np.ndarray,
    community: np.ndarray,
    deck_order: np.ndarray,
    deck_cursor: np.ndarray,
    stage: np.ndarray,
    n_raises: np.ndarray,
    player_i_index: np.ndarray,
    n_actions: np.ndarray,
    n_players_started_round: np.ndarray,
    pot_total: np.ndarray,
    history: np.ndarray,
    needs_python_showdown: np.ndarray,
    actions: np.ndarray,
    big_blind: int,
) -> tuple[int, int]:
    applied = 0
    fallback = 0
    for row in range(actions.shape[0]):
        action = int(actions[row])
        if action < 0 or stage[row] >= _SHOWDOWN:
            if needs_python_showdown[row]:
                fallback += 1
            continue
        player_i = _current_player_hu(stage[row], player_i_index[row])
        if action == 0:
            active[row, player_i] = False
        elif action == 1:
            if chips[row, player_i] > 0:
                biggest = bets[row, 0]
                if bets[row, 1] > biggest:
                    biggest = bets[row, 1]
                to_call = int(biggest - bets[row, player_i])
                if to_call > chips[row, player_i]:
                    to_call = int(chips[row, player_i])
                chips[row, player_i] -= to_call
                bets[row, player_i] += to_call
                pot_total[row] += to_call
        elif 2 <= action <= 7:
            frac = _RAISE_FRACTIONS[action - 2]
            biggest = bets[row, 0]
            if bets[row, 1] > biggest:
                biggest = bets[row, 1]
            to_call = int(biggest - bets[row, player_i])
            raise_chips = int(frac * pot_total[row]) + to_call
            min_raise = _min_raise_contribution(to_call, big_blind)
            if raise_chips < min_raise:
                raise_chips = min_raise
            if raise_chips > chips[row, player_i]:
                raise_chips = int(chips[row, player_i])
            chips[row, player_i] -= raise_chips
            bets[row, player_i] += raise_chips
            pot_total[row] += raise_chips
            n_raises[row] += 1
        elif action == 8:
            all_in_chips = int(chips[row, player_i])
            chips[row, player_i] = 0
            bets[row, player_i] += all_in_chips
            pot_total[row] += all_in_chips
            n_raises[row] += 1

        round_i = int(stage[row])
        if round_i > 3:
            round_i = 3
        if action == 1:
            history[row, round_i, 0] += 1
        elif action >= 2:
            history[row, round_i, 1] += 1
        elif action == 0:
            history[row, round_i, 2] += 1

        n_actions[row] += 1
        applied += 1
        _compiled_advance_rows(
            chips,
            bets,
            active,
            hole_cards,
            community,
            deck_order,
            deck_cursor,
            stage,
            n_raises,
            player_i_index,
            n_actions,
            n_players_started_round,
            pot_total,
            needs_python_showdown,
            row,
        )
        if needs_python_showdown[row]:
            fallback += 1
    return applied, fallback


def compiled_legal_masks(batch: CompiledFastStateBatch) -> np.ndarray:
    """Return legal action masks for every row in a compiled batch."""
    return _compiled_legal_masks(
        batch.chips,
        batch.bets,
        batch.active,
        batch.stage,
        batch.n_raises,
        batch.player_i_index,
        batch.pot_total,
        int(batch.big_blind),
    )


@njit(cache=True)
def _compiled_feature_vectors(
    chips: np.ndarray,
    bets: np.ndarray,
    active: np.ndarray,
    hole_cards: np.ndarray,
    community: np.ndarray,
    stage: np.ndarray,
    n_raises: np.ndarray,
    player_i_index: np.ndarray,
    pot_total: np.ndarray,
    history: np.ndarray,
    initial_chips: int,
) -> np.ndarray:
    features = np.zeros((chips.shape[0], N_FEATURES), dtype=np.float32)
    total_chips = float(initial_chips * 2)
    for row in range(chips.shape[0]):
        player_i = _current_player_hu(stage[row], player_i_index[row])
        for card_i in range(2):
            card = int(hole_cards[row, player_i, card_i])
            if 0 <= card < 52:
                features[row, card] = 1.0
        for card_i in range(5):
            card = int(community[row, card_i])
            if 0 <= card < 52:
                features[row, 52 + card] = 1.0

        round_idx = int(stage[row])
        if stage[row] == _TERMINAL:
            round_idx = 3
        elif round_idx > 3:
            round_idx = 3
        if stage[row] < 4 or stage[row] == _TERMINAL:
            features[row, 104 + round_idx] = 1.0

        features[row, 108] = float(pot_total[row]) / total_chips
        features[row, 109] = float(chips[row, player_i]) / float(initial_chips)
        features[row, 110] = float(bets[row, player_i]) / float(initial_chips)
        active_count = 0
        for p in range(2):
            if active[row, p]:
                active_count += 1
        features[row, 111] = float(active_count) / 2.0
        features[row, 112] = float(player_i)
        features[row, 113] = float(n_raises[row]) / 3.0

        for round_i in range(4):
            offset = 114 + round_i * 3
            features[row, offset] = float(history[row, round_i, 0]) / 2.0
            features[row, offset + 1] = float(history[row, round_i, 1]) / 3.0
            features[row, offset + 2] = float(history[row, round_i, 2]) / 2.0
    return features


def compiled_feature_vectors(batch: CompiledFastStateBatch) -> np.ndarray:
    """Return 126-feature observations for every row in a compiled batch."""
    return _compiled_feature_vectors(
        batch.chips,
        batch.bets,
        batch.active,
        batch.hole_cards,
        batch.community,
        batch.stage,
        batch.n_raises,
        batch.player_i_index,
        batch.pot_total,
        batch.history,
        int(batch.initial_chips),
    )


def compiled_apply_actions(batch: CompiledFastStateBatch, actions: np.ndarray) -> dict[str, int]:
    """Apply one action per row, returning explicit fallback counters."""
    action_array = np.asarray(actions, dtype=np.int16)
    if action_array.shape != (batch.chips.shape[0],):
        raise ValueError(f"actions must have shape {(batch.chips.shape[0],)}, got {action_array.shape}")
    applied, fallback = _compiled_apply_actions(
        batch.chips,
        batch.bets,
        batch.active,
        batch.hole_cards,
        batch.community,
        batch.deck_order,
        batch.deck_cursor,
        batch.stage,
        batch.n_raises,
        batch.player_i_index,
        batch.n_actions,
        batch.n_players_started_round,
        batch.pot_total,
        batch.history,
        batch.needs_python_showdown,
        action_array,
        int(batch.big_blind),
    )
    return {
        "applied": int(applied),
        "needs_python_showdown": int(fallback),
    }


def _new_seeded_fast_state(seed: int, game_i: int, initial_chips: int) -> FastPokerState:
    np.random.seed(int(seed) + int(game_i))
    return new_fast_game(2, initial_chips=int(initial_chips))


def _choose_benchmark_actions(masks: np.ndarray) -> np.ndarray:
    actions = np.full(masks.shape[0], -1, dtype=np.int16)
    for row_i in range(masks.shape[0]):
        if masks[row_i, 1] > 0:
            actions[row_i] = 1
        else:
            legal = np.flatnonzero(masks[row_i] > 0)
            if legal.size > 0:
                actions[row_i] = int(legal[0])
    return actions


def _compiled_transition_parity(
    *,
    n_games: int,
    max_steps_per_game: int,
    initial_chips: int,
    seed: int,
    max_mismatches: int = 10,
) -> dict[str, Any]:
    states = [_new_seeded_fast_state(seed, game_i, initial_chips) for game_i in range(n_games)]
    batch = CompiledFastStateBatch.from_fast_states(states)
    mismatches: list[str] = []
    checked_steps = 0
    for step_i in range(int(max_steps_per_game)):
        masks = compiled_legal_masks(batch)
        actions = np.full(int(n_games), -1, dtype=np.int16)
        for row_i, state in enumerate(states):
            if state.is_terminal:
                continue
            fast_mask = state.get_legal_mask()
            if not np.array_equal(masks[row_i] > 0, fast_mask > 0):
                mismatches.append(f"step={step_i} row={row_i}: legal mask mismatch")
                continue
            legal = np.flatnonzero(masks[row_i] > 0)
            action = 1 if masks[row_i, 1] > 0 else int(legal[0])
            actions[row_i] = action
            state.apply_action(action)
        result = compiled_apply_actions(batch, actions)
        for row_i, state in enumerate(states):
            if bool(batch.needs_python_showdown[row_i]):
                continue
            if not np.array_equal(batch.chips[row_i], state.chips):
                mismatches.append(f"step={step_i} row={row_i}: chips mismatch")
            if not np.array_equal(batch.bets[row_i], state.bets):
                mismatches.append(f"step={step_i} row={row_i}: bets mismatch")
            if not np.array_equal(batch.active[row_i], state.active):
                mismatches.append(f"step={step_i} row={row_i}: active mismatch")
            if not np.array_equal(batch.community[row_i], state.community):
                mismatches.append(f"step={step_i} row={row_i}: community mismatch")
            if int(batch.stage[row_i]) != int(state.stage):
                mismatches.append(f"step={step_i} row={row_i}: stage mismatch")
            if int(batch.player_i_index[row_i]) != int(state._player_i_index):
                mismatches.append(f"step={step_i} row={row_i}: player index mismatch")
            if int(batch.pot_total[row_i]) != int(state.pot_total):
                mismatches.append(f"step={step_i} row={row_i}: pot mismatch")
            if int(batch.deck_cursor[row_i]) != int(state.deck_cursor):
                mismatches.append(f"step={step_i} row={row_i}: deck cursor mismatch")
        checked_steps += int(result["applied"])
        if mismatches or result["needs_python_showdown"] > 0:
            break
        if len(mismatches) >= int(max_mismatches):
            break
    return {
        "backend": "compiled-fast-state",
        "reference_backend": "fast-state",
        "n_games": int(n_games),
        "max_steps_per_game": int(max_steps_per_game),
        "checked_steps": int(checked_steps),
        "mismatches": mismatches[: int(max_mismatches)],
        "needs_python_showdown": int(np.sum(batch.needs_python_showdown)),
        "passed": not mismatches and checked_steps > 0,
    }


def _python_transition_benchmark(
    *,
    n_games: int,
    max_steps_per_game: int,
    initial_chips: int,
    seed: int,
) -> dict[str, Any]:
    states = [_new_seeded_fast_state(seed, game_i, initial_chips) for game_i in range(n_games)]
    steps = 0
    started = time.perf_counter()
    for _ in range(int(max_steps_per_game)):
        for state in states:
            if state.is_terminal:
                continue
            mask = state.get_legal_mask()
            action = 1 if mask[1] > 0 else int(np.flatnonzero(mask > 0)[0])
            state.apply_action(action)
            steps += 1
    seconds = time.perf_counter() - started
    return {
        "backend": "fast-state-python-loop",
        "steps": int(steps),
        "seconds": float(seconds),
        "steps_per_second": float(steps / max(seconds, 1e-12)),
    }


def _compiled_transition_benchmark(
    *,
    n_games: int,
    max_steps_per_game: int,
    initial_chips: int,
    seed: int,
) -> dict[str, Any]:
    states = [_new_seeded_fast_state(seed, game_i, initial_chips) for game_i in range(n_games)]
    batch = CompiledFastStateBatch.from_fast_states(states)
    steps = 0
    started = time.perf_counter()
    for _ in range(int(max_steps_per_game)):
        masks = compiled_legal_masks(batch)
        actions = _choose_benchmark_actions(masks)
        result = compiled_apply_actions(batch, actions)
        steps += int(result["applied"])
        if result["needs_python_showdown"] > 0:
            break
    seconds = time.perf_counter() - started
    return {
        "backend": "compiled-fast-state",
        "steps": int(steps),
        "seconds": float(seconds),
        "steps_per_second": float(steps / max(seconds, 1e-12)),
        "needs_python_showdown": int(np.sum(batch.needs_python_showdown)),
    }


def run_compiled_fast_transition_benchmark(
    *,
    n_games: int = 1024,
    max_steps_per_game: int = 16,
    initial_chips: int = 1000,
    seed: int = 20260832,
    min_speedup: float = 5.0,
) -> dict[str, Any]:
    """Benchmark compiled batch transitions against the Python fast-state loop."""
    parity = _compiled_transition_parity(
        n_games=max(1, min(int(n_games), 64)),
        max_steps_per_game=int(max_steps_per_game),
        initial_chips=int(initial_chips),
        seed=int(seed),
    )
    baseline = _python_transition_benchmark(
        n_games=int(n_games),
        max_steps_per_game=int(max_steps_per_game),
        initial_chips=int(initial_chips),
        seed=int(seed) + 1,
    )
    candidate = _compiled_transition_benchmark(
        n_games=int(n_games),
        max_steps_per_game=int(max_steps_per_game),
        initial_chips=int(initial_chips),
        seed=int(seed) + 1,
    )
    speedup = float(candidate["steps_per_second"] / max(float(baseline["steps_per_second"]), 1e-12))
    pre_showdown_passed = bool(parity["passed"] and speedup >= float(min_speedup))
    terminal_payout_supported = True
    fallback_free = (
        int(parity.get("needs_python_showdown", 0)) == 0
        and int(candidate.get("needs_python_showdown", 0)) == 0
    )
    return {
        "backend": "compiled-fast-state",
        "mode": "compiled_transition_gate",
        "parity": parity,
        "baseline": baseline,
        "candidate": candidate,
        "speedup": speedup,
        "min_speedup": float(min_speedup),
        "passed": pre_showdown_passed,
        "pre_showdown_transition_passed": pre_showdown_passed,
        "fallback_free": bool(fallback_free),
        "terminal_payout_supported": terminal_payout_supported,
        "training_integration_allowed": bool(
            pre_showdown_passed
            and speedup >= 5.0
            and fallback_free
            and terminal_payout_supported
        ),
        "warning": (
            "Passing this gate proves a compiled heads-up transition and "
            "showdown-payout substrate only; it is not poker-strength evidence."
        ),
    }
