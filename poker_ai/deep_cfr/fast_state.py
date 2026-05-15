"""Lightweight numpy-array poker state for fast Deep CFR traversal.

Replaces the Python-object-based PokerState with pure numpy arrays.
State copy drops from ~2ms (deepcopy) to ~100ns (array copy).
All game logic is self-contained — no dependency on PokerEngine/Table/Pot objects.

Card index convention: card_idx = (rank - 2) * 4 + suit_idx
  rank: 2-14 (2=deuce, ..., 14=ace)
  suit_idx: 0=clubs, 1=diamonds, 2=hearts, 3=spades
"""
from __future__ import annotations

import numpy as np

from poker_ai.poker.evaluation.eval_card import EvaluationCard
from poker_ai.poker.evaluation.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Module-level constants and singletons
# ---------------------------------------------------------------------------

# Build card index -> EvaluationCard (32-bit int) lookup once at import.
_SUIT_CHARS = ["c", "d", "h", "s"]
_RANK_CHARS = list(EvaluationCard.STR_RANKS)  # "23456789TJQKA"

CARD_INDEX_TO_EVAL_CARD = np.zeros(52, dtype=np.int32)
for _ci in range(52):
    _r = _ci // 4
    _s = _ci % 4
    CARD_INDEX_TO_EVAL_CARD[_ci] = EvaluationCard.new(
        f"{_RANK_CHARS[_r]}{_SUIT_CHARS[_s]}"
    )

# Singleton evaluator (LookupTable built once).
_EVALUATOR = Evaluator()

# Feature vector size (must match N_FEATURES in state.py).
N_FEATURES = 126
N_ACTIONS = 9
RAISE_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def _min_raise_contribution(to_call: int, big_blind: int) -> int:
    """Minimum chips the acting player must add for a legal raise."""
    if to_call <= 0:
        return big_blind
    return to_call + max(to_call, big_blind)

# ---------------------------------------------------------------------------
# Precomputed player orders (shared by reference, never copied)
# ---------------------------------------------------------------------------

def _make_player_orders(n_players: int):
    """Return (preflop_order, postflop_order) arrays."""
    order = list(range(n_players))
    preflop = order[2:] + order[:2]
    # In heads-up, SB=Button acts last postflop (BB acts first).
    if n_players == 2:
        postflop = [1, 0]
    else:
        postflop = order
    return (
        np.array(preflop, dtype=np.int8),
        np.array(postflop, dtype=np.int8),
    )


# Cache orders for common player counts.
_ORDER_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _get_orders(n_players: int):
    if n_players not in _ORDER_CACHE:
        _ORDER_CACHE[n_players] = _make_player_orders(n_players)
    return _ORDER_CACHE[n_players]


# ---------------------------------------------------------------------------
# FastPokerState
# ---------------------------------------------------------------------------

class FastPokerState:
    """Numpy-array poker state.  Mutable — callers must ``copy()`` first."""

    __slots__ = (
        "chips", "bets", "active", "hole_cards", "community",
        "deck_order", "deck_cursor", "stage", "n_raises",
        "_player_i_index", "n_actions", "n_players_started_round",
        "pot_total", "history", "small_blind", "big_blind",
        "initial_chips", "n_players", "_preflop_order", "_postflop_order",
        "_skip_counter", "_winners_computed",
    )

    # Stage constants.
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4
    TERMINAL = 5

    def __init__(
        self,
        n_players: int = 2,
        small_blind: int = 50,
        big_blind: int = 100,
        initial_chips: int = 10000,
    ):
        self.n_players = n_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_chips = initial_chips

        # Per-player arrays.
        self.chips = np.full(n_players, initial_chips, dtype=np.int32)
        self.bets = np.zeros(n_players, dtype=np.int32)
        self.active = np.ones(n_players, dtype=np.bool_)
        self.hole_cards = np.full((n_players, 2), -1, dtype=np.int8)

        # Community cards (-1 = undealt).
        self.community = np.full(5, -1, dtype=np.int8)

        # Shuffled deck — shared by reference across copies.
        self.deck_order = np.random.permutation(52).astype(np.int8)
        self.deck_cursor = 0

        # Betting state.
        self.stage = self.PREFLOP
        self.n_raises = 0
        self._player_i_index = 0
        self.n_actions = 0
        self.pot_total = 0
        self._skip_counter = 0
        self._winners_computed = False

        # Per-round action counts: (4 rounds, 3 types: calls/raises/folds).
        self.history = np.zeros((4, 3), dtype=np.int8)

        # Player ordering (shared refs, never copied).
        self._preflop_order, self._postflop_order = _get_orders(n_players)

        # --- Deal hole cards ---
        for p in range(n_players):
            for c in range(2):
                self.hole_cards[p, c] = self.deck_order[self.deck_cursor]
                self.deck_cursor += 1

        # --- Post blinds ---
        sb = min(small_blind, int(self.chips[0]))
        self.chips[0] -= sb
        self.bets[0] = sb

        bb = min(big_blind, int(self.chips[1]))
        self.chips[1] -= bb
        self.bets[1] = bb

        self.pot_total = sb + bb

        # Preflop: count players who can still respond and skip to first.
        self.n_players_started_round = self._n_players_with_moves()
        self._skip_to_first_active()

    # ------------------------------------------------------------------
    # Copy (the whole point — fast!)
    # ------------------------------------------------------------------

    def copy(self) -> FastPokerState:
        """Return a shallow copy with independent numpy arrays."""
        new = FastPokerState.__new__(FastPokerState)
        new.chips = self.chips.copy()
        new.bets = self.bets.copy()
        new.active = self.active.copy()
        new.hole_cards = self.hole_cards.copy()
        new.community = self.community.copy()
        # deck_order is read-only via cursor — share by reference.
        new.deck_order = self.deck_order
        new.deck_cursor = self.deck_cursor
        new.stage = self.stage
        new.n_raises = self.n_raises
        new._player_i_index = self._player_i_index
        new.n_actions = self.n_actions
        new.n_players_started_round = self.n_players_started_round
        new.pot_total = self.pot_total
        new.history = self.history.copy()
        new.small_blind = self.small_blind
        new.big_blind = self.big_blind
        new.initial_chips = self.initial_chips
        new.n_players = self.n_players
        new._preflop_order = self._preflop_order  # shared ref
        new._postflop_order = self._postflop_order  # shared ref
        new._skip_counter = self._skip_counter
        new._winners_computed = self._winners_computed
        return new

    # ------------------------------------------------------------------
    # Player indexing
    # ------------------------------------------------------------------

    @property
    def current_player_i(self) -> int:
        """Index of the player whose turn it is."""
        if self.stage == self.PREFLOP:
            return int(self._preflop_order[self._player_i_index])
        return int(self._postflop_order[self._player_i_index])

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def get_legal_mask(self) -> np.ndarray:
        """Return (9,) float32 mask for 9-action space."""
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        pi = self.current_player_i
        if self.active[pi] and self.chips[pi] > 0:
            mask[0] = 1.0  # fold
            mask[1] = 1.0  # call
            if self.n_raises < 3:
                biggest = int(self.bets.max())
                to_call = biggest - int(self.bets[pi])
                player_chips = int(self.chips[pi])
                min_raise = _min_raise_contribution(to_call, self.big_blind)
                for i, frac in enumerate(RAISE_FRACTIONS):
                    raise_amount = int(frac * self.pot_total) + to_call
                    if raise_amount >= min_raise and raise_amount <= player_chips:
                        mask[2 + i] = 1.0
                # All-in (action 8).
                if player_chips > 0:
                    mask[8] = 1.0
        return mask

    @property
    def legal_actions(self) -> list:
        """List of legal action ints (0-8) or [None] for inactive."""
        pi = self.current_player_i
        if self.active[pi] and self.chips[pi] > 0:
            mask = self.get_legal_mask()
            return [a for a in range(N_ACTIONS) if mask[a] > 0]
        return [None]

    # ------------------------------------------------------------------
    # Terminal / payout
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.stage in (self.SHOWDOWN, self.TERMINAL)

    @property
    def payout(self) -> dict[int, int]:
        """Chip change per player (positive = won, negative = lost)."""
        if not self._winners_computed:
            self._compute_winners()
        return {
            i: int(self.chips[i]) - self.initial_chips
            for i in range(self.n_players)
        }

    # ------------------------------------------------------------------
    # Apply action (mutating — caller must copy() first)
    # ------------------------------------------------------------------

    def apply_action(self, action) -> None:
        """Apply action in-place.  action: 0-8 or None=skip."""
        pi = self.current_player_i

        if action is None:
            pass
        elif action == 0:  # fold
            self.active[pi] = False
        elif action == 1:  # call
            if self.chips[pi] > 0:
                biggest = int(self.bets.max())
                to_call = biggest - int(self.bets[pi])
                to_call = min(to_call, int(self.chips[pi]))
                self.chips[pi] -= to_call
                self.bets[pi] += to_call
                self.pot_total += to_call
        elif 2 <= action <= 7:  # fractional raise
            frac = RAISE_FRACTIONS[action - 2]
            biggest = int(self.bets.max())
            to_call = biggest - int(self.bets[pi])
            raise_chips = int(frac * self.pot_total) + to_call
            raise_chips = max(
                raise_chips,
                _min_raise_contribution(to_call, self.big_blind),
            )
            raise_chips = min(raise_chips, int(self.chips[pi]))
            self.chips[pi] -= raise_chips
            self.bets[pi] += raise_chips
            self.pot_total += raise_chips
            self.n_raises += 1
        elif action == 8:  # all-in
            all_in_chips = int(self.chips[pi])
            self.chips[pi] = 0
            self.bets[pi] += all_in_chips
            self.pot_total += all_in_chips
            self.n_raises += 1

        # Record in history (3 categories: calls, raises, folds).
        if action is not None:
            rd = min(self.stage, 3)
            if action == 1:
                self.history[rd, 0] += 1  # calls
            elif action >= 2:
                self.history[rd, 1] += 1  # raises (all sizes)
            elif action == 0:
                self.history[rd, 2] += 1  # folds

        self.n_actions += 1
        self._skip_counter = 0

        # Advance to next player / next stage.
        self._advance()

    def _advance(self):
        """Move to the next player, handling round/stage transitions."""
        while True:
            self._player_i_index = (self._player_i_index + 1) % self.n_players
            pi = self.current_player_i

            if self._n_active_players() == 1:
                self.stage = self.TERMINAL
                break

            # Check if this round of betting is finished.
            betting_done = self._is_betting_finished()
            if betting_done and self._n_players_with_moves() <= 1:
                self._deal_remaining_to_showdown()
                break

            if betting_done and self.n_actions >= self.n_players_started_round:
                self._increment_stage()
                if self.stage >= self.SHOWDOWN:
                    break
                if self._n_players_with_moves() <= 1:
                    self._deal_remaining_to_showdown()
                    break
                self._reset_round()

            pi = self.current_player_i  # may have changed after reset

            if not self.active[pi] or self.chips[pi] <= 0:
                self._skip_counter += 1
                continue

            if self.stage >= self.SHOWDOWN:
                break
            break
        if self.stage >= self.SHOWDOWN:
            self._compute_winners()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_betting_finished(self) -> bool:
        """True when active players are folded, all-in, or matched."""
        active_bets = []
        for i in range(self.n_players):
            if self.active[i]:
                active_bets.append(int(self.bets[i]))
        if len(active_bets) <= 1:
            return True
        biggest = max(active_bets)
        for i in range(self.n_players):
            if self.active[i] and self.chips[i] > 0 and int(self.bets[i]) < biggest:
                return False
        return True

    def _n_active_players(self) -> int:
        return int(self.active.sum())

    def _n_players_with_moves(self) -> int:
        return sum(
            1 for i in range(self.n_players)
            if self.active[i] and self.chips[i] > 0
        )

    def _increment_stage(self):
        if self.stage == self.PREFLOP:
            self.stage = self.FLOP
            self._deal_community(3)
        elif self.stage == self.FLOP:
            self.stage = self.TURN
            self._deal_community(1)
        elif self.stage == self.TURN:
            self.stage = self.RIVER
            self._deal_community(1)
        elif self.stage == self.RIVER:
            self.stage = self.SHOWDOWN

    def _reset_round(self):
        """Reset state for a new betting round."""
        self.n_actions = 0
        self.n_raises = 0
        self._player_i_index = 0
        self.n_players_started_round = self._n_players_with_moves()
        if self.n_players_started_round == 0:
            return
        self._skip_to_first_active()

    def _skip_to_first_active(self):
        """Advance _player_i_index to the first player with a legal move."""
        for _ in range(self.n_players):
            pi = self.current_player_i
            if self.active[pi] and self.chips[pi] > 0:
                return
            self._skip_counter += 1
            self._player_i_index += 1

    def _deal_remaining_to_showdown(self):
        """Deal all undealt board cards once no further betting is possible."""
        while self.stage < self.SHOWDOWN:
            self._increment_stage()

    def _deal_community(self, n: int):
        """Deal n cards from the deck to the community."""
        for _ in range(n):
            slot = int((self.community >= 0).sum())
            if slot >= 5:
                break
            self.community[slot] = self.deck_order[self.deck_cursor]
            self.deck_cursor += 1

    # ------------------------------------------------------------------
    # Winner computation
    # ------------------------------------------------------------------

    def _compute_winners(self):
        """Evaluate hands, distribute pot, update chips."""
        if self._winners_computed:
            return
        self._winners_computed = True

        n_active = int(self.active.sum())
        if n_active == 0:
            return

        if n_active == 1:
            winner = int(np.argmax(self.active))
            self.chips[winner] += self.pot_total
            return

        # Evaluate hands for all active players.
        board_eval = [
            int(CARD_INDEX_TO_EVAL_CARD[c])
            for c in self.community
            if c >= 0
        ]
        ranks: dict[int, int] = {}
        for p in range(self.n_players):
            if self.active[p]:
                hand_eval = [
                    int(CARD_INDEX_TO_EVAL_CARD[self.hole_cards[p, 0]]),
                    int(CARD_INDEX_TO_EVAL_CARD[self.hole_cards[p, 1]]),
                ]
                ranks[p] = _EVALUATOR.evaluate(board_eval, hand_eval)

        # Distribute side pots.
        side_pots = self._compute_side_pots()
        for pot in side_pots:
            eligible = [p for p in pot if self.active[p] and p in ranks]
            if not eligible:
                continue
            best_rank = min(ranks[p] for p in eligible)
            winners = sorted(p for p in eligible if ranks[p] == best_rank)
            total = sum(pot.values())
            per_winner = total // len(winners)
            remainder = total - per_winner * len(winners)
            for i, w in enumerate(winners):
                self.chips[w] += per_winner + (1 if i < remainder else 0)

    def _compute_side_pots(self) -> list[dict[int, int]]:
        """Compute side pots from bets array."""
        pots = []
        remaining = self.bets.copy()
        while remaining.sum() > 0:
            pot: dict[int, int] = {}
            nonzero = remaining[remaining > 0]
            if len(nonzero) == 0:
                break
            min_bet = int(nonzero.min())
            for i in range(self.n_players):
                if remaining[i] > 0:
                    contribution = min(int(remaining[i]), min_bet)
                    pot[i] = contribution
                    remaining[i] -= contribution
            pots.append(pot)
        return pots

    # ------------------------------------------------------------------
    # Feature encoding (must match state.py to_feature_vector exactly)
    # ------------------------------------------------------------------

    def to_feature_vector(self) -> np.ndarray:
        """Encode state as 126-dim float32 vector."""
        features = np.zeros(N_FEATURES, dtype=np.float32)
        pi = self.current_player_i

        # Hole cards: 52-dim binary.
        for c in self.hole_cards[pi]:
            if c >= 0:
                features[int(c)] = 1.0

        # Community cards: 52-dim binary.
        for c in self.community:
            if c >= 0:
                features[52 + int(c)] = 1.0

        # Current betting round: 4-dim one-hot.
        round_idx = min(self.stage, 3)
        if self.stage < 4:
            features[104 + round_idx] = 1.0

        # Scalar features.
        total_chips = self.initial_chips * self.n_players
        features[108] = self.pot_total / total_chips
        features[109] = self.chips[pi] / self.initial_chips
        features[110] = self.bets[pi] / self.initial_chips
        features[111] = self.active.sum() / self.n_players
        features[112] = pi / max(self.n_players - 1, 1)
        features[113] = self.n_raises / 3.0

        # Per-round action summary.
        for r in range(4):
            offset = 114 + r * 3
            features[offset] = self.history[r, 0] / max(self.n_players, 1)
            features[offset + 1] = self.history[r, 1] / 3.0
            features[offset + 2] = self.history[r, 2] / max(self.n_players, 1)

        return features


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def new_fast_game(n_players: int = 2, **kwargs) -> FastPokerState:
    """Create a new poker game."""
    return FastPokerState(n_players=n_players, **kwargs)
