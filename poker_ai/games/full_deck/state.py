"""Full-deck poker game state for Deep CFR training.

This replaces ShortDeckPokerState with a generalized state that:
- Supports any deck configuration (default: full 52-card deck)
- Does not depend on card_info_lut (no clustering needed)
- Provides to_feature_vector() for neural network input
"""
from __future__ import annotations

import collections
import copy
import json
import logging
from typing import Dict, List, Optional

import numpy as np

from poker_ai.games.full_deck.player import PokerPlayer
from poker_ai.poker.card import Card
from poker_ai.poker.engine import PokerEngine
from poker_ai.poker.pot import Pot
from poker_ai.poker.table import PokerTable

logger = logging.getLogger("poker_ai.games.full_deck.state")

# Map suit strings to indices for feature encoding.
_SUIT_TO_INDEX = {"clubs": 0, "diamonds": 1, "hearts": 2, "spades": 3}

# Number of features in the state vector.
# 52 (hole) + 52 (community) + 4 (round) + 6 (scalars) + 12 (history) = 126
N_FEATURES = 126

# Number of discrete actions (ReBeL standard).
N_ACTIONS = 9  # fold, call, 6 raise sizes, all-in

# Raise fractions for actions 2-7 (fraction of pot).
RAISE_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

# Action string to index mapping.
ACTION_TO_INDEX = {
    "fold": 0, "call": 1,
    "raise_0.25": 2, "raise_0.5": 3, "raise_0.75": 4,
    "raise_1.0": 5, "raise_1.5": 6, "raise_2.0": 7,
    "all_in": 8,
}
INDEX_TO_ACTION = {v: k for k, v in ACTION_TO_INDEX.items()}


def card_to_index(card: Card) -> int:
    """Convert a Card to a 0-51 index for the feature vector."""
    return (card.rank_int - 2) * 4 + _SUIT_TO_INDEX[card.suit]


def new_game(n_players: int, **kwargs) -> PokerState:
    """Create a new game of poker.

    Parameters
    ----------
    n_players : int
        Number of players (>= 2).
    **kwargs
        Passed to PokerState constructor (small_blind, big_blind,
        include_ranks, initial_chips).

    Returns
    -------
    state : PokerState
    """
    pot = Pot()
    initial_chips = kwargs.pop("initial_chips", 10000)
    players = [
        PokerPlayer(player_i=player_i, initial_chips=initial_chips, pot=pot)
        for player_i in range(n_players)
    ]
    return PokerState(players=players, **kwargs)


class PokerState:
    """Immutable game state for full-deck poker.

    Supports any deck size (default: full 52-card deck). New states are
    created via apply_action(). Provides to_feature_vector() for neural
    network input encoding.
    """

    def __init__(
        self,
        players: List[PokerPlayer],
        small_blind: int = 50,
        big_blind: int = 100,
        include_ranks: Optional[List[int]] = None,
    ):
        n_players = len(players)
        if n_players <= 1:
            raise ValueError(
                f"At least 2 players required but only {n_players} provided."
            )
        # Default to full 52-card deck.
        if include_ranks is None:
            include_ranks = list(range(2, 15))
        self._table = PokerTable(
            players=players, pot=players[0].pot, include_ranks=include_ranks
        )
        self._initial_n_chips = players[0].n_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self._poker_engine = PokerEngine(
            table=self._table, small_blind=small_blind, big_blind=big_blind
        )
        self._poker_engine.round_setup()
        self._table.dealer.deal_private_cards(self._table.players)
        self._history: Dict[str, List[str]] = collections.defaultdict(list)
        self._betting_stage = "pre_flop"
        self._betting_stage_to_round: Dict[str, int] = {
            "pre_flop": 0,
            "flop": 1,
            "turn": 2,
            "river": 3,
            "show_down": 4,
        }
        # Player ordering: pre-flop rotates so players after big blind act
        # first.
        player_i_order: List[int] = list(range(n_players))
        self.players[0].is_small_blind = True
        self.players[1].is_big_blind = True
        self.players[-1].is_dealer = True
        # In heads-up, SB=Button acts last postflop (BB acts first).
        if n_players == 2:
            postflop_order = [1, 0]
        else:
            postflop_order = player_i_order
        self._player_i_lut: Dict[str, List[int]] = {
            "pre_flop": player_i_order[2:] + player_i_order[:2],
            "flop": postflop_order,
            "turn": postflop_order,
            "river": postflop_order,
            "show_down": postflop_order,
            "terminal": postflop_order,
        }
        self._skip_counter = 0
        self._first_move_of_current_round = True
        self._reset_betting_round_state()
        for player in self.players:
            player.is_turn = False
        self.current_player.is_turn = True

    def __repr__(self):
        return (
            f"<PokerState player_i={self.player_i} "
            f"betting_stage={self._betting_stage}>"
        )

    def apply_action(self, action_str: Optional[str]) -> PokerState:
        """Create a new state after applying an action.

        Parameters
        ----------
        action_str : str or None
            One of {"fold", "call", "raise_X", "all_in", None}.

        Returns
        -------
        new_state : PokerState
        """
        if action_str not in self.legal_actions:
            raise ValueError(
                f"Action '{action_str}' not in legal actions: {self.legal_actions}"
            )
        new_state = copy.deepcopy(self)
        new_state._first_move_of_current_round = False
        if action_str is None:
            assert not new_state.current_player.is_active
        elif action_str == "call":
            new_state.current_player.call(players=new_state.players)
        elif action_str == "fold":
            new_state.current_player.fold()
        elif action_str == "all_in":
            all_in_amount = new_state.current_player.n_chips
            new_state.current_player.raise_to(n_chips=all_in_amount)
            new_state._n_raises += 1
        elif action_str.startswith("raise_"):
            frac = float(action_str.split("_")[1])
            biggest_bet = max(p.n_bet_chips for p in new_state.players)
            n_chips_to_call = biggest_bet - new_state.current_player.n_bet_chips
            raise_n_chips = int(frac * new_state._table.pot.total) + n_chips_to_call
            raise_n_chips = max(raise_n_chips, new_state.big_blind)
            raise_n_chips = min(raise_n_chips, new_state.current_player.n_chips)
            new_state.current_player.raise_to(n_chips=raise_n_chips)
            new_state._n_raises += 1
        else:
            raise ValueError(f"Unknown action: {action_str}")
        # Record normalized action in history (all raises → "raise").
        if action_str is not None:
            if action_str.startswith("raise_") or action_str == "all_in":
                history_action = "raise"
            else:
                history_action = action_str
        else:
            history_action = action_str
        skip_actions = ["skip" for _ in range(new_state._skip_counter)]
        new_state._history[new_state.betting_stage] += skip_actions
        new_state._history[new_state.betting_stage].append(history_action)
        new_state._n_actions += 1
        new_state._skip_counter = 0
        # Advance to next player / next stage.
        while True:
            new_state._move_to_next_player()
            finished_betting = not new_state._poker_engine.more_betting_needed
            if finished_betting and new_state.all_players_have_actioned:
                new_state._increment_stage()
                new_state._reset_betting_round_state()
                new_state._first_move_of_current_round = True
            if not new_state.current_player.is_active:
                new_state._skip_counter += 1
            elif new_state.current_player.is_active:
                if new_state._poker_engine.n_players_with_moves == 1:
                    new_state._betting_stage = "terminal"
                    if not new_state._table.community_cards:
                        new_state._poker_engine.table.dealer.deal_flop(
                            new_state._table
                        )
                if new_state._betting_stage in {"terminal", "show_down"}:
                    new_state._poker_engine.compute_winners()
                break
        for player in new_state.players:
            player.is_turn = False
        new_state.current_player.is_turn = True
        return new_state

    # ------------------------------------------------------------------
    # Feature encoding for neural networks
    # ------------------------------------------------------------------

    def to_feature_vector(self) -> np.ndarray:
        """Encode the current game state as a fixed-length feature vector.

        Returns a 126-dimensional float32 vector:
          [0:52]   - hole cards (52-dim binary)
          [52:104] - community cards (52-dim binary)
          [104:108]- current betting round (4-dim one-hot)
          [108:114]- scalar game features (6-dim)
          [114:126]- per-round action summary (4 rounds × 3 features)
        """
        features = np.zeros(N_FEATURES, dtype=np.float32)
        # Hole cards: 52-dim binary.
        for card in self.current_player.cards:
            features[card_to_index(card)] = 1.0
        # Community cards: 52-dim binary.
        for card in self._table.community_cards:
            features[52 + card_to_index(card)] = 1.0
        # Current betting round: 4-dim one-hot.
        round_idx = self._betting_stage_to_round.get(self._betting_stage, 3)
        if round_idx < 4:
            features[104 + round_idx] = 1.0
        # Scalar features (normalized to ~[0, 1]).
        n_players = len(self.players)
        total_chips = self._initial_n_chips * n_players
        features[108] = self._table.pot.total / total_chips
        features[109] = self.current_player.n_chips / self._initial_n_chips
        features[110] = self.current_player.n_bet_chips / self._initial_n_chips
        features[111] = sum(1 for p in self.players if p.is_active) / n_players
        features[112] = self.player_i / max(n_players - 1, 1)
        features[113] = self._n_raises / 3.0
        # Per-round action summary: 4 rounds × 3 features.
        stages = ["pre_flop", "flop", "turn", "river"]
        for i, stage in enumerate(stages):
            actions = self._history.get(stage, [])
            if not actions:
                continue
            n_calls = sum(1 for a in actions if a == "call")
            n_raises = sum(1 for a in actions if a == "raise")
            n_folds = sum(1 for a in actions if a == "fold")
            offset = 114 + i * 3
            features[offset] = n_calls / max(n_players, 1)
            features[offset + 1] = n_raises / 3.0
            features[offset + 2] = n_folds / max(n_players, 1)
        return features

    # ------------------------------------------------------------------
    # Info set (for debugging / compatibility — not used by Deep CFR)
    # ------------------------------------------------------------------

    @property
    def info_set(self) -> str:
        """Get an info set string encoding cards and history.

        Unlike ShortDeckPokerState, this does NOT require a card_info_lut.
        Cards are encoded directly by rank and suit.
        """
        cards = []
        for card in sorted(
            self.current_player.cards, key=lambda c: c.eval_card, reverse=True
        ):
            cards.append(f"{card.rank_int}{card.suit[0]}")
        for card in sorted(
            self._table.community_cards, key=lambda c: c.eval_card, reverse=True
        ):
            cards.append(f"{card.rank_int}{card.suit[0]}")
        info_set_dict = {
            "cards": cards,
            "history": [
                {stage: list(actions)}
                for stage, actions in self._history.items()
            ],
        }
        return json.dumps(info_set_dict, separators=(",", ":"))

    # ------------------------------------------------------------------
    # Game state properties
    # ------------------------------------------------------------------

    @property
    def community_cards(self) -> List[Card]:
        return self._table.community_cards

    @property
    def betting_stage(self) -> str:
        return self._betting_stage

    @property
    def betting_round(self) -> int:
        return self._betting_stage_to_round.get(self._betting_stage, 4)

    @property
    def all_players_have_actioned(self) -> bool:
        return self._n_actions >= self._n_players_started_round

    @property
    def player_i(self) -> int:
        return self._player_i_lut[self._betting_stage][self._player_i_index]

    @property
    def players(self) -> List[PokerPlayer]:
        return self._table.players

    @property
    def current_player(self) -> PokerPlayer:
        return self._table.players[self.player_i]

    @property
    def legal_actions(self) -> List[Optional[str]]:
        if self.current_player.is_active:
            actions: List[Optional[str]] = ["fold", "call"]
            if self._n_raises < 3:
                biggest_bet = max(p.n_bet_chips for p in self.players)
                to_call = biggest_bet - self.current_player.n_bet_chips
                player_chips = self.current_player.n_chips
                for frac in RAISE_FRACTIONS:
                    raise_amount = int(frac * self._table.pot.total) + to_call
                    if raise_amount >= self.big_blind and raise_amount <= player_chips:
                        actions.append(f"raise_{frac}")
                if player_chips > 0:
                    actions.append("all_in")
            return actions
        return [None]

    @property
    def is_terminal(self) -> bool:
        return self._betting_stage in {"show_down", "terminal"}

    @property
    def payout(self) -> Dict[int, int]:
        return {
            i: player.n_chips - self._initial_n_chips
            for i, player in enumerate(self.players)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_to_next_player(self):
        self._player_i_index += 1
        if self._player_i_index >= len(self.players):
            self._player_i_index = 0

    def _reset_betting_round_state(self):
        self._all_players_have_made_action = False
        self._n_actions = 0
        self._n_raises = 0
        self._player_i_index = 0
        self._n_players_started_round = self._poker_engine.n_active_players
        while not self.current_player.is_active:
            self._skip_counter += 1
            self._player_i_index += 1

    def _increment_stage(self):
        if self._betting_stage == "pre_flop":
            self._betting_stage = "flop"
            self._poker_engine.table.dealer.deal_flop(self._table)
        elif self._betting_stage == "flop":
            self._betting_stage = "turn"
            self._poker_engine.table.dealer.deal_turn(self._table)
        elif self._betting_stage == "turn":
            self._betting_stage = "river"
            self._poker_engine.table.dealer.deal_river(self._table)
        elif self._betting_stage == "river":
            self._betting_stage = "show_down"
        elif self._betting_stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {self._betting_stage}")
