"""Small no-limit hold'em game for the PPO-vs-R-NaD GO/NO-GO A/B (exact-NashConv tractable).

The governance bundle (autoresearch-session/poker_reviews/20260531T000000Z-ppo-vs-rnad-smallnlhe-gonogo/)
resolved the representation seam via Option A: author the small-NLHE abstraction AS AN OPENSPIEL GAME so
both arms (R-NaD via its existing OpenSpiel collector; PPO via a thin OpenSpiel adapter) train AND are
scored on ONE representation, with exact NashConv as the PRIMARY decisive metric.

This module returns that game: real NO-LIMIT betting (universal_poker betting=nolimit) on a small deck +
short stack so the whole tree enumerates and OpenSpiel exact NashConv is cheap.

MEASURED (this exact config; /tmp/up_final.py -> /tmp/up_final.json, this session):
  - num_distinct_actions = 4
  - max_game_length = 7
  - uniform-policy exact NashConv = 1.7, computed in 0.01 s
So the exact-exploitability harness runs natively + fast on this game.

NOTE on param types: universal_poker uses PER-PARAM expected types (NOT all-string). Ints:
numPlayers/numRounds/numSuits/numRanks/numHoleCards. Strings: betting/firstPlayer/numBoardCards/blind/stack.
"""
from __future__ import annotations


# Per-parameter types matter (see module docstring).
SMALL_NLHE_PARAMS = {
    "betting": "nolimit",
    "numPlayers": 2,
    "numRounds": 1,
    "blind": "1 1",
    "firstPlayer": "1",
    "numSuits": 2,
    "numRanks": 3,
    "numHoleCards": 1,
    "numBoardCards": "0",
    "stack": "4 4",
}


def load_small_nlhe():
    """Load the small no-limit hold'em OpenSpiel game (exact-NashConv tractable)."""
    import pyspiel
    return pyspiel.load_game("universal_poker", dict(SMALL_NLHE_PARAMS))
