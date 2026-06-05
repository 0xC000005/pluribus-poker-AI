"""ReBeL on a real-belief game (step 0): the turn subgame with the river as the depth-limit leaf.

The turn betting tree is the trunk; each turn-call `showdown` terminal is a CUT node where, in the
real game, the river is dealt and a river betting subgame is played. The current `StreetSolver`
approximates that leaf by averaging showdown EQUITY over the 44 runouts (no river betting); ReBeL
replaces it with the river SUBGAME value (river betting + showdown), supplied via `cut_node_fn` --
exactly DeepStack/ReBeL depth-limited solving. The exact river value is the control; a PBS value net
replaces it for efficiency (steps 1-3).

This module is built on the trusted `scripts/solver.py` (`StreetSolver`) + `scripts/fast_cfr.py`
(`solve_cfr` with the `cut_node_fn` leaf hook). Card index convention here matches solver.py:
``card = rank*4 + suit`` with ranks ``23456789TJQKA`` and suits ``cdhs`` (1 big blind = 100 chips).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# scripts/ holds solver.py + fast_cfr.py and they import each other by bare name.
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import solver as _solver  # noqa: E402
from solver import StreetSolver, BIG_BLIND  # noqa: E402

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def parse_card(s: str) -> int:
    """'Ah' -> card index (rank*4 + suit), matching solver.py's _CARD_TO_EVAL convention."""
    r, su = s[0], s[1]
    return _RANKS.index(r) * 4 + _SUITS.index(su)


def card_str(ci: int) -> str:
    return _RANKS[ci // 4] + _SUITS[ci % 4]


@dataclass
class TurnSpot:
    """A heads-up turn decision point (chips; 1 bb = BIG_BLIND)."""
    board: list          # 4 card indices
    pot: int
    hero_stack: int
    villain_stack: int
    hero_first: bool = True

    @property
    def board_str(self):
        return " ".join(card_str(c) for c in self.board)


def default_spot() -> TurnSpot:
    """Standard reproducible first instance: board Ah Kd 7c 2s, 20 bb pot, 80 bb stacks."""
    return TurnSpot(
        board=[parse_card(c) for c in ("Ah", "Kd", "7c", "2s")],
        pot=20 * BIG_BLIND,
        hero_stack=80 * BIG_BLIND,
        villain_stack=80 * BIG_BLIND,
        hero_first=True,
    )


def build_turn_solver(spot: TurnSpot) -> StreetSolver:
    return StreetSolver(spot.board, spot.pot, spot.hero_stack, spot.villain_stack, spot.hero_first)


def showdown_cut_indices(solver: StreetSolver):
    """Indices (into the tree's node array) of the turn-call `showdown` terminals -- the river
    depth-limit frontier. Fold terminals are NOT cut (the hand truly ends there)."""
    nodes = solver._tree["all_nodes"]
    return [i for i, nd in enumerate(nodes)
            if getattr(nd, "terminal_type", None) == "showdown"]


def cut_node_pots(solver: StreetSolver, cut_indices):
    """Pot/stacks at each cut node (the river subgame's starting stakes)."""
    nodes = solver._tree["all_nodes"]
    out = {}
    for i in cut_indices:
        nd = nodes[i]
        out[i] = {"pot": nd.pot, "stacks": nd.stacks}
    return out
