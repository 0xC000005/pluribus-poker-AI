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


def _average_strategy_array(strategy_sum):
    """Normalize a (n_nodes, n_actions, n) strategy-sum into per-(node,hand) average strategy.
    Unreached (all-zero) rows are left zero -> regret-matching turns them into uniform-over-legal,
    which is harmless because those nodes carry ~zero reach."""
    s = np.asarray(strategy_sum, dtype=np.float32)
    denom = s.sum(axis=1, keepdims=True)              # (n_nodes, 1, n)
    avg = np.divide(s, denom, out=np.zeros_like(s), where=denom > 1e-12)
    return avg


def river_subgame_cfv(board5, pot, hero_stack, villain_stack, hero_first,
                      hero_range, villain_range, iters=200):
    """Solve a river subgame range-vs-range and return the AVERAGE-strategy per-hand counterfactual
    values (hero_cfv, villain_cfv), each shape (n_river_hands,), in chips, opponent-reach-weighted
    (the same convention as solve_cfr's terminal hvals/vvals -- so they drop straight into a turn
    cut_node_fn). Also returns the river hand list for index mapping.

    Extraction reuses solve_cfr's exact terminal eval: solve to convergence, then re-run for ONE
    iteration with initial_regret_sum = the average strategy (regret-matching reproduces it) and
    trace the root, reading hvals[0]/vvals[0] under the average strategy.

    *** WIP / UNVERIFIED -- DO NOT USE FOR TARGETS/CONTROL YET. ***
    Verification identity (must hold for ANY strategy, since every terminal satisfies
    hero_val(a,b)+villain_val(b,a) = pot_start; coeffs at fast_cfr.py L376-391):
        sum(hero_range*hero_cfv) + sum(villain_range*villain_cfv) == pot * (hero_range @ valid @ villain_range)
    This currently FAILS (e.g. measured 806 vs expected 3663 for 1-iter uniform; ratio varies with
    strategy) -> the trace/hvals[0] extraction is not returning the full root counterfactual value as
    assumed. NEXT: instrument on a TINY river tree and compare hvals[0] to a brute-force per-hand
    value; likely a reach/strategy-weighting or root-node subtlety in the trace path. Once fixed,
    the turn cut_node_fn must also apply the convention offset (subtract hi_cut/vi_cut at the cut, in
    counterfactual form) to convert river-net-from-river into turn-net-from-turn-start."""
    rs = StreetSolver(board5, pot, hero_stack, villain_stack, hero_first)
    hr = np.asarray(hero_range, dtype=np.float32)
    vr = np.asarray(villain_range, dtype=np.float32)
    rs.solve(n_iterations=iters, hero_range=hr, villain_range=vr, backend="cpu")
    avg = _average_strategy_array(rs._strategy_sum)

    captured = {}

    def trace(*, hero_values, villain_values, **_):
        captured["h"] = np.array(hero_values[0], dtype=np.float64)
        captured["v"] = np.array(villain_values[0], dtype=np.float64)

    rs.solve(n_iterations=1, hero_range=hr, villain_range=vr, backend="cpu",
             initial_regret_sum=avg, trace_node_indices=[0], trace_node_fn=trace)
    return captured["h"], captured["v"], rs.hands

