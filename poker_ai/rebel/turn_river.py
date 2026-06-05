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


def subgame_value_pass(solver, avg, hero_range, villain_range):
    """Exact per-hand counterfactual values (hero_cfv, villain_cfv) at the ROOT of a solved
    StreetSolver, under a FIXED strategy ``avg`` (shape (n_nodes, n_actions, n)) and the given
    ranges. Mirrors solve_cfr's forward(reach)/terminal/backward(value) passes with the verified
    terminal coefficients (fast_cfr.py L376-391): value = net chips from subgame start with the
    pre-existing pot awarded -> per valid pair hero_val(a,b)+villain_val(b,a)=pot_start. Returns
    opponent-reach-weighted counterfactual values in chips (the cut_node_fn convention)."""
    t = solver._tree
    nn = t["n_nodes"]; n = solver.n
    player = t["player"]; children = t["children"]; dacts = t["decision_actions"]
    sh = t["stacks_h"]; sv = t["stacks_v"]
    win = solver.win_m; lose = solver.lose_m; tie = solver.tie_m; valid = solver.valid
    winT, loseT, tieT, validT = win.T, lose.T, tie.T, valid.T
    ps = solver.pot_start; HS = solver.hero_stack_start; VS = solver.villain_stack_start
    show = set(t["showdown_idx"].tolist())
    hfold = set(t["hero_fold_idx"].tolist())
    vfold = set(t["villain_fold_idx"].tolist())

    # Forward STRATEGY reach (start at ones -> own-range factored OUT, so the result is the
    # counterfactual value, not the contribution). hstrat carries only hero's avg strategy; vstrat
    # only villain's. The counterfactual value is then a reach-weighted sum over TERMINALS -- NOT a
    # backward aggregation (which weights opponent values by the opponent's per-hand strategy, an
    # index mismatch that is fine for CFR regrets but wrong for the absolute counterfactual value).
    hstrat = np.zeros((nn, n)); vstrat = np.zeros((nn, n))
    hstrat[0] = 1.0; vstrat[0] = 1.0
    for i in range(nn):
        if player[i] == -1:
            continue
        for a in dacts[i]:
            c = children[i, a]
            if player[i] == 0:
                hstrat[c] = hstrat[i] * avg[i, a]; vstrat[c] = vstrat[i]
            else:
                hstrat[c] = hstrat[i]; vstrat[c] = vstrat[i] * avg[i, a]

    hr0 = np.asarray(hero_range, dtype=np.float64)
    vr0 = np.asarray(villain_range, dtype=np.float64)
    hcfv = np.zeros(n); vcfv = np.zeros(n)
    for i in range(nn):                       # terminal value, reach-weighted into root counterfactual
        if player[i] != -1:
            continue
        hi = HS - sh[i]; vi = VS - sv[i]
        vfull = vr0 * vstrat[i]               # villain range * villain strat reach to terminal
        hfull = hr0 * hstrat[i]               # hero range * hero strat reach
        if i in show:
            hv = (ps + vi) * (vfull @ winT) + (-hi) * (vfull @ loseT) + ((ps + vi - hi) / 2) * (vfull @ tieT)
            vv = (ps + hi) * (hfull @ lose) + (-vi) * (hfull @ win) + ((ps + hi - vi) / 2) * (hfull @ tie)
        elif i in hfold:
            hv = (-hi) * (vfull @ validT); vv = (ps + hi) * (hfull @ valid)
        elif i in vfold:
            hv = (ps + vi) * (vfull @ validT); vv = (-vi) * (hfull @ valid)
        else:
            continue
        hcfv += hstrat[i] * hv                 # hero counterfactual value (hero range factored out)
        vcfv += vstrat[i] * vv
    return hcfv, vcfv


def turn_leaf_river_cfv(turn_board, pot, hero_stack, villain_stack, hero_first_river,
                        turn_hands, hero_reach, villain_reach, river_iters=150):
    """Exact river-continuation per-hand counterfactual values at a turn leaf, averaged over the 44
    runouts. ``turn_hands`` = the turn solver's hand list (each a (c1,c2) tuple); hero_reach/
    villain_reach are indexed to match. For each river card r not on the turn board: restrict the
    ranges to turn hands not containing r, map them onto the river subgame's hand order, solve the
    river subgame (river_subgame_cfv), and map the per-river-hand CFVs back to turn-hand indices.
    Sum over r and divide by 44 = (52 - 4 board - 2 hero - 2 villain): each valid (a,b) pair has
    exactly 44 legal runouts (the b-exclusion zeroes the rest automatically).

    Returns (hero_cfv, villain_cfv) per turn hand, in the NET-FROM-RIVER convention (they award the
    cut pot). The turn cut_node_fn must then subtract the hi_cut/vi_cut offset to get net-from-turn."""
    n = len(turn_hands)
    turn_idx = {tuple(h): i for i, h in enumerate(turn_hands)}
    cut_h = np.zeros(n); cut_v = np.zeros(n)
    hero_reach = np.asarray(hero_reach, dtype=np.float64)
    villain_reach = np.asarray(villain_reach, dtype=np.float64)
    for r in [c for c in range(52) if c not in turn_board]:
        board5 = turn_board + [r]
        river = StreetSolver(board5, pot, hero_stack, villain_stack, hero_first_river)
        rh2i = river.hand_to_idx
        hr_r = np.zeros(river.n); vr_r = np.zeros(river.n)
        back = {}
        for h, ti in turn_idx.items():
            if r in h:
                continue
            ri = rh2i.get(h)
            if ri is None:
                continue
            hr_r[ri] = hero_reach[ti]; vr_r[ri] = villain_reach[ti]; back[ri] = ti
        hcfv, vcfv, _ = river_subgame_cfv(board5, pot, hero_stack, villain_stack,
                                          hero_first_river, hr_r, vr_r, iters=river_iters)
        for ri, ti in back.items():
            cut_h[ti] += hcfv[ri]; cut_v[ti] += vcfv[ri]
    cut_h /= 44.0; cut_v /= 44.0
    return cut_h, cut_v


def river_subgame_cfv(board5, pot, hero_stack, villain_stack, hero_first,
                      hero_range, villain_range, iters=200):
    """Solve a river subgame range-vs-range and return the AVERAGE-strategy per-hand counterfactual
    values (hero_cfv, villain_cfv), each shape (n_river_hands,), in chips, opponent-reach-weighted
    (the same convention as solve_cfr's terminal hvals/vvals -- so they drop straight into a turn
    cut_node_fn). Also returns the river hand list for index mapping.

    Extraction: solve to convergence, then a VERIFIED forward-reach value pass (subgame_value_pass)
    under the average strategy. Verified via the convention identity (holds for any strategy, since
    every terminal satisfies hero_val(a,b)+villain_val(b,a)=pot_start; coeffs at fast_cfr.py L376-391):
        sum(hero_range*hero_cfv) + sum(villain_range*villain_cfv) == pot * (hero_range @ valid @ villain_range)
    (matches exactly across spots; see test_rebel_turn_river).

    NOTE for the turn cut_node_fn (next step): these are net-from-river -- they award the cut pot,
    which already includes turn investments -- so to drop into the turn solve they must be converted
    to net-from-turn-start by subtracting hi_cut/vi_cut in counterfactual (opponent-reach-weighted)
    form; that offset differs per cut node so it does NOT cancel in regrets."""
    rs = StreetSolver(board5, pot, hero_stack, villain_stack, hero_first)
    hr = np.asarray(hero_range, dtype=np.float32)
    vr = np.asarray(villain_range, dtype=np.float32)
    rs.solve(n_iterations=iters, hero_range=hr, villain_range=vr, backend="cpu")
    avg = _average_strategy_array(rs._strategy_sum)
    hcfv, vcfv = subgame_value_pass(rs, avg, hr.astype(np.float64), vr.astype(np.float64))
    return hcfv, vcfv, rs.hands

