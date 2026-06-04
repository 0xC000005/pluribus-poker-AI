"""Leaf evaluator for depth-limited solving + the PINNED CFV normalization convention.

This is must-fix #1 from the ReBeL decision doc made concrete and testable. The depth-limited trunk
solver replaces the subtree below the cut with a leaf evaluator. The value net will eventually BE
that evaluator; here an EXACT oracle stands in for it so we can verify, by a numerical round-trip,
that the convention the evaluator PRODUCES matches the convention the trunk CONSUMES.

PINNED CONVENTION
-----------------
At a public cut state (a public belief state, PBS), the leaf evaluator takes the two players'
incoming per-card *ranges* (un-normalized reach mass arriving at the cut, one scalar per private
card) and returns, for each player i and each private card c, the **normalized expected
counterfactual value**::

    v_i(c) = ( sum over opponent cards c' of  oppreach(c') * EV_i(c, c') )
             ---------------------------------------------------------------
             ( sum over opponent cards c' of  oppreach(c') )

i.e. EV to player i of holding card c, in ante units, averaged over the opponent's range and the
remaining chance (the public board), under the continuation strategy. It is the OPPONENT-range
expectation, normalized by the opponent's range mass -- NOT the un-normalized counterfactual value
(which would still carry the opponent's reach as a multiplicative factor).

The trunk consumes ``v_i(c)`` directly as the continuation value of player i holding card c
(``poker_ai/rebel/leduc.py::values_with_leaf``). The round-trip test asserts that solving / valuing
the trunk through this interface reproduces the exact full-game values. The negative control returns
the un-normalized counterfactual value instead and shows the round-trip then breaks -- which is the
whole point of pinning the convention.
"""
from __future__ import annotations

import numpy as np

from poker_ai.rebel.leduc import NCARDS


class ExactLeafOracle:
    """Exact (no-net) leaf evaluator over a Leduc tree. Given the incoming ranges at a public cut
    state and a continuation strategy, returns per-card values for both players in the pinned
    normalized convention. ``normalize=False`` returns the un-normalized counterfactual value
    instead (the negative-control convention)."""

    def __init__(self, tree, normalize: bool = True):
        self.tree = tree
        self.normalize = normalize
        self.instances = tree.cut_instances()  # key -> [(c0, c1, board_node), ...]
        # board-EV cache is per (board_node, strategy); strategy varies, so we recompute per call.

    def evaluate(self, key, range0, range1, strategy):
        """key: public cut key. range0/range1: per-card incoming reach mass (length NCARDS).
        strategy: continuation profile (indexed by infoset id, as elsewhere).
        Returns (v0[NCARDS], v1[NCARDS]) per the pinned convention (or un-normalized if
        normalize=False). Cards not reachable get value 0."""
        v0num = np.zeros(NCARDS); v0den = np.zeros(NCARDS)
        v1num = np.zeros(NCARDS); v1den = np.zeros(NCARDS)
        for c0, c1, bn in self.instances[key]:
            cont = self.tree.subtree_ev(bn, strategy)  # (2,) board+round-2 EV for (c0,c1)
            # P0 holding c0: opponent is P1 (range1 over c1). P1 holding c1: opponent is P0.
            v0num[c0] += range1[c1] * cont[0]; v0den[c0] += range1[c1]
            v1num[c1] += range0[c0] * cont[1]; v1den[c1] += range0[c0]
        if self.normalize:
            v0 = np.divide(v0num, v0den, out=np.zeros(NCARDS), where=v0den > 1e-15)
            v1 = np.divide(v1num, v1den, out=np.zeros(NCARDS), where=v1den > 1e-15)
        else:
            v0, v1 = v0num, v1num  # un-normalized counterfactual value (negative control)
        return v0, v1

    def all_leaf_values(self, reaches, strategy):
        """Evaluate every cut key. ``reaches`` = tree.cut_reaches(pol). Returns leaf_v dict:
        key -> (v0[NCARDS], v1[NCARDS]) ready for ``values_with_leaf``."""
        out = {}
        for key, (r0, r1) in reaches.items():
            out[key] = self.evaluate(key, r0, r1, strategy)
        return out
