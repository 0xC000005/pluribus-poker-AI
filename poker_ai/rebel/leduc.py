"""Tagged Leduc public tree: the small-game substrate for the ReBeL de-risk.

All game logic (betting structure, board outcomes, terminal payoffs) is taken straight from
OpenSpiel's ``leduc_poker`` by enumeration -- this module only RESTRUCTURES that tree so we can
(a) name the depth-limit cut (the public-card deal), and (b) read per-private-card reaches and
counterfactual values at the cut. There is therefore no independent game logic to mistrust; the
``values`` recursion below is the same belief-weighted-Q pass used by the trusted small-game gates
(``scripts/run_rnad_tabular_gate.py``), and ``cfr_plus`` is validated against OpenSpiel NashConv.

Leduc structure (confirmed empirically): 6 cards (rank = card // 2 -> {0,1,2}); chance deals P0's
card (1/6), then P1's card (1/5 of the remaining), then round-1 betting, then the PUBLIC card
(1/4 of the remaining) -- the depth-limit cut -- then round-2 betting, then showdown. Payoffs are
in ante units (check/check showdown = +/-1).

Node encoding (tuples, mirroring the existing gate code):
  ("term", returns[2], c0, c1, board)
  ("deal", [(card, prob, child), ...])                  # a private deal (P0 then P1)
  ("board", pub_key, c0, c1, [(card, prob, child), ...])# the CUT: public-card deal
  ("dec", player, iid, rnd, c0, c1, board, [(action, child), ...])
"""
from __future__ import annotations

import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability

NCARDS = 6


def rank(card: int) -> int:
    return card // 2


class LeducTree:
    """Precomputed tagged Leduc tree; built once, reused across all iterations (no cloning)."""

    def __init__(self):
        self.game = pyspiel.load_game("leduc_poker")
        self.infosets: dict[str, int] = {}   # infoset_str -> id
        self.iset_player: list[int] = []     # id -> player
        self.iset_actions: list[list[int]] = []  # id -> [action ids]
        self.iset_round: list[int] = []      # id -> 1 or 2
        self.iset_key: list[str] = []        # id -> the infoset string (for to_tabular)
        self.root = self._build(self.game.new_initial_state(), None, None, None, ())

    def _iset_id(self, key, player, acts, rnd):
        if key not in self.infosets:
            self.infosets[key] = len(self.iset_player)
            self.iset_player.append(player)
            self.iset_actions.append(list(acts))
            self.iset_round.append(rnd)
            self.iset_key.append(key)
        return self.infosets[key]

    def _build(self, state, c0, c1, board, r1seq):
        if state.is_terminal():
            return ("term", np.asarray(state.returns(), dtype=np.float64), c0, c1, board)
        if state.is_chance_node():
            outs = state.chance_outcomes()
            if c0 is None:           # deal P0
                kids = []
                for a, p in outs:
                    s = state.clone(); s.apply_action(a)
                    kids.append((a, p, self._build(s, a, None, None, ())))
                return ("deal", kids)
            if c1 is None:           # deal P1
                kids = []
                for a, p in outs:
                    s = state.clone(); s.apply_action(a)
                    kids.append((a, p, self._build(s, c0, a, None, ())))
                return ("deal", kids)
            # else: the public-card deal == the depth-limit CUT
            kids = []
            for a, p in outs:
                s = state.clone(); s.apply_action(a)
                kids.append((a, p, self._build(s, c0, c1, a, r1seq)))
            return ("board", tuple(r1seq), c0, c1, kids)
        # decision node
        pl = state.current_player()
        acts = state.legal_actions()
        rnd = 1 if board is None else 2
        iid = self._iset_id(state.information_state_string(), pl, acts, rnd)
        kids = []
        for a in acts:
            s = state.clone(); s.apply_action(a)
            nxt_r1 = r1seq + (a,) if rnd == 1 else r1seq
            kids.append((a, self._build(s, c0, c1, board, nxt_r1)))
        return ("dec", pl, iid, rnd, c0, c1, board, kids)

    @property
    def n_iset(self) -> int:
        return len(self.iset_player)

    # ------------------------------------------------------------------ policies
    def uniform_policy(self):
        return [np.ones(len(a)) / len(a) for a in self.iset_actions]

    def random_policy(self, rng):
        pol = []
        for a in self.iset_actions:
            x = rng.random(len(a)) + 1e-3
            pol.append(x / x.sum())
        return pol

    def _pos(self, iid):
        """action id -> position index for this infoset."""
        return {a: i for i, a in enumerate(self.iset_actions[iid])}

    def to_tabular(self, pol):
        """pol[iid] = prob array aligned with iset_actions[iid]. -> OpenSpiel TabularPolicy."""
        tp = policy_lib.TabularPolicy(self.game)
        for key, iid in self.infosets.items():
            if key in tp.state_lookup:
                row = tp.action_probability_array[tp.state_lookup[key]]
                row[:] = 0.0
                for i, a in enumerate(self.iset_actions[iid]):
                    row[a] = pol[iid][i]
        return tp

    def nash_conv(self, pol) -> float:
        return float(exploitability.nash_conv(self.game, self.to_tabular(pol)))

    # ------------------------------------------------------------------ exact values
    def values(self, pol):
        """Exact belief-weighted action values q[iid][a_pos] (normalized) and the un-normalized
        counterfactual numerators qnum + denominators qden. q = qnum / qden. Single exact pass.
        cf (belief weight) = opponent_reach * chance_reach. Mirrors the trusted gate recursion."""
        n = self.n_iset
        qnum = [np.zeros(len(a)) for a in self.iset_actions]
        qden = [0.0] * n

        def rec(node, r0, r1, rc):
            t = node[0]
            if t == "term":
                return node[1]
            if t == "deal":
                ev = np.zeros(2)
                for _a, p, ch in node[1]:
                    ev += p * rec(ch, r0, r1, rc * p)
                return ev
            if t == "board":
                ev = np.zeros(2)
                for _a, p, ch in node[4]:
                    ev += p * rec(ch, r0, r1, rc * p)
                return ev
            _, pl, iid, _rnd, _c0, _c1, _b, kids = node
            pol_i = pol[iid]
            cf = (r1 * rc) if pl == 0 else (r0 * rc)
            qden[iid] += cf
            ev = np.zeros(2)
            row = qnum[iid]
            for i, (a, ch) in enumerate(kids):
                cv = rec(ch, r0 * pol_i[i], r1, rc) if pl == 0 else rec(ch, r0, r1 * pol_i[i], rc)
                ev += pol_i[i] * cv
                row[i] += cf * cv[pl]
            return ev

        ev_root = rec(self.root, 1.0, 1.0, 1.0)
        q = []
        for iid in range(n):
            d = qden[iid]
            q.append(qnum[iid] / d if d > 1e-15 else np.zeros(len(self.iset_actions[iid])))
        return q, qnum, qden, ev_root

    # ------------------------------------------------------------------ CFR+ (full game)
    def cfr_plus(self, iters, eval_every=0, averaging="uniform"):
        """CFR+ over the full tree with ALTERNATING updates (the canonical formulation). Strategy
        averaged by OWN reach, with per-iteration weight ``w_t``: ``averaging="uniform"`` -> w_t=1
        (pinned convention, matches SampleLeaf uniform weighting; must-fix #2); ``averaging="linear"``
        -> w_t=t (the standard CFR+ weighting, kept as a faster correctness probe). Returns
        (avg_policy, history=[(t, nashconv), ...]). The AVERAGE policy is the convergent object
        (last-iterate of CFR+ does not converge)."""
        regret = [np.zeros(len(a)) for a in self.iset_actions]
        stratsum = [np.zeros(len(a)) for a in self.iset_actions]
        hist = []

        def sigma_of():
            sig = []
            for r in regret:
                pos = np.maximum(r, 0.0); s = pos.sum()
                sig.append(pos / s if s > 1e-15 else np.ones_like(r) / len(r))
            return sig

        for t in range(iters):
            upd = t % 2  # alternating: update this player's regrets/average this iteration
            sig = sigma_of()
            cfvnum = [np.zeros(len(a)) for a in self.iset_actions]
            ownreach = [0.0] * self.n_iset

            def walk(node, r0, r1, rc):
                ty = node[0]
                if ty == "term":
                    return node[1]
                if ty == "deal":
                    ev = np.zeros(2)
                    for _a, p, ch in node[1]:
                        ev += p * walk(ch, r0, r1, rc * p)
                    return ev
                if ty == "board":
                    ev = np.zeros(2)
                    for _a, p, ch in node[4]:
                        ev += p * walk(ch, r0, r1, rc * p)
                    return ev
                _, pl, iid, _rnd, _c0, _c1, _b, kids = node
                s = sig[iid]
                ownreach[iid] = r0 if pl == 0 else r1  # own reach (identical across instances)
                ev = np.zeros(2)
                cf = (r1 * rc) if pl == 0 else (r0 * rc)
                row = cfvnum[iid]
                for i, (a, ch) in enumerate(kids):
                    cv = walk(ch, r0 * s[i], r1, rc) if pl == 0 else walk(ch, r0, r1 * s[i], rc)
                    ev += s[i] * cv
                    row[i] += cf * cv[pl]
                return ev

            walk(self.root, 1.0, 1.0, 1.0)
            w = float(t + 1) if averaging == "linear" else 1.0
            for iid in range(self.n_iset):
                if self.iset_player[iid] != upd:
                    continue
                s = sig[iid]
                v = float(np.dot(s, cfvnum[iid]))
                regret[iid] = np.maximum(regret[iid] + (cfvnum[iid] - v), 0.0)
                stratsum[iid] += w * ownreach[iid] * s

            if eval_every and ((t + 1) % eval_every == 0 or t == iters - 1):
                avg = self._avg(stratsum)
                hist.append((t + 1, self.nash_conv(avg)))

        avg = self._avg(stratsum)
        if not hist:
            hist.append((iters, self.nash_conv(avg)))
        return avg, hist

    @staticmethod
    def _avg(stratsum):
        out = []
        for ss in stratsum:
            s = ss.sum()
            out.append(ss / s if s > 1e-15 else np.ones_like(ss) / len(ss))
        return out

    # ------------------------------------------------------------------ subtree EV + leaf splice
    def subtree_ev(self, node, pol):
        """Exact EV 2-vector of the subtree rooted at ``node`` under profile ``pol`` (for a board
        node: EV of board-deal + round-2 play, per its fixed (c0,c1))."""
        t = node[0]
        if t == "term":
            return node[1]
        if t == "deal":
            ev = np.zeros(2)
            for _a, p, ch in node[1]:
                ev += p * self.subtree_ev(ch, pol)
            return ev
        if t == "board":
            ev = np.zeros(2)
            for _a, p, ch in node[4]:
                ev += p * self.subtree_ev(ch, pol)
            return ev
        _, _pl, iid, _rnd, _c0, _c1, _b, kids = node
        pol_i = pol[iid]
        ev = np.zeros(2)
        for i, (_a, ch) in enumerate(kids):
            ev += pol_i[i] * self.subtree_ev(ch, pol)
        return ev

    def values_with_leaf(self, pol, leaf_v):
        """Same belief-weighted q pass as ``values``, but the depth-limit CUT (board nodes) is
        replaced by the leaf evaluator: at a board node with cards (c0,c1) and public key K, the
        continuation value 2-vector is taken to be ``[leaf_v[K][0][c0], leaf_v[K][1][c1]]`` instead
        of recursing. Returns q[iid][a_pos] for ROUND-1 infosets only (round-2 is below the cut).

        This is the trunk-side CONSUMPTION of the leaf values; the round-1 q it produces equals the
        exact ``values`` q iff the leaf values use the pinned normalization convention.
        ``leaf_v[K]`` = (v0_arr[NCARDS], v1_arr[NCARDS])."""
        n = self.n_iset
        qnum = [np.zeros(len(a)) for a in self.iset_actions]
        qden = [0.0] * n

        def rec(node, r0, r1, rc):
            t = node[0]
            if t == "term":
                return node[1]
            if t == "deal":
                ev = np.zeros(2)
                for _a, p, ch in node[1]:
                    ev += p * rec(ch, r0, r1, rc * p)
                return ev
            if t == "board":
                _, key, c0, c1, _kids = node
                v0, v1 = leaf_v[key]
                return np.array([v0[c0], v1[c1]], dtype=np.float64)
            _, pl, iid, _rnd, _c0, _c1, _b, kids = node
            pol_i = pol[iid]
            cf = (r1 * rc) if pl == 0 else (r0 * rc)
            qden[iid] += cf
            ev = np.zeros(2)
            row = qnum[iid]
            for i, (a, ch) in enumerate(kids):
                cv = rec(ch, r0 * pol_i[i], r1, rc) if pl == 0 else rec(ch, r0, r1 * pol_i[i], rc)
                ev += pol_i[i] * cv
                row[i] += cf * cv[pl]
            return ev

        rec(self.root, 1.0, 1.0, 1.0)
        q = []
        for iid in range(n):
            d = qden[iid]
            q.append(qnum[iid] / d if d > 1e-15 else np.zeros(len(self.iset_actions[iid])))
        return q

    # ------------------------------------------------------------------ cut structure (PBS)
    def cut_keys(self):
        """Set of public cut keys (round-1 betting sequences that reach the board deal)."""
        return set(self.cut_instances().keys())

    def cut_instances(self):
        """dict: public cut key -> list of (c0, c1, board_node). One board node per (c0,c1) pair
        that reaches the cut (the scalar instances that share this public belief state)."""
        out: dict[tuple, list] = {}

        def rec(node):
            t = node[0]
            if t == "term":
                return
            if t == "deal":
                for _a, _p, ch in node[1]:
                    rec(ch)
            elif t == "board":
                out.setdefault(node[1], []).append((node[2], node[3], node))
            else:
                for _a, ch in node[7]:
                    rec(ch)

        rec(self.root)
        return out

    def cut_reaches(self, pol):
        """dict: public cut key -> (reach0[NCARDS], reach1[NCARDS]) where reach_i[c] is player i's
        reach to the cut holding card c, INCLUDING the deal chance (rc) -- i.e. the un-normalized
        per-card belief mass arriving at the cut. (Per (key, c), this reach is well-defined: own
        strategy reach depends only on the player's own card and the public history.)"""
        reaches = {k: [np.zeros(NCARDS), np.zeros(NCARDS)] for k in self.cut_keys()}

        def rec(node, r0, r1, rc):
            t = node[0]
            if t == "term":
                return
            if t == "deal":
                for _a, p, ch in node[1]:
                    rec(ch, r0, r1, rc * p)
                return
            if t == "board":
                _, key, c0, c1, _kids = node
                reaches[key][0][c0] = r0 * rc   # P0 reach (incl chance) at this cut holding c0
                reaches[key][1][c1] = r1 * rc   # P1 reach (incl chance) at this cut holding c1
                return
            _, pl, iid, _rnd, _c0, _c1, _b, kids = node
            pol_i = pol[iid]
            for i, (_a, ch) in enumerate(kids):
                if pl == 0:
                    rec(ch, r0 * pol_i[i], r1, rc)
                else:
                    rec(ch, r0, r1 * pol_i[i], rc)

        rec(self.root, 1.0, 1.0, 1.0)
        return reaches
