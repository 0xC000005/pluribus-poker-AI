"""Generic depth-limited solving substrate over an OpenSpiel game (Build 3b-i): the game-agnostic
generalization of leduc.py/loop.py/leaf_eval.py's depth-limited machinery.

A ``DepthLimitedGame`` enumerates the game once into a tagged tree (terminal / chance / decision / CUT)
where CUT nodes (``is_cut_fn`` -- the one game-specific hook) carry the public-state key + each player's
PRIVATE index and retain the continuation subtree. On it:

  * ``full_values`` -- exact belief-weighted action values for ALL infosets (no cut), the ground truth;
  * ``oracle_leaf_v`` -- the EXACT leaf oracle in the PINNED normalized PBS convention
        ``v_i(priv) = sum_{opp} oppreach(opp) * EV_i(priv,opp) / sum_{opp} oppreach(opp)``
    (the generalization of ExactLeafOracle);
  * ``trunk_q_with_leaf`` -- the depth-limited trunk value pass that CONSUMES those leaf values at cuts.

The CONVENTION ROUND-TRIP gate (the Stage-0 correctness check, generalized + game-agnostic): valuing
the trunk through the leaf interface reproduces the exact full-game above-cut values to ~1e-15 with the
normalized convention, and BREAKS (large error) with the un-normalized one. This validates the generic
public/private indexing + leaf-value convention end-to-end before any net. Slumbot held-out.
"""
from __future__ import annotations

import numpy as np
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability

from poker_ai.rebel.iig_pbs import make_public_key_fn


class DepthLimitedGame:
    def __init__(self, game, is_cut_fn, public_key_fn=None):
        self.game = game
        self.is_cut = is_cut_fn
        self.public_key_fn = public_key_fn or make_public_key_fn(game)
        if self.public_key_fn is None:
            raise ValueError("no OpenSpiel public observer for this game; pass public_key_fn")
        self.infosets: dict = {}
        self.iset_player: list[int] = []
        self.iset_actions: list[list[int]] = []
        self.iset_above_cut: list[bool] = []
        self.iset_key: list[str] = []    # iid -> OpenSpiel information_state_string
        self.priv_index: dict = {}      # (public_key, player) -> {priv_key: idx}
        self.cut_nodes: list = []        # cut node tuples (for per-key oracle evaluation)
        self.root = self._build(game.new_initial_state(), above_cut=True)

    def _iset_id(self, key, player, acts, above):
        if key not in self.infosets:
            self.infosets[key] = len(self.iset_player)
            self.iset_player.append(player)
            self.iset_actions.append(list(acts))
            self.iset_above_cut.append(above)
            self.iset_key.append(key)
        return self.infosets[key]

    def _priv_idx(self, key, player, priv_key):
        m = self.priv_index.setdefault((key, player), {})
        if priv_key not in m:
            m[priv_key] = len(m)
        return m[priv_key]

    def _build(self, state, above_cut):
        if state.is_terminal():
            return ("term", np.asarray(state.returns(), dtype=np.float64))
        if above_cut and self.is_cut(state):
            key = self.public_key_fn(state)
            i0 = self._priv_idx(key, 0, state.information_state_string(0))
            i1 = self._priv_idx(key, 1, state.information_state_string(1))
            sub = self._build(state, above_cut=False)   # continuation subtree (for the oracle)
            node = ("cut", key, i0, i1, sub)
            self.cut_nodes.append(node)
            return node
        if state.is_chance_node():
            kids = [(float(p), self._build(_apply(state, a), above_cut))
                    for a, p in state.chance_outcomes()]
            return ("chance", kids)
        pl = state.current_player()
        acts = state.legal_actions()
        iid = self._iset_id(state.information_state_string(pl), pl, acts, above_cut)
        kids = [(a, self._build(_apply(state, a), above_cut)) for a in acts]
        return ("dec", pl, iid, kids)

    @property
    def n_iset(self):
        return len(self.iset_player)

    def n_priv(self, key, player):
        return len(self.priv_index.get((key, player), {}))

    def random_policy(self, rng):
        pol = []
        for a in self.iset_actions:
            x = rng.random(len(a)) + 1e-3
            pol.append(x / x.sum())
        return pol

    def uniform_policy(self):
        return [np.ones(len(a)) / len(a) for a in self.iset_actions]

    # ---- exact full-game belief-weighted action values (ground truth) ----
    def full_values(self, pol):
        qnum: dict = {}
        qden: dict = {}

        def rec(node, r0, r1, rc):
            t = node[0]
            if t == "term":
                return node[1]
            if t == "chance":
                ev = np.zeros(2)
                for p, ch in node[1]:
                    ev += p * rec(ch, r0, r1, rc * p)
                return ev
            if t == "cut":
                return rec(node[4], r0, r1, rc)   # passthrough: full game
            _, pl, iid, kids = node
            s = pol[iid]
            cf = (r1 * rc) if pl == 0 else (r0 * rc)
            qden[iid] = qden.get(iid, 0.0) + cf
            ev = np.zeros(2)
            row = qnum.setdefault(iid, np.zeros(len(kids)))
            for i, (_a, ch) in enumerate(kids):
                cv = rec(ch, r0 * s[i], r1, rc) if pl == 0 else rec(ch, r0, r1 * s[i], rc)
                ev += s[i] * cv
                row[i] += cf * cv[pl]
            return ev

        rec(self.root, 1.0, 1.0, 1.0)
        return {i: qnum[i] / qden[i] for i in qnum if qden[i] > 1e-15}

    def subtree_ev(self, node, pol):
        t = node[0]
        if t == "term":
            return node[1]
        if t == "chance":
            ev = np.zeros(2)
            for p, ch in node[1]:
                ev += p * self.subtree_ev(ch, pol)
            return ev
        if t == "cut":
            return self.subtree_ev(node[4], pol)
        _, _pl, iid, kids = node
        s = pol[iid]
        ev = np.zeros(2)
        for i, (_a, ch) in enumerate(kids):
            ev += s[i] * self.subtree_ev(ch, pol)
        return ev

    def cut_reaches(self, pol):
        """Per cut public state, each player's per-private incoming reach (incl. chance), SET per
        (public, private) instance -- the same convention as LeducTree.cut_reaches."""
        R: dict = {}

        def rec(node, r0, r1, rc):
            t = node[0]
            if t == "term":
                return
            if t == "chance":
                for p, ch in node[1]:
                    rec(ch, r0, r1, rc * p)
                return
            if t == "cut":
                key, i0, i1 = node[1], node[2], node[3]
                d = R.setdefault(key, (np.zeros(self.n_priv(key, 0)), np.zeros(self.n_priv(key, 1))))
                d[0][i0] = r0 * rc
                d[1][i1] = r1 * rc
                return
            _, pl, iid, kids = node
            s = pol[iid]
            for i, (_a, ch) in enumerate(kids):
                if pl == 0:
                    rec(ch, r0 * s[i], r1, rc)
                else:
                    rec(ch, r0, r1 * s[i], rc)

        rec(self.root, 1.0, 1.0, 1.0)
        return R

    def oracle_leaf_v(self, pol, normalize=True):
        """Exact leaf values per cut public state in the pinned normalized PBS convention (or the
        un-normalized counterfactual value if normalize=False -- the negative control)."""
        reaches = self.cut_reaches(pol)
        by_key: dict = {}
        for node in self.cut_nodes:
            by_key.setdefault(node[1], []).append(node)
        out = {}
        for key, nodes in by_key.items():
            n0, n1 = self.n_priv(key, 0), self.n_priv(key, 1)
            r0, r1 = reaches[key]
            v0n = np.zeros(n0); v0d = np.zeros(n0); v1n = np.zeros(n1); v1d = np.zeros(n1)
            for _t, _k, i0, i1, sub in nodes:
                cont = self.subtree_ev(sub, pol)
                v0n[i0] += r1[i1] * cont[0]; v0d[i0] += r1[i1]
                v1n[i1] += r0[i0] * cont[1]; v1d[i1] += r0[i0]
            if normalize:
                v0 = np.divide(v0n, v0d, out=np.zeros(n0), where=v0d > 1e-15)
                v1 = np.divide(v1n, v1d, out=np.zeros(n1), where=v1d > 1e-15)
            else:
                v0, v1 = v0n, v1n
            out[key] = (v0, v1)
        return out

    def trunk_q_with_leaf(self, pol, leaf_v):
        """Depth-limited trunk value pass: at a cut, consume leaf_v[key] = (v0[priv], v1[priv]) as the
        continuation value instead of recursing. Returns above-cut infoset action values q."""
        qnum: dict = {}
        qden: dict = {}

        def rec(node, r0, r1, rc):
            t = node[0]
            if t == "term":
                return node[1]
            if t == "chance":
                ev = np.zeros(2)
                for p, ch in node[1]:
                    ev += p * rec(ch, r0, r1, rc * p)
                return ev
            if t == "cut":
                key, i0, i1 = node[1], node[2], node[3]
                v0, v1 = leaf_v[key]
                return np.array([v0[i0], v1[i1]], dtype=np.float64)
            _, pl, iid, kids = node
            s = pol[iid]
            cf = (r1 * rc) if pl == 0 else (r0 * rc)
            qden[iid] = qden.get(iid, 0.0) + cf
            ev = np.zeros(2)
            row = qnum.setdefault(iid, np.zeros(len(kids)))
            for i, (_a, ch) in enumerate(kids):
                cv = rec(ch, r0 * s[i], r1, rc) if pl == 0 else rec(ch, r0, r1 * s[i], rc)
                ev += s[i] * cv
                row[i] += cf * cv[pl]
            return ev

        rec(self.root, 1.0, 1.0, 1.0)
        return {i: qnum[i] / qden[i] for i in qnum if qden[i] > 1e-15}

    def _trunk_cfv(self, pol, leaf_v):
        """One trunk value pass returning un-normalized cf-weighted action values + own-reach for the
        above-cut infosets (the CFR+ accumulator; cuts consume leaf_v)."""
        cfvnum: dict = {}
        ownreach: dict = {}

        def rec(node, r0, r1, rc):
            t = node[0]
            if t == "term":
                return node[1]
            if t == "chance":
                ev = np.zeros(2)
                for p, ch in node[1]:
                    ev += p * rec(ch, r0, r1, rc * p)
                return ev
            if t == "cut":
                key, i0, i1 = node[1], node[2], node[3]
                v0, v1 = leaf_v[key]
                return np.array([v0[i0], v1[i1]], dtype=np.float64)
            _, pl, iid, kids = node
            s = pol[iid]
            ownreach[iid] = r0 if pl == 0 else r1
            cf = (r1 * rc) if pl == 0 else (r0 * rc)
            row = cfvnum.setdefault(iid, np.zeros(len(kids)))
            ev = np.zeros(2)
            for i, (_a, ch) in enumerate(kids):
                cv = rec(ch, r0 * s[i], r1, rc) if pl == 0 else rec(ch, r0, r1 * s[i], rc)
                ev += s[i] * cv
                row[i] += cf * cv[pl]
            return ev

        rec(self.root, 1.0, 1.0, 1.0)
        return cfvnum, ownreach

    def trunk_solve(self, leaf_fn, iters=400, averaging="linear"):
        """Depth-limited CFR+ on the trunk (above-cut infosets) with a FIXED leaf VALUE FUNCTION
        ``leaf_fn(public_key, range0, range1) -> (v0[priv], v1[priv])`` queried at the current
        iterate's per-private reaches. The generic twin of loop.py's trunk_solve -- both the exact
        oracle and a learned PBS net plug in through ``leaf_fn``. Returns {iid: avg strategy} for the
        above-cut infosets."""
        above = [i for i in range(self.n_iset) if self.iset_above_cut[i]]
        regret = {i: np.zeros(len(self.iset_actions[i])) for i in above}
        stratsum = {i: np.zeros(len(self.iset_actions[i])) for i in above}
        keys = sorted({n[1] for n in self.cut_nodes})

        for t in range(iters):
            upd = t % 2
            sig = self.uniform_policy()
            for i in above:
                pos = np.maximum(regret[i], 0.0); ss = pos.sum()
                sig[i] = pos / ss if ss > 1e-15 else np.ones_like(pos) / len(pos)
            reaches = self.cut_reaches(sig)
            leaf_v = {k: leaf_fn(k, reaches[k][0], reaches[k][1]) for k in keys}
            cfvnum, ownreach = self._trunk_cfv(sig, leaf_v)
            w = float(t + 1) if averaging == "linear" else 1.0
            for i in above:
                if self.iset_player[i] != upd:
                    continue
                s = sig[i]
                v = float(np.dot(s, cfvnum[i]))
                regret[i] = np.maximum(regret[i] + (cfvnum[i] - v), 0.0)
                stratsum[i] += w * ownreach[i] * s

        out = {}
        for i in above:
            ss = stratsum[i].sum()
            out[i] = stratsum[i] / ss if ss > 1e-15 else np.ones(len(stratsum[i])) / len(stratsum[i])
        return out

    def blueprint_leaf_fn(self, ref_pol):
        """A consistent FIXED-strategy exact leaf: the normalized PBS value of continuing below the cut
        with a fixed reference profile ``ref_pol`` (the generic analog of loop.blueprint_leaf_fn). Cheap
        (no per-range re-solve) and a valid ``leaf_fn`` for trunk_solve."""
        by_key: dict = {}
        for node in self.cut_nodes:
            by_key.setdefault(node[1], []).append(node)

        def fn(key, range0, range1):
            n0, n1 = self.n_priv(key, 0), self.n_priv(key, 1)
            v0n = np.zeros(n0); v0d = np.zeros(n0); v1n = np.zeros(n1); v1d = np.zeros(n1)
            for _t, _k, i0, i1, sub in by_key[key]:
                cont = self.subtree_ev(sub, ref_pol)
                v0n[i0] += range1[i1] * cont[0]; v0d[i0] += range1[i1]
                v1n[i1] += range0[i0] * cont[1]; v1d[i1] += range0[i0]
            v0 = np.divide(v0n, v0d, out=np.zeros(n0), where=v0d > 1e-15)
            v1 = np.divide(v1n, v1d, out=np.zeros(n1), where=v1d > 1e-15)
            return v0, v1

        return fn

    def cfr_plus(self, iters, averaging="linear"):
        """Full-game CFR+ over the whole tree (cuts treated as passthrough) -- a near-equilibrium full
        policy (list over all infosets). Used to supply the fixed near-eq CONTINUATION strategy for
        self-play leaf targets (the below-cut/final-round play). Alternating updates, own-reach
        averaging."""
        regret = [np.zeros(len(a)) for a in self.iset_actions]
        stratsum = [np.zeros(len(a)) for a in self.iset_actions]

        def sigma_of():
            out = []
            for r in regret:
                pos = np.maximum(r, 0.0); s = pos.sum()
                out.append(pos / s if s > 1e-15 else np.ones_like(r) / len(r))
            return out

        for t in range(iters):
            upd = t % 2
            sig = sigma_of()
            cfvnum = [np.zeros(len(a)) for a in self.iset_actions]
            ownreach = [0.0] * self.n_iset

            def walk(node, r0, r1, rc):
                ty = node[0]
                if ty == "term":
                    return node[1]
                if ty == "chance":
                    ev = np.zeros(2)
                    for p, ch in node[1]:
                        ev += p * walk(ch, r0, r1, rc * p)
                    return ev
                if ty == "cut":
                    return walk(node[4], r0, r1, rc)
                _, pl, iid, kids = node
                s = sig[iid]
                ownreach[iid] = r0 if pl == 0 else r1
                cf = (r1 * rc) if pl == 0 else (r0 * rc)
                row = cfvnum[iid]
                ev = np.zeros(2)
                for i, (_a, ch) in enumerate(kids):
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

        out = []
        for ss in stratsum:
            s = ss.sum()
            out.append(ss / s if s > 1e-15 else np.ones_like(ss) / len(ss))
        return out

    # ---- exploitability of an assembled full strategy (via OpenSpiel NashConv) ----
    def to_tabular(self, pol):
        tp = policy_lib.TabularPolicy(self.game)
        for iid, key in enumerate(self.iset_key):
            if key in tp.state_lookup:
                row = tp.action_probability_array[tp.state_lookup[key]]
                row[:] = 0.0
                for i, a in enumerate(self.iset_actions[iid]):
                    row[a] = pol[iid][i]
        return tp

    def nash_conv(self, pol):
        return float(exploitability.nash_conv(self.game, self.to_tabular(pol)))

    def assemble(self, sigma1, cont_pol):
        """Full policy = trunk strategy ``sigma1`` (dict {iid: arr} over above-cut infosets) spliced
        onto the below-cut continuation ``cont_pol`` (full list)."""
        pol = list(cont_pol)
        for iid, pr in sigma1.items():
            pol[iid] = pr
        return pol

    def roundtrip_error(self, pol, normalize=True):
        """Max |full-game q - depth-limited q| over above-cut infosets (the convention round-trip)."""
        qf = self.full_values(pol)
        qt = self.trunk_q_with_leaf(pol, self.oracle_leaf_v(pol, normalize=normalize))
        err = 0.0
        for i in qt:                       # qt has exactly the above-cut infosets
            err = max(err, float(np.abs(qf[i] - qt[i]).max()))
        return err


def _apply(state, action):
    s = state.clone()
    s.apply_action(action)
    return s
