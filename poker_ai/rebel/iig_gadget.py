"""Generic DeepStack-style SAFE re-solving gadget (EA critical-path piece) -- the game-agnostic
generalization of loop.gadget_resolve (tested on Leduc: cut exploitability 0.21 -> 0.0079, 26.5x).

The fixed-continuation depth-limited agent is OFF-PATH exploitable at scale (Goofspiel-5: 0.38-0.74 exact)
because its below-cut continuation, solved at one belief, is suboptimal when an exploiter deviates above
the cut to reach it with a different range. Safe re-solving fixes this: when re-solving the subgame at a
public state for ``resolver``, give the OPPONENT a per-private FOLLOW/TERMINATE choice (terminate -> its
blueprint counterfactual value opp_cfv), so the resolver's strategy provably cannot let the opponent
exceed its blueprint CFVs -> not off-path exploitable. opp_cfv = the opponent's normalized per-private
leaf value at the cut from the trunk/blueprint (DepthLimitedGame.oracle_leaf_v). Slumbot held-out.
"""
from __future__ import annotations

import numpy as np


def _rm_plus(r):
    pos = np.maximum(r, 0.0)
    s = pos.sum()
    return pos / s if s > 1e-15 else np.ones_like(r) / len(r)


def gadget_resolve(dlg, key, resolver, range_resolver, range_opp, opp_cfv, iters=2000, averaging="linear"):
    """Safe re-solve of the below-cut subgame at public ``key`` for ``resolver`` (0 or 1), constraining the
    opponent to its blueprint CFVs ``opp_cfv`` (per opponent private, normalized convention). Returns the
    resolver's average below-cut strategy {iid: prob_array}. ``range_resolver``/``range_opp`` are per-private
    entry reaches at the cut (incl. chance)."""
    opp = 1 - resolver
    nodes = [n for n in dlg.cut_nodes if n[1] == key]            # ("cut", key, i0, i1, sub)
    iids = dlg._below_iids_for_key(key)
    res_iids = [i for i in iids if dlg.iset_player[i] == resolver]
    opp_iids = [i for i in iids if dlg.iset_player[i] == opp]
    n_opp = dlg.n_priv(key, opp)
    regret = {i: np.zeros(len(dlg.iset_actions[i])) for i in iids}
    stratsum = {i: np.zeros(len(dlg.iset_actions[i])) for i in res_iids}
    g_reg = np.zeros((n_opp, 2))                                  # opp per-private {follow, terminate}

    def g_follow():
        out = np.zeros(n_opp)
        for c in range(n_opp):
            pos = np.maximum(g_reg[c], 0.0)
            s = pos.sum()
            out[c] = (pos[0] / s) if s > 1e-15 else 0.5
        return out

    for t in range(iters):
        sig = {i: _rm_plus(regret[i]) for i in iids}
        gf = g_follow()
        cfvnum = {i: np.zeros(len(dlg.iset_actions[i])) for i in iids}
        opp_num = np.zeros(n_opp)
        opp_den = np.zeros(n_opp)

        def walk(node, r0, r1, rc):
            ty = node[0]
            if ty == "term":
                return node[1]
            if ty == "chance":
                ev = np.zeros(2)
                for p, ch in node[1]:
                    ev += p * walk(ch, r0, r1, rc * p)
                return ev
            _, pl, iid, kids = node
            s = sig[iid]
            ev = np.zeros(2)
            cf = (r1 * rc) if pl == 0 else (r0 * rc)
            row = cfvnum[iid]
            for i, (_a, ch) in enumerate(kids):
                cv = walk(ch, r0 * s[i], r1, rc) if pl == 0 else walk(ch, r0, r1 * s[i], rc)
                ev += s[i] * cv
                row[i] += cf * cv[pl]
            return ev

        for _t, _k, i0, i1, sub in nodes:
            res_idx, opp_idx = (i0, i1) if resolver == 0 else (i1, i0)
            rr = range_resolver[res_idx]
            ro = range_opp[opp_idx] * gf[opp_idx]
            r0, r1 = (rr, ro) if resolver == 0 else (ro, rr)
            ev = walk(sub, r0, r1, 1.0)
            opp_num[opp_idx] += rr * ev[opp]      # opp's FOLLOW value (resolver-range-weighted)
            opp_den[opp_idx] += rr

        w = float(t + 1) if averaging == "linear" else 1.0
        for i in res_iids:
            s = sig[i]; v = float(np.dot(s, cfvnum[i]))
            regret[i] = np.maximum(regret[i] + (cfvnum[i] - v), 0.0)
            stratsum[i] += w * s
        for i in opp_iids:
            s = sig[i]; v = float(np.dot(s, cfvnum[i]))
            regret[i] = np.maximum(regret[i] + (cfvnum[i] - v), 0.0)
        for c in range(n_opp):                    # opp gadget regret: follow=fval, terminate=opp_cfv
            if opp_den[c] <= 1e-15:
                continue
            fval = opp_num[c] / opp_den[c]
            tval = opp_cfv[c]
            ev_g = gf[c] * fval + (1.0 - gf[c]) * tval
            g_reg[c, 0] = max(g_reg[c, 0] + (fval - ev_g), 0.0)
            g_reg[c, 1] = max(g_reg[c, 1] + (tval - ev_g), 0.0)

    avg = {}
    for i in res_iids:
        ss = stratsum[i].sum()
        avg[i] = stratsum[i] / ss if ss > 1e-15 else np.ones(len(stratsum[i])) / len(stratsum[i])
    return avg


def safe_continuation(dlg, sigma1, blueprint, iters=2000):
    """Assemble a SAFE below-cut continuation (both players) via the gadget, given the trunk strategy
    ``sigma1`` and a ``blueprint`` near-eq full policy (for the entry reaches + opponent CFVs). Returns a
    full policy = trunk sigma1 + both players' safe gadget continuations. Exploitability of this is the
    safe continual-resolving agent's exploitability."""
    keys = sorted({n[1] for n in dlg.cut_nodes})
    full = dlg.assemble(sigma1, blueprint)            # start from blueprint below cut; overwrite via gadget
    reaches = dlg.cut_reaches(full)
    cfv = dlg.oracle_leaf_v(full, normalize=True)      # per key (v0, v1) normalized = blueprint CFVs
    for key in keys:
        r0, r1 = reaches[key]
        for resolver in (0, 1):
            opp = 1 - resolver
            rr, ro = (r0, r1) if resolver == 0 else (r1, r0)
            cont = gadget_resolve(dlg, key, resolver, rr, ro, cfv[key][opp], iters=iters)
            for iid, pr in cont.items():
                full[iid] = pr
    return full
