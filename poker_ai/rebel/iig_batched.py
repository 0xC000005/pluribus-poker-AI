"""Batched depth-limited PBS subgame solver (the scale-pivot PORT, Increment 1) -- the GPU-resident,
CROSS-KEY batched generalization of the per-key serial numpy ``DepthLimitedGame.solve_subgame_equilibrium``.

Premise validated across GATE 0/0b/0c/0d (RESEARCH_LOG 2026-06-07): all cut-node continuation subtrees in
a game share ONE topology (Probe A), batching tiny same-topology subgames gives ~1000x throughput (0b),
the batched CFR+ kernel is EXACT on real chance/variable-depth/imperfect-info structure at float64 (0c),
and it integrates into ``trunk_solve`` preserving the equilibrium (0d). Those gates batched only WITHIN one
key's cut nodes; this module batches ALL cut nodes across ALL public keys into a single solve (the
cross-key batching where the throughput win lives), over the union of all keys' below-cut infosets.

CONTRACT: ``solve_all_keys(dlg, ranges_by_key, iters)`` returns ``{key: {iid: avg strategy}}`` matching
per-key ``solve_subgame_equilibrium`` to float64 precision. float64 regret accumulation is MANDATORY
(0c: float32 drifts ~0.75 L1 at imperfect-info repeated infosets). Slumbot held-out; diagnostic infra.

Increment 1 (this file) reuses the proven GATE-0c lockstep tree walk for CORRECTNESS + cross-key batching.
The level-grouped flat-SoA kernel that removes the per-iteration Python walk (the G2 throughput target) is
``BatchedSubgameCompiler`` below, built + parity-gated against this reference.
"""
from __future__ import annotations

import numpy as np
import torch

_DT = torch.float64


def _regret_match(regret, mask):
    pos = (regret * mask).clamp_min(0.0)
    s = pos.sum(-1, keepdim=True)
    uni = mask / mask.sum(-1, keepdim=True).clamp_min(1.0)
    return torch.where(s > 1e-12, pos / s, uni)


def _build_global(dlg, keys):
    """Union of below-cut infosets across keys -> global local indexing + per-iid (player, legal len)."""
    iids = []
    for key in keys:
        iids.extend(dlg._below_iids_for_key(key))
    iids = sorted(set(iids))
    loc = {iid: k for k, iid in enumerate(iids)}
    alen = [len(dlg.iset_actions[i]) for i in iids]
    return iids, loc, alen


def solve_all_keys(dlg, ranges_by_key, iters, device="cuda", averaging="linear"):
    """CROSS-KEY batched CFR+ equilibrium: solve every public key's below-cut subgame at once, batched over
    all cut nodes across all keys (they share one topology). ``ranges_by_key`` = {key: (range0, range1)}.
    Returns {key: {iid: avg strategy (np)}}. Reference (proven-correct) batched solver."""
    keys = list(ranges_by_key)
    nodes, node_key = [], []
    for key in keys:
        for nd in dlg.cut_nodes:
            if nd[1] == key:
                nodes.append(nd); node_key.append(key)
    iids, loc, alen = _build_global(dlg, keys)
    n = len(iids); maxA = max(alen) if alen else 1
    player = torch.tensor([dlg.iset_player[i] for i in iids], device=device)
    mask = torch.zeros(n, maxA, dtype=_DT, device=device)
    for k, L in enumerate(alen):
        mask[k, :L] = 1.0

    B = len(nodes)
    r0_entry = torch.tensor([float(ranges_by_key[node_key[b]][0][nodes[b][2]]) for b in range(B)],
                            dtype=_DT, device=device)
    r1_entry = torch.tensor([float(ranges_by_key[node_key[b]][1][nodes[b][3]]) for b in range(B)],
                            dtype=_DT, device=device)
    subs = [nd[4] for nd in nodes]

    regret = torch.zeros(n, maxA, dtype=_DT, device=device)
    stratsum = torch.zeros(n, maxA, dtype=_DT, device=device)

    for t in range(iters):
        upd = t % 2
        sig = _regret_match(regret, mask)
        cfvnum = torch.zeros(n, maxA, dtype=_DT, device=device)
        ownreach = torch.zeros(n, dtype=_DT, device=device)

        def walk(nodes_b, r0_b, r1_b, rc_b):
            ty = nodes_b[0][0]
            if ty == "term":
                return torch.tensor(np.stack([nd[1] for nd in nodes_b]), dtype=_DT, device=device)
            if ty == "chance":
                ev = torch.zeros(len(nodes_b), 2, dtype=_DT, device=device)
                for ci in range(len(nodes_b[0][1])):
                    p_b = torch.tensor([float(nd[1][ci][0]) for nd in nodes_b], dtype=_DT, device=device)
                    ch_b = [nd[1][ci][1] for nd in nodes_b]
                    ev = ev + p_b.unsqueeze(1) * walk(ch_b, r0_b, r1_b, rc_b * p_b)
                return ev
            pl = nodes_b[0][1]; n_act = len(nodes_b[0][3])
            idx = torch.tensor([loc[nd[2]] for nd in nodes_b], device=device)
            sig_b = sig[idx][:, :n_act]
            ownreach[idx] = (r0_b if pl == 0 else r1_b)
            cf_b = (r1_b if pl == 0 else r0_b) * rc_b
            ev = torch.zeros(len(nodes_b), 2, dtype=_DT, device=device)
            cfv_b = torch.zeros(len(nodes_b), n_act, dtype=_DT, device=device)
            for k in range(n_act):
                ch_b = [nd[3][k][1] for nd in nodes_b]
                cv = (walk(ch_b, r0_b * sig_b[:, k], r1_b, rc_b) if pl == 0
                      else walk(ch_b, r0_b, r1_b * sig_b[:, k], rc_b))
                ev = ev + sig_b[:, k:k + 1] * cv
                cfv_b[:, k] = cf_b * cv[:, pl]
            pad = torch.zeros(len(nodes_b), maxA, dtype=_DT, device=device)
            pad[:, :n_act] = cfv_b
            cfvnum.index_add_(0, idx, pad)
            return ev

        walk(subs, r0_entry, r1_entry, torch.ones(B, dtype=_DT, device=device))
        w = float(t + 1) if averaging == "linear" else 1.0
        is_upd = (player == upd).to(_DT).unsqueeze(1)
        v = (sig * cfvnum).sum(-1, keepdim=True)
        regret = (regret + is_upd * mask * (cfvnum - v)).clamp_min(0.0)
        stratsum = stratsum + w * is_upd * ownreach.unsqueeze(1) * sig

    ss = stratsum.sum(-1, keepdim=True)
    avg = torch.where(ss > 1e-12, stratsum / ss, mask / mask.sum(-1, keepdim=True).clamp_min(1.0))
    avg = avg.cpu().numpy()
    out = {key: {} for key in keys}
    by_iid_key = {}
    for key in keys:
        for iid in dlg._below_iids_for_key(key):
            by_iid_key[iid] = key
    for iid in iids:
        out[by_iid_key[iid]][iid] = avg[loc[iid]][:alen[loc[iid]]].copy()
    return out


# ----------------------------------------------------------------------------------------------------
# Level-grouped flat-SoA kernel (the G2 throughput target): compile the shared topology ONCE into
# level-ordered tensor arrays, then run CFR+ as forward (reach) + backward (value) passes that are
# O(n_levels) batched gather/scatter ops -- NO per-iteration Python tree recursion. Parity-gated against
# solve_all_keys above (the proven reference). float64.
# ----------------------------------------------------------------------------------------------------

class _Compiled:
    pass


def _compile_topology(dlg, ranges_by_key, device):
    keys = list(ranges_by_key)
    nodes, node_key = [], []
    for key in keys:
        for nd in dlg.cut_nodes:
            if nd[1] == key:
                nodes.append(nd); node_key.append(key)
    iids, loc, alen = _build_global(dlg, keys)
    B = len(nodes)
    r0_entry = np.array([float(ranges_by_key[node_key[b]][0][nodes[b][2]]) for b in range(B)])
    r1_entry = np.array([float(ranges_by_key[node_key[b]][1][nodes[b][3]]) for b in range(B)])

    recs = []  # per topology node

    def comp(nodes_b, level, parent, slot):
        nid = len(recs)
        rec = {"level": level, "parent": parent, "slot": slot, "children": []}
        recs.append(rec)
        t = nodes_b[0][0]
        if t == "term":
            rec["type"] = "term"
            rec["payoff"] = np.stack([nd[1] for nd in nodes_b]).astype(np.float64)  # [B,2]
        elif t == "chance":
            nb = len(nodes_b[0][1])
            rec["type"] = "chance"; rec["nbranch"] = nb
            rec["probs"] = np.array([[float(nd[1][ci][0]) for ci in range(nb)] for nd in nodes_b])  # [B,nb]
            rec["children"] = [comp([nd[1][ci][1] for nd in nodes_b], level + 1, nid, ci) for ci in range(nb)]
        else:
            pl = nodes_b[0][1]; nact = len(nodes_b[0][3])
            rec["type"] = "dec"; rec["player"] = pl; rec["nact"] = nact
            rec["iid"] = np.array([loc[nd[2]] for nd in nodes_b])  # [B]
            rec["children"] = [comp([nd[3][k][1] for nd in nodes_b], level + 1, nid, k) for k in range(nact)]
        return nid

    comp([nd[4] for nd in nodes], 0, -1, -1)

    c = _Compiled()
    c.keys = keys; c.iids = iids; c.loc = loc; c.alen = alen; c.B = B
    c.n_iids = len(iids); c.maxA = max(alen) if alen else 1
    c.device = device; c.n_nodes = len(recs); c.recs = recs
    c.player = torch.tensor([dlg.iset_player[i] for i in iids], device=device)
    c.mask = torch.zeros(c.n_iids, c.maxA, dtype=_DT, device=device)
    for k, L in enumerate(alen):
        c.mask[k, :L] = 1.0
    c.r0_entry = torch.tensor(r0_entry, dtype=_DT, device=device)
    c.r1_entry = torch.tensor(r1_entry, dtype=_DT, device=device)
    c.dlg = dlg
    c.node_key = node_key                       # for set_entries (reuse a compiled topology w/ new ranges)
    c.node_i0 = np.array([nd[2] for nd in nodes])
    c.node_i1 = np.array([nd[3] for nd in nodes])

    maxlevel = max(r["level"] for r in recs)
    # term payoffs: [n_nodes, B, 2] (only term rows used)
    c.payoff = torch.zeros(c.n_nodes, B, 2, dtype=_DT, device=device)
    for nid, r in enumerate(recs):
        if r["type"] == "term":
            c.payoff[nid] = torch.tensor(r["payoff"], dtype=_DT, device=device)

    # FORWARD groups: children at level L, grouped by parent kind (dec player 0/1, chance).
    # each group carries tensors to compute child reach from parent reach.
    c.fwd = []  # list of (level, kind, child_idx, parent_idx, [piid B] or [prob B], slot)
    for L in range(1, maxlevel + 1):
        for kind, pred in (("d0", lambda pr: pr["type"] == "dec" and pr["player"] == 0),
                           ("d1", lambda pr: pr["type"] == "dec" and pr["player"] == 1),
                           ("ch", lambda pr: pr["type"] == "chance")):
            child_idx, parent_idx, piid, prob, slot = [], [], [], [], []
            for nid, r in enumerate(recs):
                if r["level"] != L or r["parent"] < 0:
                    continue
                pr = recs[r["parent"]]
                if not pred(pr):
                    continue
                child_idx.append(nid); parent_idx.append(r["parent"]); slot.append(r["slot"])
                if kind in ("d0", "d1"):
                    piid.append(pr["iid"])           # [B]
                else:
                    prob.append(pr["probs"][:, r["slot"]])  # [B]
            if not child_idx:
                continue
            g = {"kind": kind,
                 "child": torch.tensor(child_idx, device=device),
                 "parent": torch.tensor(parent_idx, device=device),
                 "slot": torch.tensor(slot, device=device)}
            if kind in ("d0", "d1"):
                g["piid"] = torch.tensor(np.array(piid), device=device)        # [G,B]
            else:
                g["prob"] = torch.tensor(np.array(prob), dtype=_DT, device=device)  # [G,B]
            c.fwd.append(g)

    # BACKWARD groups: non-term nodes by level (desc), grouped by (type, player, nact/nbranch).
    c.bwd = []
    for L in range(maxlevel, -1, -1):
        # decision nodes grouped by (player, nact)
        shapes = {}
        for nid, r in enumerate(recs):
            if r["level"] != L:
                continue
            if r["type"] == "dec":
                shapes.setdefault(("dec", r["player"], r["nact"]), []).append(nid)
            elif r["type"] == "chance":
                shapes.setdefault(("chance", r["nbranch"]), []).append(nid)
        for shape, nids in shapes.items():
            children = np.array([recs[n]["children"] for n in nids])  # [G, nact|nbranch]
            g = {"node": torch.tensor(nids, device=device),
                 "children": torch.tensor(children, device=device)}
            if shape[0] == "dec":
                g["kind"] = "dec"; g["player"] = shape[1]; g["nact"] = shape[2]
                g["iid"] = torch.tensor(np.array([recs[n]["iid"] for n in nids]), device=device)  # [G,B]
            else:
                g["kind"] = "chance"; g["nbranch"] = shape[1]
                g["prob"] = torch.tensor(np.array([recs[n]["probs"] for n in nids]),
                                         dtype=_DT, device=device)  # [G,B,nbranch]
            c.bwd.append(g)
    return c


def set_entries(c, ranges_by_key):
    """Update a compiled topology's entry reaches from new per-key ranges (topology fixed; reuse the
    compile across self-play iterations instead of recompiling)."""
    r0 = np.array([float(ranges_by_key[c.node_key[b]][0][c.node_i0[b]]) for b in range(c.B)])
    r1 = np.array([float(ranges_by_key[c.node_key[b]][1][c.node_i1[b]]) for b in range(c.B)])
    c.r0_entry = torch.tensor(r0, dtype=_DT, device=c.device)
    c.r1_entry = torch.tensor(r1, dtype=_DT, device=c.device)


def _ev_pass(c, sig):
    """Value-only backward pass: continuation EV at every node under strategy ``sig`` ([n_iids,maxA]).
    Returns ev[root] = [B,2], the subgame value per cut node (the leaf continuation value)."""
    ev = c.payoff.clone()
    for g in c.bwd:
        ev_ch = ev[g["children"]]                              # [G,K,B,2]
        if g["kind"] == "chance":
            pr = g["prob"].permute(0, 2, 1).unsqueeze(-1)      # [G,K,B,1]
            ev[g["node"]] = (pr * ev_ch).sum(1)
        else:
            nact = g["nact"]
            sg = sig[g["iid"]][:, :, :nact].permute(0, 2, 1)   # [G,K,B]
            ev[g["node"]] = (sg.unsqueeze(-1) * ev_ch).sum(1)
    return ev[0]                                               # [B,2]


def solve_all_keys_soa(dlg, ranges_by_key, iters, device="cuda", averaging="linear", compiled=None,
                       return_cont=False):
    """Level-grouped flat-SoA cross-key batched CFR+ (no per-iteration Python recursion). Returns the same
    {key: {iid: avg}} as solve_all_keys. Pass ``compiled`` to reuse a compiled topology across solves.
    ``return_cont=True`` also returns cont[B,2] = the subgame continuation EV per cut node under the
    average strategy (the leaf value source -- avoids a numpy subtree walk)."""
    c = compiled or _compile_topology(dlg, ranges_by_key, device)
    if compiled is not None and ranges_by_key is not None:
        set_entries(c, ranges_by_key)
    B, n, maxA = c.B, c.n_iids, c.maxA
    regret = torch.zeros(n, maxA, dtype=_DT, device=device)
    stratsum = torch.zeros(n, maxA, dtype=_DT, device=device)
    ones = torch.ones(B, dtype=_DT, device=device)

    for t in range(iters):
        upd = t % 2
        sig = _regret_match(regret, c.mask)

        # ---- forward reach pass ----
        r0 = torch.zeros(c.n_nodes, B, dtype=_DT, device=device)
        r1 = torch.zeros(c.n_nodes, B, dtype=_DT, device=device)
        rc = torch.zeros(c.n_nodes, B, dtype=_DT, device=device)
        r0[0] = c.r0_entry; r1[0] = c.r1_entry; rc[0] = ones
        for g in c.fwd:
            p = g["parent"]
            if g["kind"] == "ch":
                r0[g["child"]] = r0[p]; r1[g["child"]] = r1[p]; rc[g["child"]] = rc[p] * g["prob"]
            else:
                sg = sig[g["piid"], g["slot"].unsqueeze(1)]  # [G,B]
                if g["kind"] == "d0":
                    r0[g["child"]] = r0[p] * sg; r1[g["child"]] = r1[p]; rc[g["child"]] = rc[p]
                else:
                    r1[g["child"]] = r1[p] * sg; r0[g["child"]] = r0[p]; rc[g["child"]] = rc[p]

        # ---- backward value pass ----
        ev = c.payoff.clone()                       # term rows already set; others overwritten below
        cfvnum = torch.zeros(n, maxA, dtype=_DT, device=device)
        ownreach = torch.zeros(n, dtype=_DT, device=device)
        for g in c.bwd:
            ch = g["children"]                       # [G, K]
            ev_ch = ev[ch]                           # [G, K, B, 2]
            if g["kind"] == "chance":
                pr = g["prob"].permute(0, 2, 1).unsqueeze(-1)   # [G, K, B, 1]
                ev[g["node"]] = (pr * ev_ch).sum(1)             # [G, B, 2]
            else:
                pl = g["player"]; nact = g["nact"]
                node_iid = g["iid"]                              # [G,B]
                sg = sig[node_iid]                               # [G,B,maxA]
                sg = sg[:, :, :nact].permute(0, 2, 1)            # [G, K, B]
                ev[g["node"]] = (sg.unsqueeze(-1) * ev_ch).sum(1)   # [G,B,2]
                own = r0[g["node"]] if pl == 0 else r1[g["node"]]   # [G,B]
                opp = r1[g["node"]] if pl == 0 else r0[g["node"]]   # [G,B]
                cf = opp * rc[g["node"]]                          # [G,B]
                cfv = cf.unsqueeze(1) * ev_ch[..., pl]           # [G,K,B]
                cfv = cfv.permute(0, 2, 1)                        # [G,B,K]
                pad = torch.zeros(cfv.shape[0], B, maxA, dtype=_DT, device=device)
                pad[:, :, :nact] = cfv
                cfvnum.index_add_(0, node_iid.reshape(-1), pad.reshape(-1, maxA))
                ownreach[node_iid.reshape(-1)] = own.reshape(-1)

        w = float(t + 1) if averaging == "linear" else 1.0
        is_upd = (c.player == upd).to(_DT).unsqueeze(1)
        v = (sig * cfvnum).sum(-1, keepdim=True)
        regret = (regret + is_upd * c.mask * (cfvnum - v)).clamp_min(0.0)
        stratsum = stratsum + w * is_upd * ownreach.unsqueeze(1) * sig

    ss = stratsum.sum(-1, keepdim=True)
    avg_t = torch.where(ss > 1e-12, stratsum / ss, c.mask / c.mask.sum(-1, keepdim=True).clamp_min(1.0))
    cont = _ev_pass(c, avg_t).cpu().numpy() if return_cont else None
    avg = avg_t.cpu().numpy()
    out = {key: {} for key in c.keys}
    by_iid_key = {}
    for key in c.keys:
        for iid in dlg._below_iids_for_key(key):
            by_iid_key[iid] = key
    for iid in c.iids:
        out[by_iid_key[iid]][iid] = avg[c.loc[iid]][:c.alen[c.loc[iid]]].copy()
    return (out, cont) if return_cont else out


def _leaf_values_from_cont(dlg, c, cont, reaches, keys):
    """Normalized PBS leaf values from the per-cut-node continuation EV ``cont`` [B,2] (the SoA solver's
    root ev under the average strategy). Same convention as per_belief_equilibrium_leaf_fn but O(B) (no
    numpy subtree walk)."""
    acc = {k: (np.zeros(dlg.n_priv(k, 0)), np.zeros(dlg.n_priv(k, 0)),
               np.zeros(dlg.n_priv(k, 1)), np.zeros(dlg.n_priv(k, 1))) for k in keys}
    for b in range(c.B):
        k = c.node_key[b]; i0 = int(c.node_i0[b]); i1 = int(c.node_i1[b])
        r0, r1 = reaches[k]
        v0n, v0d, v1n, v1d = acc[k]
        v0n[i0] += r1[i1] * cont[b, 0]; v0d[i0] += r1[i1]
        v1n[i1] += r0[i0] * cont[b, 1]; v1d[i1] += r0[i0]
    leaf = {}
    for k in keys:
        v0n, v0d, v1n, v1d = acc[k]
        leaf[k] = (np.divide(v0n, v0d, out=np.zeros_like(v0n), where=v0d > 1e-15),
                   np.divide(v1n, v1d, out=np.zeros_like(v1n), where=v1d > 1e-15))
    return leaf


def trunk_solve_batched(dlg, subgame_iters=200, iters=400, device="cuda", averaging="linear"):
    """Depth-limited CFR+ on the trunk with the per-belief subgame re-solve done by the GPU cross-key
    batched SoA solver (G3). Mirrors DepthLimitedGame.trunk_solve but replaces the per-key serial numpy
    re-solve with ONE batched GPU solve of all keys per iteration. Returns {iid: avg strategy} over the
    above-cut infosets. Equilibrium-equivalent to trunk_solve(per_belief_equilibrium_leaf_fn)."""
    above = [i for i in range(dlg.n_iset) if dlg.iset_above_cut[i]]
    regret = {i: np.zeros(len(dlg.iset_actions[i])) for i in above}
    stratsum = {i: np.zeros(len(dlg.iset_actions[i])) for i in above}
    keys = sorted({n[1] for n in dlg.cut_nodes})

    compiled = None
    for t in range(iters):
        upd = t % 2
        sig = dlg.uniform_policy()
        for i in above:
            pos = np.maximum(regret[i], 0.0); ss = pos.sum()
            sig[i] = pos / ss if ss > 1e-15 else np.ones_like(pos) / len(pos)
        reaches = dlg.cut_reaches(sig)
        ranges = {k: (reaches[k][0], reaches[k][1]) for k in keys}
        if compiled is None:
            compiled = _compile_topology(dlg, ranges, device)
        _eq, cont = solve_all_keys_soa(dlg, ranges, subgame_iters, device=device, compiled=compiled,
                                       return_cont=True)
        leaf_v = _leaf_values_from_cont(dlg, compiled, cont, reaches, keys)
        cfvnum, ownreach = dlg._trunk_cfv(sig, leaf_v)
        w = float(t + 1) if averaging == "linear" else 1.0
        for i in above:
            if dlg.iset_player[i] != upd:
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
