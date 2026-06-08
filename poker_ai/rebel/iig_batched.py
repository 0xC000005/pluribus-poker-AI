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
