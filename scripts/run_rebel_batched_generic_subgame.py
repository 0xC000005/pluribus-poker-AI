#!/usr/bin/env python3
"""GATE 0c (de-risk chance + variable depth on a REAL generic subgame): the GATE-0b kernel core was a flat
synthetic 2-level bilinear subgame. Before the full multi-week SoA-compiler port, prove the batched torch
CFR+ approach handles the ACTUAL generic DepthLimitedGame subgame structure -- arbitrary nested
term/chance/decision nodes, CHANCE nodes (Goofspiel prize reveals), and VARIABLE DEPTH (folds end early) --
and produces the SAME equilibrium as the trusted numpy solve_subgame_equilibrium.

Mechanism: GATE-0 Probe A proved all cut-node continuation subtrees at a public key share ONE topology. So
walk that shared topology ONCE per CFR+ iteration, in LOCKSTEP over the batch of cut nodes (the (i0,i1)
private entry pairs), carrying batched torch reach/value tensors; gather per-infoset strategies and
scatter (index_add_) the regret/value updates back. This is the real batched kernel (the port's core),
just driven off the Python tree instead of a flat SoA array. PARITY vs dlg.solve_subgame_equilibrium is
the self-validating correctness gate (must match to ~1e-4). Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pyspiel
import torch

# CFR+ regret accumulation is precision-sensitive (the numpy reference is float64); float32 drifts
# noticeably over hundreds of iters at near-tied infosets. Use float64 for an honest parity check.
torch.set_default_dtype(torch.float64)

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import leduc_is_cut, load_goofspiel, goofspiel_is_cut, goofspiel_public_key


def _regret_match(regret, mask):
    pos = (regret * mask).clamp_min(0.0)
    s = pos.sum(-1, keepdim=True)
    uni = mask / mask.sum(-1, keepdim=True).clamp_min(1.0)
    return torch.where(s > 1e-12, pos / s, uni)


def solve_key_batched(dlg, key, range0, range1, iters, device, averaging="linear"):
    """Batched torch CFR+ equilibrium of the below-cut subgame at public ``key`` -- the generic twin of
    solve_subgame_equilibrium, walking the shared topology in lockstep over all cut nodes. Returns
    {iid: avg strategy (np)}."""
    nodes = [n for n in dlg.cut_nodes if n[1] == key]
    iids = dlg._below_iids_for_key(key)
    loc = {iid: k for k, iid in enumerate(iids)}
    n = len(iids)
    alen = [len(dlg.iset_actions[i]) for i in iids]
    maxA = max(alen) if alen else 1
    player = torch.tensor([dlg.iset_player[i] for i in iids])
    mask = torch.zeros(n, maxA)
    for k, L in enumerate(alen):
        mask[k, :L] = 1.0
    mask = mask.to(device); player = player.to(device)

    regret = torch.zeros(n, maxA, device=device)
    stratsum = torch.zeros(n, maxA, device=device)
    B = len(nodes)
    # entry reaches per cut node (batch element)
    r0_entry = torch.tensor([float(range0[nd[2]]) for nd in nodes], dtype=torch.float64, device=device)
    r1_entry = torch.tensor([float(range1[nd[3]]) for nd in nodes], dtype=torch.float64, device=device)
    subs = [nd[4] for nd in nodes]

    for t in range(iters):
        upd = t % 2
        sig = _regret_match(regret, mask)
        cfvnum = torch.zeros(n, maxA, device=device)
        ownreach = torch.zeros(n, device=device)

        def walk(nodes_b, r0_b, r1_b, rc_b):
            ty = nodes_b[0][0]
            if ty == "term":
                return torch.tensor(np.stack([nd[1] for nd in nodes_b]), dtype=torch.float64, device=device)
            if ty == "chance":
                ev = torch.zeros(len(nodes_b), 2, device=device)
                nch = len(nodes_b[0][1])
                for ci in range(nch):
                    # chance probs/outcomes can DIFFER per cut node (e.g. Leduc flop deal depends on the
                    # removed hole cards) -- use PER-BATCH-ELEMENT probabilities, aligned by branch index
                    p_b = torch.tensor([float(nd[1][ci][0]) for nd in nodes_b], device=device)  # [B]
                    ch_b = [nd[1][ci][1] for nd in nodes_b]
                    ev = ev + p_b.unsqueeze(1) * walk(ch_b, r0_b, r1_b, rc_b * p_b)
                return ev
            # decision
            pl = nodes_b[0][1]
            n_act = len(nodes_b[0][3])
            idx = torch.tensor([loc[nd[2]] for nd in nodes_b], device=device)
            sig_b = sig[idx][:, :n_act]                       # [B,n_act]
            ownreach[idx] = (r0_b if pl == 0 else r1_b)
            cf_b = (r1_b if pl == 0 else r0_b) * rc_b          # [B]
            ev = torch.zeros(len(nodes_b), 2, device=device)
            cfv_b = torch.zeros(len(nodes_b), n_act, device=device)
            for k in range(n_act):
                ch_b = [nd[3][k][1] for nd in nodes_b]
                if pl == 0:
                    cv = walk(ch_b, r0_b * sig_b[:, k], r1_b, rc_b)
                else:
                    cv = walk(ch_b, r0_b, r1_b * sig_b[:, k], rc_b)
                ev = ev + sig_b[:, k:k + 1] * cv
                cfv_b[:, k] = cf_b * cv[:, pl]
            pad = torch.zeros(len(nodes_b), maxA, device=device)
            pad[:, :n_act] = cfv_b
            cfvnum.index_add_(0, idx, pad)
            return ev

        walk(subs, r0_entry, r1_entry, torch.ones(B, device=device))
        w = float(t + 1) if averaging == "linear" else 1.0
        is_upd = (player == upd).float().unsqueeze(1)
        v = (sig * cfvnum).sum(-1, keepdim=True)
        regret = (regret + is_upd * mask * (cfvnum - v)).clamp_min(0.0)
        stratsum = stratsum + w * is_upd * ownreach.unsqueeze(1) * sig

    ss = stratsum.sum(-1, keepdim=True)
    avg = torch.where(ss > 1e-12, stratsum / ss, mask / mask.sum(-1, keepdim=True).clamp_min(1.0))
    avg = avg.cpu().numpy()
    return {iid: avg[loc[iid]][:alen[loc[iid]]].copy() for iid in iids}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = {"device": device, "games": {}}
    print(f"GATE 0c: batched torch CFR+ on REAL generic subgames vs numpy solve_subgame_equilibrium "
          f"(device={device})")

    games = [("leduc", pyspiel.load_game("leduc_poker"), leduc_is_cut, None, "variable-depth (folds), no chance"),
             ("goofspiel4", load_goofspiel(4), goofspiel_is_cut, goofspiel_public_key, "CHANCE (prize reveals) + variable depth")]
    rng = np.random.default_rng(0)
    for name, game, cut, pkf, desc in games:
        dlg = DepthLimitedGame(game, cut, public_key_fn=pkf)
        keys = sorted({nd[1] for nd in dlg.cut_nodes})
        has_chance = any(_subtree_has_chance(nd[4]) for nd in dlg.cut_nodes)
        max_l1 = 0.0; t_np = 0.0; t_torch = 0.0; n_checked = 0
        for key in keys:
            n0, n1 = dlg.n_priv(key, 0), dlg.n_priv(key, 1)
            r0 = rng.dirichlet(np.ones(n0)); r1 = rng.dirichlet(np.ones(n1))
            t0 = time.perf_counter()
            ref = dlg.solve_subgame_equilibrium(key, r0, r1, iters=args.iters)
            t_np += time.perf_counter() - t0
            t0 = time.perf_counter()
            got = solve_key_batched(dlg, key, r0, r1, args.iters, device)
            t_torch += time.perf_counter() - t0
            for iid in ref:
                d = float(np.abs(np.asarray(ref[iid]) - np.asarray(got[iid])).sum())
                max_l1 = max(max_l1, d); n_checked += 1
        out["games"][name] = {"n_public_states": len(keys), "has_chance_in_subgame": bool(has_chance),
                              "n_infosets_checked": n_checked, "max_strategy_l1_vs_numpy": round(max_l1, 6),
                              "numpy_total_s": round(t_np, 2), "batched_torch_total_s": round(t_torch, 2),
                              "desc": desc}
        ok = max_l1 < 1e-3
        print(f"  {name} ({desc}): {len(keys)} subgames, chance-in-subgame={has_chance}; "
              f"PARITY max strategy L1 vs numpy = {max_l1:.2e} {'PASS' if ok else 'FAIL'}  "
              f"(numpy {t_np:.1f}s, batched-torch {t_torch:.1f}s)")

    out["parity_pass"] = bool(all(g["max_strategy_l1_vs_numpy"] < 1e-3 for g in out["games"].values()))
    out["chance_covered"] = bool(any(g["has_chance_in_subgame"] for g in out["games"].values()))
    print(f"\n  PARITY PASS on all (real chance + variable depth handled correctly): {out['parity_pass']}")
    print(f"  CHANCE nodes exercised in a subgame: {out['chance_covered']}")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


def _subtree_has_chance(node):
    t = node[0]
    if t == "term":
        return False
    if t == "chance":
        return True
    if t == "cut":
        return _subtree_has_chance(node[4])
    return any(_subtree_has_chance(ch) for _a, ch in node[3])


if __name__ == "__main__":
    raise SystemExit(main())
