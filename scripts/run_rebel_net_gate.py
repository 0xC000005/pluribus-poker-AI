#!/usr/bin/env python3
"""ReBeL step C (completion): the LIVE net-leaf gate on a real turn+river spot.

Trains a cut-GENERAL river PBS value net (input = cut pot/stacks features + both normalized ranges)
on exact-river-leaf targets sampled across ALL cut public states of a small turn spot, then uses the
net as the LIVE showdown_leaf_fn in a full turn solve (feasible: sub-ms/leaf) and measures the
net-leaf agent's 2-street exploitability (two_street_nashconv) vs the exact-leaf control. This is the
single-value-leaf correctness check on the real game (proven on Leduc) + the efficiency payoff
(net-leaf turn solve vs the per-iteration-infeasible exact-leaf solve). Small/short-stack spot so the
BR is reliable and the exact control is computable. Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from poker_ai.rebel.turn_river import (
    TurnSpot, build_turn_solver, showdown_cut_indices, turn_leaf_river_cfv_batched,
    turn_leaf_river_cfv, make_exact_river_showdown_fn, two_street_nashconv, _average_strategy_array,
    parse_card,
)
import solver as S

POT0 = 1000.0
ST0 = 300.0


def _norm(v):
    s = v.sum()
    return v / s if s > 1e-12 else np.ones_like(v) / len(v)


def _ctx(pot, hs, vs):
    return np.array([pot / POT0, hs / ST0, vs / ST0], np.float32)


class CtxRiverNet(nn.Module):
    def __init__(self, H, n_ctx=3, hidden=512):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(nn.Linear(n_ctx + 2 * H, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * H))

    def forward(self, x):
        return self.net(x)


def sample_targets_all_cuts(turn_solver, n_strategies, seed, river_iters, batched):
    """Random turn strategies -> for each, forward to the river-deal (showdown) leaves -> per leaf,
    the exact-river target at the (normalized) ranges there, with cut pot/stacks ctx features.
    Returns X (ctx|r0n|r1n), Y (A_h|A_v for normalized ranges), W (reach weights)."""
    rng = np.random.default_rng(seed)
    t = turn_solver._tree
    player = t["player"]; children = t["children"]; dacts = t["decision_actions"]
    sh = t["stacks_h"]; sv = t["stacks_v"]; potN = t["pot"]; nn_ = t["n_nodes"]
    hands = turn_solver.hands; H = turn_solver.n
    board = list(turn_solver.board); hf = turn_solver.hero_first
    cuts = showdown_cut_indices(turn_solver)
    leaf = turn_leaf_river_cfv_batched if batched else turn_leaf_river_cfv
    Xs, Ys, Ws = [], [], []
    for _ in range(n_strategies):
        # random turn strategy
        sig = np.zeros((nn_, t["n_actions"], H))
        for i in range(nn_):
            acts = dacts[i]
            if not acts:
                continue
            a = float(rng.choice([0.3, 1.0, 3.0]))
            d = rng.dirichlet(np.full(len(acts), a), size=H)  # (H, n_acts)
            for j, act in enumerate(acts):
                sig[i, act, :] = d[:, j]
        # forward reaches (range uniform * strat)
        HR = np.zeros((nn_, H)); VR = np.zeros((nn_, H)); HR[0] = 1.0 / H; VR[0] = 1.0 / H
        for i in range(nn_):
            if player[i] == -1:
                continue
            for act in dacts[i]:
                c = children[i, act]
                if player[i] == 0:
                    HR[c] = HR[i] * sig[i, act]; VR[c] = VR[i]
                else:
                    HR[c] = HR[i]; VR[c] = VR[i] * sig[i, act]
        for ci in cuts:
            r0 = HR[ci]; r1 = VR[ci]
            if r0.sum() <= 1e-9 or r1.sum() <= 1e-9:
                continue
            r0n = _norm(r0); r1n = _norm(r1)
            A_h, A_v = leaf(board, int(potN[ci]), int(sh[ci]), int(sv[ci]), hf, hands, r0n, r1n,
                            river_iters=river_iters)
            Xs.append(np.concatenate([_ctx(potN[ci], sh[ci], sv[ci]), r0n, r1n]))
            Ys.append(np.concatenate([A_h, A_v]))
            Ws.append(np.concatenate([r0n, r1n]))
    return (np.array(Xs, np.float32), np.array(Ys, np.float32), np.array(Ws, np.float32), H)


def train(X, Y, W, H, hidden=512, epochs=800, seed=0):
    torch.manual_seed(seed)
    Xt, Yt, Wt = torch.tensor(X), torch.tensor(Y), torch.tensor(W)
    Wf = Wt  # W is already 2H = [r0n | r1n], matching Y = [A_h | A_v]
    n = Xt.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    nv = max(1, int(n * 0.2)); vi, ti = perm[:nv], perm[nv:]
    net = CtxRiverNet(H, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = (Wf[ti] * (net(Xt[ti]) - Yt[ti]) ** 2).sum() / (Wf[ti].sum() + 1e-9)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        wmae = float((Wf[vi] * (net(Xt[vi]) - Yt[vi]).abs()).sum() / (Wf[vi].sum() + 1e-9))
        scale = float((Wf * Yt.abs()).sum() / (Wf.sum() + 1e-9))
    return net, {"n": int(n), "val_mae": wmae, "scale": scale, "mae_frac": round(wmae / scale, 4)}


def make_net_leaf_fn(net, turn_solver):
    """Live net-based showdown_leaf_fn: per river-deal leaf, ctx + normalized ranges -> net -> scale
    by opponent reach mass (the net learned sum-1-range values) -> net-from-turn offset."""
    t = turn_solver._tree; sh = t["stacks_h"]; sv = t["stacks_v"]; potN = t["pot"]
    HS0 = turn_solver.hero_stack_start; VS0 = turn_solver.villain_stack_start
    valid = turn_solver.valid; validT = valid.T; H = turn_solver.n

    def fn(*, tree, showdown_indices, hero_reach, villain_reach, valid_m,
           default_hero_values, default_villain_values, pot_start, hero_stack_start, villain_stack_start):
        out_h = np.zeros((len(showdown_indices), H), np.float32)
        out_v = np.zeros((len(showdown_indices), H), np.float32)
        rows = []
        for k, ci in enumerate(showdown_indices):
            r0 = np.asarray(hero_reach[k], np.float64); r1 = np.asarray(villain_reach[k], np.float64)
            rows.append(np.concatenate([_ctx(potN[ci], sh[ci], sv[ci]), _norm(r0), _norm(r1)]))
        with torch.no_grad():
            pred = net(torch.tensor(np.array(rows, np.float32))).numpy()
        for k, ci in enumerate(showdown_indices):
            r0 = np.asarray(hero_reach[k], np.float64); r1 = np.asarray(villain_reach[k], np.float64)
            mass_h = r0.sum(); mass_v = r1.sum()
            nh = pred[k, :H] * mass_v          # un-normalize: hero value scales with villain mass
            nv = pred[k, H:] * mass_h
            hi = HS0 - int(sh[ci]); vi = VS0 - int(sv[ci])
            out_h[k] = nh - hi * (r1 @ validT)  # net-from-river -> net-from-turn
            out_v[k] = nv - vi * (r0 @ valid)
        return out_h, out_v

    return fn


def run(n_strategies=40, river_iters=200, trunk_iters=60, epochs=800, seed=0):
    spot = TurnSpot(board=[parse_card(c) for c in ("Ah", "Kd", "7c", "2s")],
                    pot=1000, hero_stack=300, villain_stack=300, hero_first=True)
    ts = build_turn_solver(spot)
    H = ts.n
    hr = np.ones(H, np.float32) / H; vr = np.ones(H, np.float32) / H
    out = {"spot": "AhKd7c2s pot10bb stacks3bb", "H": H,
           "cuts": len(showdown_cut_indices(ts)), "n_strategies": n_strategies}

    t0 = time.time()
    X, Y, W, _ = sample_targets_all_cuts(ts, n_strategies, seed, river_iters, batched=True)
    out["target_gen_s"] = round(time.time() - t0, 1); out["n_targets"] = int(X.shape[0])
    net, m = train(X, Y, W, H, hidden=512, epochs=epochs, seed=seed)
    out["net"] = m

    # exact-leaf control turn solve + its 2-street exploitability
    t0 = time.time()
    ts.solve(n_iterations=trunk_iters, hero_range=hr, villain_range=vr, backend="cpu",
             showdown_leaf_fn=make_exact_river_showdown_fn(ts, river_iters=river_iters))
    out["exact_turn_solve_s"] = round(time.time() - t0, 1)
    sig_exact = _average_strategy_array(np.asarray(ts._strategy_sum))
    r_exact = two_street_nashconv(ts, sig_exact, hr.astype(np.float64), vr.astype(np.float64),
                                  river_iters=river_iters)
    out["exact_leaf_agent_nashconv"] = r_exact["nashconv"]

    # NET-leaf live turn solve + its 2-street exploitability (measured by the EXACT river BR)
    t0 = time.time()
    ts.solve(n_iterations=trunk_iters, hero_range=hr, villain_range=vr, backend="cpu",
             showdown_leaf_fn=make_net_leaf_fn(net, ts))
    out["net_turn_solve_s"] = round(time.time() - t0, 2)
    sig_net = _average_strategy_array(np.asarray(ts._strategy_sum))
    r_net = two_street_nashconv(ts, sig_net, hr.astype(np.float64), vr.astype(np.float64),
                               river_iters=river_iters)
    out["net_leaf_agent_nashconv"] = r_net["nashconv"]

    out["turn_solve_speedup"] = round(out["exact_turn_solve_s"] / max(out["net_turn_solve_s"], 1e-3), 1)
    out["pot"] = 1000
    out["net_close_to_exact"] = bool(r_net["nashconv"] < r_exact["nashconv"] + 0.05 * 1000)
    return out, net


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-strategies", type=int, default=40)
    ap.add_argument("--river-iters", type=int, default=200)
    ap.add_argument("--trunk-iters", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out, _ = run(n_strategies=args.n_strategies, river_iters=args.river_iters,
                 trunk_iters=args.trunk_iters, epochs=args.epochs)
    print("ReBeL step C completion: LIVE net-leaf gate")
    print(f"  spot {out['spot']}  H={out['H']} hands, {out['cuts']} cut public states")
    print(f"  targets: {out['n_targets']} across cuts in {out['target_gen_s']}s; "
          f"net val MAE {out['net']['val_mae']:.1f} ({out['net']['mae_frac']:.1%} of scale)")
    print(f"  exact-leaf turn solve {out['exact_turn_solve_s']}s -> 2-street NashConv "
          f"{out['exact_leaf_agent_nashconv']:.1f} ({out['exact_leaf_agent_nashconv']/1000:.3f} pot)")
    print(f"  NET-leaf  turn solve {out['net_turn_solve_s']}s -> 2-street NashConv "
          f"{out['net_leaf_agent_nashconv']:.1f} ({out['net_leaf_agent_nashconv']/1000:.3f} pot)")
    print(f"  turn-solve speedup (net vs exact leaf): {out['turn_solve_speedup']}x")
    print(f"  net agent close to exact control: {out['net_close_to_exact']}")
    if args.output_json:
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
