"""ReBeL step 3: the self-play training loop on the turn+river real-belief game.

Closes the ReBeL loop on a real game. Each iteration:
  1. solve the TURN subgame with the current river PBS net as the depth-limit leaf
     (``make_net_leaf_fn``), giving an on-policy average turn strategy;
  2. harvest EXACT-river CFV targets at the ON-POLICY (belief-consistent) river PBSs that solve
     induces (``sample_targets_from_strategy``) plus a little random exploration for coverage
     (``sample_targets_all_cuts``);
  3. retrain the net from scratch on the accumulated reservoir;
  4. measure the 2-street best-response exploitability (``two_street_nashconv``) and watch it trend
     toward the exact-river-leaf control, aborting if it drifts up.

KEY PROPERTY: the river is the last street, so harvested targets are EXACT (``turn_leaf_river_cfv``)
-- they do NOT bootstrap through the net. The net only changes WHICH PBSs it is trained on, so the
loop refines the PBS DISTRIBUTION (a coverage / belief-consistency question), not target stability.
Single-value PBS trunk leaf; whether this closed loop converges on a real-belief game is the step-3
question (RESEARCH_LOG 20260605T143000Z NEXT). The net-leaf convention is verbatim from the verified
C-completion gate (``scripts/run_rebel_net_gate.py``). Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from poker_ai.rebel.turn_river import (
    TurnSpot, build_turn_solver, showdown_cut_indices, parse_card,
    turn_leaf_river_cfv_batched, turn_leaf_river_cfv, two_street_nashconv,
    make_exact_river_showdown_fn, _average_strategy_array,
)

# ctx normalization constants (match the C-gate spot: pot 10bb, stacks 3bb)
POT0 = 1000.0
ST0 = 300.0


def default_loop_spot() -> TurnSpot:
    """The small/short-stack spot the C-gate validated: BR is reliable and the exact control is
    computable (board AhKd7c2s, pot 10bb, stacks 3bb)."""
    return TurnSpot(board=[parse_card(c) for c in ("Ah", "Kd", "7c", "2s")],
                    pot=1000, hero_stack=300, villain_stack=300, hero_first=True)


def _norm(v):
    s = v.sum()
    return v / s if s > 1e-12 else np.ones_like(v) / len(v)


def _ctx(pot, hs, vs):
    return np.array([pot / POT0, hs / ST0, vs / ST0], np.float32)


class CtxRiverNet(nn.Module):
    """Cut-GENERAL river PBS value net: input = [pot/stack ctx | range0n | range1n], output =
    per-hand net-from-river CFVs [A_h | A_v] for the (sum-1-normalized) ranges."""

    def __init__(self, H, n_ctx=3, hidden=512):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(nn.Linear(n_ctx + 2 * H, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * H))

    def forward(self, x):
        return self.net(x)


def _reaches_for_strategy(turn_solver, sig):
    """Forward player reaches under a per-(node, action, hand) strategy `sig`, from uniform ranges.
    Returns (HR, VR) arrays of shape (n_nodes, H)."""
    t = turn_solver._tree
    player = t["player"]; children = t["children"]; dacts = t["decision_actions"]
    nn_ = t["n_nodes"]; H = turn_solver.n
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
    return HR, VR


def _targets_at_cuts(turn_solver, sig, river_iters, batched):
    """Given a turn strategy `sig`, harvest per-cut exact-river targets at the induced ranges.
    Returns lists (Xs, Ys, Ws)."""
    t = turn_solver._tree
    sh = t["stacks_h"]; sv = t["stacks_v"]; potN = t["pot"]
    hands = turn_solver.hands
    board = list(turn_solver.board); hf = turn_solver.hero_first
    cuts = showdown_cut_indices(turn_solver)
    leaf = turn_leaf_river_cfv_batched if batched else turn_leaf_river_cfv
    HR, VR = _reaches_for_strategy(turn_solver, sig)
    Xs, Ys, Ws = [], [], []
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
    return Xs, Ys, Ws


def _random_strategy(turn_solver, rng):
    """A random per-(node, action, hand) turn strategy (Dirichlet over legal actions per hand)."""
    t = turn_solver._tree
    dacts = t["decision_actions"]; nn_ = t["n_nodes"]; na = t["n_actions"]; H = turn_solver.n
    sig = np.zeros((nn_, na, H))
    for i in range(nn_):
        acts = dacts[i]
        if not acts:
            continue
        a = float(rng.choice([0.3, 1.0, 3.0]))
        d = rng.dirichlet(np.full(len(acts), a), size=H)  # (H, n_acts)
        for j, act in enumerate(acts):
            sig[i, act, :] = d[:, j]
    return sig


def sample_targets_from_strategy(turn_solver, sig, river_iters=120, batched=True):
    """ON-POLICY harvest: exact-river targets at the cut PBSs induced by a specific turn strategy.
    Returns (X, Y, W, H) with X=(ctx|r0n|r1n), Y=(A_h|A_v), W=(r0n|r1n)."""
    Xs, Ys, Ws = _targets_at_cuts(turn_solver, sig, river_iters, batched)
    return (np.array(Xs, np.float32), np.array(Ys, np.float32), np.array(Ws, np.float32), turn_solver.n)


def sample_targets_all_cuts(turn_solver, n_strategies, seed, river_iters, batched):
    """EXPLORATION harvest: exact-river targets across all cuts under `n_strategies` random turn
    strategies (coverage). Same output contract as ``sample_targets_from_strategy``; this is the
    cut-general sampler the C-gate uses."""
    rng = np.random.default_rng(seed)
    Xs, Ys, Ws = [], [], []
    for _ in range(n_strategies):
        sig = _random_strategy(turn_solver, rng)
        xs, ys, ws = _targets_at_cuts(turn_solver, sig, river_iters, batched)
        Xs += xs; Ys += ys; Ws += ws
    return (np.array(Xs, np.float32), np.array(Ys, np.float32), np.array(Ws, np.float32), turn_solver.n)


def train_ctx_net(X, Y, W, H, hidden=512, epochs=800, seed=0):
    """Fresh-net reach-weighted MSE (retrain-from-scratch each loop iter, per the project's Deep-CFR
    convention). Returns (net, metrics)."""
    torch.manual_seed(seed)
    Xt, Yt, Wt = torch.tensor(X), torch.tensor(Y), torch.tensor(W)
    Wf = Wt  # W is already 2H = [r0n | r1n], matching Y = [A_h | A_v]
    n = Xt.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    nv = max(1, int(n * 0.2)); vi, ti = perm[:nv], perm[nv:]
    if len(ti) == 0:  # tiny-n guard (smoke tests)
        ti = perm
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
    return net, {"n": int(n), "val_mae": wmae, "scale": scale, "mae_frac": round(wmae / max(scale, 1e-9), 4)}


def make_net_leaf_fn(net, turn_solver):
    """Live net-based showdown_leaf_fn (verbatim convention from the verified C-gate): per river-deal
    leaf, ctx + normalized ranges -> net -> scale by opponent reach mass -> net-from-turn offset."""
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


class ReservoirBuffer:
    """Reservoir-capped accumulation of (X, Y, W) target rows across loop iterations."""

    def __init__(self, cap, seed=0):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.X = self.Y = self.W = None

    def add(self, X, Y, W):
        if X is None or X.shape[0] == 0:
            return
        if self.X is None:
            self.X, self.Y, self.W = X, Y, W
        else:
            self.X = np.concatenate([self.X, X]); self.Y = np.concatenate([self.Y, Y])
            self.W = np.concatenate([self.W, W])
        if self.X.shape[0] > self.cap:
            idx = self.rng.choice(self.X.shape[0], self.cap, replace=False)
            self.X, self.Y, self.W = self.X[idx], self.Y[idx], self.W[idx]

    def __len__(self):
        return 0 if self.X is None else self.X.shape[0]


def run_self_play(spot=None, n_iters=8, trunk_iters=24, river_iters=120,
                  init_strategies=8, explore_strategies=2, epochs=400, hidden=512,
                  buffer_cap=4000, seed=0, drift_tol_pot=0.02, batched=True,
                  compute_control=False, verbose=True):
    """Run the turn+river ReBeL self-play loop. Returns (out_dict, net) where out_dict["history"]
    is the per-iteration 2-street NashConv trend (the convergence evidence). The loop ABORTS if
    NashConv drifts above its best-so-far by more than ``drift_tol_pot`` of the pot (the trunk-bias /
    target-drift falsification guard)."""
    spot = spot if spot is not None else default_loop_spot()
    ts = build_turn_solver(spot)
    H = ts.n
    hr = np.ones(H, np.float32) / H; vr = np.ones(H, np.float32) / H
    hr64 = hr.astype(np.float64); vr64 = vr.astype(np.float64)
    pot = float(np.asarray(ts._tree["pot"])[0])
    buf = ReservoirBuffer(buffer_cap, seed)

    # seed the buffer with exploration-only targets, train the initial net (the harvested targets are
    # EXACT regardless of the net, so a random init is fine -- only the PBS distribution is off-policy)
    Xe, Ye, We, _ = sample_targets_all_cuts(ts, init_strategies, seed, river_iters, batched)
    buf.add(Xe, Ye, We)
    net, m = train_ctx_net(buf.X, buf.Y, buf.W, H, hidden=hidden, epochs=epochs, seed=seed)

    history = []
    best_nc = float("inf"); aborted = False
    for it in range(n_iters):
        ts.solve(n_iterations=trunk_iters, hero_range=hr, villain_range=vr, backend="cpu",
                 showdown_leaf_fn=make_net_leaf_fn(net, ts))
        avg = _average_strategy_array(np.asarray(ts._strategy_sum))
        nc = float(two_street_nashconv(ts, avg, hr64, vr64, river_iters=river_iters)["nashconv"])
        rec = {"iter": it, "nashconv": nc, "nashconv_pot": nc / pot,
               "val_mae": m["val_mae"], "mae_frac": m["mae_frac"], "buffer": len(buf)}
        history.append(rec)
        if verbose:
            print(f"  iter {it}: 2-street NashConv {nc:.1f} ({nc / pot:.3f} pot)  "
                  f"net val MAE {m['val_mae']:.1f} ({m['mae_frac']:.1%})  buffer {len(buf)}", flush=True)
        if nc > best_nc + drift_tol_pot * pot:
            aborted = True
            if verbose:
                print(f"  ABORT: NashConv drifted up ({nc:.1f} > best {best_nc:.1f} + "
                      f"{drift_tol_pot:.3f}*pot)", flush=True)
            break
        best_nc = min(best_nc, nc)
        # harvest on-policy + a little exploration, retrain from scratch
        Xo, Yo, Wo, _ = sample_targets_from_strategy(ts, avg, river_iters, batched)
        buf.add(Xo, Yo, Wo)
        if explore_strategies > 0:
            Xx, Yx, Wx, _ = sample_targets_all_cuts(ts, explore_strategies, seed + 1 + it, river_iters, batched)
            buf.add(Xx, Yx, Wx)
        net, m = train_ctx_net(buf.X, buf.Y, buf.W, H, hidden=hidden, epochs=epochs, seed=seed)

    out = {"pot": pot, "n_iters": n_iters, "history": history,
           "best_nashconv": best_nc, "best_nashconv_pot": best_nc / pot, "aborted": aborted,
           "final_val_mae": m["val_mae"], "final_mae_frac": m["mae_frac"]}
    if compute_control:
        ts.solve(n_iterations=trunk_iters, hero_range=hr, villain_range=vr, backend="cpu",
                 showdown_leaf_fn=make_exact_river_showdown_fn(ts, river_iters=river_iters))
        cavg = _average_strategy_array(np.asarray(ts._strategy_sum))
        cnc = float(two_street_nashconv(ts, cavg, hr64, vr64, river_iters=river_iters)["nashconv"])
        out["control_nashconv"] = cnc; out["control_nashconv_pot"] = cnc / pot
    return out, net
