"""Generic depth-limited PBS-net self-play loop (Build 3b-ii) -- the game-agnostic ReBeL training loop,
on the iig depth-limited substrate. One PBS value net, one solver, one loop, behind the generic game
interface; the SAME code runs on any 2p0s IIG.

Each iteration: solve the TRUNK with the current net as the depth-limit leaf (``DepthLimitedGame.
trunk_solve``); harvest EXACT leaf targets -- BROAD random-range coverage (DeepStack "cover the space
CFR encounters") plus the belief-consistent on-policy ranges the solve induces -- as the normalized PBS
value of continuing under a fixed near-equilibrium round play; train the net; iterate. The net amortizes
the leaf solve (the efficiency win); coverage makes it learnable; on-policy adds belief-consistency.

IMPORTANT (honest scope): on a 2-LEVEL game (one cut -- Leduc, turn+river) the leaf is the FINAL round,
so its targets are EXACT and do NOT bootstrap through the net (the Step-3 finding). This loop validates
net-LEARNABILITY + leaf-amortization (net-leaf sigma1 -> exact-leaf control) on the generic substrate;
demonstrating self-play BOOTSTRAPPING requires a >=3-level game. Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from poker_ai.rebel.iig_solve import DepthLimitedGame


def _norm(v):
    s = v.sum()
    return v / s if s > 1e-12 else np.ones_like(v) / len(v)


class PBSNet(nn.Module):
    """Generic PBS value net: input = [public one-hot | range0 | range1] (ranges zero-padded to the max
    private dim), output = [v0 | v1] normalized per-private CFVs."""

    def __init__(self, n_public, max_priv, hidden=128):
        super().__init__()
        self.n_public = n_public
        self.P = max_priv
        self.net = nn.Sequential(
            nn.Linear(n_public + 2 * max_priv, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * max_priv),
        )

    def forward(self, x):
        return self.net(x)


def build_public_index(dlg):
    keys = sorted({n[1] for n in dlg.cut_nodes})
    return {k: i for i, k in enumerate(keys)}, keys


def _encode(pub_index, P, key, r0n, r1n):
    x = np.zeros(len(pub_index) + 2 * P, np.float32)
    x[pub_index[key]] = 1.0
    x[len(pub_index):len(pub_index) + len(r0n)] = r0n
    x[len(pub_index) + P:len(pub_index) + P + len(r1n)] = r1n
    return x


def precompute_cont(dlg, cont_pol):
    """Per cut key, the fixed per-(priv0,priv1) continuation EV under ``cont_pol`` (the near-eq
    final-round play). Computed once; leaf values for ANY range follow cheaply."""
    cache = {}
    for node in dlg.cut_nodes:
        _t, key, i0, i1, sub = node
        cache.setdefault(key, []).append((i0, i1, dlg.subtree_ev(sub, cont_pol)))
    return cache


def _leaf_value(dlg, key, cont_cache, r0n, r1n):
    n0, n1 = dlg.n_priv(key, 0), dlg.n_priv(key, 1)
    v0n = np.zeros(n0); v0d = np.zeros(n0); v1n = np.zeros(n1); v1d = np.zeros(n1)
    for i0, i1, ev in cont_cache[key]:
        v0n[i0] += r1n[i1] * ev[0]; v0d[i0] += r1n[i1]
        v1n[i1] += r0n[i0] * ev[1]; v1d[i1] += r0n[i0]
    v0 = np.divide(v0n, v0d, out=np.zeros(n0), where=v0d > 1e-15)
    v1 = np.divide(v1n, v1d, out=np.zeros(n1), where=v1d > 1e-15)
    return v0, v1


def _row(dlg, pub_index, P, key, r0n, r1n, v0, v1):
    x = _encode(pub_index, P, key, r0n, r1n)
    y = np.zeros(2 * P, np.float32); y[:len(v0)] = v0; y[P:P + len(v1)] = v1
    w = np.zeros(2 * P, np.float32); w[:len(r0n)] = r0n; w[P:P + len(r1n)] = r1n
    return x, y, w


def coverage_rows(dlg, pub_index, P, cont_cache, keys, n_per_cut, rng):
    """BROAD random-range coverage: per public state, sample n_per_cut random Dirichlet range pairs and
    the exact leaf value (the learnability driver)."""
    X, Y, W = [], [], []
    for key in keys:
        n0, n1 = dlg.n_priv(key, 0), dlg.n_priv(key, 1)
        for _ in range(n_per_cut):
            a0 = float(rng.choice([0.3, 1.0, 3.0])); a1 = float(rng.choice([0.3, 1.0, 3.0]))
            r0n = rng.dirichlet(np.full(n0, a0)); r1n = rng.dirichlet(np.full(n1, a1))
            v0, v1 = _leaf_value(dlg, key, cont_cache, r0n, r1n)
            x, y, w = _row(dlg, pub_index, P, key, r0n, r1n, v0, v1)
            X.append(x); Y.append(y); W.append(w)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(W, np.float32)


def onpolicy_rows(dlg, pub_index, P, cont_cache, keys, range_pol):
    """The belief-consistent on-policy ranges the trunk solve induces (one per public state)."""
    reaches = dlg.cut_reaches(range_pol)
    X, Y, W = [], [], []
    for key in keys:
        r0n, r1n = _norm(reaches[key][0]), _norm(reaches[key][1])
        v0, v1 = _leaf_value(dlg, key, cont_cache, r0n, r1n)
        x, y, w = _row(dlg, pub_index, P, key, r0n, r1n, v0, v1)
        X.append(x); Y.append(y); W.append(w)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(W, np.float32)


def make_net_leaf_fn(net, dlg, pub_index, P):
    def fn(key, range0, range1):
        x = _encode(pub_index, P, key, _norm(range0), _norm(range1))
        with torch.no_grad():
            y = net(torch.tensor(x).unsqueeze(0)).numpy()[0]
        return y[:len(range0)].copy(), y[P:P + len(range1)].copy()
    return fn


def _train(net, X, Y, W, epochs, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt, Yt, Wt = torch.tensor(X), torch.tensor(Y), torch.tensor(W)
    n = Xt.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    nv = max(1, int(n * 0.2)); vi, ti = perm[:nv], perm[nv:]
    if len(ti) == 0:
        ti = perm
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = (Wt[ti] * (net(Xt[ti]) - Yt[ti]) ** 2).sum() / (Wt[ti].sum() + 1e-9)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        mae = float((Wt[vi] * (net(Xt[vi]) - Yt[vi]).abs()).sum() / (Wt[vi].sum() + 1e-9))
        scale = float((Wt * Yt.abs()).sum() / (Wt.sum() + 1e-9))
    return {"val_mae": mae, "scale": scale, "mae_frac": round(mae / max(scale, 1e-9), 4)}


def _sig1_l1(a, b):
    keys = set(a) & set(b)
    return float(np.mean([np.abs(a[i] - b[i]).sum() for i in keys])) if keys else float("nan")


def self_play(game, is_cut_fn, public_key_fn=None, n_iters=6, trunk_iters=200, coverage_per_cut=150,
              train_epochs=400, cont_iters=600, buffer_cap=6000, hidden=128, seed=0, verbose=True):
    """Run the generic depth-limited PBS-net self-play loop. Returns (history, net, dlg). The control is
    trunk_solve with the EXACT blueprint leaf (same fixed near-eq continuation); the net-leaf sigma1
    should converge to it as the net learns (the leaf-amortization gate)."""
    dlg = DepthLimitedGame(game, is_cut_fn, public_key_fn)
    pub_index, keys = build_public_index(dlg)
    P = max(max(dlg.n_priv(k, 0), dlg.n_priv(k, 1)) for k in keys)

    cont_pol = dlg.cfr_plus(cont_iters)
    cont_cache = precompute_cont(dlg, cont_pol)
    exact_leaf = dlg.blueprint_leaf_fn(cont_pol)
    sigma1_exact = dlg.trunk_solve(exact_leaf, iters=trunk_iters)

    net = PBSNet(len(pub_index), P, hidden)
    rng = np.random.default_rng(seed)
    X, Y, W = coverage_rows(dlg, pub_index, P, cont_cache, keys, coverage_per_cut, rng)
    m = _train(net, X, Y, W, train_epochs, seed=seed)

    history = []
    for it in range(n_iters):
        sig1 = dlg.trunk_solve(make_net_leaf_fn(net, dlg, pub_index, P), iters=trunk_iters)
        l1 = _sig1_l1(sig1, sigma1_exact)
        history.append({"iter": it, "val_mae": m["val_mae"], "mae_frac": m["mae_frac"],
                        "sigma1_l1_vs_exact": l1, "n_targets": int(X.shape[0])})
        if verbose:
            print(f"  iter {it}: net val MAE {m['val_mae']:.4f} ({m['mae_frac']:.1%})  "
                  f"sigma1 L1 vs exact-leaf control {l1:.4f}  targets {X.shape[0]}", flush=True)
        # harvest on-policy (belief-consistent) + fresh coverage, retrain
        sig1_full = dlg.uniform_policy()
        for i, pr in sig1.items():
            sig1_full[i] = pr
        xo, yo, wo = onpolicy_rows(dlg, pub_index, P, cont_cache, keys, sig1_full)
        xc, yc, wc = coverage_rows(dlg, pub_index, P, cont_cache, keys, coverage_per_cut, rng)
        X = np.concatenate([X, xo, xc]); Y = np.concatenate([Y, yo, yc]); W = np.concatenate([W, wo, wc])
        if X.shape[0] > buffer_cap:
            idx = rng.choice(X.shape[0], buffer_cap, replace=False)
            X, Y, W = X[idx], Y[idx], W[idx]
        m = _train(net, X, Y, W, train_epochs, seed=seed)

    return history, net, dlg
