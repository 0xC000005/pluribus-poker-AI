"""ReBeL step C: a PBS value net for the river continuation at a turn leaf (HUNL).

The exact-river leaf (turn_leaf_river_cfv) is accurate but per-iteration-INFEASIBLE in the turn
trunk (step-0 finding). This trains a net to predict the same per-hand river-continuation CFVs from
the public belief state (the two ranges) in one forward pass, so it can serve as a fast
showdown_leaf_fn during the turn solve -- the efficiency thesis made real on a real-belief game.

Scope: a single cut public state (fixed board + pot + stacks); the net maps the normalized ranges
to per-hand net-from-river CFVs. Targets come from the GPU-batched exact-river leaf. Trained/judged
by REACH-WEIGHTED error (thin-reach hands down-weighted, per the Leduc finding).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from poker_ai.rebel.turn_river import turn_leaf_river_cfv_batched, turn_leaf_river_cfv


def _normalize(v):
    s = v.sum()
    return v / s if s > 1e-12 else np.ones_like(v) / len(v)


def sample_river_targets(turn_board, pot, hs, vs, hero_first, turn_hands, n_samples, seed=0,
                         river_iters=200, batched=True):
    """Sample n_samples public belief states (random Dirichlet ranges over the turn hands) and, for
    each, the exact-river continuation per-hand CFVs (net-from-river). Returns dict of arrays:
    X = [range0n | range1n] (n_samples, 2H), Y = [A_h | A_v] (n_samples, 2H), W = [range0n|range1n]
    (reach weights)."""
    rng = np.random.default_rng(seed)
    H = len(turn_hands)
    Xs, Ys, Ws = [], [], []
    leaf = turn_leaf_river_cfv_batched if batched else (
        lambda *a, **k: turn_leaf_river_cfv(*a, **k))
    for _ in range(n_samples):
        a0 = float(rng.choice([0.3, 1.0, 3.0])); a1 = float(rng.choice([0.3, 1.0, 3.0]))
        r0 = rng.dirichlet(np.full(H, a0)); r1 = rng.dirichlet(np.full(H, a1))
        if batched:
            A_h, A_v = turn_leaf_river_cfv_batched(turn_board, pot, hs, vs, hero_first, turn_hands,
                                                   r0, r1, river_iters=river_iters)
        else:
            A_h, A_v = turn_leaf_river_cfv(turn_board, pot, hs, vs, hero_first, turn_hands,
                                           r0, r1, river_iters=river_iters)
        Xs.append(np.concatenate([_normalize(r0), _normalize(r1)]))
        Ys.append(np.concatenate([A_h, A_v]))
        Ws.append(np.concatenate([_normalize(r0), _normalize(r1)]))
    return {"X": np.array(Xs, np.float32), "Y": np.array(Ys, np.float32), "W": np.array(Ws, np.float32),
            "H": H}


class RiverPBSNet(nn.Module):
    def __init__(self, H, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * H, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * H),
        )

    def forward(self, x):
        return self.net(x)


def train_river_net(data, hidden=512, epochs=400, lr=1e-3, val_frac=0.2, seed=0, device="cpu"):
    """Train with REACH-WEIGHTED MSE. Returns (net, metrics) with held-out reach-weighted MAE."""
    torch.manual_seed(seed)
    H = data["H"]
    X = torch.tensor(data["X"]); Y = torch.tensor(data["Y"]); W = torch.tensor(data["W"])
    dev = torch.device(device)
    X, Y, W = X.to(dev), Y.to(dev), W.to(dev)
    # weights are per-range (H each); duplicate to 2H to match Y layout
    Wfull = torch.cat([W[:, :H], W[:, H:]], dim=1)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    nv = max(1, int(n * val_frac)); vi, ti = perm[:nv], perm[nv:]
    net = RiverPBSNet(H, hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ep in range(epochs):
        net.train(); opt.zero_grad()
        pred = net(X[ti])
        loss = (Wfull[ti] * (pred - Y[ti]) ** 2).sum() / (Wfull[ti].sum() + 1e-9)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pv = net(X[vi])
        wmae = float((Wfull[vi] * (pv - Y[vi]).abs()).sum() / (Wfull[vi].sum() + 1e-9))
        ptr = net(X[ti])
        wmae_tr = float((Wfull[ti] * (ptr - Y[ti]).abs()).sum() / (Wfull[ti].sum() + 1e-9))
    # value scale for context (reach-weighted mean |Y|)
    scale = float((Wfull * Y.abs()).sum() / (Wfull.sum() + 1e-9))
    return net, {"n_samples": int(n), "n_val": int(nv), "val_reach_weighted_mae": wmae,
                 "train_reach_weighted_mae": wmae_tr, "value_scale": scale}
