"""ReBeL de-risk, Stage 2: a Leduc public-belief-state (PBS) value network.

The net approximates the round-2 continuation value function v(PBS) -> per-card CFVs, in the pinned
normalized convention. It is used as a FIXED leaf during each depth-limited round-1 solve (the
correct ReBeL usage established in Stage 1; see `loop.trunk_solve`). Stage 2 asks two clean
questions on our own hardware:
  (A) LEARNABILITY -- can a tiny net learn the Leduc PBS->CFV mapping (held-out reach-weighted MAE)?
  (B) SUBSTITUTION FIDELITY -- does plugging the net in as the trunk leaf reproduce the round-1
      strategy of the EXACT-leaf solve (apples-to-apples, same depth-limited algorithm)?

Targets are CFVs read off a round-2 CFR solve for sampled ranges (the consistent procedure that the
exact leaf also uses). Loss/eval are REACH-WEIGHTED so the under-determined thin-reach entries
(Stage-1 finding) are correctly down-weighted -- the net learns what matters downstream.

NOTE: end-to-end low exploitability of the played agent additionally needs SAFE re-solving (the
DeepStack/CFR-D gadget) -- Stage 1 finding #4 (un-gadgeted assembled NashConv 0.21). That is a known
technique, deferred to Stage 2b; Stage 2 here validates learnability + substitution.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from poker_ai.rebel.leduc import NCARDS
from poker_ai.rebel.leaf_eval import ExactLeafOracle
from poker_ai.rebel.loop import solve_round2_equilibrium, _listify


def _normalize(vec):
    s = vec.sum()
    return vec / s if s > 1e-12 else np.ones_like(vec) / len(vec)


def sample_pbs_dataset(tree, n_strategies=400, seed=0, round2_iters=800, ref_strategy=None):
    """Sample PBSs by drawing random round-1 strategies (varied Dirichlet concentration to cover the
    range region the trunk solve visits), then for each cut key compute the entry ranges and target
    round-2 CFVs. Returns a list of dicts with keys: key_idx, range0n, range1n (normalized inputs),
    v0, v1 (normalized targets), w0, w1 (reach weights = normalized ranges).

    Two target modes:
      * ``ref_strategy=None`` -> targets from an ISOLATED round-2 re-solve for each range. These are
        under-determined across ranges (Stage-1 finding #2) and a net underfits them.
      * ``ref_strategy`` (a full policy list, e.g. the CFR-D equilibrium) -> CONSISTENT targets: the
        CFV of continuing with that single fixed near-equilibrium round-2 strategy. Smooth and exact
        (no solve noise), and cheap (no per-range round-2 solve)."""
    rng = np.random.default_rng(seed)
    oracle = ExactLeafOracle(tree, normalize=True)
    keys = sorted(tree.cut_keys())
    key_to_idx = {k: i for i, k in enumerate(keys)}
    r1_iids = [i for i in range(tree.n_iset) if tree.iset_round[i] == 1]
    data = []
    for _ in range(n_strategies):
        alpha = float(rng.choice([0.3, 1.0, 3.0]))  # peaked / uniform-ish / smooth
        pol = tree.uniform_policy()
        for i in r1_iids:
            n = len(tree.iset_actions[i])
            pol[i] = rng.dirichlet(np.full(n, alpha))
        reaches = tree.cut_reaches(pol)
        for k in keys:
            r0, r1 = reaches[k]
            if r0.sum() <= 1e-12 or r1.sum() <= 1e-12:
                continue
            if ref_strategy is None:
                r2 = solve_round2_equilibrium(tree, k, r0, r1, round2_iters, "linear")
                v0, v1 = oracle.evaluate(k, r0, r1, _listify(tree, {}, round2={k: r2}))
            else:
                v0, v1 = oracle.evaluate(k, r0, r1, ref_strategy)
            data.append({
                "key_idx": key_to_idx[k],
                "range0n": _normalize(r0), "range1n": _normalize(r1),
                "v0": v0, "v1": v1,
                "w0": _normalize(r0), "w1": _normalize(r1),
            })
    return data, keys


def _to_tensors(data, n_keys):
    X, V, W = [], [], []
    for d in data:
        onehot = np.zeros(n_keys); onehot[d["key_idx"]] = 1.0
        X.append(np.concatenate([onehot, d["range0n"], d["range1n"]]))
        V.append(np.concatenate([d["v0"], d["v1"]]))
        W.append(np.concatenate([d["w0"], d["w1"]]))
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(V), dtype=torch.float32),
            torch.tensor(np.array(W), dtype=torch.float32))


class PBSValueNet(nn.Module):
    def __init__(self, n_keys, hidden=64):
        super().__init__()
        self.n_keys = n_keys
        d_in = n_keys + 2 * NCARDS
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * NCARDS),
        )

    def forward(self, x):
        return self.net(x)


def train_pbs_net(data, keys, hidden=64, epochs=300, lr=1e-3, val_frac=0.2, seed=0):
    """Train with REACH-WEIGHTED MSE. Returns (net, metrics) where metrics includes held-out
    reach-weighted MAE (metric A)."""
    torch.manual_seed(seed)
    n_keys = len(keys)
    X, V, W = _to_tensors(data, n_keys)
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = int(n * val_frac)
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr, Vtr, Wtr = X[ti], V[ti], W[ti]
    Xva, Vva, Wva = X[vi], V[vi], W[vi]

    net = PBSValueNet(n_keys, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ep in range(epochs):
        net.train(); opt.zero_grad()
        pred = net(Xtr)
        loss = (Wtr * (pred - Vtr) ** 2).sum() / (Wtr.sum() + 1e-9)
        loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        pva = net(Xva)
        wmae = float((Wva * (pva - Vva).abs()).sum() / (Wva.sum() + 1e-9))
        umae = float((pva - Vva).abs().mean())  # unweighted (includes thin-reach noise)
        ptr = net(Xtr)
        wmae_tr = float((Wtr * (ptr - Vtr).abs()).sum() / (Wtr.sum() + 1e-9))
    metrics = {"n_samples": int(n), "n_val": int(n_val),
               "val_reach_weighted_mae": wmae, "val_unweighted_mae": umae,
               "train_reach_weighted_mae": wmae_tr}
    return net, metrics


def net_leaf_fn(net, keys):
    """Build a ``leaf_fn(key, range0, range1) -> (v0, v1)`` for ``loop.trunk_solve`` from a trained
    net. Ranges are normalized before the forward pass; outputs are normalized CFVs (the convention
    the trunk consumes)."""
    key_to_idx = {k: i for i, k in enumerate(keys)}
    n_keys = len(keys)

    def fn(key, range0, range1):
        onehot = np.zeros(n_keys); onehot[key_to_idx[key]] = 1.0
        x = np.concatenate([onehot, _normalize(range0), _normalize(range1)]).astype(np.float32)
        with torch.no_grad():
            out = net(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()
        return out[:NCARDS].astype(np.float64), out[NCARDS:].astype(np.float64)

    return fn
