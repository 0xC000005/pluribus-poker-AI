"""Board-general river PBS value net (P3/DeepStack-lite prerequisite).

The P0b bundle's "river board-generalization curve" milestone: train ONE
principled net on float64 fused-chain V* targets sampled across the REAL
trunk-reachable river config space x random boards x Dirichlet beliefs, and
measure validation reach-weighted MAE on UNSEEN boards vs training-set size.

Target chain (ALL parity-gated, used as-is, never edited):
    targets.generate_river_targets -> lazy_subgames.PopulationQueue
        -> population_solver.solve_population (float64 batched CFR+ kernel
           + batched average-strategy root value pass), 300 iters, cuda.

Net recipe (the river_pbs_net.py prototype lifted board-general; ONE config,
no architecture sweeps -- p1_design.json / P0b governance):
    x [2706] = board 52-hot | pot/2e4, eff_stack/2e4 | r0n [1326] | r1n [1326]
    y [2652] = v0 | v1 -- per-hand V* CFVs NORMALIZED BY POT (chips / pot)
    2 hidden ReLU layers, hidden=512 (the prototype), Adam lr 1e-3
    loss = reach-weighted MSE (w = r0n|r1n) + zs_lambda * zero-sum penalty
           (per-row identity: sum(w * v_pred) == sum(w * v_target), both in
            pot units -- the turn_river.py L465 convention identity)

Conventions:
  * Validation split is BY BOARD (board-generalization = unseen boards).
  * MAE reported in pot fractions (reach-weighted |err_chips| / pot), the
    headline; the iig_selfplay mae_frac (mae / reach-weighted |Y| scale) is
    recorded alongside.
  * float64 solver targets, fixed seeds, no Slumbot anything.

Flush thresholds: PopulationQueue subclass with the EMPIRICAL per-element
memory model measured in autoresearch-session/rebel/p2_cuda_bringup.json
(peak_bytes/B at empirical B_max: 85MB@nn=21, 271MB@nn=411, 675MB@nn=1257 ->
per_elem ~= 75e6 + 0.477e6*nn bytes; budget 6.0e9 keeps >=20% headroom below
every measured OOM point). Chunking is exactness-preserving (gate P-B).

Usage:
    .venv/bin/python scripts/train_hunl_river_net.py \
        --n-specs 20000 --seed 0 --max-gen-hours 4.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from poker_ai.rebel.hunl import lazy_subgames as lzs          # noqa: E402
from poker_ai.rebel.hunl import population_solver as pop      # noqa: E402
from poker_ai.rebel.hunl import subgame_spec as sgs           # noqa: E402
from poker_ai.rebel.hunl import targets as tgt                # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CENSUS_PATH = os.path.join(
    REPO, "autoresearch-session", "rebel", "hunl_topology_census.json")
SCALE = tgt.DEFAULT_SCALE  # 20000.0 chips (200bb)
N_BOARD = tgt.N_BOARD
N_SC = tgt.N_SCALARS
NG = sgs.N_GLOBAL_HANDS


# ---------------------------------------------------------------------------
# Census config space (the REAL trunk-reachable river entries)
# ---------------------------------------------------------------------------

def load_census_configs():
    """Live river-entry configs [(pot, s0, s1, nn, topo)] + (pot,eff)->nn map."""
    with open(CENSUS_PATH) as f:
        census = json.load(f)
    rows = census["streets"]["river"]["configs"]
    live = [(int(p), int(s0), int(s1), int(nn), topo)
            for p, s0, s1, topo, nn, _depth, _nshow in rows if nn > 1]
    by_pot_eff = {}
    for p, s0, s1, nn, topo in live:
        key = (p, min(s0, s1))
        assert key not in by_pot_eff or by_pot_eff[key] == (nn, topo), key
        by_pot_eff[key] = (nn, topo)
    return live, by_pot_eff


def make_config_sampler(live_configs):
    """Uniform over the live trunk-reachable river entries (the principled
    config distribution per p1_design.json: trunk-reach, no tuned grid).
    first_to_act=1 (BB opens postflop; census protocol, symmetric entries)."""
    n = len(live_configs)

    def sampler(rng):
        p, s0, s1, _nn, _topo = live_configs[int(rng.integers(0, n))]
        return (p, s0, s1, 1)

    return sampler


def sample_boards(n_boards, seed):
    """n distinct random 5-card river boards (sorted card tuples)."""
    rng = np.random.default_rng(seed)
    boards, seen = [], set()
    while len(boards) < n_boards:
        b = tuple(sorted(rng.choice(52, size=5, replace=False).tolist()))
        if b not in seen:
            seen.add(b)
            boards.append(b)
    return boards


# ---------------------------------------------------------------------------
# Empirically-calibrated population queue
# ---------------------------------------------------------------------------

class EmpiricalBMaxQueue(lzs.PopulationQueue):
    """PopulationQueue with flush thresholds from the MEASURED p2 peaks.

    per_elem_bytes ~= 75e6 + 0.477e6 * n_nodes (linear fit of
    peak_bytes_at_B_max / B_max across the p50/p95/max bringup classes;
    residual < 1%). ``deferred=True`` makes ``flush_all`` a no-op until
    ``finalize()`` so buckets keep accumulating ACROSS generate_river_targets
    calls (the queue's intended accumulate-across-boards role); per-bucket
    flushes at B_max still happen inside ``add`` and chunking is
    exactness-preserving (gate P-B).
    """

    PER_ELEM_BASE = 75.0e6
    PER_ELEM_PER_NODE = 0.477e6
    DRAIN_RESULT_BYTES = 4.0e9  # host-RAM bound per finalized drain call

    def __init__(self, *args, budget_bytes=6.0e9, b_cap=64, deferred=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._budget_bytes = float(budget_bytes)
        self._b_cap = int(b_cap)
        self._deferred = bool(deferred)
        self._final = False

    def b_max_for_spec(self, spec):
        nn = int(spec.tree()["n_nodes"])
        per_elem = self.PER_ELEM_BASE + self.PER_ELEM_PER_NODE * nn
        return max(1, min(self._b_cap, int(self._budget_bytes // per_elem)))

    def finalize(self):
        self._final = True

    def _solve(self, specs, b_max):
        """Parent _solve with CUDA-OOM resilience: on fragmentation OOM,
        empty the cache and retry at half the fused batch size. Chunking is
        exactness-preserving (gate P-B), so results are identical."""
        b = int(b_max)
        while True:
            try:
                return super()._solve(specs, b)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if b <= 1:
                    raise
                b = max(1, b // 2)
                print(f"[queue] CUDA OOM at B={b * 2} ({len(specs)} specs); "
                      f"retrying at B={b}", flush=True)

    def flush_all(self):
        if self._deferred and not self._final:
            return []
        if not self._deferred:
            return super().flush_all()
        # finalized staged drain: each SubgameSolveResult retains [nn,A,H]
        # float64 strategy views, so draining EVERY pending bucket in one
        # generate_river_targets call would hold all of them in host RAM at
        # once; pop buckets until the estimated result footprint hits the
        # bound, return, and let the driver call again until pending == 0.
        results = []
        drained_bytes = 0.0
        for key in list(self._buckets):
            if results and drained_bytes >= self.DRAIN_RESULT_BYTES:
                break
            bucket = self._buckets.pop(key)
            nn = int(bucket["specs"][0].tree()["n_nodes"])
            per_spec = 3.0 * nn * lzs.N_ACTIONS * sgs.RIVER_H * 8
            drained_bytes += per_spec * len(bucket["specs"])
            results.extend(self._solve(bucket["specs"], bucket["b_max"]))
        return results


# ---------------------------------------------------------------------------
# Target generation (chunked driver around generate_river_targets)
# ---------------------------------------------------------------------------

def generate_rows(boards, n_beliefs, sampler, rng, queue, label,
                  boards_per_chunk=24, max_seconds=None, chunk_log=None,
                  partial_path=None):
    """Chunked generate_river_targets over ``boards``; returns (X, Y, W, info)."""
    Xs, Ys, Ws = [], [], []
    t0 = time.perf_counter()
    n_chunks = (len(boards) + boards_per_chunk - 1) // boards_per_chunk
    n_specs_enqueued = 0
    truncated = False
    for ci in range(n_chunks):
        chunk = boards[ci * boards_per_chunk:(ci + 1) * boards_per_chunk]
        tc = time.perf_counter()
        X, Y, W = tgt.generate_river_targets(
            n_beliefs=n_beliefs, boards=chunk, config_sampler=sampler,
            rng=rng, queue=queue)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        dt = time.perf_counter() - tc
        n_specs_enqueued += n_beliefs * len(chunk)
        if X.shape[0]:
            Xs.append(X)
            Ys.append(Y)
            Ws.append(W)
        elapsed = time.perf_counter() - t0
        rate = n_specs_enqueued / elapsed
        if chunk_log is not None:
            chunk_log.append({
                "label": label, "chunk": ci, "n_boards": len(chunk),
                "chunk_seconds": round(dt, 2),
                "cum_specs_enqueued": n_specs_enqueued,
                "cum_rows_emitted": int(sum(x.shape[0] for x in Xs)),
                "cum_seconds": round(elapsed, 2),
                "cum_specs_per_second": round(rate, 4),
            })
        print(f"[{label}] chunk {ci + 1}/{n_chunks}: {dt:.1f}s, "
              f"cum {n_specs_enqueued} specs in {elapsed:.0f}s "
              f"({rate:.2f} specs/s, {1.0 / rate:.2f} s/spec)", flush=True)
        if partial_path and (ci + 1) % 15 == 0 and Xs:
            tmp = partial_path + ".tmp.npz"
            np.savez(tmp, X=np.concatenate(Xs), Y=np.concatenate(Ys),
                     W=np.concatenate(Ws), chunks_done=ci + 1)
            os.replace(tmp, partial_path)
        if max_seconds is not None and ci + 1 < n_chunks:
            projected_next = elapsed + (elapsed / (ci + 1))
            if projected_next > max_seconds:
                truncated = True
                print(f"[{label}] budget guard: stopping after chunk {ci + 1} "
                      f"({elapsed:.0f}s elapsed, {max_seconds:.0f}s budget)",
                      flush=True)
                break
    # drain the deferred queue in RAM-bounded stages
    if isinstance(queue, EmpiricalBMaxQueue):
        queue.finalize()
    for _guard in range(100000):
        X, Y, W = tgt.generate_river_targets(
            n_beliefs=0, boards=[], config_sampler=sampler, rng=rng,
            queue=queue)
        if X.shape[0]:
            Xs.append(X)
            Ys.append(Y)
            Ws.append(W)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if queue.pending == 0:
            break
    else:
        raise RuntimeError("queue drain did not terminate")
    elapsed = time.perf_counter() - t0
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, tgt.X_DIM), np.float32)
    Y = np.concatenate(Ys, axis=0) if Ys else np.zeros((0, tgt.Y_DIM), np.float32)
    W = np.concatenate(Ws, axis=0) if Ws else np.zeros((0, tgt.Y_DIM), np.float32)
    info = {
        "n_rows": int(X.shape[0]),
        "wall_seconds": round(elapsed, 2),
        "specs_per_second": round(X.shape[0] / elapsed, 4) if elapsed else None,
        "seconds_per_spec": round(elapsed / X.shape[0], 4) if X.shape[0] else None,
        "gpu_seconds_per_1k_targets": (
            round(elapsed / (X.shape[0] * 2 * sgs.RIVER_H / 1000.0), 4)
            if X.shape[0] else None),
        "n_flushes": queue.n_flushes,
        "truncated_by_budget": truncated,
    }
    return X, Y, W, info


# ---------------------------------------------------------------------------
# Row metadata (recovered exactly from x: board, pot, eff stack)
# ---------------------------------------------------------------------------

def row_meta(X, by_pot_eff):
    pots = np.round(X[:, N_BOARD].astype(np.float64) * SCALE)
    effs = np.round(X[:, N_BOARD + 1].astype(np.float64) * SCALE)
    nns = np.zeros(X.shape[0], np.int64)
    for i in range(X.shape[0]):
        nns[i] = by_pot_eff[(int(pots[i]), int(effs[i]))][0]
    return pots, effs, nns


def boards_of_rows(X):
    return [tuple(np.flatnonzero(X[i, :N_BOARD]).tolist())
            for i in range(X.shape[0])]


# ---------------------------------------------------------------------------
# The net (prototype recipe, board-general input)
# ---------------------------------------------------------------------------

class BoardGeneralRiverNet(nn.Module):
    """river_pbs_net.RiverPBSNet lifted board-general: 2 hidden ReLU layers."""

    def __init__(self, d_in=tgt.X_DIM, d_out=tgt.Y_DIM, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def _wmae(pred, y, w):
    return float((w * (pred - y).abs()).sum() / (w.sum() + 1e-9))


def train_net(Xtr, Yn_tr, Wtr, Xva, Yn_va, Wva, hidden, epochs, batch_size,
              lr, zs_lambda, seed, device, eval_every=10):
    """Train the one principled config; returns (net_cpu, metrics).

    Targets/preds in POT-NORMALIZED units. Model selection: best val
    reach-weighted MAE over the eval grid (every ``eval_every`` epochs).
    """
    dev = torch.device(device)
    torch.manual_seed(seed)
    net = BoardGeneralRiverNet(hidden=hidden).to(dev)
    Xtr_t = torch.as_tensor(Xtr, device=dev)
    Ytr_t = torch.as_tensor(Yn_tr, device=dev)
    Wtr_t = torch.as_tensor(Wtr, device=dev)
    Xva_t = torch.as_tensor(Xva, device=dev)
    Yva_t = torch.as_tensor(Yn_va, device=dev)
    Wva_t = torch.as_tensor(Wva, device=dev)
    rhs_tr = (Wtr_t * Ytr_t).sum(dim=1)  # zero-sum identity RHS from targets
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = Xtr_t.shape[0]
    best = {"val_wmae": float("inf"), "epoch": -1, "state": None}
    history = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = net(Xtr_t[idx])
            w = Wtr_t[idx]
            wmse = (w * (pred - Ytr_t[idx]) ** 2).sum() / (w.sum() + 1e-9)
            zs = (((w * pred).sum(dim=1) - rhs_tr[idx]) ** 2).mean()
            (wmse + zs_lambda * zs).backward()
            opt.step()
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                pv = net(Xva_t)
                val_wmae = _wmae(pv, Yva_t, Wva_t)
                history.append({"epoch": ep + 1, "val_wmae_potfrac": round(val_wmae, 6)})
                if val_wmae < best["val_wmae"]:
                    best = {"val_wmae": val_wmae, "epoch": ep + 1,
                            "state": {k: v.detach().cpu().clone()
                                      for k, v in net.state_dict().items()}}
    net.load_state_dict(best["state"])
    net.eval()
    with torch.no_grad():
        pv = net(Xva_t)
        pt = net(Xtr_t)
        val_wmae = _wmae(pv, Yva_t, Wva_t)
        train_wmae = _wmae(pt, Ytr_t, Wtr_t)
        val_zs = float(((Wva_t * pv).sum(1) - (Wva_t * Yva_t).sum(1)).abs().max())
        # iig_selfplay mae_frac convention: mae / reach-weighted |Y| scale
        scale_iig = float((Wva_t * Yva_t.abs()).sum() / (Wva_t.sum() + 1e-9))
    metrics = {
        "n_train": int(n), "n_val": int(Xva_t.shape[0]),
        "best_epoch": best["epoch"],
        "val_reach_weighted_mae_potfrac": round(val_wmae, 6),
        "train_reach_weighted_mae_potfrac": round(train_wmae, 6),
        "val_mae_frac_iig_convention": round(val_wmae / max(scale_iig, 1e-9), 6),
        "val_value_scale_potfrac_iig": round(scale_iig, 6),
        "val_zero_sum_residual_max_potfrac": round(val_zs, 6),
        "val_history": history,
    }
    net = net.cpu()
    return net, metrics


# ---------------------------------------------------------------------------
# Per-config breakdown + sanity gate
# ---------------------------------------------------------------------------

def per_config_breakdown(net, Xva, Yn_va, Wva, pots, effs, nns, seen_configs,
                         device):
    dev = torch.device(device)
    with torch.no_grad():
        pv = net.to(dev)(torch.as_tensor(Xva, device=dev)).cpu().numpy()
    net.cpu()
    err = np.abs(pv - Yn_va)

    def wmae_mask(mask):
        w = Wva[mask]
        return float((w * err[mask]).sum() / (w.sum() + 1e-9))

    spr = effs / np.maximum(pots, 1.0)
    out = {}
    spr_edges = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0),
                 (8.0, np.inf)]
    out["by_spr"] = [
        {"spr": f"[{lo},{hi})", "n_rows": int(m.sum()),
         "val_wmae_potfrac": round(wmae_mask(m), 6)}
        for lo, hi in spr_edges if (m := (spr >= lo) & (spr < hi)).any()]
    pot_edges = [(0, 1000), (1000, 4000), (4000, 12000), (12000, 40001)]
    out["by_pot"] = [
        {"pot": f"[{lo},{hi})", "n_rows": int(m.sum()),
         "val_wmae_potfrac": round(wmae_mask(m), 6)}
        for lo, hi in pot_edges if (m := (pots >= lo) & (pots < hi)).any()]
    nn_edges = [(2, 22), (22, 100), (100, 412), (412, 1300)]
    out["by_tree_size"] = [
        {"n_nodes": f"[{lo},{hi})", "n_rows": int(m.sum()),
         "val_wmae_potfrac": round(wmae_mask(m), 6)}
        for lo, hi in nn_edges if (m := (nns >= lo) & (nns < hi)).any()]
    seen = np.array([(int(p), int(e)) in seen_configs
                     for p, e in zip(pots, effs)])
    out["by_config_seen_in_training"] = [
        {"config_seen": bool(v), "n_rows": int(m.sum()),
         "val_wmae_potfrac": round(wmae_mask(m), 6)}
        for v in (True, False) if (m := (seen == v)).any()]
    return out


def rebuild_spec_from_row(x):
    board = tuple(int(c) for c in np.flatnonzero(x[:N_BOARD]))
    pot = int(round(float(x[N_BOARD]) * SCALE))
    eff = int(round(float(x[N_BOARD + 1]) * SCALE))
    r0 = x[N_BOARD + N_SC:N_BOARD + N_SC + NG].astype(np.float64)
    r1 = x[N_BOARD + N_SC + NG:].astype(np.float64)
    return sgs.SubgameSpec(street="river", board=board, pot=pot, stack0=eff,
                           stack1=eff, first_to_act=1, r0=r0, r1=r1)


def sanity_gate(net, Xva, Yva_chips, pots, n_iterations, device, n_spots=5):
    """Fresh direct fused solves on held-out (board, belief) rows vs the net."""
    order = np.argsort(pots)
    picks = [order[int(q * (len(order) - 1))]
             for q in np.linspace(0.0, 1.0, n_spots)]
    specs = [rebuild_spec_from_row(Xva[i]) for i in picks]
    t0 = time.perf_counter()
    results = pop.solve_population(
        specs, n_iterations=n_iterations, dtype=torch.float64, device=device,
        b_max=1, compute_value_pass=True)
    solve_secs = time.perf_counter() - t0
    dev = torch.device(device)
    with torch.no_grad():
        pred_n = net.to(dev)(
            torch.as_tensor(Xva[picks], device=dev)).cpu().numpy()
    net.cpu()
    spots = []
    for j, (i, spec, res) in enumerate(zip(picks, specs, results)):
        l2g = sgs.local_to_global(spec.board)
        pot = float(spec.pot)
        v_solve = np.concatenate([res.v0, res.v1])                  # chips, local
        v_net = np.concatenate([pred_n[j, :NG][l2g],
                                pred_n[j, NG:][l2g]]) * pot          # chips
        v_row = np.concatenate([Yva_chips[i, :NG][l2g],
                                Yva_chips[i, NG:][l2g]])             # chips
        w = np.concatenate([spec.local_ranges()[0], spec.local_ranges()[1]])
        net_err = np.abs(v_net - v_solve)
        spots.append({
            "spot": j, "board": list(spec.board), "pot": spec.pot,
            "eff_stack": min(spec.stack0, spec.stack1),
            "net_vs_solve_linf_chips": round(float(net_err.max()), 4),
            "net_vs_solve_linf_over_pot": round(float(net_err.max()) / pot, 6),
            "net_vs_solve_reach_wmae_chips": round(
                float((w * net_err).sum() / (w.sum() + 1e-9)), 4),
            "net_vs_solve_reach_wmae_over_pot": round(
                float((w * net_err).sum() / (w.sum() + 1e-9)) / pot, 6),
            "storedrow_vs_fresh_solve_linf_chips": round(
                float(np.abs(v_row - v_solve).max()), 6),
        })
    return {"protocol": {
                "n_spots": n_spots,
                "selection": "held-out val rows at pot quantiles 0..1",
                "solver": f"solve_population float64 {device} B=1 "
                          f"{n_iterations} iters (fresh, end-to-end)"},
            "solve_wall_seconds": round(solve_secs, 2),
            "spots": spots}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-specs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-beliefs-per-board", type=int, default=8)
    ap.add_argument("--val-boards", type=int, default=120)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--zs-lambda", type=float, default=1.0)
    ap.add_argument("--budget-bytes", type=float, default=5.2e9)
    ap.add_argument("--boards-per-chunk", type=int, default=24)
    ap.add_argument("--max-gen-hours", type=float, default=4.0,
                    help="wall budget for TRAIN target generation (val + "
                         "training + sanity ride on top, ~0.6h)")
    ap.add_argument("--curve-cuts", default="2500,5000,10000,20000")
    ap.add_argument("--data-out", default=None)
    ap.add_argument("--json-out", default=os.path.join(
        REPO, "autoresearch-session", "rebel", "river_net_boardgen.json"))
    ap.add_argument("--ckpt-out", default=os.path.join(
        REPO, "models", "hunl_river_net_v0.pt"))
    ap.add_argument("--skip-gen", action="store_true",
                    help="reuse --data-out npz from a previous run")
    args = ap.parse_args()

    data_out = args.data_out or os.path.join(
        REPO, "autoresearch-session", "rebel",
        f"hunl_river_net_data_seed{args.seed}.npz")
    t_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    live_configs, by_pot_eff = load_census_configs()
    sampler = make_config_sampler(live_configs)
    print(f"census: {len(live_configs)} live river-entry configs", flush=True)

    chunk_log = []
    if args.skip_gen and os.path.exists(data_out):
        d = np.load(data_out, allow_pickle=True)
        Xtr, Ytr, Wtr = d["X_train"], d["Y_train"], d["W_train"]
        Xva, Yva, Wva = d["X_val"], d["Y_val"], d["W_val"]
        gen_info = json.loads(str(d["gen_info"]))
        print(f"loaded {data_out}: {Xtr.shape[0]} train / {Xva.shape[0]} val",
              flush=True)
    else:
        n_train_boards = (args.n_specs + args.n_beliefs_per_board - 1) \
            // args.n_beliefs_per_board
        all_boards = sample_boards(args.val_boards + n_train_boards,
                                   seed=args.seed + 777)
        val_boards = all_boards[:args.val_boards]
        train_boards = all_boards[args.val_boards:]

        # --- validation set first (fixed regardless of any budget trimming)
        rng_val = np.random.default_rng([args.seed, 1])
        q_val = EmpiricalBMaxQueue(
            n_iterations=args.iters, dtype=torch.float64, device=args.device,
            budget_bytes=args.budget_bytes, deferred=True)
        Xva, Yva, Wva, val_info = generate_rows(
            val_boards, args.n_beliefs_per_board, sampler, rng_val, q_val,
            "val", boards_per_chunk=args.boards_per_chunk,
            chunk_log=chunk_log)

        # --- training set (budget-guarded)
        rng_tr = np.random.default_rng([args.seed, 2])
        q_tr = EmpiricalBMaxQueue(
            n_iterations=args.iters, dtype=torch.float64, device=args.device,
            budget_bytes=args.budget_bytes, deferred=True)
        Xtr, Ytr, Wtr, tr_info = generate_rows(
            train_boards, args.n_beliefs_per_board, sampler, rng_tr, q_tr,
            "train", boards_per_chunk=args.boards_per_chunk,
            max_seconds=args.max_gen_hours * 3600.0, chunk_log=chunk_log,
            partial_path=data_out.replace(".npz", "_partial.npz"))

        gen_info = {"val": val_info, "train": tr_info,
                    "n_val_boards": len(val_boards),
                    "n_train_boards_planned": len(train_boards)}
        np.savez_compressed(
            data_out, X_train=Xtr, Y_train=Ytr, W_train=Wtr,
            X_val=Xva, Y_val=Yva, W_val=Wva,
            gen_info=json.dumps(gen_info))
        print(f"saved dataset -> {data_out}", flush=True)

    # ---- metadata, normalization, integrity checks
    pots_tr, effs_tr, nns_tr = row_meta(Xtr, by_pot_eff)
    pots_va, effs_va, nns_va = row_meta(Xva, by_pot_eff)
    val_board_set = set(boards_of_rows(Xva))
    train_board_set = set(boards_of_rows(Xtr))
    assert not (val_board_set & train_board_set), "board split leakage!"
    Yn_tr = (Ytr / pots_tr[:, None]).astype(np.float32)
    Yn_va = (Yva / pots_va[:, None]).astype(np.float32)
    # zero-sum identity of the emitted targets vs the geometric RHS
    valid_g = sgs.global_valid_matrix()
    k = min(64, Xva.shape[0])
    r0 = Xva[:k, N_BOARD + N_SC:N_BOARD + N_SC + NG].astype(np.float64)
    r1 = Xva[:k, N_BOARD + N_SC + NG:].astype(np.float64)
    lhs = (Wva[:k].astype(np.float64) * Yn_va[:k].astype(np.float64)).sum(1)
    rhs = np.einsum("bi,ij,bj->b", r0, valid_g.astype(np.float64), r1)
    zs_target_resid = float(np.abs(lhs - rhs).max())
    print(f"target zero-sum identity residual (max over {k} val rows, "
          f"pot units): {zs_target_resid:.2e}", flush=True)

    # ---- the board-generalization curve
    cuts = sorted({min(int(c), Xtr.shape[0])
                   for c in args.curve_cuts.split(",")})
    curve = []
    final_net = None
    for cut in cuts:
        t0 = time.perf_counter()
        net, m = train_net(
            Xtr[:cut], Yn_tr[:cut], Wtr[:cut], Xva, Yn_va, Wva,
            hidden=args.hidden, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            zs_lambda=args.zs_lambda, seed=args.seed, device=args.device)
        m["train_wall_seconds"] = round(time.perf_counter() - t0, 1)
        m["n_train_boards"] = len(set(boards_of_rows(Xtr[:cut])))
        curve.append(m)
        print(f"[curve] n_train={cut}: val wMAE {m['val_reach_weighted_mae_potfrac']:.4f} "
              f"pot-frac (train {m['train_reach_weighted_mae_potfrac']:.4f}, "
              f"best ep {m['best_epoch']})", flush=True)
        if cut == cuts[-1]:
            final_net = net

    # ---- per-config breakdown (final net, unseen boards)
    seen_configs = {(int(p), int(e)) for p, e in zip(pots_tr, effs_tr)}
    breakdown = per_config_breakdown(
        final_net, Xva, Yn_va, Wva, pots_va, effs_va, nns_va, seen_configs,
        args.device)

    # ---- sanity gate: 5 held-out (board, belief) pairs vs fresh fused solves
    sanity = sanity_gate(final_net, Xva, Yva, pots_va, args.iters,
                         args.device, n_spots=5)

    peak_vram = (int(torch.cuda.max_memory_allocated())
                 if torch.cuda.is_available() else None)
    wall = time.perf_counter() - t_start

    # ---- checkpoint
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({
        "state_dict": final_net.state_dict(),
        "arch": {"class": "BoardGeneralRiverNet", "d_in": tgt.X_DIM,
                 "d_out": tgt.Y_DIM, "hidden": args.hidden},
        "conventions": {
            "x": "board 52-hot | pot/2e4 | eff_stack/2e4 | r0n[1326] | r1n[1326]",
            "y": "v0|v1 per-hand V* CFVs normalized by POT (chips/pot), "
                 "global 1326 index, zero off-board",
            "value_convention": "net-from-river-start counterfactual, "
                                "opponent-reach-weighted (population_solver "
                                "value pass)",
            "denormalize": "chips = net(x) * pot",
        },
        "training": {"seed": args.seed, "epochs": args.epochs,
                     "batch_size": args.batch_size, "lr": args.lr,
                     "zs_lambda": args.zs_lambda,
                     "n_train": int(Xtr.shape[0]),
                     "target_iters": args.iters, "target_dtype": "float64"},
        "val_metrics": curve[-1],
    }, args.ckpt_out)

    # ---- JSON report
    n_params = sum(p.numel() for p in final_net.parameters())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": "P3 river board-generalization curve (P0b bundle)",
        "protocol": {
            "targets": "generate_river_targets (parity-gated fused chain), "
                       "float64, cuda, value pass on",
            "n_iterations": args.iters,
            "config_distribution": "uniform over the "
                                   f"{len(live_configs)} live trunk-reachable "
                                   "river entries (census), first_to_act=1",
            "beliefs": f"Dirichlet a0,a1 ~ choice{tgt.DIRICHLET_ALPHAS} over "
                       "local hands, per-board",
            "n_beliefs_per_board": args.n_beliefs_per_board,
            "board_split": f"val = {args.val_boards} boards fully held out "
                           "(board-disjoint, asserted)",
            "queue": "EmpiricalBMaxQueue: per-bucket B_max = min(64, "
                     f"{args.budget_bytes:.1e} / (75e6 + 0.477e6*nn)) from "
                     "the MEASURED p2 peak-bytes model; deferred flush_all "
                     "across chunks (chunk-exactness = gate P-B); CUDA-OOM "
                     "halved-B retry (exactness-preserving, gate P-B); "
                     "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "net": f"BoardGeneralRiverNet 2706->{args.hidden}x2(ReLU)->2652, "
                   f"{n_params} params (river_pbs_net prototype recipe, one "
                   "config, no sweeps)",
            "loss": "reach-weighted MSE on pot-normalized CFVs + "
                    f"{args.zs_lambda} * zero-sum-identity penalty",
            "training": f"Adam lr={args.lr}, batch={args.batch_size}, "
                        f"epochs={args.epochs}, model selection = best val "
                        "wMAE (eval every 10 epochs), fixed seed",
            "metric": "reach-weighted MAE in pot fractions (primary; "
                      "iig_selfplay mae_frac vs |Y| scale recorded alongside)",
            "seed": args.seed,
            "governance": "float64 targets; no Slumbot anything; one net "
                          "config; fixed seeds; no commits",
        },
        "throughput": {
            "gen_info": gen_info,
            "chunk_log_tail": chunk_log[-8:],
            "reference_p50_class_seconds_per_spec": 0.77,
        },
        "dataset": {
            "n_train_rows": int(Xtr.shape[0]),
            "n_train_boards": len(train_board_set),
            "n_val_rows": int(Xva.shape[0]),
            "n_val_boards": len(val_board_set),
            "n_train_configs_distinct": len(seen_configs),
            "target_zero_sum_identity_residual_max_potfrac": zs_target_resid,
        },
        "boardgen_curve": curve,
        "per_config_breakdown_unseen_boards": breakdown,
        "sanity_gate_5_spots": sanity,
        "wall_clock_seconds_total": round(wall, 1),
        "peak_vram_bytes": peak_vram,
        "files": {"checkpoint": args.ckpt_out, "dataset": data_out,
                  "report": args.json_out},
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.json_out}\nwrote {args.ckpt_out}\n"
          f"total wall {wall / 3600.0:.2f} h, peak VRAM "
          f"{(peak_vram or 0) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
