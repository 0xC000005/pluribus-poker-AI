"""Launch-amortized variants of the flat-SoA batched CFR+ kernel -- the SEQUENTIAL-arm ablation that
the v1 draft promises in §3.3(ii): how much of the fused-vs-sequential speedup survives when the per-tree
sequential arm gets CUDA-Graphs capture / ``torch.compile(mode="reduce-overhead")`` launch amortization?

This module does NOT touch ``iig_batched.py`` (the gated reference). It builds, per compiled topology:

  ARM A  ``CompiledStepSolver``  -- ``torch.compile(mode="reduce-overhead")`` on the per-iteration step.
         The step is written purely FUNCTIONALLY (state in, new state out) so Inductor's cudagraph-trees
         can manage the cross-iteration state in its own graph pool; the two iteration-indexed scalars of
         the eager loop become device tensors / tensor inputs (w as a 0-dim f64 tensor carried through the
         step; the t%2 alternating-player parity as a precomputed pair of [n,1] is_upd tensors passed in).

  ARM B  ``GraphedSolver``       -- manual ``torch.cuda.CUDAGraph`` capture. Cross-iteration state
         (regret / stratsum / w / entry reaches) lives in static pre-allocated tensors mutated strictly
         in-place; one iteration is captured TWICE (an even/odd graph pair sharing one memory pool) to bake
         the t%2 alternating update; the linear-averaging weight w = t+1 is a device scalar incremented
         in-graph; per-iteration intermediates (sig / r0 / r1 / rc / ev / cfvnum / ownreach / pads) are
         capture-time allocations from the graph's private pool, stable across replays.

Both arms share ONE step function (``_make_step``) whose arithmetic mirrors the eager
``solve_all_keys_soa`` loop body op-for-op, in the same order -- the only intended difference vs the eager
arm is launch amortization, not numerics. float64 stays mandatory (the gated-number contract of
``iig_batched.py``); parity vs the eager reference is gated by the ablation runner before any timing.
Slumbot held-out; diagnostic/efficiency infra only.
"""
from __future__ import annotations

import numpy as np
import torch

from poker_ai.rebel.iig_batched import _DT, _regret_match


# ----------------------------------------------------------------------------------------------------
# shared step (op-for-op mirror of the solve_all_keys_soa iteration body)
# ----------------------------------------------------------------------------------------------------

def _make_step(c, averaging="linear"):
    """Pure-functional ONE CFR+ iteration over compiled topology ``c``.

    ``step(regret, stratsum, w, is_upd, r0_entry, r1_entry) -> (new_regret, new_stratsum, new_w)``
    with ``w`` a 0-dim f64 device tensor (the linear-averaging weight t+1, advanced in-step) and
    ``is_upd`` the [n,1] f64 update-player mask for this iteration's parity (t%2). All other arrays are
    closed-over constants of the topology. Arithmetic order matches the eager loop body exactly.
    """
    fwd, bwd = c.fwd, c.bwd
    mask, payoff = c.mask, c.payoff
    n, maxA, B, n_nodes, device = c.n_iids, c.maxA, c.B, c.n_nodes, c.device
    ones = torch.ones(B, dtype=_DT, device=device)
    w_step = 1.0 if averaging == "linear" else 0.0

    def step(regret, stratsum, w, is_upd, r0_entry, r1_entry):
        sig = _regret_match(regret, mask)

        # ---- forward reach pass (mirror of solve_all_keys_soa) ----
        r0 = torch.zeros(n_nodes, B, dtype=_DT, device=device)
        r1 = torch.zeros(n_nodes, B, dtype=_DT, device=device)
        rc = torch.zeros(n_nodes, B, dtype=_DT, device=device)
        r0[0] = r0_entry; r1[0] = r1_entry; rc[0] = ones
        for g in fwd:
            p = g["parent"]
            if g["kind"] == "ch":
                r0[g["child"]] = r0[p]; r1[g["child"]] = r1[p]; rc[g["child"]] = rc[p] * g["prob"]
            else:
                sg = sig[g["piid"], g["slot"].unsqueeze(1)]  # [G,B]
                if g["kind"] == "d0":
                    r0[g["child"]] = r0[p] * sg; r1[g["child"]] = r1[p]; rc[g["child"]] = rc[p]
                else:
                    r1[g["child"]] = r1[p] * sg; r0[g["child"]] = r0[p]; rc[g["child"]] = rc[p]

        # ---- backward value pass (mirror) ----
        ev = payoff.clone()
        cfvnum = torch.zeros(n, maxA, dtype=_DT, device=device)
        ownreach = torch.zeros(n, dtype=_DT, device=device)
        for g in bwd:
            ch = g["children"]
            ev_ch = ev[ch]
            if g["kind"] == "chance":
                pr = g["prob"].permute(0, 2, 1).unsqueeze(-1)
                ev[g["node"]] = (pr * ev_ch).sum(1)
            else:
                pl = g["player"]; nact = g["nact"]
                node_iid = g["iid"]
                sg = sig[node_iid]
                sg = sg[:, :, :nact].permute(0, 2, 1)
                ev[g["node"]] = (sg.unsqueeze(-1) * ev_ch).sum(1)
                own = r0[g["node"]] if pl == 0 else r1[g["node"]]
                opp = r1[g["node"]] if pl == 0 else r0[g["node"]]
                cf = opp * rc[g["node"]]
                cfv = cf.unsqueeze(1) * ev_ch[..., pl]
                cfv = cfv.permute(0, 2, 1)
                pad = torch.zeros(cfv.shape[0], B, maxA, dtype=_DT, device=device)
                pad[:, :, :nact] = cfv
                cfvnum.index_add_(0, node_iid.reshape(-1), pad.reshape(-1, maxA))
                ownreach[node_iid.reshape(-1)] = own.reshape(-1)

        # ---- regret / average-strategy update (mirror; w and parity now tensors) ----
        v = (sig * cfvnum).sum(-1, keepdim=True)
        new_regret = (regret + is_upd * mask * (cfvnum - v)).clamp_min(0.0)
        new_stratsum = stratsum + w * is_upd * ownreach.unsqueeze(1) * sig
        new_w = w + w_step
        return new_regret, new_stratsum, new_w

    return step


def _is_upd_pair(c):
    """(is_upd for even t, is_upd for odd t) -- the t%2 alternating-player masks as device tensors."""
    return ((c.player == 0).to(_DT).unsqueeze(1), (c.player == 1).to(_DT).unsqueeze(1))


def _entry_arrays(c, ranges_by_key):
    """Entry-reach numpy arrays from per-key ranges (same math as iig_batched.set_entries)."""
    r0 = np.array([float(ranges_by_key[c.node_key[b]][0][c.node_i0[b]]) for b in range(c.B)])
    r1 = np.array([float(ranges_by_key[c.node_key[b]][1][c.node_i1[b]]) for b in range(c.B)])
    return r0, r1


def _finalize(c, dlg, stratsum):
    """Average-strategy extraction + per-key dict assembly (mirror of solve_all_keys_soa's tail)."""
    ss = stratsum.sum(-1, keepdim=True)
    avg_t = torch.where(ss > 1e-12, stratsum / ss, c.mask / c.mask.sum(-1, keepdim=True).clamp_min(1.0))
    avg = avg_t.cpu().numpy()
    out = {key: {} for key in c.keys}
    by_iid_key = {}
    for key in c.keys:
        for iid in dlg._below_iids_for_key(key):
            by_iid_key[iid] = key
    for iid in c.iids:
        out[by_iid_key[iid]][iid] = avg[c.loc[iid]][:c.alen[c.loc[iid]]].copy()
    return out


# ----------------------------------------------------------------------------------------------------
# ARM A: torch.compile(mode="reduce-overhead") on the per-iteration step
# ----------------------------------------------------------------------------------------------------

class CompiledStepSolver:
    """One ``torch.compile``d step instance per topology. ``solve()`` matches ``solve_all_keys_soa``'s
    contract (same {key:{iid:avg}} output, same per-solve entry update from ``ranges_by_key``)."""

    def __init__(self, dlg, c, averaging="linear", mode="reduce-overhead", fullgraph=True):
        self.dlg, self.c = dlg, c
        self.is_upd = _is_upd_pair(c)
        self.r0e = c.r0_entry.clone()
        self.r1e = c.r1_entry.clone()
        self.step = torch.compile(_make_step(c, averaging), mode=mode, fullgraph=fullgraph,
                                  dynamic=False)

    def set_entries(self, ranges_by_key):
        r0, r1 = _entry_arrays(self.c, ranges_by_key)
        self.r0e.copy_(torch.from_numpy(r0))
        self.r1e.copy_(torch.from_numpy(r1))

    def solve(self, ranges_by_key, iters):
        c = self.c
        if ranges_by_key is not None:
            self.set_entries(ranges_by_key)
        regret = torch.zeros(c.n_iids, c.maxA, dtype=_DT, device=c.device)
        stratsum = torch.zeros(c.n_iids, c.maxA, dtype=_DT, device=c.device)
        w = torch.ones((), dtype=_DT, device=c.device)
        for t in range(iters):
            # cudagraph-trees contract for output->input feedback loops: mark the step boundary and
            # clone the (graph-pool) outputs before reusing them as inputs -- the documented pattern;
            # the 3 clone kernels per iteration are honest overhead of this tooling and stay in the
            # timed region.
            torch.compiler.cudagraph_mark_step_begin()
            nr, ns, nw = self.step(regret, stratsum, w, self.is_upd[t & 1], self.r0e, self.r1e)
            regret, stratsum, w = nr.clone(), ns.clone(), nw.clone()
        return _finalize(c, self.dlg, stratsum)


# ----------------------------------------------------------------------------------------------------
# ARM B: manual torch.cuda.CUDAGraph capture (even/odd graph pair, static in-place state)
# ----------------------------------------------------------------------------------------------------

class GraphedSolver:
    """Static-buffer in-place CFR+ iteration captured once per parity into a CUDAGraph pair, replayed
    ``iters`` times per solve. State tensors are allocated once and never rebound (graphs bake their
    addresses); ``reset()`` re-zeroes them between solves; entry reaches update via ``copy_``."""

    def __init__(self, dlg, c, averaging="linear", warmup_iters=3):
        if c.device != "cuda" and not str(c.device).startswith("cuda"):
            raise RuntimeError("GraphedSolver requires a CUDA topology")
        self.dlg, self.c = dlg, c
        self.is_upd = _is_upd_pair(c)
        self._step = _make_step(c, averaging)
        # static cross-iteration state (addresses baked into the graphs)
        self.regret = torch.zeros(c.n_iids, c.maxA, dtype=_DT, device=c.device)
        self.stratsum = torch.zeros(c.n_iids, c.maxA, dtype=_DT, device=c.device)
        self.w = torch.ones((), dtype=_DT, device=c.device)
        self.r0e = c.r0_entry.clone()
        self.r1e = c.r1_entry.clone()

        # warmup on a side stream (standard capture recipe), then reset state and capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                self._body(0); self._body(1)
        torch.cuda.current_stream().wait_stream(s)
        self.reset()

        self.g_even = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_even):
            self._body(0)
        self.g_odd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_odd, pool=self.g_even.pool()):
            self._body(1)
        self._graphs = (self.g_even, self.g_odd)

    def _body(self, parity):
        """One in-place iteration: functional step, then copy new state back into the static buffers.
        Inside capture this is recorded as one fixed kernel sequence (w advances in-graph)."""
        nr, ns, nw = self._step(self.regret, self.stratsum, self.w, self.is_upd[parity],
                                self.r0e, self.r1e)
        self.regret.copy_(nr)
        self.stratsum.copy_(ns)
        self.w.copy_(nw)

    def reset(self):
        self.regret.zero_()
        self.stratsum.zero_()
        self.w.fill_(1.0)

    def set_entries(self, ranges_by_key):
        r0, r1 = _entry_arrays(self.c, ranges_by_key)
        self.r0e.copy_(torch.from_numpy(r0))
        self.r1e.copy_(torch.from_numpy(r1))

    def solve(self, ranges_by_key, iters):
        if ranges_by_key is not None:
            self.set_entries(ranges_by_key)
        self.reset()
        g = self._graphs
        for t in range(iters):
            g[t & 1].replay()
        return _finalize(self.c, self.dlg, self.stratsum)


# ----------------------------------------------------------------------------------------------------
# eager reference arm built from the same step (sanity / debugging aid; NOT the gated reference --
# the gate runs against iig_batched.solve_all_keys_soa itself)
# ----------------------------------------------------------------------------------------------------

class EagerStepSolver:
    """The same functional step run eagerly. Used only to localize numerics differences if a parity
    gate fails (separates 'step rewrite changed math' from 'compile/capture changed math')."""

    def __init__(self, dlg, c, averaging="linear"):
        self.dlg, self.c = dlg, c
        self.is_upd = _is_upd_pair(c)
        self.step = _make_step(c, averaging)
        self.r0e = c.r0_entry.clone()
        self.r1e = c.r1_entry.clone()

    def solve(self, ranges_by_key, iters):
        c = self.c
        if ranges_by_key is not None:
            r0, r1 = _entry_arrays(c, ranges_by_key)
            self.r0e.copy_(torch.from_numpy(r0))
            self.r1e.copy_(torch.from_numpy(r1))
        regret = torch.zeros(c.n_iids, c.maxA, dtype=_DT, device=c.device)
        stratsum = torch.zeros(c.n_iids, c.maxA, dtype=_DT, device=c.device)
        w = torch.ones((), dtype=_DT, device=c.device)
        for t in range(iters):
            regret, stratsum, w = self.step(regret, stratsum, w, self.is_upd[t & 1],
                                            self.r0e, self.r1e)
        return _finalize(c, self.dlg, stratsum)
