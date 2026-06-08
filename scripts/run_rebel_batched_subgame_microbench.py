#!/usr/bin/env python3
"""GATE 0b (representative): does GPU-batching a POPULATION of TINY same-topology depth-limited PBS
subgames give the large throughput win the rank-1 lever needs? GATE 0 Probe B measured the WRONG tree
(poker river, n=1081, replicated matrices = the worst case where serial GPU is already saturated and the
batch OOMs). The real generic subgames have TINY private dims (Leduc 6, Goofspiel-4 4) -- the "GPU slow on
small trees" regime where serial GPU is launch-bound and batching should help MOST.

This is the batched CFR+ KERNEL CORE (the first real increment of the eventual port), on a synthetic but
faithful depth-limited subgame: 2 public decision levels (P0 then P1, public actions observed) over a
bilinear showdown U[a,b][hero_priv, villain_priv], entered at PBS beliefs (r0, r1). It runs vanilla CFR+
fully BATCHED over B subgames (same topology, different U + beliefs) as batched matmuls. We measure
throughput (subgames/sec) for GPU-batched vs GPU-serial vs CPU-serial, with a batched-vs-serial PARITY
check and a subgame-exploitability convergence check so the speedup is from a CORRECT solve. Slumbot
held-out; diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch


def _regret_match_plus(regret):
    """regret: [B,P,A] -> strategy [B,P,A] (CFR+: regrets already kept >=0)."""
    pos = regret.clamp_min(0.0)
    s = pos.sum(-1, keepdim=True)
    A = regret.shape[-1]
    return torch.where(s > 1e-12, pos / s, torch.full_like(pos, 1.0 / A))


def solve_subgames(U, r0, r1, iters, device):
    """Batched CFR+ for B 2-level depth-limited subgames. U: [B,A,A,P,P] payoff to P0 for (a,b) over
    (hero_priv, villain_priv); r0,r1: [B,P] entry beliefs. Returns avg strategies + exploitability."""
    U = U.to(device); r0 = r0.to(device); r1 = r1.to(device)
    B, A, _, P, _ = U.shape
    reg0 = torch.zeros(B, P, A, device=device)          # P0 root infoset
    reg1 = torch.zeros(B, A, P, A, device=device)        # P1 infosets, one per P0 action a
    ssum0 = torch.zeros(B, P, A, device=device)
    ssum1 = torch.zeros(B, A, P, A, device=device)
    wsum = 0.0

    def cfvs(s0, s1):
        # v0[b,i,a] = sum_b' sum_j r1[j]*s1[b,a,j,b'] * U[b,a,b',i,j]
        vil = r1[:, None, :, None] * s1                      # [B,A,P,A] villain reach*strategy
        # contract over villain priv j and action b': einsum
        v0 = torch.einsum('bayj,bayij->bia', vil.transpose(2, 3), U)  # [B,P,A]
        # v1[b,a,j,b'] = -sum_i r0[i]*s0[b,i,a]*U[b,a,b',i,j]
        her = r0[:, :, None] * s0                            # [B,P,A] hero reach*strategy
        v1 = -torch.einsum('bia,baeij->baje', her, U)        # [B,A,P,A] (e indexes b')
        return v0, v1

    for t in range(iters):
        s0 = _regret_match_plus(reg0)
        s1 = _regret_match_plus(reg1)
        v0, v1 = cfvs(s0, s1)
        ev0 = (s0 * v0).sum(-1, keepdim=True)                # [B,P,1]
        ev1 = (s1 * v1).sum(-1, keepdim=True)                # [B,A,P,1]
        reg0 = (reg0 + (v0 - ev0)).clamp_min(0.0)
        reg1 = (reg1 + (v1 - ev1)).clamp_min(0.0)
        w = float(t + 1)                                     # linear CFR+ averaging
        ssum0 += w * r0[:, :, None] * s0
        ssum1 += w * (r1[:, None, :, None] * s1)
        wsum += w
    avg0 = ssum0 / ssum0.sum(-1, keepdim=True).clamp_min(1e-12)
    avg1 = ssum1 / ssum1.sum(-1, keepdim=True).clamp_min(1e-12)

    # subgame exploitability (best-response gap) under the average profile
    v0, v1 = cfvs(avg0, avg1)
    val0 = (r0 * (avg0 * v0).sum(-1)).sum(-1)                # [B]
    br0 = (r0 * v0.max(-1).values).sum(-1)                   # [B]
    val1 = (r1[:, None, :] * (avg1 * v1).sum(-1)).sum((-1, -2))
    br1 = (r1[:, None, :] * v1.max(-1).values).sum((-1, -2))
    exploit = ((br0 - val0) + (br1 - val1))                  # [B] subgame NashConv
    return avg0, avg1, exploit


def _make_batch(B, A, P, seed):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(B, A, A, P, P, generator=g)
    r0 = torch.rand(B, P, generator=g) + 1e-2; r0 /= r0.sum(-1, keepdim=True)
    r1 = torch.rand(B, P, generator=g) + 1e-2; r1 /= r1.sum(-1, keepdim=True)
    return U, r0, r1


def _time(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=6)        # private dim (Leduc 6, G4 4)
    ap.add_argument("--A", type=int, default=3)        # public actions per node
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-sizes", default="1,64,256,1024,4096")
    ap.add_argument("--cpu-serial-cap", type=int, default=256)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    gpu = "cuda" if torch.cuda.is_available() else None
    Bs = [int(x) for x in args.batch_sizes.split(",")]
    out = {"P": args.P, "A": args.A, "iters": args.iters, "gpu": bool(gpu), "sweep": []}
    print(f"Batched-subgame CFR+ microbench (P={args.P}, A={args.A}, iters={args.iters}, "
          f"gpu={'yes' if gpu else 'NO'})")

    # parity + convergence sanity at B=64 (batched vs serial must match; exploitability must -> ~0)
    Up, r0p, r1p = _make_batch(64, args.A, args.P, seed=1)
    a0b, a1b, expl = solve_subgames(Up, r0p, r1p, args.iters, gpu or "cpu")
    a0s, a1s, _ = [], [], None
    for b in range(8):
        x0, x1, _ = solve_subgames(Up[b:b + 1], r0p[b:b + 1], r1p[b:b + 1], args.iters, "cpu")
        a0s.append(x0); a1s.append(x1)
    par = max(float((a0b[b].cpu() - a0s[b]).abs().max()) for b in range(8))
    out["parity_batched_vs_serial_max_l_inf"] = round(par, 6)
    out["mean_subgame_exploitability"] = round(float(expl.mean()), 5)
    print(f"  sanity: batched-vs-serial strategy max|diff|={par:.2e}; mean subgame exploitability "
          f"{out['mean_subgame_exploitability']:.4f} (CFR+ -> ~0 = correct solve)")

    for B in Bs:
        row = {"B": B}
        U, r0, r1 = _make_batch(B, args.A, args.P, seed=2)
        # CPU serial (proxy for the current numpy/Python substrate's per-subgame solve loop)
        if B <= args.cpu_serial_cap:
            tc = _time(lambda: [solve_subgames(U[b:b + 1], r0[b:b + 1], r1[b:b + 1], args.iters, "cpu")
                                for b in range(B)], "cpu")
            row["cpu_serial_s"] = round(tc, 3); row["cpu_serial_subg_per_s"] = round(B / tc, 1)
        if gpu:
            tgb = _time(lambda: solve_subgames(U, r0, r1, args.iters, "cuda"), "cuda")
            row["gpu_batched_s"] = round(tgb, 3); row["gpu_batched_subg_per_s"] = round(B / tgb, 1)
            if B <= args.cpu_serial_cap:
                tgs = _time(lambda: [solve_subgames(U[b:b + 1], r0[b:b + 1], r1[b:b + 1], args.iters, "cuda")
                                     for b in range(B)], "cuda")
                row["gpu_serial_s"] = round(tgs, 3); row["gpu_serial_subg_per_s"] = round(B / tgs, 1)
                row["batched_vs_gpu_serial_x"] = round(tgs / tgb, 1)
            if "cpu_serial_s" in row:
                row["batched_vs_cpu_serial_x"] = round(row["cpu_serial_s"] / tgb, 1)
        out["sweep"].append(row)
        msg = f"  B={B:>5}: "
        if "gpu_batched_subg_per_s" in row:
            msg += f"GPU-batched {row['gpu_batched_subg_per_s']:>9.0f} subg/s"
        if "gpu_serial_subg_per_s" in row:
            msg += f" | GPU-serial {row['gpu_serial_subg_per_s']:.0f}"
        if "cpu_serial_subg_per_s" in row:
            msg += f" | CPU-serial {row['cpu_serial_subg_per_s']:.0f}"
        if "batched_vs_cpu_serial_x" in row:
            msg += f"  => {row['batched_vs_cpu_serial_x']}x vs CPU-serial"
        print(msg, flush=True)

    peak = max((r.get("gpu_batched_subg_per_s", 0) for r in out["sweep"]), default=0)
    base = next((r.get("cpu_serial_subg_per_s", 0) for r in out["sweep"] if r["B"] == 1), 0)
    out["peak_gpu_batched_subg_per_s"] = peak
    out["peak_speedup_vs_cpu_serial"] = round(peak / base, 1) if base else None
    print(f"\n  PEAK GPU-batched throughput: {peak:.0f} subgames/s; "
          f"vs CPU-serial(B=1) {base:.0f}/s -> {out['peak_speedup_vs_cpu_serial']}x")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
