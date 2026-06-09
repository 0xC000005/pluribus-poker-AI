#!/usr/bin/env python3
"""#43 HEADLINE: the cost-per-solve bake-off -- the contribution's ROBUST, deterministic efficiency claim.

The novel axis (verified unclaimed): iterated-resolving cost = (#solves)(iters/solve)(COST-PER-SOLVE). All
published lines attack the first three; the 4th -- cost-per-solve via cross-key GEMM-fusion of a POPULATION
of same-topology depth-limited PBS subgames (the inner re-solve of train-time continual resolving) -- is the
contribution. This script measures it head-to-head:

  FUSED      = solve_all_keys_soa(dlg, ALL keys, iters)            -- one level-grouped GEMM over the whole
                                                                      population of cut subgames (the mechanism)
  SEQUENTIAL = for key in keys: solve_all_keys_soa(dlg, {key}, .)  -- the SAME GPU SoA CFR+ machinery applied
                                                                      ONE subgame-tree at a time (the fair
                                                                      within-tree-GPU-CFR baseline, Kim 2408.14778)

Both arms are the GPU SoA path (NOT the Python reference walk solve_all_keys -- that would strawman the
speedup with interpreter overhead). Pre-compiled topologies are reused so we time the PURE SOLVE (the inner
re-solve cost, what iterated resolving pays per step), not one-time compile. Warmup + repeats + median; CUDA
events for GPU time + perf_counter for wall. Equilibrium PARITY is checked (max avg-strategy diff) -- a
speedup is only meaningful if both arms solve to the same answer. Slumbot held-out; uses_slumbot_data=false.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key
from poker_ai.rebel.iig_batched import _compile_topology, solve_all_keys_soa


def _uniform_ranges(dlg, keys):
    return {k: (np.ones(dlg.n_priv(k, 0)) / dlg.n_priv(k, 0),
                np.ones(dlg.n_priv(k, 1)) / dlg.n_priv(k, 1)) for k in keys}


def _time(fn, torch, repeats, warmup, device):
    """(median_ms_gpu, median_ms_wall, all_gpu) for fn(), warmup discarded."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); ev0.record()
            fn()
            ev1.record(); torch.cuda.synchronize(); t1 = time.perf_counter()
            gpu.append(ev0.elapsed_time(ev1)); wall.append((t1 - t0) * 1e3)
        else:
            t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
            wall.append((t1 - t0) * 1e3); gpu.append((t1 - t0) * 1e3)
    return statistics.median(gpu), statistics.median(wall), gpu


def _max_strategy_diff(out_a, out_b):
    """Max abs diff of average strategies between two {key:{iid:arr}} solves (equilibrium parity)."""
    md = 0.0
    for key in out_a:
        for iid, a in out_a[key].items():
            b = out_b[key][iid]
            md = max(md, float(np.max(np.abs(a - b))))
    return md


def bakeoff_one(dlg, keys, iters, torch, device, repeats, warmup):
    ranges = _uniform_ranges(dlg, keys)
    comp_all = _compile_topology(dlg, ranges, device)               # fused: one topology over all keys
    comp_key = {k: _compile_topology(dlg, {k: ranges[k]}, device) for k in keys}  # per-tree topologies
    B_total = comp_all.B                                            # population size = total cut nodes

    def run_fused():
        return solve_all_keys_soa(dlg, ranges, iters, device=device, compiled=comp_all)

    def run_seq():
        out = {}
        for k in keys:
            out.update(solve_all_keys_soa(dlg, {k: ranges[k]}, iters, device=device, compiled=comp_key[k]))
        return out

    out_f = run_fused(); out_s = run_seq()
    parity = _max_strategy_diff(out_f, out_s)
    g_f, w_f, _ = _time(run_fused, torch, repeats, warmup, device)
    g_s, w_s, _ = _time(run_seq, torch, repeats, warmup, device)
    return {"n_keys": len(keys), "B_cut_nodes": int(B_total), "iters": iters,
            "fused_gpu_ms": round(g_f, 3), "seq_gpu_ms": round(g_s, 3),
            "fused_wall_ms": round(w_f, 3), "seq_wall_ms": round(w_s, 3),
            "speedup_gpu": round(g_s / g_f, 2) if g_f > 0 else None,
            "speedup_wall": round(w_s / w_f, 2) if w_f > 0 else None,
            "parity_max_strategy_diff": round(parity, 6)}


def bakeoff_keysweep(dlg, keys, iters, torch, device, repeats, warmup):
    """CONTROLLED population-size axis: within ONE game (so per-subgame size ~fixed), sweep the NUMBER of
    co-solved subgames n=1,2,4,...,|keys| -- fused-over-first-n vs sequential-over-first-n. Isolates the
    GEMM-fusion-of-a-population benefit from subgame depth (which the cross-game sweep conflates)."""
    ranges = _uniform_ranges(dlg, keys)
    comp_key = {k: _compile_topology(dlg, {k: ranges[k]}, device) for k in keys}
    ns, n = [], 1
    while n < len(keys):
        ns.append(n); n *= 2
    ns.append(len(keys))
    rows = []
    for n in ns:
        sub = keys[:n]
        comp_all = _compile_topology(dlg, {k: ranges[k] for k in sub}, device)

        def run_fused(sub=sub, comp_all=comp_all):
            return solve_all_keys_soa(dlg, {k: ranges[k] for k in sub}, iters, device=device, compiled=comp_all)

        def run_seq(sub=sub):
            out = {}
            for k in sub:
                out.update(solve_all_keys_soa(dlg, {k: ranges[k]}, iters, device=device, compiled=comp_key[k]))
            return out

        g_f, _, _ = _time(run_fused, torch, repeats, warmup, device)
        g_s, _, _ = _time(run_seq, torch, repeats, warmup, device)
        rows.append({"n_subgames": n, "fused_gpu_ms": round(g_f, 3), "seq_gpu_ms": round(g_s, 3),
                     "speedup_gpu": round(g_s / g_f, 2) if g_f > 0 else None})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cards", type=int, nargs="+", default=[3, 4, 5],
                    help="Goofspiel sizes to sweep (population grows with size)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--key-sweep", type=int, nargs="*", default=[],
                    help="Goofspiel size(s) for the CONTROLLED within-game population-size sweep")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"#43 cost-per-solve bake-off (fused population-GEMM vs sequential per-tree GPU-CFR), device={device}",
          flush=True)

    rows = []
    for nc in args.num_cards:
        dlg = DepthLimitedGame(load_goofspiel(nc), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
        keys = sorted({n[1] for n in dlg.cut_nodes})
        r = bakeoff_one(dlg, keys, args.iters, torch, device, args.repeats, args.warmup)
        r["game"] = f"goofspiel{nc}"; r["n_iset"] = dlg.n_iset
        rows.append(r)
        print(f"  G{nc} (n_iset={dlg.n_iset}, {r['n_keys']} keys, B={r['B_cut_nodes']} cut nodes): "
              f"fused {r['fused_gpu_ms']}ms | seq {r['seq_gpu_ms']}ms | speedup {r['speedup_gpu']}x (GPU) / "
              f"{r['speedup_wall']}x (wall) | parity {r['parity_max_strategy_diff']:.2e}", flush=True)

    keysweeps = {}
    for nc in args.key_sweep:
        dlg = DepthLimitedGame(load_goofspiel(nc), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
        keys = sorted({n[1] for n in dlg.cut_nodes})
        ks = bakeoff_keysweep(dlg, keys, args.iters, torch, device, args.repeats, args.warmup)
        keysweeps[f"goofspiel{nc}"] = ks
        print(f"\n  Controlled population-size sweep on G{nc} (fixed per-subgame size, vary # co-solved):",
              flush=True)
        for r in ks:
            print(f"    n_subgames={r['n_subgames']:>3d}: fused {r['fused_gpu_ms']}ms | seq {r['seq_gpu_ms']}ms "
                  f"-> {r['speedup_gpu']}x", flush=True)

    out = {"experiment": "costpersolve_bakeoff_fused_vs_sequential", "device": device,
           "iters": args.iters, "repeats": args.repeats, "rows": rows, "keysweeps": keysweeps}
    print(f"\n  Cost-per-solve multiplier (population-GEMM fusion, GPU time):")
    for r in rows:
        print(f"    {r['game']:>11s}  B={r['B_cut_nodes']:>5d}  ->  {r['speedup_gpu']}x", flush=True)
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
