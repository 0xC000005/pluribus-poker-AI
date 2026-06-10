#!/usr/bin/env python3
"""§3.3(ii) ABLATION: the CUDA-Graphs / torch.compile launch-amortized SEQUENTIAL arm.

THE QUESTION: how much of the fused-vs-sequential cost-per-solve speedup (§4.1 of the v1 draft) survives
when the per-tree sequential arm gets modern launch-amortization tooling? Graphs/compile amortize the
WITHIN-tree per-iteration launch overhead but still replay once per tree; fusion additionally fills
occupancy ACROSS trees. The honest measured number ships, whatever it is.

ARMS (all f64, all on precompiled topologies, §4.1 protocol: 3 discarded warmups, median of 7 repeats,
torch.cuda.Event pairs bracketed by torch.cuda.synchronize, 300 CFR+ iterations):
  fused        = solve_all_keys_soa over ALL keys at once          (unchanged, iig_batched.py)
  seq_eager    = solve_all_keys_soa per key, host loop             (unchanged -- the paper's baseline)
  seq_compile  = per-key CompiledStepSolver (torch.compile reduce-overhead)   [ARM A]
  seq_graphs   = per-key GraphedSolver (manual CUDAGraph even/odd pair)       [ARM B]

PARITY GATES (before any timing; G3 + G4, 200 iters): each amortized arm vs the EAGER f64 reference at the
SAME (per-key) batching granularity -- expected <=1e-9 max avg-strategy diff (pure reordering). An
eager-vs-eager rerun diff is recorded as the CUDA-atomics nondeterminism noise floor (index_add_ ordering).
An arm that changes numerics beyond the gate is disqualified from the speedup claim and reported as such.

Setup walls (torch.compile compilation, CUDAGraph capture, topology compiles) are excluded from solve
timings -- symmetrically with the §4.1 precompiled-topology convention -- and reported separately.
Slumbot held-out; uses_slumbot_data=false. Does not modify iig_batched.py or any protected surface.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import traceback

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key
from poker_ai.rebel.iig_batched import _compile_topology, solve_all_keys_soa
from poker_ai.rebel.iig_batched_graphed import CompiledStepSolver, GraphedSolver, EagerStepSolver


def _uniform_ranges(dlg, keys):
    return {k: (np.ones(dlg.n_priv(k, 0)) / dlg.n_priv(k, 0),
                np.ones(dlg.n_priv(k, 1)) / dlg.n_priv(k, 1)) for k in keys}


def _time(fn, torch, repeats, warmup, device):
    """(median_ms_gpu, median_ms_wall, all_gpu) for fn(), warmup discarded. §4.1 protocol."""
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
    md = 0.0
    for key in out_a:
        for iid, a in out_a[key].items():
            b = out_b[key][iid]
            md = max(md, float(np.max(np.abs(a - b))))
    return md


class GameArms:
    """Per-game arm constructions over precompiled topologies (setup walls recorded, excluded from
    solve timings)."""

    def __init__(self, dlg, keys, device, build_compile=True, build_graphs=True):
        self.dlg, self.keys, self.device = dlg, keys, device
        self.ranges = _uniform_ranges(dlg, keys)
        self.setup = {}
        t0 = time.perf_counter()
        self.comp_all = _compile_topology(dlg, self.ranges, device)
        self.comp_key = {k: _compile_topology(dlg, {k: self.ranges[k]}, device) for k in keys}
        self.setup["topology_compile_s"] = round(time.perf_counter() - t0, 3)

        self.tc, self.tc_error = {}, None
        if build_compile:
            t0 = time.perf_counter()
            try:
                for k in keys:
                    s = CompiledStepSolver(dlg, self.comp_key[k])
                    s.solve({k: self.ranges[k]}, 2)   # trigger dynamo/inductor compile now
                    self.tc[k] = s
            except Exception:
                self.tc_error = traceback.format_exc()
                self.tc = {}
            self.setup["torch_compile_s"] = round(time.perf_counter() - t0, 3)

        self.gr, self.gr_error = {}, None
        if build_graphs:
            t0 = time.perf_counter()
            try:
                for k in keys:
                    self.gr[k] = GraphedSolver(dlg, self.comp_key[k])
            except Exception:
                self.gr_error = traceback.format_exc()
                self.gr = {}
            self.setup["cuda_graph_capture_s"] = round(time.perf_counter() - t0, 3)

    # ---- arm run functions over the first n keys (n=None -> all keys) ----
    def run_fused(self, comp_all, sub, iters):
        return solve_all_keys_soa(self.dlg, {k: self.ranges[k] for k in sub}, iters,
                                  device=self.device, compiled=comp_all)

    def run_seq_eager(self, sub, iters):
        out = {}
        for k in sub:
            out.update(solve_all_keys_soa(self.dlg, {k: self.ranges[k]}, iters,
                                          device=self.device, compiled=self.comp_key[k]))
        return out

    def run_seq_compile(self, sub, iters):
        out = {}
        for k in sub:
            out.update(self.tc[k].solve({k: self.ranges[k]}, iters))
        return out

    def run_seq_graphs(self, sub, iters):
        out = {}
        for k in sub:
            out.update(self.gr[k].solve({k: self.ranges[k]}, iters))
        return out


def parity_gate(arms, iters, torch):
    """Each amortized arm vs the eager per-key reference (same device, same granularity), plus the
    eager-rerun atomics noise floor and the fused-vs-sequential cross-granularity context number."""
    keys = arms.keys
    out_eager = arms.run_seq_eager(keys, iters)
    out_eager2 = arms.run_seq_eager(keys, iters)
    res = {"iters": iters,
           "eager_rerun_noise_floor": _max_strategy_diff(out_eager, out_eager2)}
    if arms.tc:
        try:
            res["seq_compile_vs_eager"] = _max_strategy_diff(arms.run_seq_compile(keys, iters),
                                                             out_eager)
        except Exception:
            res["seq_compile_error"] = traceback.format_exc(); arms.tc = {}
    if arms.gr:
        try:
            res["seq_graphs_vs_eager"] = _max_strategy_diff(arms.run_seq_graphs(keys, iters),
                                                            out_eager)
        except Exception:
            res["seq_graphs_error"] = traceback.format_exc(); arms.gr = {}
    # localization aid: the rewritten step run EAGERLY (no compile/capture) vs the reference
    es = {}
    for k in keys:
        es.update(EagerStepSolver(arms.dlg, arms.comp_key[k]).solve({k: arms.ranges[k]}, iters))
    res["eager_step_rewrite_vs_eager"] = _max_strategy_diff(es, out_eager)
    res["fused_vs_seq_eager_context"] = _max_strategy_diff(
        arms.run_fused(arms.comp_all, keys, iters), out_eager)
    return res


def time_arms(arms, sub, comp_all, iters, torch, repeats, warmup):
    row = {"n_subgames": len(sub), "iters": iters}
    g, w, _ = _time(lambda: arms.run_fused(comp_all, sub, iters), torch, repeats, warmup, arms.device)
    row["fused_gpu_ms"], row["fused_wall_ms"] = round(g, 3), round(w, 3)
    g, w, _ = _time(lambda: arms.run_seq_eager(sub, iters), torch, repeats, warmup, arms.device)
    row["seq_eager_gpu_ms"], row["seq_eager_wall_ms"] = round(g, 3), round(w, 3)
    if arms.tc:
        try:
            g, w, _ = _time(lambda: arms.run_seq_compile(sub, iters), torch, repeats, warmup,
                            arms.device)
            row["seq_compile_gpu_ms"], row["seq_compile_wall_ms"] = round(g, 3), round(w, 3)
        except Exception:
            row["seq_compile_error"] = traceback.format_exc(); arms.tc = {}
    if arms.gr:
        try:
            g, w, _ = _time(lambda: arms.run_seq_graphs(sub, iters), torch, repeats, warmup,
                            arms.device)
            row["seq_graphs_gpu_ms"], row["seq_graphs_wall_ms"] = round(g, 3), round(w, 3)
        except Exception:
            row["seq_graphs_error"] = traceback.format_exc(); arms.gr = {}
    # derived speedups (GPU-event medians)
    f = row["fused_gpu_ms"]
    for arm in ("seq_eager", "seq_compile", "seq_graphs"):
        k = f"{arm}_gpu_ms"
        if k in row and f > 0:
            row[f"speedup_fused_vs_{arm}"] = round(row[k] / f, 2)
    seq_best = min(v for k, v in row.items()
                   if k.endswith("_gpu_ms") and k.startswith("seq"))
    row["best_seq_arm"] = min((k for k in row if k.startswith("seq") and k.endswith("_gpu_ms")),
                              key=lambda k: row[k]).replace("_gpu_ms", "")
    row["speedup_fused_vs_best_seq"] = round(seq_best / f, 2) if f > 0 else None
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--parity-games", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--parity-iters", type=int, default=200)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--key-sweeps", type=str, nargs="+", default=["4:1,2,4,8,12", "5:1,2,4,8,15"],
                    help="game:N,N,... controlled population-size sweeps")
    ap.add_argument("--skip-compile-arm", action="store_true")
    ap.add_argument("--skip-graphs-arm", action="store_true")
    ap.add_argument("--output-json", default="autoresearch-session/rebel/cuda_graphs_ablation.json")
    args = ap.parse_args(argv)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "this ablation targets the CUDA launch-amortization question"
    out = {"experiment": "cuda_graphs_torchcompile_sequential_arm_ablation",
           "paper_promise": "ed_v1_draft/00_v1_r4.md section 3.3(ii)",
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "dtype": "float64 (mandatory for gated numbers)",
           "protocol": {"iters": args.iters, "repeats": args.repeats, "warmup": args.warmup,
                        "timing": "torch.cuda.Event pairs + synchronize, median of repeats, "
                                  "precompiled topologies; compile/capture walls excluded and "
                                  "reported under setup_walls"},
           "uses_slumbot_data": False,
           "parity": {}, "rows": [], "keysweeps": {}, "setup_walls": {}, "errors": {}, "notes": []}
    outp = pathlib.Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)

    sweeps = {}
    for s in args.key_sweeps:
        gname, ns = s.split(":")
        sweeps[int(gname)] = [int(x) for x in ns.split(",")]

    def flush():
        outp.write_text(json.dumps(out, indent=2))

    for nc in args.games:
        gkey = f"goofspiel{nc}"
        print(f"=== G{nc}: building game + topologies + amortized arms ===", flush=True)
        dlg = DepthLimitedGame(load_goofspiel(nc), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
        keys = sorted({n[1] for n in dlg.cut_nodes})
        arms = GameArms(dlg, keys, device,
                        build_compile=not args.skip_compile_arm,
                        build_graphs=not args.skip_graphs_arm)
        out["setup_walls"][gkey] = arms.setup
        if arms.tc_error:
            out["errors"][f"{gkey}_torch_compile"] = arms.tc_error
            print(f"  !! torch.compile arm FAILED to build:\n{arms.tc_error}", flush=True)
        if arms.gr_error:
            out["errors"][f"{gkey}_cuda_graphs"] = arms.gr_error
            print(f"  !! CUDAGraph arm FAILED to build:\n{arms.gr_error}", flush=True)
        print(f"  setup walls: {arms.setup}", flush=True)
        flush()

        if nc in args.parity_games:
            print(f"  parity gate ({args.parity_iters} iters)...", flush=True)
            par = parity_gate(arms, args.parity_iters, torch)
            out["parity"][gkey] = par
            print(f"  parity: {json.dumps(par)}", flush=True)
            flush()

        print(f"  timing cross-game point ({len(keys)} keys, B={arms.comp_all.B})...", flush=True)
        row = time_arms(arms, keys, arms.comp_all, args.iters, torch, args.repeats, args.warmup)
        row["game"] = gkey; row["n_iset"] = dlg.n_iset; row["B_cut_nodes"] = int(arms.comp_all.B)
        out["rows"].append(row)
        print(f"  {gkey}: {json.dumps(row)}", flush=True)
        flush()

        if nc in sweeps:
            ksrows = []
            for n in sweeps[nc]:
                sub = keys[:n]
                comp_n = (arms.comp_all if n == len(keys) else
                          _compile_topology(dlg, {k: arms.ranges[k] for k in sub}, device))
                r = time_arms(arms, sub, comp_n, args.iters, torch, args.repeats, args.warmup)
                ksrows.append(r)
                print(f"    sweep n={n}: {json.dumps(r)}", flush=True)
                out["keysweeps"][gkey] = ksrows
                flush()

        del arms, dlg
        torch.cuda.empty_cache()

    flush()
    print(f"wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
