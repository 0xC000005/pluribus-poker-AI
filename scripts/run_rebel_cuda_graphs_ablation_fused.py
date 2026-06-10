#!/usr/bin/env python3
"""§3.3(ii) ABLATION COMPLETION: the missing symmetric arm -- the AMORTIZED FUSED solver.

The prior run (autoresearch-session/rebel/cuda_graphs_ablation.json) applied CompiledStepSolver
(torch.compile reduce-overhead) and GraphedSolver (manual CUDAGraph capture) only to the SEQUENTIAL
(per-key) arms. But the FUSED arm (one compiled topology over ALL keys) is itself a compiled topology --
the same machinery applies directly. This runner builds the two amortized FUSED arms:

  fused_compile = CompiledStepSolver over the all-keys topology   (torch.compile reduce-overhead)
  fused_graphs  = GraphedSolver     over the all-keys topology    (manual CUDAGraph even/odd pair)

PARITY GATES (before any timing; G3/G4/G5, 200 iters, f64): each amortized fused arm vs the EAGER FUSED
f64 reference (solve_all_keys_soa over the same all-keys topology) -- expected <=1e-9 max avg-strategy
diff. Eager-fused rerun diff recorded as the CUDA-atomics noise floor; the functional step-rewrite run
eagerly is the localization aid. An arm beyond the gate is disqualified and reported as such.

TIMING (same §4.1 protocol as the prior run: 3 discarded warmups, median of 7 repeats, torch.cuda.Event
pairs bracketed by synchronize, 300 CFR+ iterations, f64, precompiled topologies; compile/capture walls
excluded and reported under setup_walls): fused_eager (re-anchored this session) + fused_compile +
fused_graphs on the G3/G4/G5 cross-game points and the G4 (N=1,2,4,8,12) / G5 (N=1,2,4,8,15) sweeps.

ANALYSIS (recomputed at the end of every invocation from whatever is present): the complete 6-arm
{fused, sequential} x {eager, compile, graphs} picture, merging this session's fused arms with the prior
artifact's sequential arms (the re-measured fused_eager vs the prior fused number is the cross-session
drift anchor); the headline AMORTIZED-FUSED vs AMORTIZED-SEQUENTIAL (best tool per arm) speedups; the
amortized-fused m_probe; and the new crossover N* (same formula as the prior artifact:
best_arm(Nmax) / best_other_arm_endpoint_slope, plus the interpolated empirical crossing).

Appends under the new top-level key 'amortized_fused' of the existing artifact; preserves all prior
content. Slumbot held-out; uses_slumbot_data=false. Does not modify iig_batched.py or any protected
surface. Run one game per process (dynamo cache hygiene + GPU memory hygiene):
    .venv/bin/python scripts/run_rebel_cuda_graphs_ablation_fused.py --games 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_rebel_cuda_graphs_ablation import _uniform_ranges, _time, _max_strategy_diff  # noqa: E402

from poker_ai.rebel.iig_solve import DepthLimitedGame                                  # noqa: E402
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key  # noqa: E402
from poker_ai.rebel.iig_batched import _compile_topology, solve_all_keys_soa           # noqa: E402
from poker_ai.rebel.iig_batched_graphed import (CompiledStepSolver, GraphedSolver,     # noqa: E402
                                                EagerStepSolver)


class FusedArms:
    """fused_eager / fused_compile / fused_graphs over ONE compiled topology covering ``sub`` keys.
    Setup walls (topology compile, torch.compile, CUDAGraph capture) recorded, excluded from timings."""

    def __init__(self, dlg, ranges, sub, device, comp=None, build_compile=True, build_graphs=True):
        self.dlg, self.ranges, self.sub, self.device = dlg, ranges, sub, device
        self.setup = {}
        t0 = time.perf_counter()
        self.comp = comp if comp is not None else _compile_topology(
            dlg, {k: ranges[k] for k in sub}, device)
        self.setup["topology_compile_s"] = round(time.perf_counter() - t0, 3)

        self.tc, self.tc_error = None, None
        if build_compile:
            t0 = time.perf_counter()
            try:
                s = CompiledStepSolver(dlg, self.comp)
                s.solve(self.rngs(), 2)   # trigger dynamo/inductor compile now (setup wall)
                self.tc = s
            except Exception:
                self.tc_error = traceback.format_exc()
            self.setup["torch_compile_s"] = round(time.perf_counter() - t0, 3)

        self.gr, self.gr_error = None, None
        if build_graphs:
            t0 = time.perf_counter()
            try:
                self.gr = GraphedSolver(dlg, self.comp)
            except Exception:
                self.gr_error = traceback.format_exc()
            self.setup["cuda_graph_capture_s"] = round(time.perf_counter() - t0, 3)

    def rngs(self):
        return {k: self.ranges[k] for k in self.sub}

    def run_eager(self, iters):
        return solve_all_keys_soa(self.dlg, self.rngs(), iters, device=self.device,
                                  compiled=self.comp)

    def run_compile(self, iters):
        return self.tc.solve(self.rngs(), iters)

    def run_graphs(self, iters):
        return self.gr.solve(self.rngs(), iters)


def parity_gate(fa, iters):
    """Each amortized FUSED arm vs the eager FUSED reference on the same all-keys topology, plus the
    eager-rerun atomics noise floor and the eager step-rewrite localization aid. 200 iters, f64."""
    out_eager = fa.run_eager(iters)
    out_eager2 = fa.run_eager(iters)
    res = {"iters": iters,
           "eager_fused_rerun_noise_floor": _max_strategy_diff(out_eager, out_eager2)}
    if fa.tc is not None:
        try:
            res["fused_compile_vs_eager_fused"] = _max_strategy_diff(
                fa.run_compile(iters), out_eager)
        except Exception:
            res["fused_compile_error"] = traceback.format_exc(); fa.tc = None
    if fa.gr is not None:
        try:
            res["fused_graphs_vs_eager_fused"] = _max_strategy_diff(
                fa.run_graphs(iters), out_eager)
        except Exception:
            res["fused_graphs_error"] = traceback.format_exc(); fa.gr = None
    res["eager_step_rewrite_vs_eager_fused"] = _max_strategy_diff(
        EagerStepSolver(fa.dlg, fa.comp).solve(fa.rngs(), iters), out_eager)
    gate = 1e-9
    res["gate"] = gate
    res["fused_compile_pass"] = (res.get("fused_compile_vs_eager_fused", float("inf")) <= gate)
    res["fused_graphs_pass"] = (res.get("fused_graphs_vs_eager_fused", float("inf")) <= gate)
    return res


def time_point(fa, iters, torch, repeats, warmup):
    row = {"n_subgames": len(fa.sub), "iters": iters}
    g, w, _ = _time(lambda: fa.run_eager(iters), torch, repeats, warmup, fa.device)
    row["fused_eager_gpu_ms"], row["fused_eager_wall_ms"] = round(g, 3), round(w, 3)
    if fa.tc is not None:
        try:
            g, w, _ = _time(lambda: fa.run_compile(iters), torch, repeats, warmup, fa.device)
            row["fused_compile_gpu_ms"], row["fused_compile_wall_ms"] = round(g, 3), round(w, 3)
        except Exception:
            row["fused_compile_error"] = traceback.format_exc(); fa.tc = None
    if fa.gr is not None:
        try:
            g, w, _ = _time(lambda: fa.run_graphs(iters), torch, repeats, warmup, fa.device)
            row["fused_graphs_gpu_ms"], row["fused_graphs_wall_ms"] = round(g, 3), round(w, 3)
        except Exception:
            row["fused_graphs_error"] = traceback.format_exc(); fa.gr = None
    cands = {a: row[f"{a}_gpu_ms"] for a in ("fused_compile", "fused_graphs")
             if f"{a}_gpu_ms" in row}
    if cands:
        best = min(cands, key=cands.get)
        row["best_fused_amortized_arm"] = best
        row["best_fused_amortized_gpu_ms"] = cands[best]
        e = row["fused_eager_gpu_ms"]
        for a, v in cands.items():
            row[f"speedup_{a}_vs_fused_eager"] = round(e / v, 2) if v > 0 else None
    return row


def measure_game(nc, args, torch, out_af, flush):
    gkey = f"goofspiel{nc}"
    device = "cuda"
    print(f"=== G{nc}: building game + fused all-keys topology + amortized fused arms ===", flush=True)
    dlg = DepthLimitedGame(load_goofspiel(nc), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    keys = sorted({n[1] for n in dlg.cut_nodes})
    ranges = _uniform_ranges(dlg, keys)
    fa = FusedArms(dlg, ranges, keys, device)
    out_af.setdefault("setup_walls", {})[gkey] = dict(fa.setup)
    if fa.tc_error:
        out_af.setdefault("errors", {})[f"{gkey}_fused_torch_compile"] = fa.tc_error
        print(f"  !! fused torch.compile arm FAILED to build:\n{fa.tc_error}", flush=True)
    if fa.gr_error:
        out_af.setdefault("errors", {})[f"{gkey}_fused_cuda_graphs"] = fa.gr_error
        print(f"  !! fused CUDAGraph arm FAILED to build:\n{fa.gr_error}", flush=True)
    print(f"  setup walls: {fa.setup}", flush=True)
    flush()

    print(f"  parity gate ({args.parity_iters} iters, fused granularity, all {len(keys)} keys)...",
          flush=True)
    par = parity_gate(fa, args.parity_iters)
    out_af.setdefault("parity", {})[gkey] = par
    print(f"  parity: {json.dumps(par)}", flush=True)
    flush()

    print(f"  timing cross-game point ({len(keys)} keys, B={fa.comp.B})...", flush=True)
    row = time_point(fa, args.iters, torch, args.repeats, args.warmup)
    row["game"] = gkey; row["n_iset"] = dlg.n_iset; row["B_cut_nodes"] = int(fa.comp.B)
    rows = out_af.setdefault("rows", [])
    rows[:] = [r for r in rows if r.get("game") != gkey] + [row]
    rows.sort(key=lambda r: r["game"])
    print(f"  {gkey}: {json.dumps(row)}", flush=True)
    flush()

    sweeps = args.sweeps.get(nc)
    if sweeps:
        ksrows = []
        for n in sweeps:
            sub = keys[:n]
            if n == len(keys):
                fan = fa
            else:
                fan = FusedArms(dlg, ranges, sub, device)
                out_af["setup_walls"].setdefault(f"{gkey}_sweep", {})[str(n)] = dict(fan.setup)
                if fan.tc_error:
                    out_af.setdefault("errors", {})[f"{gkey}_sweep{n}_fused_torch_compile"] = fan.tc_error
                if fan.gr_error:
                    out_af.setdefault("errors", {})[f"{gkey}_sweep{n}_fused_cuda_graphs"] = fan.gr_error
            r = time_point(fan, args.iters, torch, args.repeats, args.warmup)
            ksrows.append(r)
            print(f"    sweep n={n}: {json.dumps(r)}", flush=True)
            out_af.setdefault("keysweeps", {})[gkey] = ksrows
            flush()
            if fan is not fa:
                del fan
                torch.cuda.empty_cache()
    del fa, dlg
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------
# analysis: merge with the prior artifact's sequential arms -> 6-arm tables + derived + crossover
# ----------------------------------------------------------------------------------------------------

def _best(row, arms, suffix="_gpu_ms"):
    cands = {a: row[a + suffix] for a in arms if a + suffix in row}
    if not cands:
        return None, None
    b = min(cands, key=cands.get)
    return b, cands[b]


def _six_arm_row(prior_row, af_row):
    """One merged 6-arm row from a prior-artifact row (seq + prior fused eager) and the new
    amortized-fused row (this session's fused eager re-anchor + fused compile/graphs)."""
    r = {}
    if "game" in af_row:
        r["game"] = af_row["game"]
    r["n_subgames"] = af_row["n_subgames"]
    r["fused_eager_ms"] = af_row["fused_eager_gpu_ms"]
    r["fused_eager_prior_artifact_ms"] = prior_row["fused_gpu_ms"]
    r["fused_eager_drift_anchor_ratio"] = round(
        af_row["fused_eager_gpu_ms"] / prior_row["fused_gpu_ms"], 3)
    for a in ("fused_compile", "fused_graphs"):
        if a + "_gpu_ms" in af_row:
            r[a + "_ms"] = af_row[a + "_gpu_ms"]
    for a in ("seq_eager", "seq_compile", "seq_graphs"):
        if a + "_gpu_ms" in prior_row:
            r[a + "_ms"] = prior_row[a + "_gpu_ms"]
    bf, bfv = _best(af_row, ("fused_compile", "fused_graphs"))
    bs_cands = {a: prior_row[a + "_gpu_ms"] for a in ("seq_compile", "seq_graphs")
                if a + "_gpu_ms" in prior_row}
    bs = min(bs_cands, key=bs_cands.get) if bs_cands else None
    if bf is not None:
        r["best_amortized_fused_arm"] = bf
        r["best_amortized_fused_ms"] = bfv
    if bs is not None:
        r["best_amortized_seq_arm"] = bs
        r["best_amortized_seq_ms"] = bs_cands[bs]
    if bf is not None and bs is not None and bfv > 0:
        r["speedup_amortized_fused_vs_amortized_seq"] = round(bs_cands[bs] / bfv, 2)
    if bf is not None and "seq_eager_gpu_ms" in prior_row and bfv > 0:
        r["speedup_amortized_fused_vs_seq_eager"] = round(prior_row["seq_eager_gpu_ms"] / bfv, 2)
    if "seq_eager_gpu_ms" in prior_row:
        r["speedup_fused_eager_vs_seq_eager_prior"] = prior_row.get("speedup_fused_vs_seq_eager")
    return r


def _interp_crossing(ns, speedups):
    """First N where the amortized-vs-amortized speedup curve crosses 1.0 (linear interpolation)."""
    for i in range(1, len(ns)):
        a, b = speedups[i - 1], speedups[i]
        if a < 1.0 <= b:
            return round(ns[i - 1] + (ns[i] - ns[i - 1]) * (1.0 - a) / (b - a), 2)
    if speedups and speedups[0] >= 1.0:
        return float(ns[0])
    return None


def analyze(out):
    af = out.get("amortized_fused", {})
    prior_rows = {r["game"]: r for r in out.get("rows", [])}
    af_rows = {r["game"]: r for r in af.get("rows", [])}

    six = []
    for gkey in sorted(set(prior_rows) & set(af_rows)):
        six.append(_six_arm_row(prior_rows[gkey], af_rows[gkey]))
    af["six_arm_cross_game"] = six

    six_sweeps, derived = {}, {}
    for gkey in sorted(set(out.get("keysweeps", {})) & set(af.get("keysweeps", {}))):
        prior_ks = {r["n_subgames"]: r for r in out["keysweeps"][gkey]}
        af_ks = {r["n_subgames"]: r for r in af["keysweeps"][gkey]}
        ns = sorted(set(prior_ks) & set(af_ks))
        rows = []
        for n in ns:
            rr = _six_arm_row(prior_ks[n], af_ks[n])
            rr["n_subgames"] = n
            rows.append(rr)
        six_sweeps[gkey] = rows

        if len(ns) < 2:
            continue
        d = {}
        n1, n2, nmax = ns[0], ns[1], ns[-1]
        # amortized-fused m_probe (same definition as the prior artifact's fused m_probe):
        # marginal cost of the 2nd tree relative to the 1st, plus the full-span normalized slope.
        for a in ("fused_eager", "fused_compile", "fused_graphs"):
            k = a + "_gpu_ms"
            if k in af_ks[n1] and k in af_ks[n2] and k in af_ks[nmax]:
                t1, t2, tm = af_ks[n1][k], af_ks[n2][k], af_ks[nmax][k]
                d[f"{a}_m_probe_N{n2}"] = round((t2 - t1) / t1, 4)
                d[f"{a}_m_full_span"] = round((tm - t1) / t1 / (nmax - n1), 4)
        # per-tree floors at N=1 (all six arms)
        floors = {}
        for a in ("seq_eager", "seq_compile", "seq_graphs"):
            k = a + "_gpu_ms"
            if k in prior_ks[n1]:
                floors[a] = prior_ks[n1][k]
        for a in ("fused_eager", "fused_compile", "fused_graphs"):
            k = a + "_gpu_ms"
            if k in af_ks[n1]:
                floors[a + "_N1"] = af_ks[n1][k]
        d["per_tree_floor_ms"] = floors
        # endpoint slopes (prior-artifact convention: (t(Nmax)-t(N1))/(Nmax-N1))
        slopes = {}
        for src, arms in ((prior_ks, ("seq_eager", "seq_compile", "seq_graphs")),
                          (af_ks, ("fused_eager", "fused_compile", "fused_graphs"))):
            for a in arms:
                k = a + "_gpu_ms"
                if k in src[n1] and k in src[nmax]:
                    slopes[a] = round((src[nmax][k] - src[n1][k]) / (nmax - n1), 2)
        d["endpoint_slope_ms_per_tree"] = slopes
        # crossover, prior-artifact formula: best_amortized_fused(Nmax) / best_amortized_seq_slope
        bs_arm = min((a for a in ("seq_compile", "seq_graphs") if a in slopes),
                     key=lambda a: slopes[a], default=None)
        bf_arm, bf_nmax = _best(af_ks[nmax], ("fused_compile", "fused_graphs"))
        if bs_arm is not None and bf_arm is not None and slopes[bs_arm] > 0:
            d["best_amortized_seq_slope_arm"] = bs_arm
            d["best_amortized_fused_arm_at_Nmax"] = bf_arm
            d["crossover_N_amortized_fused_beats_best_amortized_seq"] = round(
                bf_nmax / slopes[bs_arm], 2)
        # empirical interpolated crossing + per-N amortized-vs-amortized speedups
        sp = {}
        sp_ns, sp_vals = [], []
        for n in ns:
            bs2_cands = {a: prior_ks[n][a + "_gpu_ms"] for a in ("seq_compile", "seq_graphs")
                         if a + "_gpu_ms" in prior_ks[n]}
            _, bfv = _best(af_ks[n], ("fused_compile", "fused_graphs"))
            if bs2_cands and bfv:
                v = round(min(bs2_cands.values()) / bfv, 2)
                sp[str(n)] = v
                sp_ns.append(n); sp_vals.append(v)
        d["speedup_amortized_fused_vs_best_amortized_seq_by_N"] = sp
        d["crossover_N_interpolated_from_measured_speedups"] = _interp_crossing(sp_ns, sp_vals)
        derived[gkey] = d

    af["six_arm_sweeps"] = six_sweeps
    af["derived"] = derived
    out["amortized_fused"] = af


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--parity-iters", type=int, default=200)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--key-sweeps", type=str, nargs="+", default=["4:1,2,4,8,12", "5:1,2,4,8,15"])
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--output-json", default="autoresearch-session/rebel/cuda_graphs_ablation.json")
    args = ap.parse_args(argv)
    args.sweeps = {}
    for s in args.key_sweeps:
        gname, ns = s.split(":")
        args.sweeps[int(gname)] = [int(x) for x in ns.split(",")]

    outp = pathlib.Path(args.output_json)
    out = json.loads(outp.read_text())
    af = out.setdefault("amortized_fused", {})
    af.setdefault("experiment", "cuda_graphs_torchcompile_AMORTIZED_FUSED_arm_completion")
    af.setdefault("uses_slumbot_data", False)

    def flush():
        analyze(out)
        outp.write_text(json.dumps(out, indent=2))

    if args.analyze_only:
        flush()
        print("analysis recomputed.", flush=True)
        return 0

    import torch
    assert torch.cuda.is_available(), "this ablation targets the CUDA launch-amortization question"
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64   # several distinct (n_iids,B) shapes per game sweep
    af["device"] = torch.cuda.get_device_name(0)
    af["torch"] = torch.__version__
    af["dtype"] = "float64 (mandatory for gated numbers)"
    af["protocol"] = {"iters": args.iters, "repeats": args.repeats, "warmup": args.warmup,
                      "timing": "torch.cuda.Event pairs + synchronize, median of repeats, "
                                "precompiled topologies; compile/capture walls excluded and "
                                "reported under setup_walls; sequential-arm numbers for the 6-arm "
                                "merge come from the prior artifact (same protocol, same device); "
                                "fused_eager re-measured this session as the cross-session drift "
                                "anchor"}

    for nc in args.games:
        measure_game(nc, args, torch, af, flush)
    flush()
    print(f"wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
