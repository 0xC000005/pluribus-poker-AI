#!/usr/bin/env python3
"""GATE 0 (no-build, existing code) for the rank-1 scale lever: BATCH a POPULATION of depth-limited PBS
subgames into one GPU GEMM. Decides -- cheaply, before any multi-week port -- whether the premise holds.

Three probes:
  A. TOPOLOGY MULTIPLICITY (generic substrate; the make-or-break "buckets of size 1" risk): over the
     depth-limited subgames the method actually solves (the below-cut continuation per PUBLIC state), how
     many share an IDENTICAL tree TOPOLOGY (-> batchable into one GEMM) vs are unique (-> bucket size 1,
     no GEMM win)? Reported per game (Leduc, Goofspiel-4): #public-states, #distinct-topologies,
     multiplicity, bucket-size histogram, subgame sizes.
  C. RE-SOLVE COST (generic substrate; what batching would attack): per trunk solve the method issues
     n_public_states x trunk_iters per-belief subgame re-solves. Time trunk_solve with the per-belief
     RE-SOLVE leaf vs the FIXED blueprint leaf -> the re-solve overhead ratio + per-subgame-solve time +
     the available batch size (subgame solves per trunk iter = #public states sharing topology).
  B. GPU BATCHED-vs-SERIAL CROSSOVER (existing tested fast_cfr.py:933 batched same-topology solver, via
     StreetSolver replication -- no protected files modified): does batching B same-topology trees beat B
     serial solves on the GPU, and at what B*? Establishes AXIS-2's raw win. Guarded against OOM.

Go/No-Go: KILL the rank-1 premise cheaply if topology multiplicity ~1 (every PBS root a unique shape) OR
the GPU crossover never materializes. GREENLIGHT the multi-week generic port if multiplicity is high AND
batched throughput >> serial at feasible B. Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np
import pyspiel

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import (leduc_is_cut, load_goofspiel,
                                     goofspiel_public_key, goofspiel_is_cut)


def topo_sig(node):
    """Tree-SHAPE signature of a below-cut subgame (ignores infoset ids + terminal values; captures node
    types, branching, player-to-act, action counts) -- two subgames with the same sig are GEMM-batchable."""
    t = node[0]
    if t == "term":
        return ("T",)
    if t == "chance":
        return ("C", tuple(topo_sig(ch) for _p, ch in node[1]))
    if t == "cut":  # not expected below a cut, but be safe
        return ("X",)
    _, pl, _iid, kids = node
    return ("D", int(pl), len(kids), tuple(topo_sig(ch) for _a, ch in kids))


def _subgame_nnodes(node):
    t = node[0]
    if t == "term":
        return 1
    if t == "chance":
        return 1 + sum(_subgame_nnodes(ch) for _p, ch in node[1])
    if t == "cut":
        return 1
    return 1 + sum(_subgame_nnodes(ch) for _a, ch in node[3])


def probe_multiplicity(dlg):
    by_key = collections.defaultdict(list)
    for node in dlg.cut_nodes:
        by_key[node[1]].append(node)
    keys = sorted(by_key)
    sig_of_key = {}
    nodes_of_key = {}
    for key in keys:
        sub = by_key[key][0][4]  # all cut_nodes at a key share public topology; take a representative
        sig_of_key[key] = hash(topo_sig(sub))
        nodes_of_key[key] = _subgame_nnodes(sub)
    buckets = collections.Counter(sig_of_key.values())
    bucket_sizes = sorted(buckets.values(), reverse=True)
    n_keys = len(keys)
    n_distinct = len(buckets)
    # batch size available = how many public states share the most common topology
    return {
        "n_public_states": n_keys,
        "n_distinct_topologies": n_distinct,
        "multiplicity_mean": round(n_keys / n_distinct, 2) if n_distinct else 0.0,
        "largest_bucket": bucket_sizes[0] if bucket_sizes else 0,
        "buckets_of_size_1": int(sum(1 for s in bucket_sizes if s == 1)),
        "frac_keys_in_size1_buckets": round(sum(s for s in bucket_sizes if s == 1) / max(n_keys, 1), 3),
        "bucket_size_hist_top": bucket_sizes[:10],
        "subgame_nnodes_min_med_max": [min(nodes_of_key.values()), int(np.median(list(nodes_of_key.values()))),
                                       max(nodes_of_key.values())] if nodes_of_key else [],
        "max_priv_dims": [max(dlg.n_priv(k, 0) for k in keys), max(dlg.n_priv(k, 1) for k in keys)],
    }


def probe_resolve_cost(dlg, trunk_iters=8, subgame_iters=80, cont_iters=200):
    keys = sorted({n[1] for n in dlg.cut_nodes})
    ref = dlg.cfr_plus(cont_iters)
    fixed = dlg.blueprint_leaf_fn(ref)
    t0 = time.perf_counter()
    dlg.trunk_solve(fixed, iters=trunk_iters)
    t_fixed = time.perf_counter() - t0

    resolve = dlg.per_belief_equilibrium_leaf_fn(iters=subgame_iters)
    t0 = time.perf_counter()
    dlg.trunk_solve(resolve, iters=trunk_iters)
    t_resolve = time.perf_counter() - t0

    n_solves = len(keys) * trunk_iters  # one subgame re-solve per public key per trunk iter
    return {
        "n_public_states": len(keys),
        "trunk_iters": trunk_iters,
        "subgame_iters": subgame_iters,
        "subgame_solves_per_trunk_solve": n_solves,
        "subgame_solves_per_trunk_iter": len(keys),
        "t_fixed_leaf_s": round(t_fixed, 3),
        "t_resolve_leaf_s": round(t_resolve, 3),
        "resolve_overhead_x": round(t_resolve / max(t_fixed, 1e-6), 1),
        "ms_per_subgame_solve": round(1000.0 * (t_resolve - t_fixed) / max(n_solves, 1), 3),
    }


def probe_gpu_crossover(batch_sizes, iters=25):
    """Batched-vs-serial throughput on the existing tested solver via river-solver replication."""
    import torch
    from solver import StreetSolver
    from fast_cfr import (solve_cfr_levelsync_torch,
                          solve_cfr_levelsync_torch_batched_same_topology)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    board = [0, 1, 2, 3, 4]  # arbitrary fixed river board -> fixed topology
    s = StreetSolver(board, 200, 19900, 19900, True)
    n_nodes = int(s._tree["n_nodes"]); n_hands = int(s.n)
    args = (s._tree, s.n, s.win_m, s.lose_m, s.tie_m, s.valid,
            s.pot_start, s.hero_stack_start, s.villain_stack_start)
    # warm up
    solve_cfr_levelsync_torch(*args, n_iterations=1, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    out = {"device": device, "river_n_nodes": n_nodes, "river_n_hands": n_hands, "iters": iters, "sweep": []}
    for B in batch_sizes:
        try:
            t0 = time.perf_counter()
            for _ in range(B):
                solve_cfr_levelsync_torch(*args, n_iterations=iters, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            serial = time.perf_counter() - t0

            t0 = time.perf_counter()
            solve_cfr_levelsync_torch_batched_same_topology(
                [s._tree] * B, s.n, [s.win_m] * B, [s.lose_m] * B, [s.tie_m] * B, [s.valid] * B,
                [s.pot_start] * B, [s.hero_stack_start] * B, [s.villain_stack_start] * B,
                n_iterations=iters, terminal_eval_mode="batched", device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            batched = time.perf_counter() - t0
            out["sweep"].append({
                "B": B, "serial_s": round(serial, 3), "batched_s": round(batched, 3),
                "speedup": round(serial / max(batched, 1e-6), 2),
                "serial_subgames_per_s": round(B / max(serial, 1e-6), 1),
                "batched_subgames_per_s": round(B / max(batched, 1e-6), 1)})
        except RuntimeError as e:
            out["sweep"].append({"B": B, "error": str(e)[:120]})
            if device == "cuda":
                torch.cuda.empty_cache()
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--batch-sizes", default="1,8,32,64,128")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = {}

    games = [("leduc", pyspiel.load_game("leduc_poker"), leduc_is_cut, None),
             ("goofspiel4", load_goofspiel(4), goofspiel_is_cut, goofspiel_public_key)]
    print("GATE 0: batched-subgame premise probe\n")
    print("== Probe A: topology multiplicity (batchability) ==")
    out["multiplicity"] = {}
    out["resolve_cost"] = {}
    for name, game, cut, pkf in games:
        dlg = DepthLimitedGame(game, cut, public_key_fn=pkf)
        a = probe_multiplicity(dlg)
        out["multiplicity"][name] = a
        print(f"  {name}: {a['n_public_states']} public states -> {a['n_distinct_topologies']} distinct "
              f"topologies (mult {a['multiplicity_mean']}x; largest bucket {a['largest_bucket']}; "
              f"{a['frac_keys_in_size1_buckets']:.0%} of states in size-1 buckets); subgame nodes "
              f"{a['subgame_nnodes_min_med_max']}; max priv {a['max_priv_dims']}")
        c = probe_resolve_cost(dlg)
        out["resolve_cost"][name] = c
        print(f"     re-solve: {c['subgame_solves_per_trunk_iter']} solves/trunk-iter, re-solve leaf is "
              f"{c['resolve_overhead_x']}x the fixed leaf, {c['ms_per_subgame_solve']}ms/subgame-solve")

    if not args.skip_gpu:
        print("\n== Probe B: GPU batched-vs-serial crossover (existing solver, river replication) ==")
        try:
            bs = [int(x) for x in args.batch_sizes.split(",")]
            out["gpu_crossover"] = probe_gpu_crossover(bs)
            for r in out["gpu_crossover"]["sweep"]:
                if "error" in r:
                    print(f"  B={r['B']}: OOM/err ({r['error']})")
                else:
                    print(f"  B={r['B']:>4}: serial {r['serial_s']}s vs batched {r['batched_s']}s -> "
                          f"{r['speedup']}x  ({r['batched_subgames_per_s']} vs {r['serial_subgames_per_s']} subgames/s)")
        except Exception as e:  # noqa: BLE001
            out["gpu_crossover"] = {"error": str(e)[:200]}
            print(f"  Probe B unavailable: {str(e)[:200]}")

    # crude go/no-go summary
    mult_ok = all(out["multiplicity"][g]["multiplicity_mean"] >= 4 for g in out["multiplicity"])
    gpu = out.get("gpu_crossover", {})
    best_speedup = max((r.get("speedup", 0) for r in gpu.get("sweep", [])), default=0)
    out["batchable_premise_holds"] = bool(mult_ok)
    out["gpu_best_speedup"] = best_speedup
    print(f"\n  TOPOLOGY MULTIPLICITY high (>=4x, batchable): {mult_ok}")
    print(f"  GPU best batched speedup: {best_speedup}x")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
