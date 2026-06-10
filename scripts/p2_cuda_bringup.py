#!/usr/bin/env python
"""P2 CUDA bring-up of the HUNL population solver + cost-per-target profiling.

Phases (run independently; each merges its section into
autoresearch-session/rebel/p2_cuda_bringup.json):

  parity     CUDA float64 vs CPU float64 reference on a fixed spot set
             (3 river boards x 3 (pot,stack) configs incl. one all-in-truncated,
             B>1 fused buckets, 250 iters). Measures the EMPIRICAL cuda-vs-cpu
             drift (index_add_/atomics nondeterminism) + cuda run-to-run spread.
             Also profiles the float32 CUDA variant -- EXPERIMENTAL, parity
             recorded separately, float64 stays the mandated production dtype.
  bmax       Empirical OOM boundary on the 8GB card for the river p50/p95/max
             topology classes at H=1081 float64 (binary search B, clean OOM
             catch, empty_cache between) vs the census arithmetic table.
  cost       generate_river_targets end-to-end on cuda at 300/600 subgame
             iters (beliefs x boards filling real buckets via PopulationQueue):
             targets/sec + GPU-seconds per 1k river V* targets, fused vs B=1
             sequential on the same workload. THE P4 budget datum.
  turn       STEP-10 smoke: one turn subgame solved on cuda with the exact-river
             leaf hook (48 runout river specs per cut via the shared expansion),
             root values verified against the turn_river.py single-spot
             reference machinery at the SAME average strategy.

Protected surfaces (scripts/solver.py, scripts/fast_cfr.py) are imported via
poker_ai.rebel.hunl.tree_builder, never edited. float64 regrets are mandatory
on the parity path; float32 rows are labeled experimental.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import time

import numpy as np
import torch

from poker_ai.rebel.hunl import tree_builder as tb  # wires scripts/ into sys.path
from poker_ai.rebel.hunl import subgame_spec as sgs
from poker_ai.rebel.hunl import terminal_eval as tev
from poker_ai.rebel.hunl import population_solver as pop
from poker_ai.rebel.hunl import lazy_subgames as lzs
from poker_ai.rebel.hunl import targets as tgt
from poker_ai.rebel import turn_river as trv

import fast_cfr  # noqa: E402  (protected surface; imported, never edited)
import solver as slv  # noqa: E402  (protected surface; imported, never edited)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO, "autoresearch-session", "rebel", "p2_cuda_bringup.json")


def _board(*cards):
    return tuple(trv.parse_card(c) for c in cards)


RIVER_BOARDS = [
    _board("Ah", "Kd", "7c", "2s", "Jh"),
    _board("Qs", "Js", "9s", "3d", "2c"),
    _board("8c", "8d", "Kh", "4s", "4d"),
]
TURN_BOARD = _board("Ah", "Kd", "7c", "2s")

# Parity spot set: 3 boards x 3 configs (incl. one all-in-truncated), each
# bucket fused at B = 3 boards x 2 belief seeds = 6 (> 1).
PARITY_CONFIGS = {
    "mid_nn75": dict(pot=400, stack0=400, stack1=400, first_to_act=0),
    "census_p95_nn411": dict(pot=4850, stack0=17575, stack1=17575, first_to_act=1),
    "allin_trunc_nn10": dict(pot=3000, stack0=150, stack1=600, first_to_act=1),
}

# Census river class examples (hunl_topology_census.json b_max_table rows;
# postflop first-to-act = player1 per the census protocol).
CENSUS_RIVER_CLASSES = {
    "p50": dict(pot=20002, stack0=9999, stack1=9999, first_to_act=1),   # nn=21
    "p95": dict(pot=4850, stack0=17575, stack1=17575, first_to_act=1),  # nn=411
    "max": dict(pot=400, stack0=19800, stack1=19800, first_to_act=1),   # nn=1257
}
CENSUS_ARITHMETIC_B_MAX = {"p50": 154, "p95": 51, "max": 21}  # 6.5GB, f32 terminals


def merge_out(section, payload):
    data = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as fh:
            data = json.load(fh)
    data[section] = payload
    data["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    data["env"] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_vram_bytes": (torch.cuda.get_device_properties(0).total_memory
                             if torch.cuda.is_available() else None),
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[merged section {section!r} -> {OUT_PATH}]")


def _global_ranges(seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        r = rng.random(sgs.N_GLOBAL_HANDS)
        out.append((r / r.sum()).astype(np.float64))
    return out[0], out[1]


def _make_spec(board, config, seed, street="river"):
    r0, r1 = _global_ranges(seed)
    return sgs.SubgameSpec(
        street=street, board=tuple(board), pot=config["pot"],
        stack0=config["stack0"], stack1=config["stack1"],
        first_to_act=config["first_to_act"], r0=r0, r1=r1)


def _free_mb():
    free, total = torch.cuda.mem_get_info()
    return free / 1e6, total / 1e6


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _diff_stats(res_a, res_b):
    """Per-spec-list max diffs between two solve_population result lists."""
    s = v = 0.0
    v_rel = 0.0
    for a, b in zip(res_a, res_b, strict=True):
        s = max(s, float(np.max(np.abs(a.avg_strategy - b.avg_strategy))))
        dv = max(float(np.max(np.abs(a.v0 - b.v0))),
                 float(np.max(np.abs(a.v1 - b.v1))))
        v = max(v, dv)
        v_rel = max(v_rel, dv / a.spec.pot)
    return {"avg_strategy_linf": s, "value_linf_chips": v,
            "value_linf_over_pot": v_rel}


# ---------------------------------------------------------------------------
# Phase 1: CUDA parity vs CPU float64 reference
# ---------------------------------------------------------------------------

def phase_parity(n_iterations=250, n_seeds=2):
    specs, labels = [], []
    for cname, cfg in PARITY_CONFIGS.items():
        for bi, board in enumerate(RIVER_BOARDS):
            for s in range(n_seeds):
                specs.append(_make_spec(board, cfg, seed=1000 + 97 * bi + s))
                labels.append(cname)
    b_max = 32  # every bucket (B=6) solves as one fused chunk

    def run(device, dtype):
        _cleanup()
        t0 = time.perf_counter()
        out = pop.solve_population(specs, n_iterations=n_iterations,
                                   dtype=dtype, device=device, b_max=b_max)
        if device == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    print(f"parity: {len(specs)} specs, {n_iterations} iters, buckets "
          f"{sorted(set(labels))} at B={len(specs)//len(PARITY_CONFIGS)} each")
    cpu64, t_cpu = run("cpu", torch.float64)
    print(f"  cpu  float64: {t_cpu:.1f}s")
    cuda64_a, t_cuda_a = run("cuda", torch.float64)
    print(f"  cuda float64 (run 1): {t_cuda_a:.1f}s")
    cuda64_b, t_cuda_b = run("cuda", torch.float64)
    print(f"  cuda float64 (run 2): {t_cuda_b:.1f}s")
    cuda32, t_cuda32 = run("cuda", torch.float32)
    print(f"  cuda float32 (EXPERIMENTAL): {t_cuda32:.1f}s")

    per_config = {}
    for cname in PARITY_CONFIGS:
        idx = [i for i, l in enumerate(labels) if l == cname]
        sub = lambda res: [res[i] for i in idx]  # noqa: E731
        per_config[cname] = {
            "config": PARITY_CONFIGS[cname],
            "n_elems_fused": len(idx),
            "cuda64_vs_cpu64": _diff_stats(sub(cuda64_a), sub(cpu64)),
            "cuda64_run_to_run": _diff_stats(sub(cuda64_a), sub(cuda64_b)),
            "cuda32_vs_cpu64_EXPERIMENTAL": _diff_stats(sub(cuda32), sub(cpu64)),
        }

    # zero-sum convention identity on the cuda float64 results
    ident = 0.0
    for res in cuda64_a:
        spec = res.spec
        valid64 = sgs.local_valid_matrix(spec.board).astype(np.float64)
        r0l, r1l = spec.local_ranges()
        lhs = float(r0l @ res.v0 + r1l @ res.v1)
        rhs = float(spec.pot * (r0l @ valid64 @ r1l))
        ident = max(ident, abs(lhs - rhs))

    overall = {
        "cuda64_vs_cpu64": _diff_stats(cuda64_a, cpu64),
        "cuda64_run_to_run": _diff_stats(cuda64_a, cuda64_b),
        "cuda32_vs_cpu64_EXPERIMENTAL": _diff_stats(cuda32, cpu64),
    }
    payload = {
        "protocol": {
            "n_specs": len(specs), "n_iterations": n_iterations,
            "boards": [list(b) for b in RIVER_BOARDS],
            "configs": PARITY_CONFIGS, "b_max": b_max,
            "note": ("CPU float64 = the reference (itself gated against "
                     "fast_cfr.solve_cfr by test_hunl_population_parity.py); "
                     "cuda drift comes from index_add_/bmm atomics + "
                     "non-deterministic reduction order. float32 rows are "
                     "EXPERIMENTAL profiling only -- float64 stays mandatory."),
        },
        "wall_seconds": {"cpu_float64": t_cpu, "cuda_float64_run1": t_cuda_a,
                         "cuda_float64_run2": t_cuda_b,
                         "cuda_float32_EXPERIMENTAL": t_cuda32},
        "overall_max_diffs": overall,
        "per_config": per_config,
        "cuda64_zero_sum_identity_residual_max": ident,
    }
    merge_out("parity", payload)
    print(json.dumps(overall, indent=2))
    print(f"identity residual (cuda64): {ident:.3e}")


# ---------------------------------------------------------------------------
# Phase 2: empirical B_max (OOM boundary) per river topology class
# ---------------------------------------------------------------------------

def _try_b(config, B, n_iterations=3):
    """One fused river solve at batch size B; returns (ok, peak_bytes, secs)."""
    board = RIVER_BOARDS[0]
    specs = [_make_spec(board, config, seed=5000 + i) for i in range(B)]
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pop.solve_population(specs, n_iterations=n_iterations,
                             dtype=torch.float64, device="cuda", b_max=B)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return True, int(torch.cuda.max_memory_allocated()), dt
    except torch.cuda.OutOfMemoryError:
        return False, None, None
    except RuntimeError as exc:  # cublas etc. can surface OOM as RuntimeError
        if "out of memory" in str(exc).lower():
            return False, None, None
        raise
    finally:
        del specs
        _cleanup()


def phase_bmax(hi_cap=192):
    results = {}
    for cls, cfg in CENSUS_RIVER_CLASSES.items():
        tree = tb.build_street_tree(cfg["pot"], cfg["stack0"], cfg["stack1"],
                                    cfg["first_to_act"], "river")
        nn = int(tree["n_nodes"])
        print(f"\n[bmax/{cls}] config={cfg} nn={nn}")
        # exponential probe up, then binary search the boundary
        attempts = []
        lo, lo_peak, lo_t = 0, None, None
        b = 1
        hi = None
        while b <= hi_cap:
            ok, peak, dt = _try_b(cfg, b)
            attempts.append({"B": b, "ok": ok, "peak_bytes": peak, "secs": dt})
            print(f"  B={b}: {'ok' if ok else 'OOM'}"
                  + (f" peak={peak/1e9:.2f}GB {dt:.1f}s" if ok else ""))
            if ok:
                lo, lo_peak, lo_t = b, peak, dt
                b *= 2
            else:
                hi = b
                break
        if hi is None:
            hi = hi_cap + 1  # never OOMed below cap
        while hi - lo > 1:
            mid = (lo + hi) // 2
            ok, peak, dt = _try_b(cfg, mid)
            attempts.append({"B": mid, "ok": ok, "peak_bytes": peak, "secs": dt})
            print(f"  B={mid}: {'ok' if ok else 'OOM'}"
                  + (f" peak={peak/1e9:.2f}GB {dt:.1f}s" if ok else ""))
            if ok:
                lo, lo_peak, lo_t = mid, peak, dt
            else:
                hi = mid
        free_mb, total_mb = _free_mb()
        arith = {
            "census_6p5GB_f32_terminals": CENSUS_ARITHMETIC_B_MAX[cls],
            "recomputed_6p5GB_f32_terminals": lzs.b_max_for_tree(tree, "river", 6.5),
            "recomputed_6p5GB_f64_terminals_actual_port": lzs.b_max_for_tree(
                tree, "river", 6.5, terminal_value_bytes=8),
            "recomputed_7p8GB_f64_terminals_actual_port": lzs.b_max_for_tree(
                tree, "river", 7.8, terminal_value_bytes=8),
        }
        results[cls] = {
            "config": cfg, "n_nodes": nn,
            "empirical_B_max": lo,
            "first_oom_B": (hi if hi <= hi_cap else None),
            "hit_search_cap": hi > hi_cap,
            "peak_bytes_at_B_max": lo_peak,
            "secs_at_B_max_3iters": lo_t,
            "arithmetic": arith,
            "attempts": attempts,
            "free_total_mb_after": [free_mb, total_mb],
        }
        print(f"  -> empirical B_max={lo} (census arithmetic "
              f"{CENSUS_ARITHMETIC_B_MAX[cls]}, f64-terminal arithmetic "
              f"{arith['recomputed_6p5GB_f64_terminals_actual_port']})")
    payload = {
        "protocol": {
            "H": 1081, "dtype": "float64 (terminals cast to float64 by the "
            "current port -- census table assumed float32 terminals)",
            "n_iterations_per_attempt": 3, "compute_value_pass": True,
            "search": "exponential doubling then binary search; "
            "torch.cuda.empty_cache between attempts", "hi_cap": hi_cap,
        },
        "classes": results,
    }
    merge_out("bmax", payload)


# ---------------------------------------------------------------------------
# Phase 3: cost-per-target (THE P4 budget datum)
# ---------------------------------------------------------------------------

def _random_boards(n, seed):
    rng = np.random.default_rng(seed)
    boards = set()
    while len(boards) < n:
        boards.add(tuple(sorted(rng.choice(52, size=5, replace=False).tolist())))
    return sorted(boards)


def _cost_run(cfg, n_boards, n_beliefs, n_iterations, b_max, dtype, label):
    boards = _random_boards(n_boards, seed=20260610)
    sampler = lambda rng: (cfg["pot"], cfg["stack0"], cfg["stack1"],  # noqa: E731
                           cfg["first_to_act"])
    queue = lzs.PopulationQueue(n_iterations=n_iterations, dtype=dtype,
                                device="cuda", b_max=b_max)
    rng = np.random.default_rng(424242)
    _cleanup()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    X, Y, W = tgt.generate_river_targets(
        n_beliefs=n_beliefs, boards=boards, config_sampler=sampler,
        rng=rng, queue=queue)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_specs = X.shape[0]
    n_targets = n_specs * 2 * sgs.RIVER_H  # per-hand V* targets (v0+v1 per spec)
    row = {
        "label": label, "n_boards": n_boards, "n_beliefs_per_board": n_beliefs,
        "n_specs": n_specs, "n_iterations": n_iterations, "b_max": b_max,
        "dtype": str(dtype).replace("torch.", ""),
        "n_flushes": queue.n_flushes,
        "wall_seconds": dt,
        "targets_per_second": n_targets / dt,
        "gpu_seconds_per_1k_targets": dt / (n_targets / 1000.0),
        "specs_per_second": n_specs / dt,
        "peak_bytes": int(torch.cuda.max_memory_allocated()),
    }
    print(f"  {label}: {n_specs} specs, {queue.n_flushes} flushes, {dt:.1f}s "
          f"-> {row['targets_per_second']:.0f} targets/s, "
          f"{row['gpu_seconds_per_1k_targets']:.3f} GPU-s/1k")
    return row, Y


def phase_cost(b_safe_p50, b_safe_p95):
    rows = []
    checks = {}
    # p50 class (the trunk-reach median): the headline workload
    cfg50 = CENSUS_RIVER_CLASSES["p50"]
    cfg95 = CENSUS_RIVER_CLASSES["p95"]
    for iters in (300, 600):
        print(f"\n[cost/p50 nn=21] {iters} iters")
        fused, y_f = _cost_run(cfg50, 16, 8, iters, b_safe_p50, torch.float64,
                               f"p50_fused_B{b_safe_p50}_{iters}it")
        seq, y_s = _cost_run(cfg50, 16, 8, iters, 1, torch.float64,
                             f"p50_sequential_B1_{iters}it")
        fused["speedup_vs_sequential"] = seq["wall_seconds"] / fused["wall_seconds"]
        checks[f"p50_{iters}it_fused_vs_seq_target_linf"] = float(
            np.max(np.abs(y_f - y_s)))
        rows += [fused, seq]
    # p95 class at 300 iters (coverage of the expensive tail)
    print("\n[cost/p95 nn=411] 300 iters")
    fused, y_f = _cost_run(cfg95, 8, 3, 300, b_safe_p95, torch.float64,
                           f"p95_fused_B{b_safe_p95}_300it")
    seq, y_s = _cost_run(cfg95, 8, 3, 300, 1, torch.float64,
                         "p95_sequential_B1_300it")
    fused["speedup_vs_sequential"] = seq["wall_seconds"] / fused["wall_seconds"]
    checks["p95_300it_fused_vs_seq_target_linf"] = float(np.max(np.abs(y_f - y_s)))
    rows += [fused, seq]
    # EXPERIMENTAL float32 fused row (profiling only; parity gated separately)
    print("\n[cost/p50 float32 EXPERIMENTAL] 300 iters")
    f32, _ = _cost_run(cfg50, 16, 8, 300, b_safe_p50, torch.float32,
                       f"p50_fused_float32_EXPERIMENTAL_B{b_safe_p50}_300it")
    rows.append(f32)
    payload = {
        "protocol": {
            "definition": "1 target = one per-hand V* value (v0 or v1 entry); "
            "one river spec yields 2*1081 = 2162 targets",
            "workload": "generate_river_targets end-to-end (belief sampling + "
            "spec build + terminal matrices + queue bucketing + fused CUDA "
            "solves + value pass + row emission), fixed census-class config, "
            "random boards x Dirichlet beliefs, identical rng for fused/seq",
            "b_safe": {"p50": b_safe_p50, "p95": b_safe_p95},
            "dtype": "float64 (mandated); float32 rows EXPERIMENTAL",
        },
        "rows": rows,
        "fused_vs_sequential_target_checks_linf_chips": checks,
    }
    merge_out("cost", payload)


# ---------------------------------------------------------------------------
# Phase 4: STEP-10 turn smoke (exact-river leaf hook on cuda)
# ---------------------------------------------------------------------------

def phase_turn(turn_iters=24, river_iters=100):
    pot, s0, s1, first = 800, 500, 500, 0
    rng = np.random.default_rng(55)
    n_turn = len(sgs.local_hands(TURN_BOARD))
    hr = rng.random(n_turn) * 0.5
    vr = rng.random(n_turn) * 0.5
    spec = sgs.SubgameSpec(
        street="turn", board=TURN_BOARD, pot=pot, stack0=s0, stack1=s1,
        first_to_act=first,
        r0=sgs.scatter_global(TURN_BOARD, hr), r1=sgs.scatter_global(TURN_BOARD, vr))
    leaf = tev.make_exact_river_leaf(
        n_iterations=river_iters, dtype=torch.float64, device="cuda", b_max=48)

    _cleanup()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = pop.solve_population([spec], n_iterations=turn_iters, leaf=leaf,
                               dtype=torch.float64, device="cuda")[0]
    torch.cuda.synchronize()
    t_solve = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated())
    print(f"[turn] cuda solve: {t_solve:.1f}s, peak {peak/1e9:.2f}GB")

    # Reference: turn_river.py single-spot machinery at the SAME avg strategy.
    # Forward avg reaches; per river-deal cut, turn_leaf_river_cfv (float32 CPU
    # rivers) + the L241-242 net-from-turn offset; subgame_value_pass with the
    # overrides. Per-cut documented band: 1e-3 * pot_cut (gate P-E).
    t0 = time.perf_counter()
    solver = slv.StreetSolver(list(TURN_BOARD), pot, s0, s1, hero_first=(first == 0))
    assert tb.topology_key(solver._tree) == res.topology_key
    avg = res.avg_strategy
    t = solver._tree
    nn, n = int(t["n_nodes"]), solver.n
    player, children, dacts = t["player"], t["children"], t["decision_actions"]
    HR = np.zeros((nn, n)); VR = np.zeros((nn, n))
    HR[0], VR[0] = hr, vr
    for i in range(nn):
        if player[i] == -1:
            continue
        for a in dacts[i]:
            c = children[i, a]
            if player[i] == 0:
                HR[c] = HR[i] * avg[i, a]; VR[c] = VR[i]
            else:
                HR[c] = HR[i]; VR[c] = VR[i] * avg[i, a]
    valid = solver.valid; validT = valid.T
    sh, sv, potN = t["stacks_h"], t["stacks_v"], t["pot"]
    ov = {}
    max_pot_cut = 0
    for ci in t["showdown_idx"].tolist():
        P_cut, hs, vs = int(potN[ci]), int(sh[ci]), int(sv[ci])
        max_pot_cut = max(max_pot_cut, P_cut)
        hi_cut, vi_cut = s0 - hs, s1 - vs
        ch, cv = trv.turn_leaf_river_cfv(
            list(TURN_BOARD), P_cut, hs, vs, first == 0, solver.hands,
            HR[ci], VR[ci], river_iters=river_iters, backend="cpu")
        ov[ci] = (ch - hi_cut * (VR[ci] @ validT), cv - vi_cut * (HR[ci] @ valid))
    ref_v0, ref_v1 = trv.subgame_value_pass(solver, avg, hr, vr, showdown_override=ov)
    t_ref = time.perf_counter() - t0

    d_v = max(float(np.max(np.abs(res.v0 - ref_v0))),
              float(np.max(np.abs(res.v1 - ref_v1))))
    ev0_pop, ev1_pop = float(hr @ res.v0), float(vr @ res.v1)
    ev0_ref, ev1_ref = float(hr @ ref_v0), float(vr @ ref_v1)
    valid64 = valid.astype(np.float64)
    ident = abs((hr @ res.v0 + vr @ res.v1) - pot * (hr @ valid64 @ vr))
    tol = 1e-3 * max_pot_cut  # gate P-E's documented per-cut band
    n_cuts = len(ov)
    payload = {
        "protocol": {
            "spot": dict(pot=pot, stack0=s0, stack1=s1, first_to_act=first,
                         board=list(TURN_BOARD)),
            "turn_iters": turn_iters, "river_iters": river_iters,
            "n_river_deal_cuts": n_cuts,
            "leaf": "make_exact_river_leaf (48-runout fused river populations "
            "per cut, b_max=48, float64, cuda)",
            "reference": "turn_river.turn_leaf_river_cfv (float32 CPU rivers) "
            "+ L241-242 offset + subgame_value_pass at the SAME avg strategy",
            "documented_tolerance_chips": tol,
        },
        "wall_seconds_turn_solve_cuda": t_solve,
        "wall_seconds_reference_value_cpu": t_ref,
        "peak_bytes": peak,
        "root_value_linf_vs_reference_chips": d_v,
        "root_ev": {"pop_v0": ev0_pop, "ref_v0": ev0_ref,
                    "pop_v1": ev1_pop, "ref_v1": ev1_ref,
                    "ev0_diff": abs(ev0_pop - ev0_ref),
                    "ev1_diff": abs(ev1_pop - ev1_ref)},
        "zero_sum_identity_residual_chips": float(ident),
        "within_documented_tolerance": bool(d_v <= tol),
    }
    merge_out("turn_smoke", payload)
    print(f"[turn] root-value L-inf vs reference: {d_v:.4f} chips "
          f"(tol {tol:.1f}), identity residual {ident:.3e}, "
          f"ref pass {t_ref:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["parity", "bmax", "cost", "turn"])
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--b-safe-p50", type=int, default=None)
    ap.add_argument("--b-safe-p95", type=int, default=None)
    ap.add_argument("--turn-iters", type=int, default=24)
    ap.add_argument("--river-iters", type=int, default=100)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "this harness needs the GPU"
    if args.phase == "parity":
        phase_parity(n_iterations=args.iters)
    elif args.phase == "bmax":
        phase_bmax()
    elif args.phase == "cost":
        if args.b_safe_p50 is None or args.b_safe_p95 is None:
            raise SystemExit("cost phase needs --b-safe-p50/--b-safe-p95 "
                             "(from the bmax phase output)")
        phase_cost(args.b_safe_p50, args.b_safe_p95)
    elif args.phase == "turn":
        phase_turn(turn_iters=args.turn_iters, river_iters=args.river_iters)


if __name__ == "__main__":
    main()
