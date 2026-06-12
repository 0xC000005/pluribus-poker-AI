#!/usr/bin/env python3
"""B0 (cheapest decisive probe, gates A-vs-B): does training the PBS value net on PER-BELIEF RE-SOLVED
targets (V*(belief) -- re-solve the subgame to equilibrium AT each sampled belief via the batched solver)
break the 0.21 single-cut exploitability floor that the FIXED-continuation targets plateau at?

EC found: with a FIXED below-cut continuation as the net target, G5 exploitability plateaus ~0.21 (more net
training did not help). The hypothesis: the floor is the fixed continuation's off-path INCOHERENCE, so
learning the true equilibrium value function V*(belief) (per-belief re-solved targets) should give a more
coherent leaf and a lower exploitability -- OR it still plateaus, which would prove the single-cut leaf is
intrinsically the final-round value (no bootstrapping) and breaking 0.21 REQUIRES nested multi-level cuts
(the multi-week build). Either outcome decisively informs A (write up) vs B (close the loop). Reuses the
existing batched solver + gadget; exact nash_conv metric on small games. Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key
from poker_ai.rebel.iig_selfplay import PBSNet, build_public_index, make_net_leaf_fn, _train, _row
from poker_ai.rebel.iig_batched import _compile_topology, solve_all_keys_soa, set_entries, _leaf_values_from_cont
from poker_ai.rebel.iig_batched_graphed import CompiledStepSolver, GraphedSolver
from poker_ai.rebel.iig_gadget import safe_continuation


def measure_onpolicy(dlg, net, pub_index, P, keys, compiled, subgame_iters, trunk_iters, device):
    """One curve point: net-leaf trunk sigma1, on-policy batched re-solve continuation, EXACT nash_conv.
    Returns (nc_onpolicy, sig1, cont_pol) -- sig1/cont_pol reused for the final-round gadget."""
    sig1 = dlg.trunk_solve(make_net_leaf_fn(net, dlg, pub_index, P), iters=trunk_iters)
    reaches = dlg.cut_reaches(dlg.assemble(sig1, dlg.uniform_policy()))
    ranges = {k: (reaches[k][0], reaches[k][1]) for k in keys}
    eq = solve_all_keys_soa(dlg, ranges, max(subgame_iters, 600), device=device, compiled=compiled)
    cont_pol = dlg.uniform_policy()
    for k in keys:
        for iid, pr in eq[k].items():
            cont_pol[iid] = pr
    nc = dlg.nash_conv(dlg.assemble(sig1, cont_pol))
    return nc, sig1, cont_pol


def _inner_batching(inner_mode):
    return "sequential" if inner_mode.startswith("sequential") else "fused"


def _inner_tool(inner_mode):
    if inner_mode.endswith("-compile"):
        return "compile"
    if inner_mode.endswith("-graphs"):
        return "graphs"
    return "eager"


def _build_amortized_solver(dlg, comp, tool, ranges):
    if tool == "compile":
        solver = CompiledStepSolver(dlg, comp)
        solver.solve(ranges, 2)
        return solver
    if tool == "graphs":
        return GraphedSolver(dlg, comp)
    return None


def _resolve_one_belief(dlg, ranges, keys, inner_mode, comp_all, comp_key, subgame_iters, device,
                        solver_all=None, solver_key=None):
    """The inner re-solve at one sampled belief -> {key: (v0*, v1*)}. The ONLY behavioral difference
    between the two timing arms; both call the SAME GPU SoA kernel solve_all_keys_soa (float64), differing
    only in batching granularity. FUSED = one level-grouped GEMM over the whole population of cut subgames;
    SEQUENTIAL = the SAME kernel per-key (the fair within-tree-GPU-CFR baseline -- NOT solve_all_keys, the
    Python reference walk, which would strawman the speedup with interpreter overhead)."""
    batching = _inner_batching(inner_mode)
    tool = _inner_tool(inner_mode)
    if batching == "fused":
        if tool == "eager":
            _eq, cont = solve_all_keys_soa(dlg, ranges, subgame_iters, device=device, compiled=comp_all,
                                           return_cont=True)
        else:
            _eq, cont = solver_all.solve(ranges, subgame_iters, return_cont=True)
        return _leaf_values_from_cont(dlg, comp_all, cont, ranges, keys)
    assert solve_all_keys_soa.__name__ == "solve_all_keys_soa"  # guard: sequential is the SoA kernel per-key
    leaf = {}
    for k in keys:
        if tool == "eager":
            _eqk, contk = solve_all_keys_soa(dlg, {k: ranges[k]}, subgame_iters, device=device,
                                             compiled=comp_key[k], return_cont=True)
        else:
            _eqk, contk = solver_key[k].solve({k: ranges[k]}, subgame_iters, return_cont=True)
        leaf[k] = _leaf_values_from_cont(dlg, comp_key[k], contk, {k: ranges[k]}, [k])[k]
    return leaf


def perbelief_targets(dlg, pub_index, P, keys, n_samples, subgame_iters, device, rng,
                      inner_mode, comp_all, comp_key, torch, solver_all=None, solver_key=None):
    """Training rows whose targets are V*(belief): for each sampled belief, RE-SOLVE all subgames to eq via
    the batched solver (fused or sequential, see _resolve_one_belief) and read the normalized equilibrium
    continuation value. Times ONLY the inner re-solve (+ leaf reconstruction) -- the quantity the mechanism
    reduces; the _row / X-Y-W host assembly is OUTSIDE the timed region (identical work in both arms).
    Returns (X, Y, W, inner_gpu_ms, inner_wall_ms) where the times are summed over this round's resolves."""
    X, Y, W = [], [], []
    inner_gpu_ms = 0.0
    inner_wall_ms = 0.0
    for _ in range(n_samples):
        ranges = {}
        for k in keys:
            a0 = float(rng.choice([0.3, 1.0, 3.0])); a1 = float(rng.choice([0.3, 1.0, 3.0]))
            ranges[k] = (rng.dirichlet(np.full(dlg.n_priv(k, 0), a0)),
                         rng.dirichlet(np.full(dlg.n_priv(k, 1), a1)))
        if device == "cuda":
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); ev0.record()
            leaf = _resolve_one_belief(dlg, ranges, keys, inner_mode, comp_all, comp_key, subgame_iters, device,
                                       solver_all=solver_all, solver_key=solver_key)
            ev1.record(); torch.cuda.synchronize(); t1 = time.perf_counter()
            inner_gpu_ms += ev0.elapsed_time(ev1); inner_wall_ms += (t1 - t0) * 1e3
        else:
            t0 = time.perf_counter()
            leaf = _resolve_one_belief(dlg, ranges, keys, inner_mode, comp_all, comp_key, subgame_iters, device,
                                       solver_all=solver_all, solver_key=solver_key)
            dt = (time.perf_counter() - t0) * 1e3
            inner_gpu_ms += dt; inner_wall_ms += dt
        for k in keys:
            v0, v1 = leaf[k]
            x, y, w = _row(dlg, pub_index, P, k, ranges[k][0], ranges[k][1], v0, v1)
            X.append(x); Y.append(y); W.append(w)
    return (np.array(X, np.float32), np.array(Y, np.float32), np.array(W, np.float32),
            inner_gpu_ms, inner_wall_ms)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cards", type=int, default=4)
    ap.add_argument("--n-target-rounds", type=int, default=8)
    ap.add_argument("--samples-per-round", type=int, default=40)
    ap.add_argument("--measure-every", type=int, default=2,
                    help="measure exact on-policy nash_conv every N rounds (final round always measured)")
    ap.add_argument("--subgame-iters", type=int, default=300)
    ap.add_argument("--trunk-iters", type=int, default=200)
    ap.add_argument("--train-epochs", type=int, default=400)
    ap.add_argument("--gadget-iters", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--buffer-cap", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds numpy + torch (+cuda) -- net INIT is otherwise unseeded, the dominant "
                         "source of absolute-exploitability run-to-run variance at G5 (the within-run "
                         "decreasing-with-compute TREND is robust; absolute values need seeded multi-runs)")
    ap.add_argument("--inner-mode", choices=("fused", "sequential", "fused-compile",
                                             "sequential-compile", "fused-graphs",
                                             "sequential-graphs"), default="fused",
                    help="inner re-solve execution: fused/sequential keep the original eager arms; "
                         "*-compile uses torch.compile(reduce-overhead); *-graphs uses CUDA Graphs. "
                         "The batching axis remains fused vs sequential.")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="cpu forces exact-parity mode (no atomics) for the determinism gate; "
                         "cuda for the timing comparison")
    ap.add_argument("--skip-gadget", action="store_true",
                    help="skip the final safe-resolving gadget pass (pure-Python, ~40min at G5). "
                         "Protocol-clean for the e2e TIMING runs: the gadget is outside total_wall_s and "
                         "unused by the e2e analysis; band-characterization runs should keep it")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dlg = DepthLimitedGame(load_goofspiel(args.num_cards), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    pub_index, keys = build_public_index(dlg)
    P = max(max(dlg.n_priv(k, 0), dlg.n_priv(k, 1)) for k in keys)
    rng = np.random.default_rng(args.seed)
    init = {k: (np.ones(dlg.n_priv(k, 0)) / dlg.n_priv(k, 0), np.ones(dlg.n_priv(k, 1)) / dlg.n_priv(k, 1)) for k in keys}
    # compile ONCE before the loop (one-time infra, excluded from the inner-resolve timer): comp_all for the
    # fused arm (+ the always-fused metrology), comp_key for the sequential arm.
    batching = _inner_batching(args.inner_mode)
    tool = _inner_tool(args.inner_mode)
    comp_all = _compile_topology(dlg, init, device)
    comp_key = {k: _compile_topology(dlg, {k: init[k]}, device) for k in keys} if batching == "sequential" else None
    setup_t0 = time.perf_counter()
    if tool == "graphs" and device != "cuda":
        raise RuntimeError("--inner-mode *-graphs requires CUDA")
    solver_all = _build_amortized_solver(dlg, comp_all, tool, init) if batching == "fused" else None
    solver_key = None
    if batching == "sequential" and tool != "eager":
        solver_key = {k: _build_amortized_solver(dlg, comp_key[k], tool, {k: init[k]}) for k in keys}
    amortized_setup_wall_s = time.perf_counter() - setup_t0
    print(f"B0 per-belief V* targets: Goofspiel-{args.num_cards} (n_iset={dlg.n_iset}, device={device}, "
          f"inner_mode={args.inner_mode}, batching={batching}, tool={tool})", flush=True)

    net = PBSNet(len(pub_index), P, args.hidden)
    X = Y = W = None
    history = []
    sig1 = cont_pol = None  # final measured trunk + continuation, reused for the gadget
    t_loop0 = time.perf_counter()
    for it in range(args.n_target_rounds):
        xi, yi, wi, ig, iw = perbelief_targets(dlg, pub_index, P, keys, args.samples_per_round,
                                               args.subgame_iters, device, rng,
                                               args.inner_mode, comp_all, comp_key, torch,
                                               solver_all=solver_all, solver_key=solver_key)
        X, Y, W = (xi, yi, wi) if X is None else (np.concatenate([X, xi]), np.concatenate([Y, yi]), np.concatenate([W, wi]))
        if X.shape[0] > args.buffer_cap:
            idx = rng.choice(X.shape[0], args.buffer_cap, replace=False); X, Y, W = X[idx], Y[idx], W[idx]
        t_tr0 = time.perf_counter()
        m = _train(net, X, Y, W, args.train_epochs, seed=0)
        train_wall_ms = (time.perf_counter() - t_tr0) * 1e3
        rec = {"round": it, "val_mae_frac": m["mae_frac"], "n": int(X.shape[0]),
               "n_beliefs_cumulative": (it + 1) * args.samples_per_round,  # target-compute axis (# resolved beliefs)
               "inner_resolve_gpu_ms": round(ig, 3), "inner_resolve_wall_ms": round(iw, 3),
               "train_wall_ms": round(train_wall_ms, 1), "measure_wall_ms": 0.0}
        is_final = (it == args.n_target_rounds - 1)
        if is_final or (it + 1) % args.measure_every == 0:   # exact on-policy curve point (ALWAYS fused = comp_all)
            t_me0 = time.perf_counter()
            nc, sig1, cont_pol = measure_onpolicy(dlg, net, pub_index, P, keys, comp_all,
                                                  args.subgame_iters, args.trunk_iters, device)
            rec["measure_wall_ms"] = round((time.perf_counter() - t_me0) * 1e3, 1)
            rec["exploit_onpolicy"] = round(nc, 5)
            print(f"  round {it}: V*-net val MAE {m['mae_frac']:.1%}  on-policy exploit {nc:.4f}  "
                  f"(beliefs {rec['n_beliefs_cumulative']}, inner {iw:.0f}ms/{ig:.0f}ms gpu, train "
                  f"{train_wall_ms:.0f}ms)", flush=True)
        else:
            print(f"  round {it}: V*-net val MAE {m['mae_frac']:.1%}  (inner {iw:.0f}ms wall / {ig:.0f}ms gpu, "
                  f"train {train_wall_ms:.0f}ms)", flush=True)
        history.append(rec)
    total_wall_s = time.perf_counter() - t_loop0

    # final round always measured -> sig1/cont_pol set; gadget safe continuation on top, exact nash_conv
    nc_onpolicy = history[-1]["exploit_onpolicy"]
    if args.skip_gadget:
        nc_gadget = None
    else:
        full_gadget = safe_continuation(dlg, sig1, cont_pol, iters=args.gadget_iters)
        nc_gadget = dlg.nash_conv(full_gadget)
    nash_floor = dlg.nash_conv(dlg.cfr_plus(600)) if args.num_cards <= 4 else None

    curve = [(h["n_beliefs_cumulative"], h["exploit_onpolicy"]) for h in history if "exploit_onpolicy" in h]
    inner_gpu_total = sum(h["inner_resolve_gpu_ms"] for h in history)
    inner_wall_total = sum(h["inner_resolve_wall_ms"] for h in history)
    train_wall_total = sum(h["train_wall_ms"] for h in history)
    out = {"game": f"goofspiel{args.num_cards}", "n_iset": dlg.n_iset, "target_type": "per_belief_resolved_Vstar",
           "inner_mode": args.inner_mode, "device": device, "seed": args.seed,
           "inner_batching": batching, "inner_tool": tool,
           "amortized_setup_wall_s_excluded": round(amortized_setup_wall_s, 3),
           "final_val_mae": history[-1]["val_mae_frac"], "nashconv_on_policy": round(nc_onpolicy, 5),
           "nashconv_gadget": (round(nc_gadget, 5) if nc_gadget is not None else None),
           "nash_floor": (round(nash_floor, 5) if nash_floor else None),
           "curve_beliefs_vs_exploit": curve,
           "curve_decreasing": bool(len(curve) >= 2 and curve[-1][1] < curve[0][1] - 1e-3),
           # timing: inner-resolve (the accelerated term) vs train (unaccelerated, identical across arms) vs total loop wall
           "inner_resolve_gpu_ms_total": round(inner_gpu_total, 1),
           "inner_resolve_wall_s_total": round(inner_wall_total / 1e3, 3),
           "train_wall_s_total": round(train_wall_total / 1e3, 3),
           "total_wall_s": round(total_wall_s, 3),
           "inner_resolve_wall_s_per_round": [round(h["inner_resolve_wall_ms"] / 1e3, 4) for h in history],
           "train_wall_s_per_round": [round(h["train_wall_ms"] / 1e3, 4) for h in history],
           "measure_wall_s_per_round": [round(h["measure_wall_ms"] / 1e3, 4) for h in history],
           "history": history}
    print(f"\n  B0 curve (Goofspiel-{args.num_cards}, V*-targets) on-policy exploit vs # resolved beliefs:")
    for nb, ex in curve:
        print(f"    {nb:>5d} beliefs -> {ex:.4f}")
    print(f"  final: on-policy {nc_onpolicy:.4f} | gadget "
          + (f"{nc_gadget:.4f}" if nc_gadget is not None else "skipped")
          + (f" | Nash floor {nash_floor:.4f}" if nash_floor else "")
          + (f"  [vs fixed-continuation-target gadget 0.21 on G5]" if args.num_cards == 5 else ""))
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
