#!/usr/bin/env python3
"""ReBeL step C (completion): the LIVE net-leaf gate on a real turn+river spot.

Trains a cut-GENERAL river PBS value net (input = cut pot/stacks features + both normalized ranges)
on exact-river-leaf targets sampled across ALL cut public states of a small turn spot, then uses the
net as the LIVE showdown_leaf_fn in a full turn solve (feasible: sub-ms/leaf) and measures the
net-leaf agent's 2-street exploitability (two_street_nashconv) vs the exact-leaf control. This is the
single-value-leaf correctness check on the real game (proven on Leduc) + the efficiency payoff
(net-leaf turn solve vs the per-iteration-infeasible exact-leaf solve). Small/short-stack spot so the
BR is reliable and the exact control is computable. Slumbot held-out; diagnostic only.

The reusable pieces (CtxRiverNet, the cut-general sampler, training, the net leaf adapter) live in
``poker_ai.rebel.self_play`` -- single source of truth shared with the step-3 self-play loop.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from poker_ai.rebel.turn_river import (
    TurnSpot, build_turn_solver, showdown_cut_indices, make_exact_river_showdown_fn,
    two_street_nashconv, _average_strategy_array, parse_card,
)
from poker_ai.rebel.self_play import (
    sample_targets_all_cuts, make_net_leaf_fn, train_ctx_net as train,
)


def run(n_strategies=40, river_iters=200, trunk_iters=60, epochs=800, seed=0):
    spot = TurnSpot(board=[parse_card(c) for c in ("Ah", "Kd", "7c", "2s")],
                    pot=1000, hero_stack=300, villain_stack=300, hero_first=True)
    ts = build_turn_solver(spot)
    H = ts.n
    hr = np.ones(H, np.float32) / H; vr = np.ones(H, np.float32) / H
    out = {"spot": "AhKd7c2s pot10bb stacks3bb", "H": H,
           "cuts": len(showdown_cut_indices(ts)), "n_strategies": n_strategies}

    t0 = time.time()
    X, Y, W, _ = sample_targets_all_cuts(ts, n_strategies, seed, river_iters, batched=True)
    out["target_gen_s"] = round(time.time() - t0, 1); out["n_targets"] = int(X.shape[0])
    net, m = train(X, Y, W, H, hidden=512, epochs=epochs, seed=seed)
    out["net"] = m

    # exact-leaf control turn solve + its 2-street exploitability
    t0 = time.time()
    ts.solve(n_iterations=trunk_iters, hero_range=hr, villain_range=vr, backend="cpu",
             showdown_leaf_fn=make_exact_river_showdown_fn(ts, river_iters=river_iters))
    out["exact_turn_solve_s"] = round(time.time() - t0, 1)
    sig_exact = _average_strategy_array(np.asarray(ts._strategy_sum))
    r_exact = two_street_nashconv(ts, sig_exact, hr.astype(np.float64), vr.astype(np.float64),
                                  river_iters=river_iters)
    out["exact_leaf_agent_nashconv"] = r_exact["nashconv"]

    # NET-leaf live turn solve + its 2-street exploitability (measured by the EXACT river BR)
    t0 = time.time()
    ts.solve(n_iterations=trunk_iters, hero_range=hr, villain_range=vr, backend="cpu",
             showdown_leaf_fn=make_net_leaf_fn(net, ts))
    out["net_turn_solve_s"] = round(time.time() - t0, 2)
    sig_net = _average_strategy_array(np.asarray(ts._strategy_sum))
    r_net = two_street_nashconv(ts, sig_net, hr.astype(np.float64), vr.astype(np.float64),
                               river_iters=river_iters)
    out["net_leaf_agent_nashconv"] = r_net["nashconv"]

    out["turn_solve_speedup"] = round(out["exact_turn_solve_s"] / max(out["net_turn_solve_s"], 1e-3), 1)
    out["pot"] = 1000
    out["net_close_to_exact"] = bool(r_net["nashconv"] < r_exact["nashconv"] + 0.05 * 1000)
    return out, net


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-strategies", type=int, default=40)
    ap.add_argument("--river-iters", type=int, default=200)
    ap.add_argument("--trunk-iters", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out, _ = run(n_strategies=args.n_strategies, river_iters=args.river_iters,
                 trunk_iters=args.trunk_iters, epochs=args.epochs)
    print("ReBeL step C completion: LIVE net-leaf gate")
    print(f"  spot {out['spot']}  H={out['H']} hands, {out['cuts']} cut public states")
    print(f"  targets: {out['n_targets']} across cuts in {out['target_gen_s']}s; "
          f"net val MAE {out['net']['val_mae']:.1f} ({out['net']['mae_frac']:.1%} of scale)")
    print(f"  exact-leaf turn solve {out['exact_turn_solve_s']}s -> 2-street NashConv "
          f"{out['exact_leaf_agent_nashconv']:.1f} ({out['exact_leaf_agent_nashconv']/1000:.3f} pot)")
    print(f"  NET-leaf  turn solve {out['net_turn_solve_s']}s -> 2-street NashConv "
          f"{out['net_leaf_agent_nashconv']:.1f} ({out['net_leaf_agent_nashconv']/1000:.3f} pot)")
    print(f"  turn-solve speedup (net vs exact leaf): {out['turn_solve_speedup']}x")
    print(f"  net agent close to exact control: {out['net_close_to_exact']}")
    if args.output_json:
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
