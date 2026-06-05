#!/usr/bin/env python3
"""ReBeL de-risk, Stage 2c (Leduc): end-to-end exploitability of the learned + gadget agent.

Stage 1/2 established: (i) the exact-oracle control = full CFR-D NashConv ~8e-4; (ii) a value net
learns the PBS->CFV map when targets are consistent; (iii) naive re-solving is off-path exploitable
(0.21) but the DeepStack safe-resolving GADGET fixes it (validated: exact-CFV gadget agent ~0.008).

This script measures the REAL metric (exploitability via OpenSpiel NashConv) for the agent that
plays round-1 from a depth-limited trunk solve and round-2 by GADGET-safe re-solving, comparing:
  * EXACT leaf  -> gadget agent (the depth-limited+gadget control ceiling), and
  * NET leaf    -> gadget agent (the end-to-end learned agent).
Both use the leaf values (exact or net) BOTH as the trunk leaf (shaping round-1) and as the gadget's
opponent-CFV constraints (making round-2 safe). PASS = the net agent's exploitability is close to the
exact-leaf gadget control (and far below the un-gadgeted 0.21 and the net-only R-NaD ceiling).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from poker_ai.rebel.leduc import LeducTree
from poker_ai.rebel.leaf_eval import ExactLeafOracle
from poker_ai.rebel.loop import (trunk_solve, exact_leaf_fn, gadget_resolve, _listify,
                                 _full_pol_with_round2)
from poker_ai.rebel.pbs_value_net import sample_pbs_dataset, train_pbs_net, net_leaf_fn


def gadget_agent_exploitability(tree, sigma1, leaf_fn, gadget_iters=2000):
    """Exploitability (OpenSpiel NashConv) of the agent: round-1 = sigma1 ({iid:arr}); round-2 =
    GADGET-safe re-solve at sigma1's induced ranges, with the opponent constrained to the leaf values
    leaf_fn produces (the blueprint CFVs). Returns (nashconv, full_policy)."""
    reaches = tree.cut_reaches(_listify(tree, sigma1))
    round2 = {}
    for k in sorted(tree.cut_keys()):
        r0, r1 = reaches[k]
        v0, v1 = leaf_fn(k, r0, r1)
        round2.update(gadget_resolve(tree, k, 0, r0, r1, v1, iters=gadget_iters))  # P0 safe vs P1
        round2.update(gadget_resolve(tree, k, 1, r1, r0, v0, iters=gadget_iters))  # P1 safe vs P0
    full = _full_pol_with_round2(tree, sigma1, {0: round2})
    return tree.nash_conv(full), full


def run(n_strategies=1000, trunk_iters=200, epochs=1500, round2_iters_ctrl=1500,
        gadget_iters=2000, seed=0):
    tree = LeducTree()
    out = {"config": {"n_strategies": n_strategies, "trunk_iters": trunk_iters, "epochs": epochs,
                      "round2_iters_ctrl": round2_iters_ctrl, "gadget_iters": gadget_iters}}

    star, hist = tree.cfr_plus(2000, eval_every=2000, averaging="linear")
    out["full_cfrd_nashconv"] = hist[-1][1]

    # net trained on CONSISTENT blueprint-continuation targets (star round-2 fixed)
    t0 = time.time()
    data, keys = sample_pbs_dataset(tree, n_strategies=n_strategies, seed=seed, ref_strategy=star)
    net, mA = train_pbs_net(data, keys, epochs=epochs, seed=seed)
    out["data_gen_seconds"] = round(time.time() - t0, 1)
    out["net_val_reach_weighted_mae"] = mA["val_reach_weighted_mae"]

    # control: EXACT leaf -> trunk sigma1, then gadget agent
    t0 = time.time()
    exact_lf = exact_leaf_fn(tree, round2_iters=round2_iters_ctrl)
    s1_exact = trunk_solve(tree, exact_lf, trunk_iters=trunk_iters, averaging="linear")
    nc_exact, _ = gadget_agent_exploitability(tree, s1_exact, exact_lf, gadget_iters)
    out["exact_gadget_seconds"] = round(time.time() - t0, 1)
    out["exact_leaf_gadget_nashconv"] = nc_exact

    # end-to-end: NET leaf -> trunk sigma1, then gadget agent with NET CFV constraints
    net_lf = net_leaf_fn(net, keys)
    s1_net = trunk_solve(tree, net_lf, trunk_iters=trunk_iters, averaging="linear")
    nc_net, _ = gadget_agent_exploitability(tree, s1_net, net_lf, gadget_iters)
    out["net_leaf_gadget_nashconv"] = nc_net

    out["pass"] = {
        "gadget_beats_naive": nc_exact < 0.05,            # exact gadget agent is safe
        "net_close_to_exact": nc_net < 3 * max(nc_exact, 1e-4) + 0.01,
    }
    out["all_pass"] = bool(out["pass"]["gadget_beats_naive"] and out["pass"]["net_close_to_exact"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-strategies", type=int, default=1000)
    ap.add_argument("--trunk-iters", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--gadget-iters", type=int, default=2000)
    ap.add_argument("--round2-iters-ctrl", type=int, default=1000)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = run(n_strategies=args.n_strategies, trunk_iters=args.trunk_iters, epochs=args.epochs,
              gadget_iters=args.gadget_iters, round2_iters_ctrl=args.round2_iters_ctrl)

    print("ReBeL Stage-2c (Leduc) end-to-end exploitability (real metric: OpenSpiel NashConv)")
    print(f"  full CFR-D (ceiling)              = {out['full_cfrd_nashconv']:.5f}")
    print(f"  EXACT leaf + gadget agent         = {out['exact_leaf_gadget_nashconv']:.5f}")
    print(f"  NET   leaf + gadget agent (E2E)   = {out['net_leaf_gadget_nashconv']:.5f}")
    print(f"  (net held-out reach-weighted MAE  = {out['net_val_reach_weighted_mae']:.4f})")
    print(f"  PASS: {out['pass']}  ALL={out['all_pass']}")
    if args.output_json:
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
