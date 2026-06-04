#!/usr/bin/env python3
"""ReBeL de-risk, Stage 1 (Leduc): depth-limited solving correctness + resolving-safety findings.

Consolidates, into one reproducible artifact, what depth-limited solving on Leduc establishes BEFORE
training a value net (Stage 2). Four measurements:

  1. EXACT-ORACLE CONTROL = full CFR+ over the public tree (co-evolving subgame == CFR-D). This is
     the depth-limited control ceiling; the learned-leaf Stage 2 is judged by the gap to THIS.
     -> NashConv ~5e-4.

  2. LEAF-VALUE (UN)RELIABILITY of an ISOLATED re-solve: the per-card values from solving the
     round-2 subgame in isolation for fixed ranges only match the true full-game continuation CFVs
     at the MOST-reached entries; they diverge as reach drops -- up to ~0.09 even at >10% reach and
     ~0.77 at ~4% reach. This is NOT slow convergence (stable across 3k/12k/40k iters) -> the values
     are UNDER-DETERMINED: different valid subgame equilibria for the same ranges assign different
     values to thinly-reached hands. -> isolated subgame re-solving is the wrong way to generate
     value targets; ReBeL instead reads self-consistent CFVs off the trunk CFR solve, averaged over
     the visited PBS distribution (with a value net that smooths low-reach PBSs). A scale-up risk to
     watch, not a blocker (Leduc still solves to ~8e-4 via proper CFR-D, finding #1).

  3. ISOLATED-ORACLE TRUNK BIAS (pitfall): re-solving the subgame to equilibrium each trunk
     iteration biases the trunk regrets (the opponent re-adapts inside the leaf value), so the trunk
     does NOT recover the equilibrium round-1 strategy. -> round-1 L1 ~0.15 (large). The correct
     mechanic is co-evolving CFR-D (#1) / a value net FIXED per solve.

  4. RESOLVING SAFETY: a frozen isolated subgame strategy (solved at the equilibrium ranges, then
     held fixed) is off-path exploitable. -> assembled NashConv ~0.21 (>> 5e-4). -> scale-up must
     use continual re-solving at play, not a frozen subgame strategy.

CONCLUSION (design constraint for Stage 2): use the net as a FIXED PBS-value function during each
depth-limited solve (co-evolving / CFR-D), train it on the accurate leaf values (#2), and re-solve
subgames at play time. Stage 2 PASS = depth-limited solve with the learned net stays within a small
gap of the exact-oracle control (#1).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from poker_ai.rebel.leduc import LeducTree
from poker_ai.rebel.leaf_eval import ExactLeafOracle
from poker_ai.rebel.loop import (
    solve_round2_equilibrium,
    depth_limited_solve_isolated_oracle,
    _listify,
    _full_pol_with_round2,
)


def run(control_iters=1500, eq_iters=2000, trunk_iters=60, round2_iters=120,
        resolve_iters=3000, well_reached_frac=0.10):
    tree = LeducTree()
    r1_iids = [i for i in range(tree.n_iset) if tree.iset_round[i] == 1]
    out = {}

    # 1. exact-oracle control = full CFR+ (CFR-D with co-evolving subgame)
    _, hist = tree.cfr_plus(control_iters, eval_every=control_iters, averaging="linear")
    control_nc = hist[-1][1]
    out["control_full_cfr_nashconv"] = control_nc

    # reference equilibrium for the value/strategy comparisons
    star, _ = tree.cfr_plus(eq_iters, eval_every=0, averaging="linear")
    star_r1 = {i: star[i] for i in r1_iids}
    reaches = tree.cut_reaches(star)
    keys = sorted(tree.cut_keys())
    oracle = ExactLeafOracle(tree, normalize=True)

    # 2. leaf-value accuracy: well-reached (reach frac > thresh) vs low-reach degradation
    well_err = 0.0          # max err over entries with reach fraction > well_reached_frac
    worst_err = 0.0         # worst err over any positive-reach entry
    worst_frac = 1.0        # reach fraction at that worst entry
    for k in keys:
        r0, r1 = reaches[k]
        v0t, v1t = oracle.evaluate(k, r0, r1, star)                       # truth
        r2 = solve_round2_equilibrium(tree, k, r0, r1, resolve_iters, "linear")
        pol = _listify(tree, star_r1, round2={k: r2})
        v0, v1 = oracle.evaluate(k, r0, r1, pol)                          # via isolated re-solve
        for vt, v, rr in ((v0t, v0, r0), (v1t, v1, r1)):
            tot = rr.sum()
            if tot <= 0:
                continue
            frac = rr / tot
            d = np.abs(vt - v)
            well = frac > well_reached_frac
            if well.any():
                well_err = max(well_err, float(np.max(d[well])))
            pos = rr > 0
            if pos.any():
                i = int(np.argmax(np.where(pos, d, -1.0)))
                if d[i] > worst_err:
                    worst_err = float(d[i]); worst_frac = float(frac[i])
    out["leaf_value_well_reached_max_err"] = well_err
    out["leaf_value_worst_err"] = worst_err
    out["leaf_value_worst_err_reach_frac"] = worst_frac
    out["well_reached_frac_threshold"] = well_reached_frac

    # 4. resolving safety: frozen isolated subgame at equilibrium ranges
    r2_by_key = {}
    for k in keys:
        r0, r1 = reaches[k]
        r2_by_key[k] = solve_round2_equilibrium(tree, k, r0, r1, resolve_iters, "linear")
    frozen = _full_pol_with_round2(tree, star_r1, r2_by_key)
    out["frozen_subgame_assembled_nashconv"] = tree.nash_conv(frozen)

    # 3. isolated-oracle trunk bias: does the depth-limited trunk recover star's round-1 strategy?
    iso_full, _ = depth_limited_solve_isolated_oracle(
        tree, trunk_iters=trunk_iters, round2_iters=round2_iters,
        averaging="linear", round2_averaging="linear", eval_every=0)
    r1_l1 = float(np.mean([np.sum(np.abs(iso_full[i] - star[i])) for i in r1_iids]))
    out["isolated_oracle_trunk_round1_L1"] = r1_l1

    # verdict flags -- each is the EXPECTED de-risk finding (True = confirmed as expected)
    out["findings"] = {
        "control_low": control_nc < 1e-2,                                    # CFR-D control solves
        "isolated_resolve_values_unreliable": well_err > 1e-2,               # not a usable target source
        "isolated_trunk_biased": r1_l1 > 0.05,                              # per-iter re-solve is wrong
        "frozen_subgame_unsafe": out["frozen_subgame_assembled_nashconv"] > 1e-2,  # resolving-safety
    }
    out["conclusion"] = ("isolated subgame re-solving is the wrong primitive; Stage 2 must use a "
                         "value net FIXED per solve (CFR-D), targets read off the trunk solve, and "
                         "continual re-solving at play")
    out["all_findings_as_expected"] = all(out["findings"].values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-iters", type=int, default=1500)
    ap.add_argument("--trunk-iters", type=int, default=60)
    ap.add_argument("--round2-iters", type=int, default=120)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = run(control_iters=args.control_iters, trunk_iters=args.trunk_iters,
              round2_iters=args.round2_iters)

    print("ReBeL Stage-1 (Leduc) depth-limited solving findings")
    print(f"  1. exact-oracle control (full CFR+ / CFR-D) NashConv = {out['control_full_cfr_nashconv']:.5f}")
    print(f"  2. isolated-resolve leaf-value err: at frac>{out['well_reached_frac_threshold']:.0%} = "
          f"{out['leaf_value_well_reached_max_err']:.5f}   worst = {out['leaf_value_worst_err']:.5f} "
          f"@ reachfrac {out['leaf_value_worst_err_reach_frac']:.3f} (under-determined, not iter-fixable)")
    print(f"  3. isolated-oracle trunk round-1 L1 vs equilibrium = {out['isolated_oracle_trunk_round1_L1']:.5f}"
          f"  (biased -> wrong algorithm)")
    print(f"  4. frozen isolated subgame assembled NashConv = {out['frozen_subgame_assembled_nashconv']:.5f}"
          f"  (off-path exploitable -> resolving-safety)")
    print(f"\n  findings as expected: {out['all_findings_as_expected']}  {out['findings']}")
    print("  => Stage 2 must use a value net FIXED per solve (CFR-D) trained on trunk-solve CFV "
          "targets (not isolated re-solves), with continual re-solving at play.")

    if args.output_json:
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
