#!/usr/bin/env python3
"""ReBeL de-risk, Stage 0: the constant-oracle round-trip on Leduc (no training).

Isolates the two correctness must-fixes from the ReBeL decision doc BEFORE any value-net work:
  (#1) the CFV normalization convention (reach-weighted vs normalized), verified by a NUMERICAL
       round-trip rather than a shape test;
  (#2) uniform averaging (value-/policy-average and SampleLeaf weights identical), via CFR+.

Three checks:
  A. ENGINE TRUST + AVERAGING: full CFR+ over the tagged Leduc tree -> the AVERAGE policy's exact
     NashConv (OpenSpiel) descends to ~0. Linear averaging hits ~1e-3 (proves the tree + values +
     regret machinery are exact); uniform averaging (the pinned convention) descends more slowly.
  B. CONVENTION ROUND-TRIP: for random strategies, compute per-card continuation values at every
     public cut state through the ExactLeafOracle (pinned normalized convention), feed them back as
     the leaf evaluator, and assert the reconstructed ROUND-1 q-values equal the exact full-game
     q-values to ~1e-12.
  C. NEGATIVE CONTROL: the same with the un-normalized (counterfactual) convention produced but
     consumed as if normalized -> the round-trip breaks (large q error). Proves the test has teeth.

PASS = A converges low AND B matches to tol AND C clearly breaks.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from poker_ai.rebel.leduc import LeducTree
from poker_ai.rebel.leaf_eval import ExactLeafOracle


def max_round1_q_error(tree, q_full, q_dl):
    """Max abs difference over ROUND-1 infoset q-values (the trunk values the leaf feeds)."""
    err = 0.0
    for iid in range(tree.n_iset):
        if tree.iset_round[iid] != 1:
            continue
        err = max(err, float(np.max(np.abs(q_full[iid] - q_dl[iid]))))
    return err


def run(seeds=(0, 1, 2, 3, 4), cfr_iters=1000, tol=1e-9):
    tree = LeducTree()
    out = {"n_iset": tree.n_iset,
           "round1_infosets": int(sum(r == 1 for r in tree.iset_round)),
           "round2_infosets": int(sum(r == 2 for r in tree.iset_round)),
           "cut_keys": sorted(map(list, tree.cut_keys()))}

    # ---- A. engine trust + averaging
    _, hist_lin = tree.cfr_plus(cfr_iters, eval_every=cfr_iters // 4, averaging="linear")
    _, hist_uni = tree.cfr_plus(cfr_iters, eval_every=cfr_iters // 4, averaging="uniform")
    out["A_engine_trust"] = {
        "linear_nashconv": [[s, n] for s, n in hist_lin],
        "uniform_nashconv": [[s, n] for s, n in hist_uni],
        "linear_final": hist_lin[-1][1],
        "uniform_final": hist_uni[-1][1],
        "pass": hist_lin[-1][1] < 0.01,  # linear CFR+ must reach low -> engine exact
    }

    # ---- B / C. convention round-trip on random strategies
    oracle_ok = ExactLeafOracle(tree, normalize=True)
    oracle_bad = ExactLeafOracle(tree, normalize=False)
    b_errs, c_errs = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        pol = tree.random_policy(rng)
        q_full, _, _, _ = tree.values(pol)
        reaches = tree.cut_reaches(pol)
        leaf_ok = oracle_ok.all_leaf_values(reaches, pol)
        leaf_bad = oracle_bad.all_leaf_values(reaches, pol)
        q_ok = tree.values_with_leaf(pol, leaf_ok)
        q_bad = tree.values_with_leaf(pol, leaf_bad)
        b_errs.append(max_round1_q_error(tree, q_full, q_ok))
        c_errs.append(max_round1_q_error(tree, q_full, q_bad))

    out["B_convention_roundtrip"] = {
        "max_round1_q_error_per_seed": b_errs,
        "max_error": max(b_errs),
        "tol": tol,
        "pass": max(b_errs) < tol,
    }
    out["C_negative_control"] = {
        "max_round1_q_error_per_seed": c_errs,
        "min_error": min(c_errs),
        "pass": min(c_errs) > 1e-3,  # the wrong convention must clearly break the round-trip
    }
    out["verdict_pass"] = bool(out["A_engine_trust"]["pass"]
                               and out["B_convention_roundtrip"]["pass"]
                               and out["C_negative_control"]["pass"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfr-iters", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = run(tuple(args.seeds), args.cfr_iters, args.tol)

    A, B, C = out["A_engine_trust"], out["B_convention_roundtrip"], out["C_negative_control"]
    print(f"infosets: {out['n_iset']} (round1={out['round1_infosets']}, round2={out['round2_infosets']})")
    print(f"cut public states: {len(out['cut_keys'])}")
    print(f"\nA. ENGINE TRUST + AVERAGING (CFR+ avg-policy NashConv):")
    print(f"   linear  -> {A['linear_nashconv']}  final {A['linear_final']:.6f}")
    print(f"   uniform -> {A['uniform_nashconv']}  final {A['uniform_final']:.6f}")
    print(f"   {'PASS' if A['pass'] else 'FAIL'} (linear CFR+ reaches low -> tree/values/regrets exact)")
    print(f"\nB. CONVENTION ROUND-TRIP (normalized): max round-1 q error = {B['max_error']:.2e} "
          f"(tol {B['tol']:.0e})  {'PASS' if B['pass'] else 'FAIL'}")
    print(f"C. NEGATIVE CONTROL (un-normalized consumed as normalized): min q error = "
          f"{C['min_error']:.3f}  {'PASS (breaks, as required)' if C['pass'] else 'FAIL (did not break)'}")
    print(f"\nVERDICT: {'PASS' if out['verdict_pass'] else 'FAIL'}")

    if args.output_json:
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.output_json}")
    return 0 if out["verdict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
