#!/usr/bin/env python3
"""ReBeL de-risk, Stage 2 (Leduc): a learned PBS value net as the depth-limited trunk leaf.

Two clean metrics (see `poker_ai/rebel/pbs_value_net.py`):
  A. LEARNABILITY -- train a tiny net on round-2 PBS->CFV targets; report held-out REACH-WEIGHTED
     MAE (thin-reach under-determined entries down-weighted, per Stage-1 finding #2).
  B. SUBSTITUTION FIDELITY -- run the depth-limited round-1 solve with the NET leaf and with the
     EXACT leaf (same `loop.trunk_solve` algorithm, fixed-function leaf), and report the round-1
     strategy L1 between them. Small => the net is a faithful drop-in for the exact leaf. For
     context, both are also compared to the full CFR-D round-1 reference.

PASS = A small (net learns the value function) AND B small (net reproduces the exact-leaf trunk).
Absolute end-to-end exploitability additionally needs safe re-solving (Stage 2b; Stage-1 finding #4).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from poker_ai.rebel.leduc import LeducTree
from poker_ai.rebel.loop import trunk_solve, exact_leaf_fn, blueprint_leaf_fn
from poker_ai.rebel.pbs_value_net import sample_pbs_dataset, train_pbs_net, net_leaf_fn


def round1_l1(tree, polA, polB):
    """Mean and reach-weighted L1 over round-1 infosets between two {iid: prob} strategy dicts.
    Reach weight = own reach of the infoset under polA (uniform fallback)."""
    r1 = [i for i in range(tree.n_iset) if tree.iset_round[i] == 1]
    l1 = {i: float(np.sum(np.abs(polA[i] - polB[i]))) for i in r1}
    return float(np.mean(list(l1.values()))), l1


def run(n_strategies=250, round2_iters_data=600, round2_iters_ctrl=600,
        trunk_iters=150, epochs=400, hidden=64, seed=0, target_mode="resolve"):
    tree = LeducTree()
    out = {"config": {"n_strategies": n_strategies, "round2_iters_data": round2_iters_data,
                      "round2_iters_ctrl": round2_iters_ctrl, "trunk_iters": trunk_iters,
                      "epochs": epochs, "hidden": hidden, "seed": seed, "target_mode": target_mode}}

    # full CFR-D round-1 reference (near-Nash); its round-2 is the blueprint continuation strategy
    star, _ = tree.cfr_plus(2000, eval_every=0, averaging="linear")
    star_r1 = {i: star[i] for i in range(tree.n_iset) if tree.iset_round[i] == 1}
    ref = star if target_mode == "blueprint" else None

    # data generation
    t0 = time.time()
    data, keys = sample_pbs_dataset(tree, n_strategies=n_strategies, seed=seed,
                                    round2_iters=round2_iters_data, ref_strategy=ref)
    out["data_gen_seconds"] = round(time.time() - t0, 1)

    # A. train + held-out learnability
    net, mA = train_pbs_net(data, keys, hidden=hidden, epochs=epochs, seed=seed)
    out["A_learnability"] = mA

    # B. substitution fidelity: net-leaf trunk vs the MATCHING exact-leaf trunk (and vs full CFR-D).
    # The exact leaf is the same value function the targets were drawn from (apples-to-apples).
    t0 = time.time()
    if target_mode == "blueprint":
        ctrl_leaf = blueprint_leaf_fn(tree, star)
    else:
        ctrl_leaf = exact_leaf_fn(tree, round2_iters=round2_iters_ctrl)
    exact_pol = trunk_solve(tree, ctrl_leaf, trunk_iters=trunk_iters, averaging="linear")
    out["exact_trunk_seconds"] = round(time.time() - t0, 1)
    net_pol = trunk_solve(tree, net_leaf_fn(net, keys), trunk_iters=trunk_iters, averaging="linear")

    net_vs_exact, _ = round1_l1(tree, net_pol, exact_pol)
    net_vs_star, _ = round1_l1(tree, net_pol, star_r1)
    exact_vs_star, _ = round1_l1(tree, exact_pol, star_r1)
    out["B_substitution_fidelity"] = {
        "round1_L1_net_vs_exact_leaf": net_vs_exact,
        "round1_L1_net_vs_full_cfrd": net_vs_star,
        "round1_L1_exact_leaf_vs_full_cfrd": exact_vs_star,
    }

    out["pass"] = {
        "A_learnable": mA["val_reach_weighted_mae"] < 0.05,
        "B_faithful_drop_in": net_vs_exact < 0.05,
    }
    out["all_pass"] = bool(out["pass"]["A_learnable"] and out["pass"]["B_faithful_drop_in"])
    return out, net


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-strategies", type=int, default=250)
    ap.add_argument("--round2-iters-data", type=int, default=600)
    ap.add_argument("--trunk-iters", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--target-mode", choices=["resolve", "blueprint"], default="resolve")
    ap.add_argument("--output-json")
    ap.add_argument("--save-net")
    args = ap.parse_args(argv)
    out, net = run(n_strategies=args.n_strategies, round2_iters_data=args.round2_iters_data,
                   trunk_iters=args.trunk_iters, epochs=args.epochs, target_mode=args.target_mode)

    A, B = out["A_learnability"], out["B_substitution_fidelity"]
    print("ReBeL Stage-2 (Leduc) learned PBS value net")
    print(f"  data: {A['n_samples']} PBSs ({out['data_gen_seconds']}s gen)")
    print(f"  A. learnability: held-out reach-weighted MAE = {A['val_reach_weighted_mae']:.4f} "
          f"(train {A['train_reach_weighted_mae']:.4f}, unweighted {A['val_unweighted_mae']:.4f})")
    print(f"  B. substitution fidelity (round-1 L1):")
    print(f"       net  vs exact-leaf  = {B['round1_L1_net_vs_exact_leaf']:.4f}")
    print(f"       net  vs full CFR-D  = {B['round1_L1_net_vs_full_cfrd']:.4f}")
    print(f"       exact vs full CFR-D = {B['round1_L1_exact_leaf_vs_full_cfrd']:.4f}")
    print(f"  PASS: {out['pass']}  ALL={out['all_pass']}")

    if args.save_net:
        import torch
        torch.save(net.state_dict(), args.save_net)
        print(f"saved net -> {args.save_net}")
    if args.output_json:
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
