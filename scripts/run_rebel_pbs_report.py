#!/usr/bin/env python3
"""Generic PBS-structure report: the depth-limit cut + public-belief-state dimensions derived by ONE
game-agnostic abstraction (poker_ai/rebel/iig_pbs) across games, with a bit-identical parity check vs
the trusted LeducTree. The belief dimension (private states per public state) is the quantity that
scales from ~few on small games to >1000 on HUNL -- the single-GPU scaling driver. Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyspiel

from poker_ai.rebel.leduc import LeducTree
from poker_ai.rebel.iig_pbs import PBSStructure, leduc_is_cut, first_decision_is_cut, uniform_policy_fn


def leduc_parity():
    lt = LeducTree()
    lr = lt.cut_reaches(lt.uniform_policy())
    gr = PBSStructure(pyspiel.load_game("leduc_poker"), leduc_is_cut).cut_reaches(uniform_policy_fn)
    diffs = []
    for side in (0, 1):
        L = np.array(sorted(v for arr in lr.values() for v in arr[side] if v > 1e-15))
        G = np.array(sorted(v for c in gr.values() for v in c[side].values() if v > 1e-15))
        diffs.append(float(np.abs(L - G).max()) if L.shape == G.shape else float("nan"))
    return {"n_public_leduc": len(lr), "n_public_generic": len(gr), "max_abs_diff": max(diffs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    games = [
        ("kuhn_poker", None, first_decision_is_cut, None),
        ("leduc_poker", None, leduc_is_cut, None),
        ("liars_dice", {"dice_sides": 3}, first_decision_is_cut, (lambda s: "root")),
        ("liars_dice", None, first_decision_is_cut, (lambda s: "root")),
    ]
    out = {"games": [], "leduc_parity": leduc_parity()}
    print("Generic PBS-structure report (one abstraction, many games)")
    for name, params, cut, pkf in games:
        g = pyspiel.load_game(name, params) if params else pyspiel.load_game(name)
        dims = PBSStructure(g, cut, public_key_fn=pkf).belief_dims()
        rec = {"game": name, "params": params, **dims}
        out["games"].append(rec)
        label = name + (f" {params}" if params else "")
        print(f"  {label:28s} n_public={dims['n_public_states']:4d} "
              f"belief_dim(p0 max/mean)={dims['priv_dim_p0_max']}/{dims['priv_dim_p0_mean']:.1f}")
    p = out["leduc_parity"]
    print(f"  Leduc PBS parity vs trusted LeducTree: n_public {p['n_public_generic']}=={p['n_public_leduc']}, "
          f"max|reach diff|={p['max_abs_diff']:.2e}")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
