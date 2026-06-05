#!/usr/bin/env python3
"""ReBeL general-method self-play driver (Build 3b-ii): the game-agnostic depth-limited PBS-net
self-play loop on Leduc. Trains ONE generic PBS net as the depth-limit leaf via trunk-solve + exact
leaf-target harvest (broad coverage + on-policy), and reports the net-learnability trend + the net-leaf
sigma1 vs the exact-leaf control.

HONEST SCOPE: Leduc is a 2-LEVEL game (one cut) -> the leaf is the final round -> targets are EXACT and
do NOT bootstrap (the Step-3 finding). This validates net-learnability + leaf-amortization on the generic
substrate; self-play BOOTSTRAPPING (beating supervised coverage) needs a >=3-level game. Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import pyspiel

from poker_ai.rebel.iig_selfplay import self_play
from poker_ai.rebel.iig_pbs import leduc_is_cut


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=8)
    ap.add_argument("--trunk-iters", type=int, default=200)
    ap.add_argument("--coverage-per-cut", type=int, default=200)
    ap.add_argument("--train-epochs", type=int, default=500)
    ap.add_argument("--cont-iters", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    print("ReBeL general-method self-play (Leduc; one generic PBS net, one solver, one loop)")
    hist, net, dlg = self_play(
        pyspiel.load_game("leduc_poker"), leduc_is_cut,
        n_iters=args.n_iters, trunk_iters=args.trunk_iters, coverage_per_cut=args.coverage_per_cut,
        train_epochs=args.train_epochs, cont_iters=args.cont_iters, hidden=args.hidden,
        seed=args.seed, verbose=True)
    out = {"game": "leduc_poker", "history": hist,
           "final_mae_frac": hist[-1]["mae_frac"], "final_sigma1_l1": hist[-1]["sigma1_l1_vs_exact"]}
    print(f"  net learnability: val MAE {hist[0]['mae_frac']:.1%} -> {hist[-1]['mae_frac']:.1%}; "
          f"net-leaf sigma1 L1 vs exact-leaf control {hist[0]['sigma1_l1_vs_exact']:.3f} -> "
          f"{hist[-1]['sigma1_l1_vs_exact']:.3f}")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
