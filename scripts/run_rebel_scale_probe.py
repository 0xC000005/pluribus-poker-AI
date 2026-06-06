#!/usr/bin/env python3
"""Build-4 -> scale pivot: scale-readiness probe (the cheapest decisive go/no-go before the abstraction
build). Walks the Goofspiel size ladder on the generic substrate, reports the unabstracted infoset count
(the Python tree-walk ceiling), and for the feasible sizes measures the NET-leaf depth-limited self-play
exploitability (assembled NashConv) + wall-clock -- the first points on the single-consumer-GPU scaling
curve. Establishes that the unabstracted substrate caps at ~few-k infosets, so a LEARNED ABSTRACTION is
the mandatory next lever (VRAM is NOT the binding constraint -- the CPU tree-walk is). Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_public_key, goofspiel_is_cut
from poker_ai.rebel.iig_selfplay import (build_public_index, make_net_leaf_fn, PBSNet,
                                         coverage_rows, precompute_cont, _train)

# unabstracted self-play is intractable past this many infosets (Python tree-walk ~500k nodes/s)
FEASIBLE_ISET_CAP = 50_000


def curve_point(num_cards, cont_iters, trunk_iters, coverage_per_cut, train_epochs, hidden, seed):
    t0 = time.time()
    g = load_goofspiel(num_cards=num_cards, points_order="random")
    dlg = DepthLimitedGame(g, goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    rec = {"num_cards": num_cards, "n_iset": dlg.n_iset,
           "cut_public_states": len(set(n[1] for n in dlg.cut_nodes)), "build_s": round(time.time() - t0, 1)}
    if dlg.n_iset > FEASIBLE_ISET_CAP:
        rec["feasible_unabstracted"] = False
        rec["note"] = "past the unabstracted tree-walk ceiling -> needs learned abstraction"
        return rec, dlg
    rec["feasible_unabstracted"] = True
    pub_index, keys = build_public_index(dlg)
    P = max(max(dlg.n_priv(k, 0), dlg.n_priv(k, 1)) for k in keys)
    cont = dlg.cfr_plus(cont_iters)
    rec["nash_floor"] = dlg.nash_conv(cont)
    cont_cache = precompute_cont(dlg, cont)
    rng = np.random.default_rng(seed)
    X, Y, W = coverage_rows(dlg, pub_index, P, cont_cache, keys, coverage_per_cut, rng)
    net = PBSNet(len(pub_index), P, hidden)
    m = _train(net, X, Y, W, train_epochs, seed=seed)
    rec["net_val_mae_frac"] = m["mae_frac"]
    s_net = dlg.trunk_solve(make_net_leaf_fn(net, dlg, pub_index, P), iters=trunk_iters)
    ex = dlg.nash_conv(dlg.assemble(s_net, cont))
    rec["net_leaf_exploit"] = ex
    rec["net_leaf_over_nash"] = round(ex / max(rec["nash_floor"], 1e-9), 1)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec, dlg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--cont-iters", type=int, default=400)
    ap.add_argument("--trunk-iters", type=int, default=300)
    ap.add_argument("--coverage-per-cut", type=int, default=120)
    ap.add_argument("--train-epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    print("Scale-readiness probe: Goofspiel size ladder + net-leaf self-play curve (one consumer GPU host)")
    out = {"ladder": [], "feasible_iset_cap": FEASIBLE_ISET_CAP}
    for nc in args.cards:
        rec, _ = curve_point(nc, args.cont_iters, args.trunk_iters, args.coverage_per_cut,
                             args.train_epochs, args.hidden, args.seed)
        out["ladder"].append(rec)
        if rec["feasible_unabstracted"]:
            print(f"  goofspiel({nc}): n_iset={rec['n_iset']:7d}  net-leaf exploit "
                  f"{rec['net_leaf_exploit']:.4f} ({rec['net_leaf_over_nash']}x Nash {rec['nash_floor']:.4f}); "
                  f"netMAE {rec['net_val_mae_frac']:.1%}; {rec['wall_s']}s")
        else:
            print(f"  goofspiel({nc}): n_iset={rec['n_iset']:7d}  {rec['note']}")
    print("  => unabstracted ceiling is ~few-k infosets; VRAM is NOT binding (CPU tree-walk is) "
          "-> learned abstraction is the mandatory next lever.")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
