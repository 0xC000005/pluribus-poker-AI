#!/usr/bin/env python3
"""Build-4 kill-switch: does a depth-limited value-function leaf at a NON-TERMINAL cut clear the trunk
wall on a >=3-level game (Goofspiel), which 2-level Leduc structurally could not?

On Goofspiel the cut at the prize-2 reveal is NON-TERMINAL (its continuation is rounds 2..C, multi-level).
We run the exact Phase-1 comparison here: single-value (blueprint) leaf vs the per-belief EQUILIBRIUM leaf
(the strongest leaf, what a perfect PBS net learns) vs the full-CFR Nash floor + the base (uniform) policy.

DECISIVE: if the per-belief-equilibrium leaf sigma1 drops well below the single-value wall AND far below
the Leduc residual ratio (~57x Nash), the non-terminal-cut depth limit is much more sound -> bootstrapping
is worth pursuing -> greenlight the single-GPU efficiency/scaling pivot. If it stays ~57x like Leduc, the
single-value-leaf bias is depth-independent -> the mechanism needs multi-valued/gadget regardless (a kill,
learned cheaply). Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_public_key, goofspiel_is_cut


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cards", type=int, default=4)
    ap.add_argument("--cont-iters", type=int, default=300)
    ap.add_argument("--trunk-iters", type=int, default=120)
    ap.add_argument("--subgame-iters", type=int, default=120)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    t0 = time.time()
    g = load_goofspiel(num_cards=args.num_cards, points_order="random")
    dlg = DepthLimitedGame(g, goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    out = {"game": f"goofspiel({args.num_cards})_turnbased_random", "n_iset": dlg.n_iset,
           "cut_public_states": len(set(n[1] for n in dlg.cut_nodes))}
    print(f"Goofspiel({args.num_cards}) kill-switch: n_iset={dlg.n_iset} "
          f"cut_public_states={out['cut_public_states']} (non-terminal cut at prize-2 reveal)")

    cont = dlg.cfr_plus(args.cont_iters)
    out["nash_floor"] = dlg.nash_conv(cont)
    out["base_uniform_exploit"] = dlg.nash_conv(dlg.uniform_policy())

    s_single = dlg.trunk_solve(dlg.blueprint_leaf_fn(cont), iters=args.trunk_iters)
    out["single_value_leaf_exploit"] = dlg.nash_conv(dlg.assemble(s_single, cont))

    s_pbe = dlg.trunk_solve(dlg.per_belief_equilibrium_leaf_fn(iters=args.subgame_iters),
                            iters=args.trunk_iters)
    out["per_belief_equilibrium_leaf_exploit"] = dlg.nash_conv(dlg.assemble(s_pbe, cont))

    out["single_over_nash"] = round(out["single_value_leaf_exploit"] / max(out["nash_floor"], 1e-9), 1)
    out["pbe_over_nash"] = round(out["per_belief_equilibrium_leaf_exploit"] / max(out["nash_floor"], 1e-9), 1)
    out["leduc_single_over_nash_ref"] = 96.0   # Leduc single-value leaf 0.44 / Nash 0.0046
    out["seconds"] = round(time.time() - t0, 1)

    print(f"  Nash floor            : {out['nash_floor']:.4f}")
    print(f"  base (uniform)        : {out['base_uniform_exploit']:.4f}")
    print(f"  single-value leaf σ1  : {out['single_value_leaf_exploit']:.4f}  "
          f"({out['single_over_nash']}x Nash; Leduc 2-level was ~96x)")
    print(f"  per-belief EQ leaf σ1 : {out['per_belief_equilibrium_leaf_exploit']:.4f}  "
          f"({out['pbe_over_nash']}x Nash; live re-solve = Stage-1 re-solve bias)")
    # PASS = the FIXED value-function leaf (what a net learns) is near-Nash on this >=3-level game,
    # i.e. depth-limited soundness holds at >=3 levels (unlike 2-level Leduc).
    passed = (out["single_value_leaf_exploit"] < max(3.0 * out["nash_floor"], 0.05)
              and out["single_value_leaf_exploit"] < 0.1 * out["base_uniform_exploit"])
    out["killswitch_pass"] = bool(passed)
    print(f"  KILL-SWITCH PASS (fixed value-fn leaf near-Nash at >=3 levels): {passed}  "
          f"({out['seconds']}s)")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
