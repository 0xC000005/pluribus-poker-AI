#!/usr/bin/env python3
"""ReBeL real-game step 0: turn+river harness + the exact-oracle feasibility finding.

Builds the default turn subgame, reports its structure (hands / nodes / river-cut frontier), and
quantifies the cost of an EXACT river leaf (river betting subgame solved per runout) used as the
depth-limit `cut_node_fn`. This establishes the harness and the efficiency motivation: the exact
oracle is infeasible per-iteration in the trunk, so a learned PBS value net is load-bearing (steps
1-3 replace the exact leaf with multi-valued states then a net).
"""
from __future__ import annotations

import argparse
import json
import time

from poker_ai.rebel.turn_river import (
    default_spot, build_turn_solver, showdown_cut_indices, cut_node_pots, card_str,
)


def run(river_iters=100, backend="cpu"):
    import solver as S
    spot = default_spot()
    out = {"board": spot.board_str, "pot_bb": spot.pot // 100,
           "stack_bb": spot.hero_stack // 100, "river_iters": river_iters, "backend": backend}

    t0 = time.time()
    ts = build_turn_solver(spot)
    out["turn_build_s"] = round(time.time() - t0, 2)
    out["turn_hands"] = ts.n
    out["turn_nodes"] = int(ts._tree["n_nodes"])
    cuts = showdown_cut_indices(ts)
    out["cut_nodes"] = len(cuts)
    pots = cut_node_pots(ts, cuts)
    out["distinct_cut_stakes"] = len(set((p["pot"], p["stacks"]) for p in pots.values()))

    # time one river solve at a representative cut node
    ci = cuts[0]
    pot, (hs, vs) = pots[ci]["pot"], pots[ci]["stacks"]
    river_card = [c for c in range(52) if c not in spot.board][0]
    board5 = spot.board + [river_card]
    t0 = time.time(); rs = S.StreetSolver(board5, pot, hs, vs, hero_first=True)
    build_s = time.time() - t0
    t0 = time.time(); rs.solve(n_iterations=river_iters, backend=backend)
    solve_s = time.time() - t0
    out["river_hands"] = rs.n
    out["river_one_solve_s"] = round(build_s + solve_s, 3)
    out["exact_leaf_eval_s"] = round(44 * (build_s + solve_s), 1)          # one cut node, 44 runouts
    out["per_iter_all_cuts_s"] = round(out["exact_leaf_eval_s"] * len(cuts), 1)
    out["turn_solve_100it_hours"] = round(out["per_iter_all_cuts_s"] * 100 / 3600, 1)
    out["verdict"] = ("exact river oracle is INFEASIBLE per-iteration in the trunk "
                      "-> learned PBS value net is load-bearing, not optional")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--river-iters", type=int, default=100)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = run(river_iters=args.river_iters, backend=args.backend)
    print("ReBeL real-game step 0: turn+river harness")
    print(f"  spot: board {out['board']}  pot {out['pot_bb']}bb  stacks {out['stack_bb']}bb")
    print(f"  turn subgame: {out['turn_hands']} hands, {out['turn_nodes']} nodes, "
          f"{out['cut_nodes']} river-cut nodes ({out['distinct_cut_stakes']} distinct stakes)")
    print(f"  one river solve: {out['river_hands']} hands, {out['river_one_solve_s']}s "
          f"({args.river_iters} it, {args.backend})")
    print(f"  EXACT river leaf eval (44 runouts): {out['exact_leaf_eval_s']}s/cut")
    print(f"  per CFR iteration over all cuts: {out['per_iter_all_cuts_s']}s")
    print(f"  => full 100-iter turn solve with exact oracle: {out['turn_solve_100it_hours']} HOURS")
    print(f"  VERDICT: {out['verdict']}")
    if args.output_json:
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
