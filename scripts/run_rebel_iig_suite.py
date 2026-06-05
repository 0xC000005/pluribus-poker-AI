#!/usr/bin/env python3
"""Generic IIG solver suite: ONE game-agnostic CFR+ implementation (poker_ai/rebel/iig.PublicTree)
run unchanged across multiple OpenSpiel games -> exact NashConv each. The "same computation solves any
imperfect-info game" generality proof: Kuhn + Leduc (with a PARITY check vs the trusted LeducTree) +
Liar's Dice (a NON-poker game, small and full 6-sided). This is the foundation for the general,
single-GPU, tabula-rasa ReBeL method (the depth-limited PBS net leaf builds on this tree next).
Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from poker_ai.rebel.iig import PublicTree
from poker_ai.rebel.leduc import LeducTree

# (game_name, params, iters)
GAMES = [
    ("kuhn_poker", None, 1000),
    ("leduc_poker", None, 600),
    ("liars_dice", {"dice_sides": 3}, 400),
    ("liars_dice", None, 300),   # full 6-sided -- the standard ReBeL benchmark game
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--averaging", default="linear")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    out = {"averaging": args.averaging, "games": []}
    print("Generic IIG solver suite (one CFR+ implementation, many games)")
    for name, params, iters in GAMES:
        t0 = time.time()
        tree = PublicTree(name, params)
        avg, _ = tree.cfr_plus(iters, averaging=args.averaging)
        nc = tree.nash_conv(avg)
        dt = time.time() - t0
        rec = {"game": name, "params": params, "iters": iters, "n_iset": tree.n_iset,
               "n_nodes": tree.n_nodes, "nashconv": nc, "seconds": round(dt, 1)}
        if name == "leduc_poker":   # parity vs the trusted hand-crafted substrate
            lt = LeducTree()
            lnc = lt.nash_conv(lt.cfr_plus(iters, averaging=args.averaging)[0])
            rec["leduc_parity_nashconv"] = lnc
            rec["parity_abs_diff"] = abs(nc - lnc)
        out["games"].append(rec)
        label = f"{name}{params or ''}"
        extra = f"  parity|diff|={rec['parity_abs_diff']:.2e}" if "parity_abs_diff" in rec else ""
        print(f"  {label:26s} n_iset={tree.n_iset:6d} iters={iters:5d} -> NashConv {nc:.5f} ({dt:.1f}s){extra}")

    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
