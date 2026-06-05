#!/usr/bin/env python3
"""ReBeL step 3 driver: run the turn+river SELF-PLAY loop, print the 2-street exploitability trend,
and write JSON.

The loop solves the turn subgame with the current river PBS net as the depth-limit leaf, harvests
EXACT-river CFV targets at the on-policy PBSs the solve induces (+ exploration), retrains the net,
and iterates -- the closed ReBeL loop on a real-belief game. Success = the net-leaf agent's 2-street
NashConv trends down toward the exact-river-leaf control (C-gate: 0.041 pot) and below the single-shot
net result (C-gate: 0.076 pot); the loop aborts if exploitability drifts up. Slumbot held-out;
diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from poker_ai.rebel.self_play import run_self_play


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=8)
    ap.add_argument("--trunk-iters", type=int, default=24)
    ap.add_argument("--river-iters", type=int, default=120)
    ap.add_argument("--init-strategies", type=int, default=8)
    ap.add_argument("--explore-strategies", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-batched", action="store_true", help="use the CPU (non-batched) river leaf")
    ap.add_argument("--control", action="store_true",
                    help="also compute the exact-leaf control NashConv (slow, ~1600s)")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    print("ReBeL step 3: turn+river self-play loop (spot AhKd7c2s pot10bb stacks3bb)")
    out, _ = run_self_play(
        n_iters=args.n_iters, trunk_iters=args.trunk_iters, river_iters=args.river_iters,
        init_strategies=args.init_strategies, explore_strategies=args.explore_strategies,
        epochs=args.epochs, seed=args.seed, batched=not args.no_batched,
        compute_control=args.control, verbose=True)

    h = out["history"]
    if h:
        print(f"  trend: iter0 NashConv {h[0]['nashconv']:.1f} ({h[0]['nashconv_pot']:.3f} pot) -> "
              f"best {out['best_nashconv']:.1f} ({out['best_nashconv_pot']:.3f} pot)"
              f"{'  [ABORTED on drift]' if out['aborted'] else ''}")
    if "control_nashconv" in out:
        print(f"  exact-leaf control NashConv {out['control_nashconv']:.1f} "
              f"({out['control_nashconv_pot']:.3f} pot)")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
