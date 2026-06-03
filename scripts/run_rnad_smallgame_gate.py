#!/usr/bin/env python3
"""B Stage-1 gate: does CANONICAL R-NaD (DeepNash) converge to LOW EXACT exploitability
on small games? Runs the vendored OpenSpiel R-NaD (scripts/vendor/rnad.py — last-maintained
Jan-2025 commit, removed from OpenSpiel 2025-05-27) on Kuhn/Leduc and measures exact NashConv.

PASS = NashConv drives toward ~0 (the mechanism converges -> NLHE port justified).
This is the canonical reference; the PyTorch/cuda NLHE port is validated against it next.
Diagnostic only; no Slumbot; no protected-surface edit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyspiel
from open_spiel.python.algorithms import exploitability
from vendor import rnad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="leduc_poker")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--entropy-schedule-size", type=int, default=20000)
    ap.add_argument("--entropy-schedule-sizes", default=None,
                    help="Comma-separated increasing phase sizes (overrides --entropy-schedule-size).")
    ap.add_argument("--entropy-schedule-repeats", default=None,
                    help="Comma-separated repeats parallel to --entropy-schedule-sizes (last must be 1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    if args.entropy_schedule_sizes:
        sched_sizes = tuple(int(x) for x in args.entropy_schedule_sizes.split(","))
        if args.entropy_schedule_repeats:
            sched_repeats = tuple(int(x) for x in args.entropy_schedule_repeats.split(","))
        else:
            sched_repeats = tuple(1 for _ in sched_sizes)
    else:
        sched_sizes = (int(args.entropy_schedule_size),)
        sched_repeats = (1,)
    cfg = rnad.RNaDConfig(
        game_name=args.game,
        batch_size=int(args.batch_size),
        eta_reward_transform=float(args.eta),
        entropy_schedule_size=sched_sizes,
        entropy_schedule_repeats=sched_repeats,
        seed=int(args.seed),
    )
    solver = rnad.RNaDSolver(cfg)
    game = pyspiel.load_game(args.game)
    hist = []
    t0 = time.time()
    nc0 = exploitability.nash_conv(game, solver)
    hist.append((0, float(nc0)))
    print(f"{args.game}: step 0  NashConv={nc0:.4f}  (uniform-ish init)")
    for it in range(1, int(args.steps) + 1):
        solver.step()
        if it % int(args.eval_every) == 0 or it == int(args.steps):
            nc = exploitability.nash_conv(game, solver)
            hist.append((it, float(nc)))
            print(f"{args.game}: step {it}  NashConv={nc:.4f}  ({(time.time()-t0):.0f}s)")
    best = min(nc for _, nc in hist)
    last = hist[-1][1]
    out = {"game": args.game, "steps": int(args.steps), "eta": args.eta,
           "entropy_schedule_sizes": list(sched_sizes),
           "entropy_schedule_repeats": list(sched_repeats),
           "nashconv_init": float(nc0), "nashconv_last": float(last), "nashconv_best": float(best),
           "history": hist, "seconds": round(time.time() - t0, 1)}
    print(f"\n{args.game}: init {nc0:.4f} -> last {last:.4f} (best {best:.4f}) in {out['seconds']}s")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
