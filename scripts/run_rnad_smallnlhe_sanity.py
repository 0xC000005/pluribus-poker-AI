#!/usr/bin/env python3
"""R-NaD convergence-sanity on the small-NLHE OpenSpiel game (Arm-B validation for the GO/NO-GO).

The corrected GO/NO-GO bundle requires the ACTUAL Arm-B path (RNaDSolver + the small-NLHE collector) to
show monotone exact-NashConv descent on the small-NLHE game BEFORE the A/B is scored — not merely reuse the
kuhn/leduc gate. This runs that: RNaDSolver on the pre-built small-NLHE game object, exact NashConv via
OpenSpiel every eval_every steps, and reports the trajectory + whether it descended below a threshold.

CPU, cheap (637-node game). Writes metrics JSON for the gov cycle.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _obs_lookup(game):
    by = {}
    stack, seen = [game.new_initial_state()], set()
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                c = s.clone(); c.apply_action(a); stack.append(c)
            continue
        k = s.information_state_string()
        if k not in by:
            by[k] = (np.asarray(s.information_state_tensor(), np.float32),
                     np.asarray(s.legal_actions_mask(), np.float32))
        for a in s.legal_actions():
            c = s.clone(); c.apply_action(a)
            if c.history_str() not in seen:
                seen.add(c.history_str()); stack.append(c)
    return by


def _nashconv(game, solver, by, policy_lib, exploitability):
    tp = policy_lib.TabularPolicy(game)
    for key, idx in tp.state_lookup.items():
        if key in by:
            o, l = by[key]
            pi = solver.action_probabilities(o[None], l[None])[0]
            r = tp.action_probability_array[idx]
            r[:] = 0.0
            r[: len(pi)] = pi
    return float(exploitability.nash_conv(game, tp))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--reset-every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--layers", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad import RNaDConfig, RNaDSolver, LeducTreeCollector
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    game = load_small_nlhe()
    by = _obs_lookup(game)
    col = LeducTreeCollector(game, device="cpu")  # pre-built game object
    cfg = RNaDConfig(batch_size=args.batch_size, trajectory_max=max(8, game.max_game_length() + 1),
                     policy_network_layers=tuple(args.layers), learning_rate=args.lr,
                     entropy_schedule_size=(args.reset_every,), entropy_schedule_repeats=(1,), seed=args.seed)
    solver = RNaDSolver(cfg, col, device="cpu")

    hist = [(0, _nashconv(game, solver, by, policy_lib, exploitability))]
    print(f"[smallnlhe] step    0  NashConv={hist[0][1]:.4f}", flush=True)
    t0 = time.time()
    for i in range(1, args.steps + 1):
        solver.step()
        if i % args.eval_every == 0 or i == args.steps:
            nc = _nashconv(game, solver, by, policy_lib, exploitability)
            hist.append((i, nc))
            print(f"[smallnlhe] step {i:4d}  NashConv={nc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    ncs = [h[1] for h in hist]
    out = {
        "game": "small_nlhe_universal_poker",
        "history": hist,
        "start": ncs[0], "last": ncs[-1], "best": min(ncs),
        "descended": ncs[-1] < ncs[0],
        "frac_of_start": round(ncs[-1] / ncs[0], 4) if ncs[0] else None,
        "monotone_nonincreasing_evals": all(b <= a + 1e-6 for a, b in zip(ncs, ncs[1:])),
        "seconds": round(time.time() - t0, 1),
        "config": {"steps": args.steps, "batch": args.batch_size, "reset_every": args.reset_every,
                   "lr": args.lr, "layers": args.layers, "seed": args.seed},
    }
    print(f"  => start {out['start']:.4f} -> last {out['last']:.4f} (best {out['best']:.4f}); "
          f"descended={out['descended']}; {out['seconds']}s", flush=True)
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"WROTE {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
