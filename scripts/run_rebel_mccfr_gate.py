#!/usr/bin/env python3
"""Scale lever B (MCCFR/sampling) -- premise gate. The unabstracted full-tree substrate caps at ~few-k
infosets (Goofspiel(5)=236k is intractable to enumerate). This gate checks the lever-B premise with
OpenSpiel's PRODUCTION external-sampling MCCFR (zero reimplementation risk at this stage):
  (a) FIDELITY: on Goofspiel(4), sampling reaches the same near-Nash as our full-tree cfr_plus.
  (b) SCALING: on Goofspiel(5) (236k infosets -- our enumerator cannot build it), MCCFR (on-the-fly,
      no enumeration) still drives NashConv down -> sampling scales past the enumeration ceiling.
If both hold, sampling is the sound scale lever -> build our own sampled depth-limited PBS solver next.
Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr as esm
from open_spiel.python.algorithms import exploitability

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_public_key, goofspiel_is_cut


def mccfr_nashconv(game, iters):
    solver = esm.ExternalSamplingSolver(game, esm.AverageType.SIMPLE)
    for _ in range(iters):
        solver.iteration()
    return float(exploitability.nash_conv(game, solver.average_policy()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mccfr-iters", type=int, default=3000)
    ap.add_argument("--cfr-iters", type=int, default=600)
    ap.add_argument("--g5-iters", type=int, default=3000)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = {}

    # (a) fidelity on Goofspiel(4): MCCFR vs our full-tree cfr_plus
    t0 = time.time()
    g4 = load_goofspiel(num_cards=4, points_order="random")
    nc_cfr = DepthLimitedGame(g4, goofspiel_is_cut, public_key_fn=goofspiel_public_key).nash_conv(
        DepthLimitedGame(g4, goofspiel_is_cut, public_key_fn=goofspiel_public_key).cfr_plus(args.cfr_iters))
    nc_mccfr = mccfr_nashconv(g4, args.mccfr_iters)
    out["goofspiel4"] = {"cfr_plus_nashconv": nc_cfr, "mccfr_nashconv": nc_mccfr,
                         "seconds": round(time.time() - t0, 1)}
    print(f"(a) FIDELITY goofspiel(4): full-tree cfr_plus NashConv={nc_cfr:.4f}  "
          f"MCCFR NashConv={nc_mccfr:.4f}  ({out['goofspiel4']['seconds']}s)")

    # (b) scaling on Goofspiel(5): 236k infosets, beyond our enumerator; MCCFR is on-the-fly
    t0 = time.time()
    g5 = load_goofspiel(num_cards=5, points_order="random")
    nc5 = mccfr_nashconv(g5, args.g5_iters)
    out["goofspiel5"] = {"mccfr_nashconv": nc5, "n_iset": 236450, "seconds": round(time.time() - t0, 1)}
    print(f"(b) SCALING goofspiel(5) (~236k infosets, enumerator-intractable): "
          f"MCCFR NashConv={nc5:.4f}  ({out['goofspiel5']['seconds']}s)")

    fidelity_ok = abs(nc_mccfr - nc_cfr) < max(0.05, 0.5 * nc_cfr) or nc_mccfr < 0.05
    scaling_ok = nc5 < 0.5 * 1.4167   # well below the base/uniform exploitability
    out["premise_pass"] = bool(fidelity_ok and scaling_ok)
    print(f"  LEVER-B PREMISE PASS (sampling near-Nash on G4 + scales on G5): {out['premise_pass']}")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
