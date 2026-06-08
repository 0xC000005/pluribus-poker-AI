#!/usr/bin/env python3
"""PORT G4 (scale headline): solve Goofspiel-5 (236k infosets -- past the dense per-iteration CPU walk
ceiling that capped every prior scale lever) via depth-limited self-play using the GPU cross-key batched
SoA solver (poker_ai/rebel/iig_batched). Reports the assembled-strategy exact NashConv + wall-clock, vs
the uniform baseline. The dense per-iteration CFR walk is intractable on G5 single-core; this shows the
batched GPU solver makes the 236k-infoset solve feasible on one consumer GPU. Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key
from poker_ai.rebel.iig_batched import trunk_solve_batched, solve_all_keys_soa, _compile_topology, set_entries


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cards", type=int, default=5)
    ap.add_argument("--trunk-iters", type=int, default=40)
    ap.add_argument("--subgame-iters", type=int, default=150)
    ap.add_argument("--final-subgame-iters", type=int, default=600)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    dlg = DepthLimitedGame(load_goofspiel(args.num_cards), goofspiel_is_cut,
                           public_key_fn=goofspiel_public_key)
    t_build = time.time() - t0
    keys = sorted({n[1] for n in dlg.cut_nodes})
    print(f"G4 scale: Goofspiel-{args.num_cards}  n_iset={dlg.n_iset}  keys={len(keys)}  "
          f"cut_nodes={len(dlg.cut_nodes)}  (build {t_build:.1f}s, device={device})")

    # uniform baseline NashConv
    t0 = time.time()
    nc_uniform = dlg.nash_conv(dlg.uniform_policy())
    t_unif = time.time() - t0
    print(f"  uniform-policy NashConv = {nc_uniform:.4f}  ({t_unif:.1f}s)")

    # depth-limited self-play: trunk CFR+ with the GPU batched per-belief re-solve leaf
    t0 = time.time()
    sig1 = trunk_solve_batched(dlg, subgame_iters=args.subgame_iters, iters=args.trunk_iters, device=device)
    t_trunk = time.time() - t0
    print(f"  trunk_solve_batched ({args.trunk_iters} trunk x {args.subgame_iters} subgame iters): "
          f"{t_trunk:.1f}s  ({t_trunk/args.trunk_iters:.1f}s/trunk-iter; {len(sig1)} above-cut infosets)")

    # final below-cut continuation: solve all subgames at the on-policy (sigma1) beliefs, then assemble
    reaches = dlg.cut_reaches(dlg.assemble(sig1, dlg.uniform_policy()))
    ranges = {k: (reaches[k][0], reaches[k][1]) for k in keys}
    t0 = time.time()
    eq = solve_all_keys_soa(dlg, ranges, args.final_subgame_iters, device=device)
    t_final = time.time() - t0
    cont_pol = dlg.uniform_policy()
    for key in keys:
        for iid, pr in eq[key].items():
            cont_pol[iid] = pr
    full = dlg.assemble(sig1, cont_pol)
    t0 = time.time()
    nc = dlg.nash_conv(full)
    t_nc = time.time() - t0

    out = {"game": f"goofspiel{args.num_cards}", "n_iset": dlg.n_iset, "n_keys": len(keys),
           "n_cut_nodes": len(dlg.cut_nodes), "device": device, "build_s": round(t_build, 1),
           "trunk_iters": args.trunk_iters, "subgame_iters": args.subgame_iters,
           "final_subgame_iters": args.final_subgame_iters,
           "nashconv_assembled": round(nc, 5), "nashconv_uniform": round(nc_uniform, 5),
           "reduction_vs_uniform_x": round(nc_uniform / nc, 1) if nc > 1e-9 else None,
           "t_trunk_s": round(t_trunk, 1), "t_final_solve_s": round(t_final, 1), "t_nashconv_s": round(t_nc, 1)}
    print(f"  ASSEMBLED depth-limited NashConv = {nc:.4f}  (uniform {nc_uniform:.4f}; "
          f"{out['reduction_vs_uniform_x']}x below uniform)  [final solve {t_final:.1f}s, NashConv {t_nc:.1f}s]")
    print(f"  G4: solved {dlg.n_iset} infosets depth-limited on {device} -- the dense per-iteration CPU "
          f"walk was intractable here.")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
