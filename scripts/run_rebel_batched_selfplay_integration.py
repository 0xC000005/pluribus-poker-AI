#!/usr/bin/env python3
"""GATE 0d (integration): wire the validated GATE-0c batched CFR+ kernel in as the per-belief RE-SOLVE
LEAF inside the depth-limited trunk solve, and confirm the end-to-end equilibrium is PRESERVED on a
>=3-level game (Goofspiel-4) -- i.e. swapping the serial numpy subgame solver for the batched torch one
changes neither the trunk strategy nor the assembled NashConv.

The serial path (dlg.per_belief_equilibrium_leaf_fn) re-solves each public-state subgame with numpy
solve_subgame_equilibrium at every trunk iteration. The batched path uses solve_key_batched (the 0c
kernel, float64) for the same re-solve. GATE 0c proved per-key parity; this proves it composes correctly
through the full trunk_solve -> assemble -> NashConv pipeline (the self-play inner loop). Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pyspiel

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# NB: importing this sets torch default dtype to float64 (required for CFR+ parity -- see GATE 0c).
from run_rebel_batched_generic_subgame import solve_key_batched  # noqa: E402
from poker_ai.rebel.iig_solve import DepthLimitedGame  # noqa: E402
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key  # noqa: E402


def make_batched_per_belief_leaf_fn(dlg, iters, device):
    """Drop-in replacement for dlg.per_belief_equilibrium_leaf_fn, using the batched torch kernel for the
    subgame re-solve (everything else -- the normalized PBS leaf-value computation -- is identical)."""
    by_key = {}
    for n in dlg.cut_nodes:
        by_key.setdefault(n[1], []).append(n)

    def fn(key, range0, range1):
        eq = solve_key_batched(dlg, key, range0, range1, iters, device)
        pol = dlg.uniform_policy()
        for i, pr in eq.items():
            pol[i] = pr
        n0, n1 = dlg.n_priv(key, 0), dlg.n_priv(key, 1)
        v0n = np.zeros(n0); v0d = np.zeros(n0); v1n = np.zeros(n1); v1d = np.zeros(n1)
        for _t, _k, i0, i1, sub in by_key[key]:
            cont = dlg.subtree_ev(sub, pol)
            v0n[i0] += range1[i1] * cont[0]; v0d[i0] += range1[i1]
            v1n[i1] += range0[i0] * cont[1]; v1d[i1] += range0[i0]
        v0 = np.divide(v0n, v0d, out=np.zeros(n0), where=v0d > 1e-15)
        v1 = np.divide(v1n, v1d, out=np.zeros(n1), where=v1d > 1e-15)
        return v0, v1

    return fn


def _sig_l1(a, b):
    return max(float(np.abs(np.asarray(a[i]) - np.asarray(b[i])).sum()) for i in (set(a) & set(b)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk-iters", type=int, default=50)
    ap.add_argument("--subgame-iters", type=int, default=150)
    ap.add_argument("--cont-iters", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    dlg = DepthLimitedGame(load_goofspiel(4), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    print(f"GATE 0d: batched re-solve leaf inside trunk_solve on Goofspiel-4 (>=3-level); device={device}")
    print(f"  trunk_iters={args.trunk_iters}, subgame_iters={args.subgame_iters}")

    serial_leaf = dlg.per_belief_equilibrium_leaf_fn(iters=args.subgame_iters)
    batched_leaf = make_batched_per_belief_leaf_fn(dlg, args.subgame_iters, device)

    t0 = time.time()
    sig_serial = dlg.trunk_solve(serial_leaf, iters=args.trunk_iters)
    t_serial = time.time() - t0
    t0 = time.time()
    sig_batched = dlg.trunk_solve(batched_leaf, iters=args.trunk_iters)
    t_batched = time.time() - t0

    sig_l1 = _sig_l1(sig_serial, sig_batched)

    # end-to-end: assemble each trunk strategy onto a fixed near-eq continuation, compare NashConv
    cont_pol = dlg.cfr_plus(args.cont_iters)
    nc_serial = dlg.nash_conv(dlg.assemble(sig_serial, cont_pol))
    nc_batched = dlg.nash_conv(dlg.assemble(sig_batched, cont_pol))

    out = {"game": "goofspiel4_>=3level", "device": device, "trunk_iters": args.trunk_iters,
           "subgame_iters": args.subgame_iters, "trunk_sigma1_l1_serial_vs_batched": round(sig_l1, 9),
           "nashconv_serial": round(nc_serial, 6), "nashconv_batched": round(nc_batched, 6),
           "nashconv_abs_diff": round(abs(nc_serial - nc_batched), 9),
           "t_serial_s": round(t_serial, 1), "t_batched_s": round(t_batched, 1)}
    out["integration_pass"] = bool(sig_l1 < 1e-4 and abs(nc_serial - nc_batched) < 1e-4)
    print(f"  trunk sigma1 L1 (serial vs batched leaf): {sig_l1:.2e}")
    print(f"  assembled NashConv: serial {nc_serial:.5f}  batched {nc_batched:.5f}  "
          f"(|diff| {abs(nc_serial - nc_batched):.2e})")
    print(f"  wall-clock: serial {t_serial:.1f}s, batched {t_batched:.1f}s")
    print(f"  INTEGRATION PASS (equilibrium preserved through trunk_solve->assemble->NashConv): "
          f"{out['integration_pass']}")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
