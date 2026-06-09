#!/usr/bin/env python3
"""B0 (cheapest decisive probe, gates A-vs-B): does training the PBS value net on PER-BELIEF RE-SOLVED
targets (V*(belief) -- re-solve the subgame to equilibrium AT each sampled belief via the batched solver)
break the 0.21 single-cut exploitability floor that the FIXED-continuation targets plateau at?

EC found: with a FIXED below-cut continuation as the net target, G5 exploitability plateaus ~0.21 (more net
training did not help). The hypothesis: the floor is the fixed continuation's off-path INCOHERENCE, so
learning the true equilibrium value function V*(belief) (per-belief re-solved targets) should give a more
coherent leaf and a lower exploitability -- OR it still plateaus, which would prove the single-cut leaf is
intrinsically the final-round value (no bootstrapping) and breaking 0.21 REQUIRES nested multi-level cuts
(the multi-week build). Either outcome decisively informs A (write up) vs B (close the loop). Reuses the
existing batched solver + gadget; exact nash_conv metric on small games. Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key
from poker_ai.rebel.iig_selfplay import PBSNet, build_public_index, make_net_leaf_fn, _train, _row
from poker_ai.rebel.iig_batched import _compile_topology, solve_all_keys_soa, set_entries, _leaf_values_from_cont
from poker_ai.rebel.iig_gadget import safe_continuation


def perbelief_targets(dlg, pub_index, P, keys, compiled, n_samples, subgame_iters, device, rng):
    """Training rows whose targets are V*(belief): for each sampled belief, RE-SOLVE all subgames to eq via
    the batched solver and read the normalized equilibrium continuation value (cont under the avg strategy)."""
    X, Y, W = [], [], []
    for _ in range(n_samples):
        ranges = {}
        for k in keys:
            a0 = float(rng.choice([0.3, 1.0, 3.0])); a1 = float(rng.choice([0.3, 1.0, 3.0]))
            ranges[k] = (rng.dirichlet(np.full(dlg.n_priv(k, 0), a0)),
                         rng.dirichlet(np.full(dlg.n_priv(k, 1), a1)))
        _eq, cont = solve_all_keys_soa(dlg, ranges, subgame_iters, device=device, compiled=compiled,
                                       return_cont=True)
        leaf = _leaf_values_from_cont(dlg, compiled, cont, ranges, keys)   # {key: (v0*, v1*)} at this belief
        for k in keys:
            v0, v1 = leaf[k]
            x, y, w = _row(dlg, pub_index, P, k, ranges[k][0], ranges[k][1], v0, v1)
            X.append(x); Y.append(y); W.append(w)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(W, np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cards", type=int, default=4)
    ap.add_argument("--n-target-rounds", type=int, default=8)
    ap.add_argument("--samples-per-round", type=int, default=40)
    ap.add_argument("--subgame-iters", type=int, default=300)
    ap.add_argument("--trunk-iters", type=int, default=200)
    ap.add_argument("--train-epochs", type=int, default=400)
    ap.add_argument("--gadget-iters", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--buffer-cap", type=int, default=8000)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dlg = DepthLimitedGame(load_goofspiel(args.num_cards), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    pub_index, keys = build_public_index(dlg)
    P = max(max(dlg.n_priv(k, 0), dlg.n_priv(k, 1)) for k in keys)
    rng = np.random.default_rng(0)
    init = {k: (np.ones(dlg.n_priv(k, 0)) / dlg.n_priv(k, 0), np.ones(dlg.n_priv(k, 1)) / dlg.n_priv(k, 1)) for k in keys}
    compiled = _compile_topology(dlg, init, device)
    print(f"B0 per-belief V* targets: Goofspiel-{args.num_cards} (n_iset={dlg.n_iset}, device={device})", flush=True)

    net = PBSNet(len(pub_index), P, args.hidden)
    X = Y = W = None
    history = []
    for it in range(args.n_target_rounds):
        xi, yi, wi = perbelief_targets(dlg, pub_index, P, keys, compiled, args.samples_per_round,
                                       args.subgame_iters, device, rng)
        X, Y, W = (xi, yi, wi) if X is None else (np.concatenate([X, xi]), np.concatenate([Y, yi]), np.concatenate([W, wi]))
        if X.shape[0] > args.buffer_cap:
            idx = rng.choice(X.shape[0], args.buffer_cap, replace=False); X, Y, W = X[idx], Y[idx], W[idx]
        m = _train(net, X, Y, W, args.train_epochs, seed=0)
        history.append({"round": it, "val_mae_frac": m["mae_frac"], "n": int(X.shape[0])})
        print(f"  round {it}: V*-net val MAE {m['mae_frac']:.1%}  (targets {X.shape[0]})", flush=True)

    # measure: net-leaf trunk sigma1, then gadget safe continuation; exact nash_conv
    sig1 = dlg.trunk_solve(make_net_leaf_fn(net, dlg, pub_index, P), iters=args.trunk_iters)
    # cont_pol for the gadget's blueprint CFVs: the V*-net's on-policy continuation (batched re-solve at sigma1 belief)
    reaches = dlg.cut_reaches(dlg.assemble(sig1, dlg.uniform_policy()))
    ranges = {k: (reaches[k][0], reaches[k][1]) for k in keys}
    eq = solve_all_keys_soa(dlg, ranges, max(args.subgame_iters, 600), device=device, compiled=compiled)
    cont_pol = dlg.uniform_policy()
    for k in keys:
        for iid, pr in eq[k].items():
            cont_pol[iid] = pr
    nc_onpolicy = dlg.nash_conv(dlg.assemble(sig1, cont_pol))
    full_gadget = safe_continuation(dlg, sig1, cont_pol, iters=args.gadget_iters)
    nc_gadget = dlg.nash_conv(full_gadget)
    nash_floor = dlg.nash_conv(dlg.cfr_plus(600)) if args.num_cards <= 4 else None

    out = {"game": f"goofspiel{args.num_cards}", "n_iset": dlg.n_iset, "target_type": "per_belief_resolved_Vstar",
           "final_val_mae": history[-1]["val_mae_frac"], "nashconv_on_policy": round(nc_onpolicy, 5),
           "nashconv_gadget": round(nc_gadget, 5), "nash_floor": (round(nash_floor, 5) if nash_floor else None),
           "history": history}
    print(f"\n  B0 result (Goofspiel-{args.num_cards}, V*-targets): on-policy {nc_onpolicy:.4f} | gadget "
          f"{nc_gadget:.4f}" + (f" | Nash floor {nash_floor:.4f}" if nash_floor else "")
          + (f"  [vs fixed-continuation-target gadget 0.21 on G5]" if args.num_cards == 5 else ""))
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
