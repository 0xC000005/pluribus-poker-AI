#!/usr/bin/env python3
"""EA/EB decisive G5 test: re-state the Goofspiel-5 exploitability with the RIGHT ingredients. EB-step-2
got 0.74 using a UNIFORM-belief continuation; the G4 gadget test showed a CONSISTENT value-fn-leaf sigma1
+ ON-POLICY re-solve continuation is near-Nash on G4 (0.0048), and that the per-belief-re-solve LEAF
biases the trunk (early Stage-1 finding). So compare, on G5 (236k, exact nash_conv), the assembled
exploitability under three below-cut continuations, all with the SAME learned-net-leaf trunk sigma1:
  (1) UNIFORM-belief batched eq (EB-step-2 baseline)
  (2) ON-POLICY per-belief re-solve (batched solver at sigma1's reaches) -- the strong continuation
  (3) GADGET safe re-solve (iig_gadget.safe_continuation) -- robust off-path
Finding => which continuation gives a low G5 number, and whether the gadget is needed at scale. Slumbot
held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key
from poker_ai.rebel.iig_selfplay import (PBSNet, build_public_index, precompute_cont, coverage_rows,
                                         onpolicy_rows, make_net_leaf_fn, _train)
from poker_ai.rebel.iig_batched import solve_all_keys_soa
from poker_ai.rebel.iig_gadget import safe_continuation


def _batched_cont_pol(dlg, keys, iters, device):
    ranges = {k: (np.ones(dlg.n_priv(k, 0)) / dlg.n_priv(k, 0),
                  np.ones(dlg.n_priv(k, 1)) / dlg.n_priv(k, 1)) for k in keys}
    eq = solve_all_keys_soa(dlg, ranges, iters, device=device)
    pol = dlg.uniform_policy()
    for k in keys:
        for iid, pr in eq[k].items():
            pol[iid] = pr
    return pol


def _splice(dlg, sig1, eq):
    pol = dlg.uniform_policy()
    for k, d in eq.items():
        for iid, pr in d.items():
            pol[iid] = pr
    full = dlg.assemble(sig1, pol)
    return full


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cards", type=int, default=5)
    ap.add_argument("--train-iters", type=int, default=5)
    ap.add_argument("--trunk-iters", type=int, default=200)
    ap.add_argument("--coverage-per-cut", type=int, default=200)
    ap.add_argument("--train-epochs", type=int, default=400)
    ap.add_argument("--cont-iters", type=int, default=600)
    ap.add_argument("--gadget-iters", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dlg = DepthLimitedGame(load_goofspiel(args.num_cards), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    pub_index, keys = build_public_index(dlg)
    P = max(max(dlg.n_priv(k, 0), dlg.n_priv(k, 1)) for k in keys)
    print(f"EA/EB G5-resolve test: Goofspiel-{args.num_cards} (n_iset={dlg.n_iset}, device={device})", flush=True)

    cont_pol = _batched_cont_pol(dlg, keys, args.cont_iters, device)   # fixed ref for targets (consistent)
    cont_cache = precompute_cont(dlg, cont_pol)
    rng = np.random.default_rng(0)
    net = PBSNet(len(pub_index), P, args.hidden)
    X, Y, W = coverage_rows(dlg, pub_index, P, cont_cache, keys, args.coverage_per_cut, rng)
    m = _train(net, X, Y, W, args.train_epochs, seed=0)
    for it in range(args.train_iters):
        sig1 = dlg.trunk_solve(make_net_leaf_fn(net, dlg, pub_index, P), iters=args.trunk_iters)
        sig1_full = dlg.uniform_policy()
        for i, pr in sig1.items():
            sig1_full[i] = pr
        xo, yo, wo = onpolicy_rows(dlg, pub_index, P, cont_cache, keys, sig1_full)
        xc, yc, wc = coverage_rows(dlg, pub_index, P, cont_cache, keys, args.coverage_per_cut, rng)
        X = np.concatenate([X, xo, xc]); Y = np.concatenate([Y, yo, yc]); W = np.concatenate([W, wo, wc])
        m = _train(net, X, Y, W, args.train_epochs, seed=0)
        print(f"  train iter {it}: net val MAE {m['mae_frac']:.1%}", flush=True)

    # final net-leaf trunk strategy (consistent value-fn leaf -> good sigma1)
    sig1 = dlg.trunk_solve(make_net_leaf_fn(net, dlg, pub_index, P), iters=args.trunk_iters)
    reaches = dlg.cut_reaches(dlg.assemble(sig1, cont_pol))
    ranges = {k: (reaches[k][0], reaches[k][1]) for k in keys}

    out = {"game": f"goofspiel{args.num_cards}", "n_iset": dlg.n_iset, "net_val_mae": m["mae_frac"]}
    # (1) uniform-belief continuation (EB-step-2 baseline)
    out["uniform_belief"] = round(dlg.nash_conv(dlg.assemble(sig1, cont_pol)), 5)
    print(f"  (1) net-sigma1 + UNIFORM-belief continuation: exploit {out['uniform_belief']:.4f}", flush=True)
    # (2) on-policy per-belief re-solve continuation (the strong continuation)
    eq_on = solve_all_keys_soa(dlg, ranges, max(args.cont_iters, 600), device=device)
    out["on_policy_resolve"] = round(dlg.nash_conv(_splice(dlg, sig1, eq_on)), 5)
    print(f"  (2) net-sigma1 + ON-POLICY re-solve continuation: exploit {out['on_policy_resolve']:.4f}", flush=True)
    # (3) gadget safe continuation
    full_gadget = safe_continuation(dlg, sig1, cont_pol, iters=args.gadget_iters)
    out["gadget_safe"] = round(dlg.nash_conv(full_gadget), 5)
    print(f"  (3) net-sigma1 + GADGET safe continuation: exploit {out['gadget_safe']:.4f}", flush=True)

    out["best"] = min(out["uniform_belief"], out["on_policy_resolve"], out["gadget_safe"])
    print(f"\n  G5 re-stated: uniform {out['uniform_belief']:.4f} | on-policy {out['on_policy_resolve']:.4f} | "
          f"gadget {out['gadget_safe']:.4f}  (EB-step-2 was 0.74)")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
