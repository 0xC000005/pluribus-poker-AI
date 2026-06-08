#!/usr/bin/env python3
"""EB (tabula-rasa self-play) -- step 1: certify a LEARNED PBS value-net leaf on a >=3-level game with the
EA exact-exploitability metric. EA established that the value-FUNCTION leaf is the sound low-exploitability
path on >=3-level games (Goofspiel-4: 0.019 vs naive re-solve 0.2); EB's learned net must GENERALIZE that
value function. Here: run the generic depth-limited PBS-net self-play loop (poker_ai/rebel/iig_selfplay) on
Goofspiel-4, and measure the EXACT exploitability (nash_conv of assemble(net-leaf sigma1, continuation))
vs TRAINING ITERATION. ACCEPT (EB-step-1): exploitability DECREASES with training toward the value-fn-leaf
floor -- i.e. the learned leaf is sound, certified by EA's metric. (EB-step-2 = batched-solver target
generation for scale to G5; this step proves correctness/soundness first.) Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key
from poker_ai.rebel.iig_selfplay import (PBSNet, build_public_index, precompute_cont, coverage_rows,
                                         onpolicy_rows, make_net_leaf_fn, _train, _norm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=8)
    ap.add_argument("--trunk-iters", type=int, default=200)
    ap.add_argument("--coverage-per-cut", type=int, default=150)
    ap.add_argument("--train-epochs", type=int, default=400)
    ap.add_argument("--cont-iters", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--buffer-cap", type=int, default=6000)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    dlg = DepthLimitedGame(load_goofspiel(4), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    pub_index, keys = build_public_index(dlg)
    P = max(max(dlg.n_priv(k, 0), dlg.n_priv(k, 1)) for k in keys)

    cont_pol = dlg.cfr_plus(args.cont_iters)
    cont_cache = precompute_cont(dlg, cont_pol)
    nash_floor = dlg.nash_conv(cont_pol)
    # value-fn-leaf floor (exact-leaf control): the sound leaf EA identified, the net's target ceiling
    vf_sig1 = dlg.trunk_solve(dlg.blueprint_leaf_fn(cont_pol), iters=args.trunk_iters)
    vf_exploit = dlg.nash_conv(dlg.assemble(vf_sig1, cont_pol))
    print(f"EB step 1: net-leaf self-play on Goofspiel-4 (>=3-level). Nash floor {nash_floor:.4f}; "
          f"value-fn-leaf floor (target) {vf_exploit:.4f}")

    net = PBSNet(len(pub_index), P, args.hidden)
    rng = np.random.default_rng(0)
    X, Y, W = coverage_rows(dlg, pub_index, P, cont_cache, keys, args.coverage_per_cut, rng)
    m = _train(net, X, Y, W, args.train_epochs, seed=0)

    history = []
    for it in range(args.n_iters):
        net_leaf = make_net_leaf_fn(net, dlg, pub_index, P)
        sig1 = dlg.trunk_solve(net_leaf, iters=args.trunk_iters)
        exploit = dlg.nash_conv(dlg.assemble(sig1, cont_pol))
        history.append({"iter": it, "val_mae_frac": m["mae_frac"], "exploit": round(exploit, 5)})
        print(f"  iter {it}: net val MAE {m['mae_frac']:.1%}  EXACT exploit {exploit:.4f}  "
              f"(target {vf_exploit:.4f}, Nash {nash_floor:.4f})", flush=True)
        # harvest on-policy (belief-consistent) + fresh coverage, retrain
        sig1_full = dlg.uniform_policy()
        for i, pr in sig1.items():
            sig1_full[i] = pr
        xo, yo, wo = onpolicy_rows(dlg, pub_index, P, cont_cache, keys, sig1_full)
        xc, yc, wc = coverage_rows(dlg, pub_index, P, cont_cache, keys, args.coverage_per_cut, rng)
        X = np.concatenate([X, xo, xc]); Y = np.concatenate([Y, yo, yc]); W = np.concatenate([W, wo, wc])
        if X.shape[0] > args.buffer_cap:
            idx = rng.choice(X.shape[0], args.buffer_cap, replace=False)
            X, Y, W = X[idx], Y[idx], W[idx]
        m = _train(net, X, Y, W, args.train_epochs, seed=0)

    exploits = [h["exploit"] for h in history]
    out = {"game": "goofspiel4_>=3level", "nash_floor": round(nash_floor, 5),
           "valuefn_leaf_floor": round(vf_exploit, 5), "history": history,
           "first_exploit": exploits[0], "best_exploit": min(exploits), "last_exploit": exploits[-1],
           "decreased_with_training": bool(min(exploits) < exploits[0] - 1e-3),
           "reached_valuefn_floor_factor": round(min(exploits) / max(vf_exploit, 1e-9), 2)}
    print(f"\n  EB step 1: exploit {exploits[0]:.4f} -> best {min(exploits):.4f} (target {vf_exploit:.4f}); "
          f"decreased_with_training={out['decreased_with_training']}; "
          f"{out['reached_valuefn_floor_factor']}x the value-fn floor")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
