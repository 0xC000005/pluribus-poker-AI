#!/usr/bin/env python3
"""Scale lever -> abstraction gate: is a CHEAP, game-agnostic infoset clustering a SOUND abstraction
signal? Clusters Goofspiel(4) infosets two ways and measures within-cluster equilibrium-strategy
INCOHERENCE (L1 deviation from the cluster mean; low = mergeable without breaking the solution):
  - FEATURE clustering: by the OpenSpiel information-state TENSOR (cheap, game-agnostic, no solve).
  - VALUE clustering: by the equilibrium action-value vector (the right signal, but needs a solve ->
    circular on big games).
DECISIVE: if FEATURE clustering is incoherent (would break soundness) while VALUE clustering is much
tighter, the cheap game-agnostic abstraction the scale plan assumed does NOT work -> the lever needs a
value signal (circular) or a different scale approach (e.g. MCCFR sampling). Slumbot held-out; diagnostic.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_public_key, goofspiel_is_cut
from poker_ai.rebel.iig_clustering import collect_features, cluster_infosets, strategy_incoherence


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cards", type=int, default=4)
    ap.add_argument("--cfr-iters", type=int, default=800)
    ap.add_argument("--ratios", type=int, nargs="+", default=[5, 20, 50])
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    t0 = time.time()
    g = load_goofspiel(num_cards=args.num_cards, points_order="random")
    dlg = DepthLimitedGame(g, goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    pol = dlg.cfr_plus(args.cfr_iters)
    q = dlg.full_values(pol)
    feats = collect_features(g, dlg)
    qfeat = {i: np.asarray(q[i], np.float32) for i in q}

    out = {"game": f"goofspiel({args.num_cards})", "n_iset": dlg.n_iset, "rows": []}
    print(f"Abstraction gate: goofspiel({args.num_cards}) n_iset={dlg.n_iset} ({time.time()-t0:.0f}s)")
    for ratio in args.ratios:
        cf, nf = cluster_infosets(dlg, feats, ratio)
        cq, nq = cluster_infosets(dlg, qfeat, ratio)
        incf = strategy_incoherence(dlg, pol, cf)
        incq = strategy_incoherence(dlg, pol, cq)
        rec = {"ratio": ratio, "feature_incoherence": incf, "value_incoherence": incq,
               "feature_compression": round(dlg.n_iset / nf, 1), "value_compression": round(dlg.n_iset / nq, 1)}
        out["rows"].append(rec)
        print(f"  ratio~{ratio:2d}: FEATURE incoherence={incf:.3f} ({rec['feature_compression']}x) | "
              f"VALUE incoherence={incq:.3f} ({rec['value_compression']}x)")
    f0, v0 = out["rows"][0]["feature_incoherence"], out["rows"][0]["value_incoherence"]
    out["verdict"] = ("feature_clustering_INSUFFICIENT (high+flat incoherence); value signal better but "
                      "circular-on-big-games + still lossy" if (f0 > 0.15 and f0 > 1.5 * v0)
                      else "feature_clustering_viable")
    print(f"  verdict: {out['verdict']}")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
