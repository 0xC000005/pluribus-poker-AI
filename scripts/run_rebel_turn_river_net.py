#!/usr/bin/env python3
"""ReBeL step C: train a river PBS value net + measure learnability and inference throughput.

Generates exact-river-leaf targets at a single cut public state (check-check river-deal leaf) over
sampled ranges (GPU-batched), trains RiverPBSNet, and reports (1) held-out reach-weighted MAE vs the
sample budget (learnability trend) and (2) INFERENCE THROUGHPUT: net forward pass vs the exact-river
leaf (the efficiency thesis -- the net replaces a ~10s/eval exact solve with a sub-ms forward pass).
Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from poker_ai.rebel.river_pbs_net import sample_river_targets, train_river_net, RiverPBSNet
from poker_ai.rebel.turn_river import parse_card, turn_leaf_river_cfv_batched


def run(n_samples=300, river_iters=200, hidden=512, epochs=600, seed=0):
    import solver as S
    tb = [parse_card(c) for c in ("Ah", "Kd", "7c", "2s")]
    POT, HS, VS = 4000, 3000, 3000          # check-check cut: pot=turn pot, full stacks
    ts = S.StreetSolver(tb, POT, HS, VS, True)
    hands = ts.hands; H = ts.n
    out = {"spot": "AhKd7c2s pot40bb stacks30bb (check-check cut)", "H": H,
           "n_samples": n_samples, "river_iters": river_iters, "hidden": hidden}

    t0 = time.time()
    data = sample_river_targets(tb, POT, HS, VS, True, hands, n_samples=n_samples, seed=seed,
                                river_iters=river_iters, batched=True)
    out["target_gen_seconds"] = round(time.time() - t0, 1)
    out["sec_per_target"] = round(out["target_gen_seconds"] / n_samples, 2)

    # learnability vs budget: train on increasing prefixes (held-out is the last 20%)
    trend = []
    for frac in (0.34, 0.67, 1.0):
        k = max(8, int(n_samples * frac))
        sub = {"X": data["X"][:k], "Y": data["Y"][:k], "W": data["W"][:k], "H": H}
        _, m = train_river_net(sub, hidden=hidden, epochs=epochs, seed=seed)
        trend.append({"n": k, "val_mae": m["val_reach_weighted_mae"],
                      "train_mae": m["train_reach_weighted_mae"], "value_scale": m["value_scale"]})
    out["learnability_trend"] = trend
    net, m_full = train_river_net(data, hidden=hidden, epochs=epochs, seed=seed)
    out["final"] = m_full
    out["val_mae_frac_of_scale"] = round(m_full["val_reach_weighted_mae"] / m_full["value_scale"], 4)

    # inference throughput: net forward vs exact-river leaf
    r0 = np.ones(H) / H; r1 = np.ones(H) / H
    x = torch.tensor(np.concatenate([r0, r1]), dtype=torch.float32).unsqueeze(0)
    net.eval()
    with torch.no_grad():
        for _ in range(3):
            net(x)
        t0 = time.time()
        for _ in range(100):
            net(x)
        net_ms = (time.time() - t0) / 100 * 1000
    t0 = time.time()
    turn_leaf_river_cfv_batched(tb, POT, HS, VS, True, hands, r0, r1, river_iters=river_iters)
    exact_ms = (time.time() - t0) * 1000
    out["net_inference_ms"] = round(net_ms, 3)
    out["exact_leaf_ms"] = round(exact_ms, 1)
    out["inference_speedup"] = round(exact_ms / net_ms, 1)
    return out, net


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--river-iters", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--output-json")
    ap.add_argument("--save-net")
    args = ap.parse_args(argv)
    out, net = run(n_samples=args.n_samples, river_iters=args.river_iters, epochs=args.epochs)
    print("ReBeL step C: river PBS value net")
    print(f"  spot: {out['spot']}  H={out['H']} hands")
    print(f"  target gen: {out['n_samples']} targets in {out['target_gen_seconds']}s "
          f"({out['sec_per_target']}s/target)")
    print("  learnability (reach-weighted MAE vs budget):")
    for r in out["learnability_trend"]:
        print(f"    n={r['n']:4d}: val_mae={r['val_mae']:.2f} train_mae={r['train_mae']:.2f} "
              f"(scale {r['value_scale']:.0f})")
    print(f"  final val MAE = {out['final']['val_reach_weighted_mae']:.2f} "
          f"({out['val_mae_frac_of_scale']:.1%} of value scale)")
    print(f"  INFERENCE: net {out['net_inference_ms']}ms vs exact leaf {out['exact_leaf_ms']}ms "
          f"-> {out['inference_speedup']}x faster (the efficiency thesis)")
    if args.save_net:
        torch.save(net.state_dict(), args.save_net); print(f"  saved net -> {args.save_net}")
    if args.output_json:
        import pathlib
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
