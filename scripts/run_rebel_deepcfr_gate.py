#!/usr/bin/env python3
"""Scale lever D (Deep CFR) -- premise gate via OpenSpiel's TESTED DeepCFRSolver (low reimplementation
risk, like the MCCFR gate). Question: does neural-regret Deep CFR DEFEAT the variance that sank naive
external-sampling MCCFR -- reaching near-Nash on small games where MCCFR was variance-dominated
(Goofspiel-4: MCCFR 0.21 after 4000 it vs cfr_plus oracle 0.0015), and scaling on Goofspiel-5 (236k
infosets, MCCFR 0.93)? If yes, Deep CFR (the field's proven scalable CFR, a non-lossy implicit
abstraction) is the scale lever -> build the generic depth-limited-PBS integration next. Slumbot held-out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import pyspiel
import torch
from open_spiel.python.pytorch import deep_cfr
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_public_key, goofspiel_is_cut

# --- OpenSpiel CUDA compat shim ---------------------------------------------------------------------
# OpenSpiel's reference DeepCFRSolver builds every input tensor with the LEGACY constructor
# torch.FloatTensor(np_array, device=self._device), which raises "legacy constructor expects device type:
# cpu but device type: cuda was passed" -- so upstream it only runs on CPU. The fix is one-line: the
# legacy ctor ignores `device`; route through a CPU tensor then .to(device). We patch torch.FloatTensor to
# a device-aware wrapper (all 7 call sites share this exact pattern; no isinstance checks rely on the
# type). This lives in our repo (reproducible, survives a venv reinstall) instead of hand-editing
# site-packages, and lets the trusted reference solver run on the GPU.
_orig_float_tensor = torch.FloatTensor


def _float_tensor_compat(*args, **kwargs):
    device = kwargs.pop("device", None)
    t = _orig_float_tensor(*args, **kwargs)
    return t.to(device) if device is not None else t


torch.FloatTensor = _float_tensor_compat

# With the shim the reference solver runs on whatever hardware is present. The gate's question (does Deep
# CFR beat the MCCFR variance floor + scale?) is a NashConv result, hardware-independent -- GPU just makes
# the net train/backward steps faster (the pyspiel traversal stays CPU-bound Python either way).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def deepcfr_nashconv(game, iters, traversals, layers=(64, 64), lr=1e-3, mem=1_000_000):
    solver = deep_cfr.DeepCFRSolver(
        game, policy_network_layers=layers, advantage_network_layers=layers,
        num_iterations=iters, num_traversals=traversals, learning_rate=lr,
        batch_size_advantage=2048, batch_size_strategy=2048, memory_capacity=mem, device=DEVICE)
    solver.solve()
    tp = policy_lib.tabular_policy_from_callable(game, solver.action_probabilities)
    return float(exploitability.nash_conv(game, tp))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--traversals", type=int, default=1500)
    ap.add_argument("--g5-iters", type=int, default=150)
    ap.add_argument("--g5-traversals", type=int, default=1500)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = {"device": DEVICE}
    print(f"Deep CFR gate (net training on device={DEVICE})")

    # sanity: Kuhn (Deep CFR is sample-hungry; needs adequate budget)
    t0 = time.time()
    nck = deepcfr_nashconv(pyspiel.load_game("kuhn_poker"), 100, 1000, layers=(32, 32))
    out["kuhn"] = {"deepcfr_nashconv": nck, "s": round(time.time() - t0, 1)}
    print(f"sanity kuhn: DeepCFR NashConv={nck:.4f} (target <0.1; CFR+ ~0)")

    # gate: Goofspiel(4) vs the exact cfr_plus oracle + the naive-MCCFR variance floor
    g4 = load_goofspiel(4)
    nc_oracle = DepthLimitedGame(g4, goofspiel_is_cut, public_key_fn=goofspiel_public_key).nash_conv(
        DepthLimitedGame(g4, goofspiel_is_cut, public_key_fn=goofspiel_public_key).cfr_plus(600))
    t0 = time.time()
    nc4 = deepcfr_nashconv(g4, args.iters, args.traversals)
    out["goofspiel4"] = {"deepcfr_nashconv": nc4, "cfr_plus_oracle": nc_oracle,
                         "naive_mccfr_ref": 0.21, "s": round(time.time() - t0, 1)}
    print(f"GATE goofspiel(4): DeepCFR NashConv={nc4:.4f}  (cfr_plus oracle {nc_oracle:.4f}; "
          f"naive MCCFR 0.21)  {out['goofspiel4']['s']}s")

    # scaling: Goofspiel(5) = 236k infosets (enumerator-intractable; naive MCCFR stuck at 0.93)
    g5 = load_goofspiel(5)
    t0 = time.time()
    nc5 = deepcfr_nashconv(g5, args.g5_iters, args.g5_traversals)
    out["goofspiel5"] = {"deepcfr_nashconv": nc5, "naive_mccfr_ref": 0.93, "base_uniform": 1.4167,
                         "s": round(time.time() - t0, 1)}
    print(f"SCALING goofspiel(5) (236k): DeepCFR NashConv={nc5:.4f}  (naive MCCFR 0.93; uniform 1.42)  "
          f"{out['goofspiel5']['s']}s")

    out["gate_pass"] = bool(nck < 0.15 and nc4 < 0.5 * 0.21 and nc5 < 0.5 * 0.93)
    print(f"  DEEP-CFR GATE PASS (defeats MCCFR variance on G4 + scales on G5): {out['gate_pass']}")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
