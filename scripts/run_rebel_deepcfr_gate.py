#!/usr/bin/env python3
"""Scale lever D (Deep CFR) -- premise gate via OpenSpiel's TESTED DeepCFRSolver (low reimplementation
risk, like the MCCFR gate). Question: with a perfect simulator, external-sampling Deep CFR (the field's
proven scalable CFR, a non-lossy implicit abstraction) is LOW-VARIANCE and sound -- does it reach
near-Nash where naive external-sampling MCCFR was variance-dominated (Goofspiel-4: MCCFR 0.21 vs cfr_plus
oracle 0.0015), and scale on Goofspiel-5 (236k infosets, MCCFR 0.93)? If yes -> build the generic
depth-limited-PBS integration next. Slumbot held-out.

v2 (valid-budget re-run, 2026-06-07): the v1 run FAILED but was SAMPLE-STARVED -- it left
policy_network_train_steps / advantage_network_train_steps at the OpenSpiel DEFAULT of 1 (one gradient
step per iter), so the nets were essentially untrained (Kuhn 0.611 where Deep CFR trivially -> ~0). This
re-run sets the train-step knobs and other config to OpenSpiel's OWN canonical example values
(open_spiel/python/examples/deep_cfr_{pytorch,tf2}.py: policy_train_steps 5000, adv_train_steps 500-750,
batch 2048, mem 1e6, lr 1e-3, deeper nets) -- this is RESOURCING THE TEST TO VALIDITY (Kuhn->~0,
Leduc->near-Nash are the precondition), NOT a benchmark-hacking sweep: one principled config from the
reference impl, run once. Leduc is added as a stronger poker-class validity anchor.
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


def deepcfr_nashconv(game, iters, traversals, layers=(64, 64, 64, 64), lr=1e-3,
                     mem=1_000_000, policy_steps=5000, adv_steps=500, batch=2048):
    """Run OpenSpiel's tested DeepCFRSolver and return the average-policy NashConv.

    The train-step knobs (policy_steps / adv_steps) are the v1 bug fix: their OpenSpiel default is 1, which
    leaves the nets untrained. Values here track the reference examples (deep_cfr_{pytorch,tf2}.py).
    """
    solver = deep_cfr.DeepCFRSolver(
        game, policy_network_layers=layers, advantage_network_layers=layers,
        num_iterations=iters, num_traversals=traversals, learning_rate=lr,
        batch_size_advantage=batch, batch_size_strategy=batch, memory_capacity=mem,
        policy_network_train_steps=policy_steps, advantage_network_train_steps=adv_steps,
        reinitialize_advantage_networks=True, device=DEVICE)
    solver.solve()
    tp = policy_lib.tabular_policy_from_callable(game, solver.action_probabilities)
    return float(exploitability.nash_conv(game, tp))


def main(argv=None):
    ap = argparse.ArgumentParser()
    # Defaults track OpenSpiel's canonical example configs; G5 is bounded for wall-clock on the consumer GPU.
    ap.add_argument("--g4-iters", type=int, default=100)
    ap.add_argument("--g4-traversals", type=int, default=1000)
    ap.add_argument("--g5-iters", type=int, default=60)
    ap.add_argument("--g5-traversals", type=int, default=800)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = {"device": DEVICE,
           "config_provenance": "OpenSpiel examples/deep_cfr_{pytorch,tf2}.py: policy_steps=5000, "
                                "adv_steps=500-750, batch=2048, mem=1e6, lr=1e-3; train-step knobs are the "
                                "v1 fix (default was 1 -> untrained nets)."}
    print(f"Deep CFR gate v2 (valid budget; net training on device={DEVICE})")

    # VALIDITY anchor 1 -- Kuhn (12 infosets; canonical pytorch example config). Must -> ~0.
    t0 = time.time()
    nck = deepcfr_nashconv(pyspiel.load_game("kuhn_poker"), 100, 1000,
                           layers=(64,), policy_steps=5000, adv_steps=750)
    out["kuhn"] = {"deepcfr_nashconv": nck, "target": 0.1, "s": round(time.time() - t0, 1)}
    print(f"VALIDITY kuhn: DeepCFR NashConv={nck:.4f} (target <0.1; CFR+ ~0)  {out['kuhn']['s']}s")

    # VALIDITY anchor 2 -- Leduc (poker: chance + 2 betting rounds; canonical tf2 example config).
    t0 = time.time()
    ncl = deepcfr_nashconv(pyspiel.load_game("leduc_poker"), 100, 1000,
                           layers=(64, 64, 64, 64), policy_steps=5000, adv_steps=500)
    out["leduc"] = {"deepcfr_nashconv": ncl, "target": 0.5, "cfr_plus_ref": 0.0046,
                    "s": round(time.time() - t0, 1)}
    print(f"VALIDITY leduc: DeepCFR NashConv={ncl:.4f} (target <0.5; CFR+ 0.0046)  {out['leduc']['s']}s")

    # PREMISE -- Goofspiel(4) vs the exact cfr_plus oracle + the naive-MCCFR variance floor (0.21)
    g4 = load_goofspiel(4)
    nc_oracle = DepthLimitedGame(g4, goofspiel_is_cut, public_key_fn=goofspiel_public_key).nash_conv(
        DepthLimitedGame(g4, goofspiel_is_cut, public_key_fn=goofspiel_public_key).cfr_plus(600))
    t0 = time.time()
    nc4 = deepcfr_nashconv(g4, args.g4_iters, args.g4_traversals,
                           layers=(64, 64, 64, 64), policy_steps=5000, adv_steps=500)
    out["goofspiel4"] = {"deepcfr_nashconv": nc4, "cfr_plus_oracle": nc_oracle,
                         "naive_mccfr_ref": 0.21, "target": 0.105, "s": round(time.time() - t0, 1)}
    print(f"PREMISE goofspiel(4): DeepCFR NashConv={nc4:.4f}  (cfr_plus oracle {nc_oracle:.4f}; "
          f"naive MCCFR 0.21; target <0.105)  {out['goofspiel4']['s']}s")

    # SCALING -- Goofspiel(5) = 236k infosets (enumerator-intractable; naive MCCFR stuck at 0.93)
    g5 = load_goofspiel(5)
    t0 = time.time()
    nc5 = deepcfr_nashconv(g5, args.g5_iters, args.g5_traversals,
                           layers=(64, 64, 64, 64), policy_steps=5000, adv_steps=500)
    out["goofspiel5"] = {"deepcfr_nashconv": nc5, "naive_mccfr_ref": 0.93, "base_uniform": 1.4167,
                         "target": 0.465, "s": round(time.time() - t0, 1)}
    print(f"SCALING goofspiel(5) (236k): DeepCFR NashConv={nc5:.4f}  (naive MCCFR 0.93; uniform 1.42; "
          f"target <0.465)  {out['goofspiel5']['s']}s")

    validity_ok = bool(nck < 0.1 and ncl < 0.5)
    premise_ok = bool(nc4 < 0.105 and nc5 < 0.465)
    out["validity_ok"] = validity_ok
    out["premise_ok"] = premise_ok
    out["gate_pass"] = bool(validity_ok and premise_ok)
    print(f"  VALIDITY (kuhn->~0 + leduc near-Nash, test is well-resourced): {validity_ok}")
    print(f"  PREMISE  (G4 beats MCCFR floor 0.21 + G5 beats MCCFR 0.93, both 2x): {premise_ok}")
    print(f"  DEEP-CFR GATE PASS: {out['gate_pass']}")
    if not validity_ok:
        print("  WARNING: validity anchors did not converge -> the premise numbers are NOT a valid test "
              "(still under-resourced or a solver issue), do not read G4/G5 as a method verdict.")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
