#!/usr/bin/env python3
"""Scale lever D disambiguation: is the valid-budget Deep CFR PLATEAU (Leduc 0.37 / Goofspiel-4 0.29,
vs CFR+ ~0.005) caused by the lossy two-net AVERAGE-STRATEGY NET, or by the underlying CFR process
(advantage-net regret quality + iteration count)?

OpenSpiel's DeepCFRSolver reads its average policy from a separate POLICY NET trained to regress the
sampled, iteration-weighted average-strategy targets it stores in `_strategy_memories`. SD-CFR (Steinberger
2019) removes that net's approximation error. We approximate the SD-CFR question cheaply + with the TESTED
solver (no new solver to validate): from the SAME solve, compare
  (a) NET average  = NashConv of solver.action_probabilities (the policy net), vs
  (b) TABULAR avg  = NashConv of the iteration-weighted mean of the strategy_action_probs stored in
                     _strategy_memories (the IDEAL target the net is trying to fit).
If (b) << (a): the policy NET is the lossy component -> Deep CFR convergence is SALVAGEABLE (SD-CFR / a
better/longer-trained net fixes it). If (b) ~= (a) (~0.3): the targets themselves are ~0.3 -> the upstream
CFR process is the limiter, removing the net won't help -> Deep CFR is genuinely inadequate here.
Coverage (sampled infosets / total) is reported -- the tabular reconstruction only covers sampled
infosets (rest stay uniform), valid on Leduc/Goofspiel-4 where coverage is high. Slumbot held-out.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np
import pyspiel
import torch
from open_spiel.python.pytorch import deep_cfr
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability

from poker_ai.rebel.iig_pbs import load_goofspiel

# OpenSpiel DeepCFRSolver CUDA fix (see scripts/run_rebel_deepcfr_gate.py): legacy torch.FloatTensor(np,
# device=cuda) raises; route through a CPU tensor then .to(device).
_orig_float_tensor = torch.FloatTensor


def _float_tensor_compat(*args, **kwargs):
    device = kwargs.pop("device", None)
    t = _orig_float_tensor(*args, **kwargs)
    return t.to(device) if device is not None else t


torch.FloatTensor = _float_tensor_compat
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _tabular_avg_from_memories(game, solver):
    """Iteration-weighted mean of the stored strategy targets, assembled into a TabularPolicy.

    Returns (nash_conv, coverage_fraction). Mirrors the net's training objective (weight ~ iteration):
    for each infoset the minimizer of sum_t iter*(pred - target)^2 is sum(iter*target)/sum(iter).
    """
    buf = solver._strategy_memories  # ReservoirBuffer of StrategyMemory(info_state, iteration, probs)
    n = len(buf)
    feats = np.asarray(buf.experience.info_state[:n], dtype=np.float32)
    iters = np.asarray(buf.experience.iteration[:n], dtype=np.float64).reshape(-1)
    sprobs = np.asarray(buf.experience.strategy_action_probs[:n], dtype=np.float64)

    wsum = collections.defaultdict(lambda: np.zeros(sprobs.shape[1]))
    wtot = collections.defaultdict(float)
    for i in range(n):
        k = feats[i].tobytes()
        w = float(iters[i])
        wsum[k] += w * sprobs[i]
        wtot[k] += w

    tp = policy_lib.TabularPolicy(game)
    covered = total = 0

    def walk(state):
        nonlocal covered, total
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a))
            return
        cur = state.current_player()
        key = state.information_state_string(cur)
        if key in tp.state_lookup:
            total += 1
            k = np.asarray(state.information_state_tensor(cur), dtype=np.float32).tobytes()
            if k in wtot and wtot[k] > 0:
                avg = wsum[k] / wtot[k]
                legal = state.legal_actions()
                row = tp.action_probability_array[tp.state_lookup[key]]
                row[:] = 0.0
                for a in legal:
                    row[a] = max(avg[a], 0.0)
                s = row.sum()
                if s > 1e-12:
                    row /= s
                    covered += 1
                else:
                    for a in legal:  # degenerate target -> uniform over legal
                        row[a] = 1.0 / len(legal)
        for a in state.legal_actions():
            walk(state.child(a))

    walk(game.new_initial_state())
    nc = float(exploitability.nash_conv(game, tp))
    return nc, (covered / total if total else 0.0)


def solve_and_compare(game, iters, traversals, layers, policy_steps, adv_steps,
                      mem=1_000_000, batch=2048, lr=1e-3):
    solver = deep_cfr.DeepCFRSolver(
        game, policy_network_layers=layers, advantage_network_layers=layers,
        num_iterations=iters, num_traversals=traversals, learning_rate=lr,
        batch_size_advantage=batch, batch_size_strategy=batch, memory_capacity=mem,
        policy_network_train_steps=policy_steps, advantage_network_train_steps=adv_steps,
        reinitialize_advantage_networks=True, device=DEVICE)
    solver.solve()
    net_nc = float(exploitability.nash_conv(
        game, policy_lib.tabular_policy_from_callable(game, solver.action_probabilities)))
    tab_nc, coverage = _tabular_avg_from_memories(game, solver)
    return {"net_avg_nashconv": net_nc, "tabular_avg_nashconv": tab_nc, "coverage": coverage}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--traversals", type=int, default=1000)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = {"device": DEVICE, "question": "policy-NET error vs upstream-process limiter"}
    print(f"Deep CFR avg-net disambiguation (device={DEVICE})")

    cfg = dict(layers=(64, 64, 64, 64), policy_steps=5000, adv_steps=500)
    for name, game, ref in [("leduc", pyspiel.load_game("leduc_poker"), 0.0046),
                            ("goofspiel4", load_goofspiel(4), 0.0015)]:
        t0 = time.time()
        r = solve_and_compare(game, args.iters, args.traversals, **cfg)
        r["cfr_plus_ref"] = ref
        r["s"] = round(time.time() - t0, 1)
        r["net_minus_tabular"] = round(r["net_avg_nashconv"] - r["tabular_avg_nashconv"], 4)
        out[name] = r
        print(f"  {name}: NET avg NashConv={r['net_avg_nashconv']:.4f}  vs  TABULAR avg "
              f"(ideal target)={r['tabular_avg_nashconv']:.4f}  (CFR+ {ref}; coverage {r['coverage']:.2f})"
              f"  {r['s']}s")

    # Verdict: if the tabular target is much tighter than the net on BOTH, the net is the lossy part.
    net_better = all(out[g]["tabular_avg_nashconv"] < 0.6 * out[g]["net_avg_nashconv"]
                     for g in ("leduc", "goofspiel4"))
    tab_also_plateaus = all(out[g]["tabular_avg_nashconv"] > 0.15 for g in ("leduc", "goofspiel4"))
    out["net_is_lossy_component"] = bool(net_better)
    out["process_is_limiter"] = bool(tab_also_plateaus and not net_better)
    print(f"  POLICY-NET is the lossy component (tabular target <0.6x net on both -> SD-CFR salvageable): "
          f"{out['net_is_lossy_component']}")
    print(f"  UPSTREAM PROCESS is the limiter (tabular target ALSO plateaus): {out['process_is_limiter']}")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
