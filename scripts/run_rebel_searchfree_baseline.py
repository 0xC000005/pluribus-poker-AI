#!/usr/bin/env python3
"""#43 SEARCH-FREE BOUNDARY: the per-compute boundary the resolving apparatus must justify itself against.
The contribution claims the batched inner re-solve makes train-time iterated continual resolving reach a
target exploitability faster; this baseline asks the prior question -- "does ANY re-solving buy
exploitability-per-second that a search-free direct-policy learner cannot?"

PRIMARY method = NFSP (Heinrich & Silver 2016), the canonical search-free near-Nash function-approx learner.
CRITICAL FAIRNESS POINT (discovered empirically this session): regret/quantal PG (RPG/QPG, Srinivasan 2018)
keep NO averaged policy, so their CURRENT-policy exploitability oscillates near UNIFORM (G4: ~1.43 vs uniform
1.42) -- reporting that as 'the search-free boundary' would be a STRAWMAN that inflates the resolving method's
advantage (benchmark-hacking in our own favor). NFSP maintains an AVERAGE-policy network, so its
exploitability is the meaningful quantity (evaluated in MODE.AVERAGE_POLICY). --method rpg|qpg remain
available ONLY as an explicitly-labelled current-policy contrast, never as the headline boundary.

Scored on the SAME game object (load_goofspiel) by the SAME exact axis (exploitability.nash_conv ==
DepthLimitedGame.nash_conv), SAME hardware. ONE principled OpenSpiel-published config (NOT swept). A plateau
ABOVE the band is a LEGITIMATE result that justifies the resolving apparatus -- NOT a bug to tune away.
Slumbot held-out; uses_slumbot_data=false.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key


# OpenSpiel-published NFSP config (Leduc-class defaults; NOT swept, NOT Goofspiel-tuned).
NFSP_CONFIG = dict(hidden_layers_sizes=[128], reservoir_buffer_capacity=200000, anticipatory_param=0.1,
                   batch_size=128, rl_learning_rate=0.01, sl_learning_rate=0.01,
                   min_buffer_size_to_learn=1000, learn_every=64, optimizer_str="sgd")
PG_CONFIG = dict(hidden_layers_sizes=(128, 128), batch_size=128, entropy_cost=0.01,
                 critic_learning_rate=0.01, pi_learning_rate=0.01, num_critic_before_pi=4, optimizer_str="sgd")
REMOVAL_CRITERION = ("Retire this boundary iff, after a compute budget equal to the resolving arm's total "
                     "wall-to-final-round at G4, NFSP average-policy nash_conv > 5x the band ceiling AND a "
                     "second search-free family confirms it -- a falsification of 'search-free is a "
                     "meaningful boundary', NOT a tuning knob.")


def _build(method, g, iss, na, n_players, seed):
    import torch  # noqa: F401 (seeded in main)
    if method == "nfsp":
        from open_spiel.python.pytorch import nfsp
        agents = [nfsp.NFSP(i, iss, na, **NFSP_CONFIG, seed=seed) for i in range(n_players)]
        mode = nfsp.MODE.AVERAGE_POLICY
        return agents, ("nfsp", mode)
    from open_spiel.python.pytorch.policy_gradient import PolicyGradient
    agents = [PolicyGradient(player_id=i, info_state_size=iss, num_actions=na, loss_str=method, **PG_CONFIG)
              for i in range(n_players)]
    return agents, ("pg", None)


def _make_wrapper(g, agents, n_players, kind, num_actions):
    from open_spiel.python import policy as policy_lib
    from open_spiel.python import rl_environment

    class _W(policy_lib.Policy):
        def __init__(self):
            super().__init__(g, list(range(n_players)))

        def action_probabilities(self, state, player_id=None):
            cur = state.current_player()
            legal = state.legal_actions(cur)
            info_state = state.information_state_tensor(cur)
            if kind[0] == "nfsp":   # FAIR: evaluate the AVERAGE policy (_act wants a boolean legal MASK,
                mask = np.zeros(num_actions, dtype=bool); mask[legal] = True   # returns (action_values, action, probs)
                with agents[cur].temp_mode_as(kind[1]):
                    probs = agents[cur]._act(np.asarray(info_state, dtype=np.float32), mask)[2]
                p = np.asarray([probs[a] for a in legal], dtype=np.float64)
            else:                   # PG: current policy (oscillating; contrast only)
                obs = {"info_state": [None] * n_players, "legal_actions": [None] * n_players, "current_player": cur}
                obs["info_state"][cur] = info_state; obs["legal_actions"][cur] = legal
                ts = rl_environment.TimeStep(observations=obs, rewards=None, discounts=None,
                                             step_type=rl_environment.StepType.MID)
                pr = agents[cur].step(ts, is_evaluation=True).probs
                p = np.asarray([pr[a] for a in legal], dtype=np.float64)
            s = p.sum()
            p = p / s if s > 1e-12 else np.ones(len(legal)) / len(legal)
            return {a: float(pi) for a, pi in zip(legal, p)}

    return _W()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("nfsp", "rpg", "qpg"), default="nfsp",
                    help="nfsp = fair averaged-policy search-free boundary; rpg/qpg = current-policy contrast only")
    ap.add_argument("--eval-grid-seconds", default="2,5,10,30,60,120,180")
    ap.add_argument("--max-seconds", type=float, default=240.0)
    ap.add_argument("--band-ceiling", type=float, default=0.05)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    from open_spiel.python import rl_environment
    from open_spiel.python.algorithms import exploitability

    g = load_goofspiel(args.num_cards)
    dlg = DepthLimitedGame(g, goofspiel_is_cut, public_key_fn=goofspiel_public_key)  # shared game object only
    env = rl_environment.Environment(g)
    n_players = env.num_players
    iss = env.observation_spec()["info_state"][0]
    na = env.action_spec()["num_actions"]
    agents, kind = _build(args.method, g, iss, na, n_players, args.seed)
    wrapper = _make_wrapper(g, agents, n_players, kind, na)
    grid = sorted(float(s) for s in args.eval_grid_seconds.split(","))
    cfg = NFSP_CONFIG if args.method == "nfsp" else {**PG_CONFIG, "loss_str": args.method}
    print(f"search-free boundary [{args.method}] on Goofspiel-{args.num_cards} (n_iset={dlg.n_iset}); "
          f"{'AVERAGE-policy (fair)' if kind[0]=='nfsp' else 'current-policy (contrast)'} exact nash_conv", flush=True)

    curve = []
    elapsed = 0.0; gi = 0; n_eps = 0
    while elapsed < args.max_seconds:
        t0 = time.perf_counter()
        ts = env.reset()
        while not ts.last():
            cur = ts.observations["current_player"]
            out = agents[cur].step(ts)
            ts = env.step([out.action])
        for ag in agents:
            ag.step(ts)
        elapsed += time.perf_counter() - t0
        n_eps += 1
        if gi < len(grid) and elapsed >= grid[gi]:
            nc = float(exploitability.nash_conv(g, wrapper))   # eval EXCLUDED from elapsed (metric, both arms)
            curve.append((round(elapsed, 3), round(nc, 5)))
            print(f"  t={elapsed:7.2f}s  ({n_eps} eps)  nash_conv {nc:.4f}", flush=True)
            gi += 1
            while gi < len(grid) and elapsed >= grid[gi]:
                gi += 1

    in_band = [(t, nc) for t, nc in curve if nc <= args.band_ceiling]
    out = {"game": f"goofspiel{args.num_cards}", "method": f"search_free_{args.method}",
           "policy_evaluated": ("average" if kind[0] == "nfsp" else "current"), "seed": args.seed,
           "n_iset": dlg.n_iset, "config": cfg, "band_ceiling": args.band_ceiling,
           "curve_seconds_vs_nashconv": curve, "time_to_band_s": (in_band[0][0] if in_band else None),
           "sustained_in_band": bool(curve and curve[-1][1] <= args.band_ceiling), "n_episodes": n_eps,
           "removal_criterion": REMOVAL_CRITERION, "uses_slumbot_data": False}
    print(f"\n  {args.method} G{args.num_cards}: best nash_conv {min(nc for _, nc in curve):.4f}; "
          f"time-to-band(<= {args.band_ceiling}) {out['time_to_band_s']}; sustained {out['sustained_in_band']} "
          f"(plateau ABOVE band legitimately justifies resolving)")
    if args.output_json:
        pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
