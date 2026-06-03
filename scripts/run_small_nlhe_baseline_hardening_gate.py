#!/usr/bin/env python3
"""Small-NLHE harness fingerprint plus stronger PPO-vs-R-NaD baseline gate.

This is the bounded follow-up to the corrected small-NLHE GO/NO-GO:

1. Fingerprint the exact OpenSpiel small no-limit hold'em harness before scoring.
2. Re-run the R-NaD incumbent under exact NashConv.
3. Compare PPO with a frozen historical opponent pool using both FIFO and K-best retention.

The only hardening knob here is the pool-retention rule. It does not touch Slumbot, full-HUNL, or
cross-environment transfer; exact NashConv on one small OpenSpiel game is the primary metric.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from statistics import fmean

import numpy as np
import torch


def _obs_lookup(game):
    by = {}
    stack, seen = [game.new_initial_state()], set()
    while stack:
        state = stack.pop()
        if state.is_terminal():
            continue
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                stack.append(child)
            continue
        key = state.information_state_string()
        if key not in by:
            by[key] = (
                np.asarray(state.information_state_tensor(), np.float32),
                np.asarray(state.legal_actions_mask(), np.float32),
            )
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            hist = child.history_str()
            if hist not in seen:
                seen.add(hist)
                stack.append(child)
    return by


def _nashconv_from_policy_fn(game, by, action_prob_fn, policy_lib, exploitability) -> float:
    tabular = policy_lib.TabularPolicy(game)
    for key, row_idx in tabular.state_lookup.items():
        if key not in by:
            continue
        obs, legal = by[key]
        pi = np.asarray(action_prob_fn(obs[None], legal[None])[0], dtype=np.float64)
        row = tabular.action_probability_array[row_idx]
        row[:] = 0.0
        row[: len(pi)] = pi
    return float(exploitability.nash_conv(game, tabular))


def _nashconv_from_solver(game, by, solver, policy_lib, exploitability) -> float:
    return _nashconv_from_policy_fn(game, by, solver.action_probabilities, policy_lib, exploitability)


def _stats(values: list[float]) -> dict:
    mean = float(fmean(values)) if values else math.nan
    var = float(fmean([(v - mean) ** 2 for v in values])) if values else math.nan
    return {"mean": mean, "std": math.sqrt(var) if values else math.nan}


def _net_policy_fn(net):
    @torch.no_grad()
    def fn(obs_np, legal_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        legal = torch.as_tensor(legal_np, dtype=torch.float32)
        pi, _, _, _ = net(obs, legal)
        return pi.detach().cpu().numpy()

    return fn


def _rnad_policy_fn(solver, source: str):
    if source == "target":
        return solver.action_probabilities
    if source == "learner":
        return _net_policy_fn(solver.net)
    raise ValueError("rnad policy source must be one of: target, learner")


def _ppo_step_with_opponent(solver, opponent_policy_fn=None) -> dict:
    traj = solver.collector.collect(
        solver._policy_fn,
        solver.config.batch_size,
        solver.config.trajectory_max,
        solver._rng,
        opponent_policy_fn=opponent_policy_fn,
    ).to(solver.device)
    logs = {}
    for _ in range(solver.config.ppo_epochs):
        solver.optimizer.zero_grad(set_to_none=True)
        loss, logs = solver.loss_on_trajectory(traj)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.net.parameters(), solver.config.clip_gradient)
        solver.optimizer.step()
    solver.learner_steps += 1
    logs.update(loss=float(loss.detach().cpu()), learner_steps=solver.learner_steps)
    return logs


def _fingerprint(game, policy_lib, exploitability):
    from poker_ai.rnad import LeducTreeCollector
    from poker_ai.rnad.small_nlhe import SMALL_NLHE_PARAMS

    collector = LeducTreeCollector(game, device="cpu")
    uniform = float(exploitability.nash_conv(game, policy_lib.TabularPolicy(game)))
    kind = collector.kind_arr
    return {
        "game": "small_nlhe_universal_poker",
        "params": dict(SMALL_NLHE_PARAMS),
        "num_players": int(game.num_players()),
        "num_distinct_actions": int(game.num_distinct_actions()),
        "max_game_length": int(game.max_game_length()),
        "information_state_tensor_size": int(game.information_state_tensor_size()),
        "observation_tensor_size": int(game.observation_tensor_size()),
        "tree_nodes": int(len(kind)),
        "terminal_nodes": int((kind == 0).sum()),
        "chance_nodes": int((kind == 1).sum()),
        "decision_nodes": int((kind == 2).sum()),
        "uniform_nashconv": uniform,
    }


def _rnad_schedule(args) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if args.reset_sizes:
        sizes = tuple(int(x) for x in args.reset_sizes.split(","))
        if args.reset_repeats:
            repeats = tuple(int(x) for x in args.reset_repeats.split(","))
        else:
            repeats = tuple(1 for _ in sizes)
    else:
        sizes, repeats = (int(args.reset_every),), (1,)
    return sizes, repeats


def _run_rnad_seed(args, seed: int, game, by, policy_lib, exploitability) -> dict:
    from poker_ai.rnad import LeducTreeCollector, RNaDConfig, RNaDSolver

    collector = LeducTreeCollector(game, device="cpu")
    sched_sizes, sched_repeats = _rnad_schedule(args)
    cfg = RNaDConfig(
        batch_size=args.batch_size,
        trajectory_max=max(8, game.max_game_length() + 1),
        policy_network_layers=tuple(args.layers),
        learning_rate=args.rnad_lr,
        entropy_schedule_size=sched_sizes,
        entropy_schedule_repeats=sched_repeats,
        cix_eta=args.cix_eta,
        seed=seed,
    )
    solver = RNaDSolver(cfg, collector, device="cpu")
    policy_fn = _rnad_policy_fn(solver, args.rnad_policy_source)
    history = [(0, _nashconv_from_policy_fn(game, by, policy_fn, policy_lib, exploitability))]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        solver.step()
        if step % args.eval_every == 0 or step == args.steps:
            policy_fn = _rnad_policy_fn(solver, args.rnad_policy_source)
            history.append((step, _nashconv_from_policy_fn(game, by, policy_fn, policy_lib, exploitability)))
    return {
        "seed": seed,
        "policy_source": str(args.rnad_policy_source),
        "history": history,
        "start": history[0][1],
        "last": history[-1][1],
        "best": min(v for _, v in history),
        "descended": history[-1][1] < history[0][1],
        "seconds": round(time.time() - t0, 3),
    }


def _run_ppo_seed(args, seed: int, pool_kind: str, game, by, policy_lib, exploitability) -> dict:
    from poker_ai.rnad.ppo import PPOConfig, PPOSolver
    from poker_ai.rnad.seat_collector import SeatAwarePyspielCollector

    collector = SeatAwarePyspielCollector(game, device="cpu")
    cfg = PPOConfig(
        batch_size=args.batch_size,
        trajectory_max=max(8, game.max_game_length() + 1),
        policy_network_layers=tuple(args.layers),
        learning_rate=args.ppo_lr,
        seed=seed,
    )
    solver = PPOSolver(cfg, collector, device="cpu")
    history = [(0, _nashconv_from_solver(game, by, solver, policy_lib, exploitability))]
    pool: list[dict] = []
    pool_trace: list[dict] = []
    t0 = time.time()

    def add_snapshot(step: int, score: float) -> None:
        snapshot = copy.deepcopy(solver.net).to("cpu")
        snapshot.eval()
        for param in snapshot.parameters():
            param.requires_grad_(False)
        pool.append({"step": step, "nashconv": score, "net": snapshot})
        if pool_kind == "kbest":
            pool.sort(key=lambda x: x["nashconv"])
            del pool[args.pool_size :]
        else:
            del pool[: max(0, len(pool) - args.pool_size)]
        pool_trace.append(
            {
                "step": step,
                "snapshot_nashconv": score,
                "retained": [{"step": p["step"], "nashconv": p["nashconv"]} for p in pool],
            }
        )

    for step in range(1, args.steps + 1):
        opponent_policy_fn = None
        if pool:
            idx = int(solver._rng.randint(len(pool)))
            opponent_policy_fn = _net_policy_fn(pool[idx]["net"])
        _ppo_step_with_opponent(solver, opponent_policy_fn)
        if step % args.eval_every == 0 or step == args.steps:
            history.append((step, _nashconv_from_solver(game, by, solver, policy_lib, exploitability)))
        if step % args.snapshot_every == 0 or step == args.steps:
            score = _nashconv_from_solver(game, by, solver, policy_lib, exploitability)
            add_snapshot(step, score)

    return {
        "seed": seed,
        "pool_kind": pool_kind,
        "history": history,
        "pool_trace": pool_trace,
        "start": history[0][1],
        "last": history[-1][1],
        "best": min(v for _, v in history),
        "descended": history[-1][1] < history[0][1],
        "seconds": round(time.time() - t0, 3),
    }


def _summarize_arm(name: str, runs: list[dict]) -> dict:
    last = [float(r["last"]) for r in runs]
    best = [float(r["best"]) for r in runs]
    last_stats = _stats(last)
    best_stats = _stats(best)
    return {
        "name": name,
        "seeds": [int(r["seed"]) for r in runs],
        "last_nashconv": last,
        "best_nashconv": best,
        "mean_last_nashconv": last_stats["mean"],
        "std_last_nashconv": last_stats["std"],
        "mean_best_nashconv": best_stats["mean"],
        "std_best_nashconv": best_stats["std"],
        "runs": runs,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--rnad-lr", type=float, default=0.005)
    parser.add_argument("--cix-eta", type=float, default=0.0,
                        help="NeuRD-CIX cap on importance weight 1/(mu+cix_eta); 0 = exact NeuRD.")
    parser.add_argument("--rnad-policy-source", choices=("target", "learner"), default="target")
    parser.add_argument("--ppo-lr", type=float, default=0.005)
    parser.add_argument("--reset-every", type=int, default=1000)
    parser.add_argument("--reset-sizes", default=None,
                        help="Comma-separated increasing R-NaD entropy phase sizes (overrides --reset-every).")
    parser.add_argument("--reset-repeats", default=None,
                        help="Comma-separated repeats parallel to --reset-sizes (last must be 1; default all 1s).")
    parser.add_argument("--snapshot-every", type=int, default=200)
    parser.add_argument("--pool-size", type=int, default=3)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    game = load_small_nlhe()
    by = _obs_lookup(game)
    fp = _fingerprint(game, policy_lib, exploitability)

    print(
        "[fingerprint] "
        f"actions={fp['num_distinct_actions']} max_len={fp['max_game_length']} "
        f"nodes={fp['tree_nodes']} uniform_nashconv={fp['uniform_nashconv']:.4f}",
        flush=True,
    )
    print(
        f"[config] steps={args.steps} seeds={seeds} batch={args.batch_size} "
        f"layers={args.layers} ppo_lr={args.ppo_lr} pool_size={args.pool_size}",
        flush=True,
    )

    t0 = time.time()
    arms = {
        "rnad": _summarize_arm(
            "rnad",
            [_run_rnad_seed(args, seed, game, by, policy_lib, exploitability) for seed in seeds],
        ),
        "ppo_fifo": _summarize_arm(
            "ppo_fifo",
            [_run_ppo_seed(args, seed, "fifo", game, by, policy_lib, exploitability) for seed in seeds],
        ),
        "ppo_kbest": _summarize_arm(
            "ppo_kbest",
            [_run_ppo_seed(args, seed, "kbest", game, by, policy_lib, exploitability) for seed in seeds],
        ),
    }

    lead = min(arms, key=lambda key: arms[key]["mean_last_nashconv"])
    rnad_mean = arms["rnad"]["mean_last_nashconv"]
    best_ppo = min(("ppo_fifo", "ppo_kbest"), key=lambda key: arms[key]["mean_last_nashconv"])
    ppo_mean = arms[best_ppo]["mean_last_nashconv"]
    decision = {
        "primary_metric": "exact_open_spiel_nashconv_lower_is_better",
        "lead": lead,
        "best_ppo_arm": best_ppo,
        "ppo_beats_rnad": bool(ppo_mean < rnad_mean),
        "harness_passed": bool(
            fp["num_players"] == 2
            and fp["num_distinct_actions"] == 4
            and fp["max_game_length"] == 7
            and abs(fp["uniform_nashconv"] - 1.7) < 1e-3
        ),
        "slumbot_full_hunl_blocked": True,
        "interpretation": (
            "Promote PPO-style net-only self-play only if its hardened population baseline beats R-NaD "
            "on exact NashConv; otherwise keep R-NaD as the small-game incumbent and do not advance to Slumbot."
        ),
    }
    out = {
        "gate": "small_nlhe_baseline_hardening",
        "created_at_unix": int(time.time()),
        "seconds": round(time.time() - t0, 3),
        "fingerprint": fp,
        "config": {
            "steps": args.steps,
            "eval_every": args.eval_every,
            "batch_size": args.batch_size,
            "seeds": seeds,
            "layers": args.layers,
            "rnad_lr": args.rnad_lr,
            "rnad_policy_source": args.rnad_policy_source,
            "ppo_lr": args.ppo_lr,
            "reset_every": args.reset_every,
            "snapshot_every": args.snapshot_every,
            "pool_size": args.pool_size,
        },
        "arms": arms,
        "decision": decision,
    }

    for key, arm in arms.items():
        print(
            f"[{key}] last={arm['last_nashconv']} "
            f"mean={arm['mean_last_nashconv']:.4f} std={arm['std_last_nashconv']:.4f}",
            flush=True,
        )
    print(f"[decision] lead={lead} best_ppo={best_ppo} ppo_beats_rnad={decision['ppo_beats_rnad']}", flush=True)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
