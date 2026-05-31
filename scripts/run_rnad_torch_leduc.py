#!/usr/bin/env python3
"""Validate the PyTorch R-NaD port on the simplified card game (Leduc/Kuhn): CORRECTNESS + SPEED.

Two things, the user's stated priorities:
  CORRECTNESS — run the torch R-NaD solver and track EXACT NashConv via OpenSpiel. It must descend
                (the JAX reference Finding 12 reached ~0.6 on Leduc; the port must match that trend).
  SPEED       — benchmark wall-clock per learner step of the torch port vs the canonical JAX reference
                (scripts/vendor/rnad.py) at MATCHED config (same batch_size, trajectory_max, layers,
                schedule). The speedup is reported as steps/sec ratio + per-step ms.

The JAX reference uses a per-state pyspiel python collector; the torch port uses the precomputed-tree
vectorized collector. Both run the SAME algorithm (parity-tested in test/unit/test_rnad_torch.py), so a
fair matched-config wall-clock comparison isolates the harness efficiency.

Usage:
  .venv/bin/python scripts/run_rnad_torch_leduc.py --game leduc_poker --steps 2000 --device cpu \
      --bench-jax-steps 50 --output-json autoresearch-session/rnad_torch_leduc.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _all_decision_states(game):
    out, stack, seen = [], [game.new_initial_state()], set()
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                c = s.clone(); c.apply_action(a); stack.append(c)
            continue
        out.append(s)
        for a in s.legal_actions():
            c = s.clone(); c.apply_action(a)
            if c.history_str() not in seen:
                seen.add(c.history_str()); stack.append(c)
    return out


def _obs_lookup(game):
    by_key = {}
    for st in _all_decision_states(game):
        k = st.information_state_string()
        if k not in by_key:
            by_key[k] = (np.asarray(st.information_state_tensor(), dtype=np.float32),
                         np.asarray(st.legal_actions_mask(), dtype=np.float32))
    return by_key


def _nashconv(game, solver, obs_by_key):
    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    tp = policy_lib.TabularPolicy(game)
    for key, idx in tp.state_lookup.items():
        if key in obs_by_key:
            obs, legal = obs_by_key[key]
            pi = solver.action_probabilities(obs[None, :], legal[None, :])[0]
            row = tp.action_probability_array[idx]
            row[:] = 0.0
            row[: len(pi)] = pi
    return float(exploitability.nash_conv(game, tp))


def run_torch(game_name, steps, device, cfg_kwargs, eval_every, log=True, collector="cpu"):
    import pyspiel
    from poker_ai.rnad import RNaDConfig, RNaDSolver, LeducTreeCollector, GPUTreeCollector
    game = pyspiel.load_game(game_name)
    obs_by_key = _obs_lookup(game)
    if collector == "gpu":
        col = GPUTreeCollector(game_name, device=device)
    else:
        col = LeducTreeCollector(game_name, device=device)
    cfg = RNaDConfig(**cfg_kwargs)
    solver = RNaDSolver(cfg, col, device=device)

    import torch
    is_cuda = str(device).startswith("cuda")

    history = [(0, _nashconv(game, solver, obs_by_key))]
    if log:
        print(f"[torch:{game_name}] step    0  NashConv={history[0][1]:.4f}")
    solver.step()  # warmup (lazy CUDA init / first-touch allocs) — excluded from timing
    # Time ONLY solver.step() (exclude NashConv eval); sync CUDA so async kernels are counted.
    step_seconds = 0.0
    wall0 = time.time()
    for i in range(1, steps + 1):
        if is_cuda:
            torch.cuda.synchronize()
        ts = time.time()
        solver.step()
        if is_cuda:
            torch.cuda.synchronize()
        step_seconds += time.time() - ts
        if i % eval_every == 0 or i == steps:
            nc = _nashconv(game, solver, obs_by_key)
            history.append((i, nc))
            if log:
                print(f"[torch:{game_name}] step {i:4d}  NashConv={nc:.4f}  "
                      f"({step_seconds:.1f}s step, {time.time()-wall0:.1f}s wall)")
    elapsed = step_seconds
    return {
        "history": history,
        "last": history[-1][1],
        "best": min(h[1] for h in history),
        "seconds": round(elapsed, 2),
        "steps": steps,
        "steps_per_sec": round(steps / elapsed, 2),
        "ms_per_step": round(1000 * elapsed / steps, 2),
        "n_infosets": len(obs_by_key),
    }


def bench_jax(game_name, steps, cfg_kwargs):
    """Time the canonical jax reference per-step at MATCHED config (no NashConv eval, pure throughput)."""
    spec = importlib.util.spec_from_file_location("vendor_rnad", str(ROOT / "scripts" / "vendor" / "rnad.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    cfg = ref.RNaDConfig(
        game_name=game_name,
        trajectory_max=cfg_kwargs["trajectory_max"],
        policy_network_layers=tuple(cfg_kwargs["policy_network_layers"]),
        batch_size=cfg_kwargs["batch_size"],
        learning_rate=cfg_kwargs["learning_rate"],
        entropy_schedule_size=tuple(cfg_kwargs["entropy_schedule_size"]),
        entropy_schedule_repeats=tuple(cfg_kwargs["entropy_schedule_repeats"]),
        eta_reward_transform=cfg_kwargs["eta_reward_transform"],
        c_vtrace=cfg_kwargs["c_vtrace"],
        seed=cfg_kwargs["seed"],
    )
    solver = ref.RNaDSolver(cfg)
    solver.step()  # warmup: triggers jit compile (excluded from timing)
    t0 = time.time()
    for _ in range(steps):
        solver.step()
    elapsed = time.time() - t0
    return {
        "seconds": round(elapsed, 2),
        "steps": steps,
        "steps_per_sec": round(steps / elapsed, 2),
        "ms_per_step": round(1000 * elapsed / steps, 2),
        "note": "jit warmup step excluded; per-state pyspiel python collector",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="leduc_poker")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--trajectory-max", type=int, default=10)
    ap.add_argument("--layers", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--lr", type=float, default=0.00005)
    ap.add_argument("--reset-every", type=int, default=1000, help="entropy_schedule_size (reference reset cadence)")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--bench-jax-steps", type=int, default=0,
                    help="if >0, also time the jax reference for this many steps at matched config")
    ap.add_argument("--collector", choices=["cpu", "gpu"], default="cpu",
                    help="cpu = numpy tree collector; gpu = GPU-resident tree collector")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    cfg_kwargs = dict(
        batch_size=args.batch_size, trajectory_max=args.trajectory_max,
        policy_network_layers=tuple(args.layers), learning_rate=args.lr,
        entropy_schedule_size=(args.reset_every,), entropy_schedule_repeats=(1,),
        eta_reward_transform=0.2, c_vtrace=1.0, seed=42,
    )

    out = {"game": args.game, "device": args.device, "config": {**cfg_kwargs,
           "policy_network_layers": list(cfg_kwargs["policy_network_layers"]),
           "entropy_schedule_size": list(cfg_kwargs["entropy_schedule_size"]),
           "entropy_schedule_repeats": list(cfg_kwargs["entropy_schedule_repeats"])}}

    out["collector"] = args.collector
    print(f"=== PyTorch R-NaD on {args.game} ({args.steps} steps, device={args.device}, "
          f"collector={args.collector}) ===")
    torch_res = run_torch(args.game, args.steps, args.device, cfg_kwargs, args.eval_every,
                          collector=args.collector)
    out["torch"] = torch_res
    print(f"  => torch: NashConv {torch_res['history'][0][1]:.4f} -> {torch_res['last']:.4f} "
          f"(best {torch_res['best']:.4f}); {torch_res['ms_per_step']} ms/step, "
          f"{torch_res['steps_per_sec']} steps/s\n")

    if args.bench_jax_steps > 0:
        print(f"=== JAX reference throughput ({args.bench_jax_steps} steps, matched config) ===")
        jax_res = bench_jax(args.game, args.bench_jax_steps, cfg_kwargs)
        out["jax"] = jax_res
        speedup = jax_res["ms_per_step"] / torch_res["ms_per_step"]
        out["speedup_torch_over_jax"] = round(speedup, 2)
        print(f"  => jax: {jax_res['ms_per_step']} ms/step, {jax_res['steps_per_sec']} steps/s")
        print(f"  => SPEEDUP (torch/jax): {speedup:.2f}x faster per learner step\n")

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
