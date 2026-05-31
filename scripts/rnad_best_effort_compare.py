#!/usr/bin/env python3
"""CPU best-effort vs GPU best-effort R-NaD: wall-clock to reach the SAME Nash equilibrium.

NOT a matched-config comparison. Each platform uses whatever it does best, and we report the minimum
wall-clock to reach a target exact NashConv on kuhn/leduc:
  CPU best effort  = vectorized numpy tree collector + torch on CPU using ALL cores (set_num_threads),
                     swept over batch; pick the config with least wall-clock-to-target.
  GPU best effort  = GPU-resident tree collector + torch.compile(reduce-overhead) on the learner,
                     swept over batch; pick the config with least wall-clock-to-target.

KEY EMPIRICAL FACT (measured): NashConv-per-step is ~batch-independent past ~b1024, so convergence is
step-bound. Therefore the best-effort config is the SMALLEST batch that still converges well at the
lowest per-step cost — NOT the largest batch. The sweep finds this automatically.

Honesty guards: exclude NashConv eval from timing; torch.cuda.synchronize bracket; warmup step
(incl. compile) excluded; report the config chosen for each platform; same algo/seed/schedule.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _obs_lookup(game):
    by_key = {}
    stack, seen = [game.new_initial_state()], set()
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                c = s.clone(); c.apply_action(a); stack.append(c)
            continue
        k = s.information_state_string()
        if k not in by_key:
            by_key[k] = (np.asarray(s.information_state_tensor(), np.float32),
                         np.asarray(s.legal_actions_mask(), np.float32))
        for a in s.legal_actions():
            c = s.clone(); c.apply_action(a)
            if c.history_str() not in seen:
                seen.add(c.history_str()); stack.append(c)
    return by_key


def _nashconv(game, solver, obs_by_key, policy_lib, exploitability):
    tp = policy_lib.TabularPolicy(game)
    for key, idx in tp.state_lookup.items():
        if key in obs_by_key:
            o, l = obs_by_key[key]
            pi = solver.action_probabilities(o[None], l[None])[0]
            r = tp.action_probability_array[idx]
            r[:] = 0.0
            r[: len(pi)] = pi
    return float(exploitability.nash_conv(game, tp))


def run_to_target(game_name, *, platform, batch, target, max_steps, reset_every, layers, lr,
                  eval_every, threads, seed=1):
    """Run one config; return (wall_clock_to_target, steps_to_target, final_nc, ms_per_step)."""
    import torch
    import pyspiel
    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad import RNaDConfig, RNaDSolver, LeducTreeCollector, GPUTreeCollector

    if platform == "cpu":
        torch.set_num_threads(threads)
        device = "cpu"
        col = LeducTreeCollector(game_name, device=device)
    else:
        device = "cuda"
        col = GPUTreeCollector(game_name, device=device)

    game = pyspiel.load_game(game_name)
    obs_by_key = _obs_lookup(game)
    cfg = RNaDConfig(batch_size=batch, trajectory_max=10, policy_network_layers=tuple(layers),
                     learning_rate=lr, entropy_schedule_size=(reset_every,),
                     entropy_schedule_repeats=(1,), seed=seed)
    solver = RNaDSolver(cfg, col, device=device)

    if platform == "gpu":
        # GPU best-effort lever: compile the learner loss (reduce-overhead -> CUDA graph + fusion).
        try:
            solver.loss_on_trajectory = torch.compile(solver.loss_on_trajectory, mode="reduce-overhead")
        except Exception as e:  # pragma: no cover
            print(f"  [warn] torch.compile failed, running eager: {repr(e)[:120]}")

    is_cuda = (device == "cuda")
    # warmup (lazy init + compile) — excluded from timing
    solver.step()

    step_seconds = 0.0
    nc_hit = None
    steps_hit = None
    final_nc = None
    for i in range(1, max_steps + 1):
        if is_cuda:
            torch.cuda.synchronize()
        t = time.time()
        solver.step()
        if is_cuda:
            torch.cuda.synchronize()
        step_seconds += time.time() - t
        if i % eval_every == 0 or i == max_steps:
            nc = _nashconv(game, solver, obs_by_key, policy_lib, exploitability)
            final_nc = nc
            if nc <= target and nc_hit is None:
                nc_hit = step_seconds
                steps_hit = i
                break
    ms = 1000.0 * step_seconds / (steps_hit or max_steps)
    return {"reached": nc_hit is not None, "wall_to_target_s": round(nc_hit, 2) if nc_hit else None,
            "steps_to_target": steps_hit, "final_nc": round(final_nc, 4), "ms_per_step": round(ms, 2),
            "batch": batch}


def best_effort(game_name, *, platform, target, batches, **kw):
    print(f"\n--- {platform.upper()} best-effort sweep on {game_name} (target NashConv <= {target}) ---")
    results = []
    for b in batches:
        r = run_to_target(game_name, platform=platform, batch=b, target=target, **kw)
        results.append(r)
        status = (f"{r['wall_to_target_s']}s @ {r['steps_to_target']} steps"
                  if r["reached"] else f"NOT reached (final {r['final_nc']})")
        print(f"  batch={b:6d}: {status}  ({r['ms_per_step']} ms/step)")
    reached = [r for r in results if r["reached"]]
    best = min(reached, key=lambda r: r["wall_to_target_s"]) if reached else None
    return {"sweep": results, "best": best}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="leduc_poker")
    ap.add_argument("--target", type=float, default=2.5, help="target exact NashConv to reach")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--reset-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.00005)
    ap.add_argument("--layers", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--threads", type=int, default=20, help="CPU torch threads (best effort)")
    ap.add_argument("--cpu-batches", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--gpu-batches", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    common = dict(target=args.target, max_steps=args.max_steps, reset_every=args.reset_every,
                  layers=args.layers, lr=args.lr, eval_every=args.eval_every, threads=args.threads)

    cpu = best_effort(args.game, platform="cpu", batches=args.cpu_batches, **common)
    gpu = best_effort(args.game, platform="gpu", batches=args.gpu_batches, **common)

    out = {"game": args.game, "target_nashconv": args.target, "max_steps": args.max_steps,
           "reset_every": args.reset_every, "layers": args.layers, "cpu_threads": args.threads,
           "cpu_best_effort": cpu["best"], "gpu_best_effort": gpu["best"],
           "cpu_sweep": cpu["sweep"], "gpu_sweep": gpu["sweep"]}
    if cpu["best"] and gpu["best"]:
        sp = cpu["best"]["wall_to_target_s"] / gpu["best"]["wall_to_target_s"]
        out["best_effort_speedup_gpu_over_cpu"] = round(sp, 2)
        print(f"\n=== BEST-EFFORT RESULT ({args.game}, NashConv<= {args.target}) ===")
        print(f"  CPU best: batch={cpu['best']['batch']}  {cpu['best']['wall_to_target_s']}s "
              f"({cpu['best']['steps_to_target']} steps, {cpu['best']['ms_per_step']} ms/step)")
        print(f"  GPU best: batch={gpu['best']['batch']}  {gpu['best']['wall_to_target_s']}s "
              f"({gpu['best']['steps_to_target']} steps, {gpu['best']['ms_per_step']} ms/step)")
        print(f"  SPEEDUP (GPU best / CPU best): {sp:.2f}x")
    else:
        print("\n[!] one platform did not reach the target; raise --max-steps or --target")

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
