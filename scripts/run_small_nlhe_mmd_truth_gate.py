#!/usr/bin/env python3
"""Exact small-NLHE truth gate for OpenSpiel MMD/dilated-entropy policy dynamics.

This gate is intentionally narrow. It checks whether the reviewed
MMD/NashPG-family update moves a locked, tiny imperfect-information poker game
toward lower exact NashConv before any native HUNL scaling attempt. It is not a
Slumbot trainer, not a checkpoint selector, and not promotion evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from run_small_nlhe_baseline_hardening_gate import _fingerprint


def _eval_mmd(game, solver, exploitability) -> dict:
    current_nc = float(exploitability.nash_conv(game, solver.get_policies()))
    average_nc = float(exploitability.nash_conv(game, solver.get_avg_policies()))
    gap = float(solver.get_gap())
    return {
        "current_nashconv": current_nc,
        "average_nashconv": average_nc,
        "regularized_gap": gap,
    }


def _load_baseline(path: str | None, arm: str) -> dict | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    arms = payload.get("arms", {})
    if arm not in arms:
        known = ", ".join(sorted(arms))
        raise ValueError(f"baseline arm '{arm}' not present in {path}; known arms: {known}")
    selected = arms[arm]
    return {
        "path": path,
        "arm": arm,
        "mean_last_nashconv": float(selected["mean_last_nashconv"]),
        "mean_best_nashconv": float(selected["mean_best_nashconv"]),
        "source_gate": payload.get("gate"),
    }


def _finite(value: float) -> bool:
    return bool(math.isfinite(float(value)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--stepsize", type=float, default=None)
    parser.add_argument("--baseline-json")
    parser.add_argument("--baseline-arm", default="rnad")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive so the regularized gap is defined")

    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability, mmd_dilated
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    game = load_small_nlhe()
    fp = _fingerprint(game, policy_lib, exploitability)
    baseline = _load_baseline(args.baseline_json, args.baseline_arm)

    t0 = time.time()
    solver = mmd_dilated.MMDDilatedEnt(game, alpha=args.alpha, stepsize=args.stepsize)
    history = [{"step": 0, **_eval_mmd(game, solver, exploitability)}]
    for step in range(1, args.steps + 1):
        solver.update_sequences()
        if step % args.eval_every == 0 or step == args.steps:
            history.append({"step": step, **_eval_mmd(game, solver, exploitability)})

    current_values = [float(row["current_nashconv"]) for row in history]
    average_values = [float(row["average_nashconv"]) for row in history]
    gap_values = [float(row["regularized_gap"]) for row in history]
    best_current = min(current_values)
    best_average = min(average_values)
    last_current = current_values[-1]
    last_average = average_values[-1]
    harness_passed = bool(
        fp["num_players"] == 2
        and fp["num_distinct_actions"] == 4
        and fp["max_game_length"] == 7
        and abs(fp["uniform_nashconv"] - 1.7) < 1e-3
    )
    all_finite = all(_finite(v) for v in current_values + average_values + gap_values)
    improved_from_uniform = bool(best_current < fp["uniform_nashconv"] and best_average < fp["uniform_nashconv"])
    compared_to_baseline = None
    if baseline is not None:
        compared_to_baseline = {
            "baseline_arm": baseline["arm"],
            "baseline_mean_last_nashconv": baseline["mean_last_nashconv"],
            "mmd_best_current_beats_baseline_last": bool(best_current < baseline["mean_last_nashconv"]),
            "mmd_best_average_beats_baseline_last": bool(best_average < baseline["mean_last_nashconv"]),
        }

    decision = {
        "primary_metric": "exact_open_spiel_nashconv_lower_is_better",
        "harness_passed": harness_passed,
        "all_metrics_finite": all_finite,
        "small_game_exact_only": True,
        "improved_from_uniform": improved_from_uniform,
        "candidate_truth_gate_passed": bool(harness_passed and all_finite and improved_from_uniform),
        "slumbot_full_hunl_blocked": True,
        "compared_to_baseline": compared_to_baseline,
        "interpretation": (
            "MMD/dilated entropy is an exact small-game reference for the regularized "
            "policy-gradient family. Passing this gate authorizes further mechanism work; "
            "it does not promote any native HUNL checkpoint."
        ),
    }
    out = {
        "gate": "small_nlhe_mmd_truth_gate",
        "algorithm": "open_spiel_mmd_dilated",
        "candidate_family": "mmd_nashpg_style_regularized_policy_dynamics",
        "created_at_unix": int(time.time()),
        "seconds": round(time.time() - t0, 3),
        "uses_slumbot_training_data": False,
        "promotion": False,
        "fingerprint": fp,
        "config": {
            "steps": args.steps,
            "eval_every": args.eval_every,
            "alpha": args.alpha,
            "stepsize": solver.stepsize,
            "baseline_json": args.baseline_json,
            "baseline_arm": args.baseline_arm,
        },
        "baseline": baseline,
        "history": history,
        "summary": {
            "start_current_nashconv": current_values[0],
            "last_current_nashconv": last_current,
            "best_current_nashconv": best_current,
            "start_average_nashconv": average_values[0],
            "last_average_nashconv": last_average,
            "best_average_nashconv": best_average,
            "last_regularized_gap": gap_values[-1],
            "best_regularized_gap": min(gap_values),
        },
        "decision": decision,
    }

    print(
        "[fingerprint] "
        f"actions={fp['num_distinct_actions']} max_len={fp['max_game_length']} "
        f"nodes={fp['tree_nodes']} uniform_nashconv={fp['uniform_nashconv']:.4f}",
        flush=True,
    )
    print(
        f"[mmd] alpha={args.alpha} steps={args.steps} stepsize={solver.stepsize:.6g} "
        f"best_current={best_current:.6f} last_current={last_current:.6f} "
        f"last_avg={last_average:.6f} gap={gap_values[-1]:.6g}",
        flush=True,
    )
    print(
        f"[decision] passed={decision['candidate_truth_gate_passed']} "
        f"small_game_exact_only={decision['small_game_exact_only']}",
        flush=True,
    )

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
