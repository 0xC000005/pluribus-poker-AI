#!/usr/bin/env python3
"""Fit a neural policy to exact small-NLHE MMD policies as a representation probe.

This diagnostic is intentionally small-game only. It asks whether the neural
policy class can represent the exact MMD/dilated-entropy policy on the locked
small-NLHE harness. Passing this probe does not train or promote a HUNL agent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_small_nlhe_baseline_hardening_gate import (  # noqa: E402
    _fingerprint,
    _nashconv_from_policy_fn,
    _obs_lookup,
)


def _parse_layers(values: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(int(v) for v in values)
    if not layers or any(v <= 0 for v in layers):
        raise ValueError("--layers must contain positive hidden sizes")
    return layers


def _mmd_tabular_policy(game, *, alpha: float, mmd_steps: int, stepsize: float | None):
    from open_spiel.python.algorithms import mmd_dilated

    solver = mmd_dilated.MMDDilatedEnt(game, alpha=float(alpha), stepsize=stepsize)
    for _ in range(int(mmd_steps)):
        solver.update_sequences()
    return solver.get_policies(), solver


def _dataset_from_tabular_policy(game, by: dict, tabular_policy) -> dict[str, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    legal_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for key, row_idx in tabular_policy.state_lookup.items():
        if key not in by:
            continue
        obs, legal = by[key]
        target = np.asarray(tabular_policy.action_probability_array[row_idx], dtype=np.float32)
        legal = np.asarray(legal, dtype=np.float32)
        target = np.where(legal > 0.0, target[: game.num_distinct_actions()], 0.0)
        total = float(target.sum())
        if total <= 0.0:
            target = legal / max(float(legal.sum()), 1.0)
        else:
            target = target / total
        obs_rows.append(np.asarray(obs, dtype=np.float32))
        legal_rows.append(legal)
        target_rows.append(target.astype(np.float32, copy=False))
    if not obs_rows:
        raise ValueError("empty policy-fit dataset")
    return {
        "obs": np.stack(obs_rows).astype(np.float32),
        "legal": np.stack(legal_rows).astype(np.float32),
        "target": np.stack(target_rows).astype(np.float32),
    }


def _train_policy_fit(
    *,
    dataset: dict[str, np.ndarray],
    layers: tuple[int, ...],
    steps: int,
    lr: float,
    seed: int,
) -> tuple[torch.nn.Module, dict]:
    from poker_ai.rnad.network import RNaDNetwork

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32)
    legal = torch.as_tensor(dataset["legal"], dtype=torch.float32)
    target = torch.as_tensor(dataset["target"], dtype=torch.float32)
    net = RNaDNetwork(obs.shape[1], legal.shape[1], layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
    losses: list[float] = []
    for _ in range(int(steps)):
        pi, _value, _log_pi, _logits = net(obs, legal)
        loss = -(target * torch.log(pi.clamp_min(1e-9))).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        pi, _value, _log_pi, _logits = net(obs, legal)
        pred = pi.detach().cpu().numpy()
    target_np = dataset["target"]
    kl = target_np * (np.log(np.maximum(target_np, 1e-9)) - np.log(np.maximum(pred, 1e-9)))
    metrics = {
        "final_cross_entropy": losses[-1] if losses else math.nan,
        "start_cross_entropy": losses[0] if losses else math.nan,
        "mean_target_kl_to_fit": float(np.sum(kl, axis=1).mean()),
        "max_abs_policy_error": float(np.max(np.abs(pred - target_np))),
        "mean_l1_policy_error": float(np.mean(np.sum(np.abs(pred - target_np), axis=1))),
    }
    return net, metrics


def _net_policy_fn(net):
    @torch.no_grad()
    def action_probabilities(obs_np, legal_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        legal = torch.as_tensor(legal_np, dtype=torch.float32)
        pi, _value, _log_pi, _logits = net(obs, legal)
        return pi.detach().cpu().numpy()

    return action_probabilities


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmd-steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--stepsize", type=float, default=None)
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--max-fit-nashconv", type=float, default=0.15)
    parser.add_argument("--max-mean-target-kl", type=float, default=0.02)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    if args.mmd_steps < 0:
        raise ValueError("--mmd-steps must be non-negative")
    if args.fit_steps <= 0:
        raise ValueError("--fit-steps must be positive")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive")

    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    t0 = time.time()
    game = load_small_nlhe()
    by = _obs_lookup(game)
    fp = _fingerprint(game, policy_lib, exploitability)
    tabular_policy, solver = _mmd_tabular_policy(
        game,
        alpha=float(args.alpha),
        mmd_steps=int(args.mmd_steps),
        stepsize=args.stepsize,
    )
    target_nashconv = float(exploitability.nash_conv(game, tabular_policy))
    dataset = _dataset_from_tabular_policy(game, by, tabular_policy)
    net, fit_metrics = _train_policy_fit(
        dataset=dataset,
        layers=_parse_layers(args.layers),
        steps=int(args.fit_steps),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    fit_nashconv = _nashconv_from_policy_fn(
        game,
        by,
        _net_policy_fn(net),
        policy_lib,
        exploitability,
    )
    harness_passed = bool(
        fp["num_players"] == 2
        and fp["num_distinct_actions"] == 4
        and fp["max_game_length"] == 7
        and abs(fp["uniform_nashconv"] - 1.7) < 1e-3
    )
    passed = bool(
        harness_passed
        and fit_nashconv <= float(args.max_fit_nashconv)
        and fit_metrics["mean_target_kl_to_fit"] <= float(args.max_mean_target_kl)
    )
    out = {
        "gate": "small_nlhe_mmd_policy_fit_probe",
        "algorithm": "neural_fit_to_exact_mmd_policy",
        "candidate_family": "mmd_nashpg_style_regularized_policy_dynamics",
        "created_at_unix": int(time.time()),
        "seconds": round(time.time() - t0, 3),
        "uses_slumbot_training_data": False,
        "promotion": False,
        "small_game_exact_only": True,
        "fingerprint": fp,
        "config": {
            "mmd_steps": int(args.mmd_steps),
            "alpha": float(args.alpha),
            "stepsize": getattr(solver, "stepsize", args.stepsize),
            "fit_steps": int(args.fit_steps),
            "lr": float(args.lr),
            "layers": list(_parse_layers(args.layers)),
            "seed": int(args.seed),
            "max_fit_nashconv": float(args.max_fit_nashconv),
            "max_mean_target_kl": float(args.max_mean_target_kl),
        },
        "dataset": {
            "n_information_states": int(dataset["obs"].shape[0]),
            "obs_dim": int(dataset["obs"].shape[1]),
            "num_actions": int(dataset["legal"].shape[1]),
        },
        "summary": {
            "target_mmd_nashconv": target_nashconv,
            "fit_policy_nashconv": fit_nashconv,
            **fit_metrics,
        },
        "decision": {
            "primary_metric": "exact_open_spiel_nashconv_and_policy_kl",
            "harness_passed": harness_passed,
            "policy_fit_probe_passed": passed,
            "interpretation": (
                "If this probe passes, the small-game neural class can represent the exact "
                "MMD policy; native failures are more likely update/trajectory fidelity "
                "problems than pure representation capacity."
            ),
            "slumbot_full_hunl_blocked": True,
        },
    }
    print(
        f"[fit] target_nc={target_nashconv:.6f} fit_nc={fit_nashconv:.6f} "
        f"kl={fit_metrics['mean_target_kl_to_fit']:.6g} "
        f"max_err={fit_metrics['max_abs_policy_error']:.6g} passed={passed}",
        flush=True,
    )
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"WROTE {path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
