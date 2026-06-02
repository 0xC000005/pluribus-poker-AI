#!/usr/bin/env python3
"""Compare neural rollout updates with exact MMD local policy movement.

This is a small-NLHE diagnostic only. It starts from a neural policy fitted to
an exact MMD policy, applies the existing neural reference-regularized rollout
update, and checks whether that update moves toward the exact next MMD policy.
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

from scripts.analyze_small_nlhe_mmd_policy_fit import (  # noqa: E402
    _dataset_from_tabular_policy,
    _mmd_tabular_policy,
    _parse_layers,
)
from scripts.run_small_nlhe_baseline_hardening_gate import (  # noqa: E402
    _fingerprint,
    _nashconv_from_policy_fn,
    _obs_lookup,
)
from scripts.run_small_nlhe_neural_nashpg_gate import NeuralReferencePGSolver  # noqa: E402


def _policy_arrays(net, dataset: dict[str, np.ndarray]) -> np.ndarray:
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32)
    legal = torch.as_tensor(dataset["legal"], dtype=torch.float32)
    with torch.no_grad():
        pi, _value, _log_pi, _logits = net(obs, legal)
    return pi.detach().cpu().numpy().astype(np.float64)


def _target_kl(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    kl = target * (np.log(np.maximum(target, 1e-12)) - np.log(np.maximum(pred, 1e-12)))
    return float(np.sum(kl, axis=1).mean())


def _delta_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _infostate_player_lookup(game) -> dict[str, int]:
    by: dict[str, int] = {}
    stack = [game.new_initial_state()]
    seen = set()
    while stack:
        state = stack.pop()
        if state.is_terminal():
            continue
        if state.is_chance_node():
            for action, _prob in state.chance_outcomes():
                child = state.clone()
                child.apply_action(action)
                key = child.history_str()
                if key not in seen:
                    seen.add(key)
                    stack.append(child)
            continue
        player = int(state.current_player())
        by.setdefault(str(state.information_state_string(player)), player)
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            key = child.history_str()
            if key not in seen:
                seen.add(key)
                stack.append(child)
    return by


def _infostate_weights(
    *,
    game,
    solver,
    keys: Sequence[str],
    mode: str,
) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(keys), dtype=np.float32)
    if mode != "parent-sequence":
        raise ValueError("infostate_weight_mode must be one of: uniform, parent-sequence")
    player_by_key = _infostate_player_lookup(game)
    weights: list[float] = []
    for key in keys:
        player = player_by_key.get(str(key))
        if player is None:
            weights.append(1.0)
            continue
        if str(key) not in solver.infoset_action_maps[player]:
            weights.append(1.0)
            continue
        weights.append(max(float(solver.get_parent_seq(player, str(key))), 0.0))
    arr = np.asarray(weights, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    if mean > 0.0:
        arr = arr / mean
    return arr.astype(np.float32, copy=False)


def _apply_exact_infostate_ce_update(
    solver: NeuralReferencePGSolver,
    dataset: dict[str, np.ndarray],
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int | str]:
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=solver.device)
    legal = torch.as_tensor(dataset["legal"], dtype=torch.float32, device=solver.device)
    target_t = torch.as_tensor(target, dtype=torch.float32, device=solver.device)
    weight_t = torch.as_tensor(weights, dtype=torch.float32, device=solver.device)
    if obs.shape[0] != target_t.shape[0] or obs.shape[0] != weight_t.shape[0]:
        raise ValueError("exact infostate update tensors must share the same row count")
    denom = weight_t.sum().clamp_min(1.0)
    pi, _value, _log_pi, _logits = solver.net(obs, legal)
    row_ce = -(target_t * torch.log(pi.clamp_min(1e-9))).sum(dim=1)
    loss = (row_ce * weight_t).sum() / denom
    solver.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(solver.net.parameters(), 10.0)
    solver.optimizer.step()
    solver.learner_steps += 1
    return {
        "bridge_update_mode": "exact-infostate-ce",
        "loss": float(loss.detach().cpu()),
        "n_information_states": int(obs.shape[0]),
        "mean_infostate_weight": float(weight_t.detach().mean().cpu()) if int(weight_t.numel()) else 0.0,
        "max_infostate_weight": float(weight_t.detach().max().cpu()) if int(weight_t.numel()) else 0.0,
        "learner_steps": int(solver.learner_steps),
    }


def _fit_solver_policy(
    solver: NeuralReferencePGSolver,
    dataset: dict[str, np.ndarray],
    *,
    fit_steps: int,
    lr: float,
) -> dict[str, float]:
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=solver.device)
    legal = torch.as_tensor(dataset["legal"], dtype=torch.float32, device=solver.device)
    target = torch.as_tensor(dataset["target"], dtype=torch.float32, device=solver.device)
    optimizer = torch.optim.Adam(solver.net.parameters(), lr=float(lr))
    losses: list[float] = []
    for _ in range(int(fit_steps)):
        pi, _value, _log_pi, _logits = solver.net(obs, legal)
        loss = -(target * torch.log(pi.clamp_min(1e-9))).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    solver.reference_net.load_state_dict(solver.net.state_dict())
    return {
        "fit_start_cross_entropy": losses[0] if losses else math.nan,
        "fit_final_cross_entropy": losses[-1] if losses else math.nan,
    }


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
    parser.add_argument("--start-mmd-steps", type=int, default=0)
    parser.add_argument("--target-mmd-delta", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--stepsize", type=float, default=None)
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--fit-lr", type=float, default=0.01)
    parser.add_argument("--update-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--reference-update-every", type=int, default=1000000)
    parser.add_argument("--advantage-target", choices=("terminal", "gae"), default="terminal")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--bridge-update-mode",
        choices=("rollout", "exact-infostate-ce"),
        default="rollout",
    )
    parser.add_argument(
        "--infostate-weight-mode",
        choices=("uniform", "parent-sequence"),
        default="parent-sequence",
    )
    parser.add_argument(
        "--decision-weight-mode",
        choices=("uniform", "inverse-own-reach"),
        default="uniform",
    )
    parser.add_argument("--max-decision-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--max-fit-current-kl", type=float, default=0.02)
    parser.add_argument("--min-target-kl-reduction", type=float, default=1e-5)
    parser.add_argument("--min-update-delta-cosine", type=float, default=0.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    if args.start_mmd_steps < 0:
        raise ValueError("--start-mmd-steps must be non-negative")
    if args.target_mmd_delta <= 0:
        raise ValueError("--target-mmd-delta must be positive")
    if args.fit_steps <= 0 or args.update_steps <= 0:
        raise ValueError("--fit-steps and --update-steps must be positive")
    if not (0.0 <= float(args.gamma) <= 1.0):
        raise ValueError("--gamma must be in [0, 1]")
    if not (0.0 <= float(args.gae_lambda) <= 1.0):
        raise ValueError("--gae-lambda must be in [0, 1]")
    if float(args.max_decision_weight) <= 0.0:
        raise ValueError("--max-decision-weight must be positive")

    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad.seat_collector import SeatAwarePyspielCollector
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    t0 = time.time()
    game = load_small_nlhe()
    by = _obs_lookup(game)
    fp = _fingerprint(game, policy_lib, exploitability)
    current_policy, current_solver = _mmd_tabular_policy(
        game,
        alpha=float(args.alpha),
        mmd_steps=int(args.start_mmd_steps),
        stepsize=args.stepsize,
    )
    target_policy, target_solver = _mmd_tabular_policy(
        game,
        alpha=float(args.alpha),
        mmd_steps=int(args.start_mmd_steps + args.target_mmd_delta),
        stepsize=args.stepsize,
    )
    current_data = _dataset_from_tabular_policy(game, by, current_policy)
    target_data = _dataset_from_tabular_policy(game, by, target_policy)

    collector = SeatAwarePyspielCollector(game, device="cpu")
    solver = NeuralReferencePGSolver(
        collector=collector,
        layers=_parse_layers(args.layers),
        lr=float(args.lr),
        reference_kl_weight=float(args.reference_kl_weight),
        entropy_weight=float(args.entropy_weight),
        value_weight=float(args.value_weight),
        reference_update_every=int(args.reference_update_every),
        seed=int(args.seed),
        advantage_target=str(args.advantage_target),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        decision_weight_mode=str(args.decision_weight_mode),
        max_decision_weight=float(args.max_decision_weight),
    )
    fit_stats = _fit_solver_policy(
        solver,
        current_data,
        fit_steps=int(args.fit_steps),
        lr=float(args.fit_lr),
    )
    before = _policy_arrays(solver.net, current_data)
    target = np.asarray(target_data["target"], dtype=np.float64)
    current = np.asarray(current_data["target"], dtype=np.float64)
    before_current_kl = _target_kl(current, before)
    before_target_kl = _target_kl(target, before)
    trajectory_max = max(8, game.max_game_length() + 1)
    step_logs = []
    for _ in range(int(args.update_steps)):
        if args.bridge_update_mode == "rollout":
            step_logs.append(
                solver.step(
                    batch_size=int(args.batch_size),
                    trajectory_max=trajectory_max,
                )
            )
        else:
            weights = _infostate_weights(
                game=game,
                solver=current_solver,
                keys=[str(key) for key in current_data["keys"]],
                mode=str(args.infostate_weight_mode),
            )
            step_logs.append(
                _apply_exact_infostate_ce_update(
                    solver,
                    current_data,
                    target,
                    weights,
                )
            )
    after = _policy_arrays(solver.net, current_data)
    after_target_kl = _target_kl(target, after)
    after_current_kl = _target_kl(current, after)
    target_kl_reduction = float(before_target_kl - after_target_kl)
    exact_delta = target - current
    neural_delta = after - before
    cosine = _delta_cosine(exact_delta, neural_delta)
    fit_nashconv = _nashconv_from_policy_fn(
        game,
        by,
        _net_policy_fn(solver.net),
        policy_lib,
        exploitability,
    )
    current_nashconv = float(exploitability.nash_conv(game, current_policy))
    target_nashconv = float(exploitability.nash_conv(game, target_policy))
    harness_passed = bool(
        fp["num_players"] == 2
        and fp["num_distinct_actions"] == 4
        and fp["max_game_length"] == 7
        and abs(fp["uniform_nashconv"] - 1.7) < 1e-3
    )
    passed = bool(
        harness_passed
        and before_current_kl <= float(args.max_fit_current_kl)
        and target_kl_reduction >= float(args.min_target_kl_reduction)
        and cosine >= float(args.min_update_delta_cosine)
    )
    out = {
        "gate": "small_nlhe_sequence_form_mmd_update_fidelity_bridge",
        "algorithm": "neural_rollout_update_vs_exact_mmd_delta",
        "candidate_family": "mmd_nashpg_style_regularized_policy_dynamics",
        "created_at_unix": int(time.time()),
        "seconds": round(time.time() - t0, 3),
        "uses_slumbot_training_data": False,
        "promotion": False,
        "small_game_exact_only": True,
        "fingerprint": fp,
        "config": {
            "start_mmd_steps": int(args.start_mmd_steps),
            "target_mmd_delta": int(args.target_mmd_delta),
            "alpha": float(args.alpha),
            "stepsize": getattr(current_solver, "stepsize", args.stepsize),
            "target_stepsize": getattr(target_solver, "stepsize", args.stepsize),
            "fit_steps": int(args.fit_steps),
            "fit_lr": float(args.fit_lr),
            "update_steps": int(args.update_steps),
            "batch_size": int(args.batch_size),
            "layers": list(_parse_layers(args.layers)),
            "lr": float(args.lr),
            "reference_kl_weight": float(args.reference_kl_weight),
            "entropy_weight": float(args.entropy_weight),
            "value_weight": float(args.value_weight),
            "bridge_update_mode": str(args.bridge_update_mode),
            "infostate_weight_mode": str(args.infostate_weight_mode),
            "advantage_target": str(args.advantage_target),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "decision_weight_mode": str(args.decision_weight_mode),
            "max_decision_weight": float(args.max_decision_weight),
            "seed": int(args.seed),
            "max_fit_current_kl": float(args.max_fit_current_kl),
            "min_target_kl_reduction": float(args.min_target_kl_reduction),
            "min_update_delta_cosine": float(args.min_update_delta_cosine),
        },
        "summary": {
            **fit_stats,
            "current_mmd_nashconv": current_nashconv,
            "target_mmd_nashconv": target_nashconv,
            "after_neural_update_nashconv": fit_nashconv,
            "before_current_kl": before_current_kl,
            "after_current_kl": after_current_kl,
            "before_target_kl": before_target_kl,
            "after_target_kl": after_target_kl,
            "target_kl_reduction": target_kl_reduction,
            "update_delta_cosine": cosine,
            "exact_delta_l2": float(np.linalg.norm(exact_delta.reshape(-1))),
            "neural_delta_l2": float(np.linalg.norm(neural_delta.reshape(-1))),
            "last_step_logs": step_logs[-1] if step_logs else {},
        },
        "decision": {
            "primary_metric": "target_kl_reduction_and_delta_cosine",
            "harness_passed": harness_passed,
            "update_fidelity_passed": passed,
            "interpretation": (
                "Pass means the existing neural rollout update moves in the same "
                "direction as exact sequence-form MMD on the locked small game. "
                "Fail means native scaling should change the update estimator."
            ),
            "slumbot_full_hunl_blocked": True,
        },
    }
    print(
        f"[bridge] before_target_kl={before_target_kl:.6g} "
        f"after_target_kl={after_target_kl:.6g} reduction={target_kl_reduction:.6g} "
        f"cos={cosine:.6g} passed={passed}",
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
