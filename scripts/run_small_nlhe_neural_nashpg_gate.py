#!/usr/bin/env python3
"""Neural NashPG/MMD-style truth gate on the locked small-NLHE harness.

This gate asks whether a neural reference-regularized policy-gradient update
can reproduce the exact small-game improvement signal before another native
HUNL scaling attempt. It trains only from local self-play trajectories and
scores with exact OpenSpiel NashConv. It is not Slumbot training and not
promotion evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from statistics import fmean
from typing import Sequence

import numpy as np
import torch

from poker_ai.rnad.network import RNaDNetwork
from run_small_nlhe_baseline_hardening_gate import (
    _fingerprint,
    _nashconv_from_solver,
    _obs_lookup,
    _stats,
)
from run_small_nlhe_mmd_truth_gate import _load_baseline


def _parse_layers(values: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(int(v) for v in values)
    if not layers or any(v <= 0 for v in layers):
        raise ValueError("--layers must contain positive hidden sizes")
    return layers


class NeuralReferencePGSolver:
    """Small-game neural policy-gradient learner with a moving reference policy."""

    def __init__(
        self,
        *,
        collector,
        layers: tuple[int, ...],
        lr: float,
        reference_kl_weight: float,
        entropy_weight: float,
        value_weight: float,
        reference_update_every: int,
        seed: int,
        advantage_target: str = "terminal",
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        decision_weight_mode: str = "uniform",
        max_decision_weight: float = 10.0,
        collector_mode: str = "seat-aware",
    ) -> None:
        if str(advantage_target) not in {"terminal", "gae", "player-gae"}:
            raise ValueError("advantage_target must be one of: terminal, gae, player-gae")
        if str(decision_weight_mode) not in {"uniform", "inverse-own-reach"}:
            raise ValueError("decision_weight_mode must be one of: uniform, inverse-own-reach")
        if not (0.0 <= float(gamma) <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if not (0.0 <= float(gae_lambda) <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1]")
        if float(max_decision_weight) <= 0.0:
            raise ValueError("max_decision_weight must be positive")
        self.collector = collector
        self.device = torch.device("cpu")
        self.n_actions = int(collector.n_actions)
        self.obs_dim = int(collector.obs_dim)
        self.num_players = int(collector.n_players)
        torch.manual_seed(int(seed))
        self.net = RNaDNetwork(self.obs_dim, self.n_actions, layers).to(self.device)
        self.reference_net = copy.deepcopy(self.net).to(self.device)
        self.reference_net.eval()
        for param in self.reference_net.parameters():
            param.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        self.reference_kl_weight = float(reference_kl_weight)
        self.entropy_weight = float(entropy_weight)
        self.value_weight = float(value_weight)
        self.reference_update_every = max(1, int(reference_update_every))
        self.advantage_target = str(advantage_target)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.decision_weight_mode = str(decision_weight_mode)
        self.max_decision_weight = float(max_decision_weight)
        self.collector_mode = str(collector_mode)
        self._rng = np.random.RandomState(int(seed))
        self.learner_steps = 0

    @torch.no_grad()
    def _policy_fn(self, obs_np, legal_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(legal_np, dtype=torch.float32, device=self.device)
        pi, _, _, _ = self.net(obs, legal)
        return pi.detach().cpu().numpy()

    @torch.no_grad()
    def action_probabilities(self, obs_np, legal_np):
        return self._policy_fn(obs_np, legal_np)

    def _loss_on_trajectory(self, traj) -> tuple[torch.Tensor, dict]:
        safe_legal = torch.where(
            traj.legal.sum(-1, keepdim=True) > 0,
            traj.legal,
            torch.ones_like(traj.legal),
        )
        pi, value, log_pi, _ = self.net(traj.obs, safe_legal)
        with torch.no_grad():
            ref_pi, _, _, _ = self.reference_net(traj.obs, safe_legal)

        valid = traj.valid.to(pi.dtype)
        decision_weights = _decision_weights(
            mode=self.decision_weight_mode,
            action_oh=traj.action_oh,
            behavior_policy=traj.policy,
            valid=valid,
            max_decision_weight=self.max_decision_weight,
        ).to(pi.dtype)
        weighted_valid = valid * decision_weights
        mask = weighted_valid > 0
        denom = weighted_valid.sum().clamp_min(1.0)
        logp_action = (traj.action_oh * log_pi).sum(-1)

        player = traj.player_id.long().clamp(min=0)
        final_returns = traj.rewards[-1]
        actor_return = torch.gather(
            final_returns.unsqueeze(0).expand(traj.rewards.shape[0], -1, self.num_players),
            2,
            player.unsqueeze(-1),
        ).squeeze(-1)
        actor_reward = torch.gather(
            traj.rewards,
            2,
            player.unsqueeze(-1),
        ).squeeze(-1)
        value_flat = value.squeeze(-1)
        if self.advantage_target == "terminal":
            value_target = actor_return
            advantage = value_target - value_flat.detach()
        elif self.advantage_target == "gae":
            value_target, advantage = _linked_bootstrap_targets_and_advantages(
                values=value_flat.detach(),
                terminal_returns=actor_return,
                valid=valid,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        else:
            value_target, advantage = _player_perspective_bootstrap_targets_and_advantages(
                values=value_flat.detach(),
                rewards=actor_reward,
                terminal_returns=actor_return,
                valid=valid,
                player_id=traj.player_id,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        if bool(mask.any()):
            adv_valid = advantage[mask]
            advantage = (advantage - adv_valid.mean()) / (adv_valid.std(unbiased=False) + 1e-5)
        policy_loss = -((logp_action * advantage) * weighted_valid).sum() / denom

        value_loss = (((value_flat - value_target) ** 2) * weighted_valid).sum() / denom
        entropy = -((pi.clamp_min(1e-9).log() * pi * safe_legal).sum(-1) * weighted_valid).sum() / denom
        kl = (
            (
                pi
                * (pi.clamp_min(1e-9).log() - ref_pi.clamp_min(1e-9).log())
                * safe_legal
            ).sum(-1)
            * weighted_valid
        ).sum() / denom
        observed_weights = decision_weights[valid > 0]
        mean_weight = (
            float(observed_weights.detach().mean().cpu())
            if int(observed_weights.numel()) > 0
            else 0.0
        )
        max_weight = (
            float(observed_weights.detach().max().cpu())
            if int(observed_weights.numel()) > 0
            else 0.0
        )
        loss = policy_loss + self.value_weight * value_loss + self.reference_kl_weight * kl - self.entropy_weight * entropy
        logs = {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "reference_kl": float(kl.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "collector_mode": self.collector_mode,
            "mean_valid_decisions": float(valid.sum().detach().cpu()) / max(1, int(valid.shape[1])),
            "advantage_target": self.advantage_target,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "decision_weight_mode": self.decision_weight_mode,
            "mean_decision_weight": mean_weight,
            "max_decision_weight": max_weight,
        }
        return loss, logs

    def step(self, *, batch_size: int, trajectory_max: int) -> dict:
        traj = self.collector.collect(
            self._policy_fn,
            int(batch_size),
            int(trajectory_max),
            self._rng,
        ).to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss, logs = self._loss_on_trajectory(traj)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()
        self.learner_steps += 1
        if self.learner_steps % self.reference_update_every == 0:
            self.reference_net.load_state_dict(self.net.state_dict())
        logs["learner_steps"] = int(self.learner_steps)
        return logs


def _linked_bootstrap_targets_and_advantages(
    *,
    values: torch.Tensor,
    terminal_returns: torch.Tensor,
    valid: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE-style targets over same-trajectory learner decision links."""
    if values.shape != terminal_returns.shape or values.shape != valid.shape:
        raise ValueError("values, terminal_returns, and valid must have matching shapes")
    targets = torch.zeros_like(terminal_returns)
    advantages = torch.zeros_like(terminal_returns)
    valid_bool = valid > 0
    for batch_i in range(values.shape[1]):
        next_i = -1
        next_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
        for time_i in range(values.shape[0] - 1, -1, -1):
            if not bool(valid_bool[time_i, batch_i]):
                continue
            if next_i >= 0:
                delta = float(gamma) * values[next_i, batch_i] - values[time_i, batch_i]
                advantages[time_i, batch_i] = (
                    delta + float(gamma) * float(gae_lambda) * next_advantage
                )
                targets[time_i, batch_i] = advantages[time_i, batch_i] + values[time_i, batch_i]
            else:
                advantages[time_i, batch_i] = (
                    terminal_returns[time_i, batch_i] - values[time_i, batch_i]
                )
                targets[time_i, batch_i] = terminal_returns[time_i, batch_i]
            next_advantage = advantages[time_i, batch_i]
            next_i = time_i
    return targets.detach(), advantages.detach()


def _player_perspective_bootstrap_targets_and_advantages(
    *,
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminal_returns: torch.Tensor,
    valid: torch.Tensor,
    player_id: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bootstrapped targets with zero-sum sign flips across player turns."""
    if (
        values.shape != rewards.shape
        or values.shape != terminal_returns.shape
        or values.shape != valid.shape
        or values.shape != player_id.shape
    ):
        raise ValueError("values, rewards, terminal_returns, valid, and player_id must match")
    targets = torch.zeros_like(terminal_returns)
    advantages = torch.zeros_like(terminal_returns)
    valid_bool = valid > 0
    for batch_i in range(values.shape[1]):
        next_i = -1
        next_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
        for time_i in range(values.shape[0] - 1, -1, -1):
            if not bool(valid_bool[time_i, batch_i]):
                continue
            if next_i >= 0:
                same_player = bool(player_id[next_i, batch_i] == player_id[time_i, batch_i])
                sign = 1.0 if same_player else -1.0
                delta = (
                    rewards[time_i, batch_i]
                    + float(gamma) * sign * values[next_i, batch_i]
                    - values[time_i, batch_i]
                )
                advantages[time_i, batch_i] = (
                    delta + float(gamma) * float(gae_lambda) * sign * next_advantage
                )
                targets[time_i, batch_i] = advantages[time_i, batch_i] + values[time_i, batch_i]
            else:
                advantages[time_i, batch_i] = (
                    terminal_returns[time_i, batch_i] - values[time_i, batch_i]
                )
                targets[time_i, batch_i] = terminal_returns[time_i, batch_i]
            next_advantage = advantages[time_i, batch_i]
            next_i = time_i
    return targets.detach(), advantages.detach()


def _decision_weights(
    *,
    mode: str,
    action_oh: torch.Tensor,
    behavior_policy: torch.Tensor,
    valid: torch.Tensor,
    max_decision_weight: float,
) -> torch.Tensor:
    if str(mode) == "uniform":
        return torch.where(valid > 0, torch.ones_like(valid), torch.zeros_like(valid))
    if str(mode) != "inverse-own-reach":
        raise ValueError("decision_weight_mode must be one of: uniform, inverse-own-reach")
    return _decision_weights_from_own_reach(
        action_oh=action_oh,
        behavior_policy=behavior_policy,
        valid=valid,
        max_decision_weight=float(max_decision_weight),
    )


def _decision_weights_from_own_reach(
    *,
    action_oh: torch.Tensor,
    behavior_policy: torch.Tensor,
    valid: torch.Tensor,
    max_decision_weight: float,
) -> torch.Tensor:
    """Approximate counterfactual occupancy by removing sampled learner reach."""
    if action_oh.shape != behavior_policy.shape:
        raise ValueError("action_oh and behavior_policy must have matching shapes")
    if action_oh.shape[:2] != valid.shape:
        raise ValueError("valid must match the [time, batch] trajectory shape")
    if float(max_decision_weight) <= 0.0:
        raise ValueError("max_decision_weight must be positive")
    weights = torch.zeros_like(valid, dtype=behavior_policy.dtype, device=behavior_policy.device)
    valid_bool = valid > 0
    eps = torch.tensor(1.0e-8, dtype=behavior_policy.dtype, device=behavior_policy.device)
    cap = torch.tensor(float(max_decision_weight), dtype=behavior_policy.dtype, device=behavior_policy.device)
    for batch_i in range(valid.shape[1]):
        own_reach = torch.ones((), dtype=behavior_policy.dtype, device=behavior_policy.device)
        for time_i in range(valid.shape[0]):
            if not bool(valid_bool[time_i, batch_i]):
                continue
            weights[time_i, batch_i] = torch.minimum(1.0 / torch.maximum(own_reach, eps), cap)
            action_prob = torch.sum(
                action_oh[time_i, batch_i] * behavior_policy[time_i, batch_i]
            )
            own_reach = own_reach * torch.maximum(action_prob, eps)
    observed = weights[valid_bool]
    if int(observed.numel()) > 0:
        weights = weights / observed.mean().clamp_min(1.0e-8)
    return torch.where(valid_bool, weights, torch.zeros_like(weights))


def _run_seed(args, seed: int, game, by, policy_lib, exploitability) -> dict:
    if args.collector_mode == "seat-aware":
        from poker_ai.rnad.seat_collector import SeatAwarePyspielCollector

        collector = SeatAwarePyspielCollector(game, device="cpu")
    elif args.collector_mode == "full-self-play":
        from poker_ai.rnad import LeducTreeCollector

        collector = LeducTreeCollector(game, device="cpu")
    else:
        raise ValueError("--collector-mode must be one of: seat-aware, full-self-play")
    solver = NeuralReferencePGSolver(
        collector=collector,
        layers=_parse_layers(args.layers),
        lr=float(args.lr),
        reference_kl_weight=float(args.reference_kl_weight),
        entropy_weight=float(args.entropy_weight),
        value_weight=float(args.value_weight),
        reference_update_every=int(args.reference_update_every),
        seed=int(seed),
        advantage_target=str(args.advantage_target),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        decision_weight_mode=str(args.decision_weight_mode),
        max_decision_weight=float(args.max_decision_weight),
        collector_mode=str(args.collector_mode),
    )
    trajectory_max = max(8, game.max_game_length() + 1)
    history = [(0, _nashconv_from_solver(game, by, solver, policy_lib, exploitability))]
    losses: list[dict] = []
    t0 = time.time()
    for step in range(1, int(args.steps) + 1):
        losses.append(solver.step(batch_size=args.batch_size, trajectory_max=trajectory_max))
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            history.append((step, _nashconv_from_solver(game, by, solver, policy_lib, exploitability)))
    values = [float(v) for _, v in history]
    return {
        "seed": int(seed),
        "history": history,
        "start": float(values[0]),
        "last": float(values[-1]),
        "best": float(min(values)),
        "descended": bool(values[-1] < values[0]),
        "loss_last": losses[-1] if losses else {},
        "seconds": round(time.time() - t0, 3),
    }


def _summarize(name: str, runs: list[dict]) -> dict:
    last = [float(run["last"]) for run in runs]
    best = [float(run["best"]) for run in runs]
    return {
        "name": name,
        "seeds": [int(run["seed"]) for run in runs],
        "last_nashconv": last,
        "best_nashconv": best,
        "mean_last_nashconv": _stats(last)["mean"],
        "std_last_nashconv": _stats(last)["std"],
        "mean_best_nashconv": _stats(best)["mean"],
        "std_best_nashconv": _stats(best)["std"],
        "runs": runs,
    }


def _finite_values(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _candidate_truth_gate_passed(
    *,
    harness_passed: bool,
    all_metrics_finite: bool,
    improved_from_uniform: bool,
    mean_best_nashconv: float,
    mean_last_nashconv: float,
    baseline_mean_last_nashconv: float | None,
) -> bool:
    if not (bool(harness_passed) and bool(all_metrics_finite) and bool(improved_from_uniform)):
        return False
    if baseline_mean_last_nashconv is None:
        return True
    return bool(
        float(mean_best_nashconv) < float(baseline_mean_last_nashconv)
        and float(mean_last_nashconv) < float(baseline_mean_last_nashconv)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--reference-update-every", type=int, default=200)
    parser.add_argument("--advantage-target", choices=("terminal", "gae", "player-gae"), default="terminal")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--decision-weight-mode",
        choices=("uniform", "inverse-own-reach"),
        default="uniform",
    )
    parser.add_argument("--max-decision-weight", type=float, default=10.0)
    parser.add_argument(
        "--collector-mode",
        choices=("seat-aware", "full-self-play"),
        default="seat-aware",
        help=(
            "seat-aware trains only the selected learner seat; full-self-play "
            "uses the vectorized tree collector and trains all acting-player decisions."
        ),
    )
    parser.add_argument("--baseline-json")
    parser.add_argument("--baseline-arm", default="rnad")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.reference_kl_weight < 0:
        raise ValueError("--reference-kl-weight must be non-negative")
    if args.reference_update_every <= 0:
        raise ValueError("--reference-update-every must be positive")
    if not (0.0 <= float(args.gamma) <= 1.0):
        raise ValueError("--gamma must be in [0, 1]")
    if not (0.0 <= float(args.gae_lambda) <= 1.0):
        raise ValueError("--gae-lambda must be in [0, 1]")
    if float(args.max_decision_weight) <= 0.0:
        raise ValueError("--max-decision-weight must be positive")
    if args.collector_mode == "full-self-play" and args.advantage_target == "gae":
        parser.error(
            "full-self-play GAE is unsupported by this current-player value convention; "
            "use terminal returns or implement player-perspective bootstrapping first"
        )

    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    game = load_small_nlhe()
    by = _obs_lookup(game)
    fp = _fingerprint(game, policy_lib, exploitability)
    baseline = _load_baseline(args.baseline_json, args.baseline_arm)

    t0 = time.time()
    runs = [_run_seed(args, seed, game, by, policy_lib, exploitability) for seed in seeds]
    arm = _summarize("neural_nashpg", runs)
    all_values = [float(v) for run in runs for _, v in run["history"]]
    harness_passed = bool(
        fp["num_players"] == 2
        and fp["num_distinct_actions"] == 4
        and fp["max_game_length"] == 7
        and abs(fp["uniform_nashconv"] - 1.7) < 1e-3
    )
    improved_from_uniform = bool(arm["mean_best_nashconv"] < fp["uniform_nashconv"])
    compared_to_baseline = None
    best_beats_baseline = None
    last_beats_baseline = None
    baseline_mean_last = None
    if baseline is not None:
        baseline_mean_last = float(baseline["mean_last_nashconv"])
        best_beats_baseline = bool(arm["mean_best_nashconv"] < baseline_mean_last)
        last_beats_baseline = bool(arm["mean_last_nashconv"] < baseline_mean_last)
        compared_to_baseline = {
            "baseline_arm": baseline["arm"],
            "baseline_mean_last_nashconv": baseline_mean_last,
            "neural_best_beats_baseline_last": best_beats_baseline,
            "neural_last_beats_baseline_last": last_beats_baseline,
        }
    all_metrics_finite = _finite_values(all_values)
    candidate_passed = _candidate_truth_gate_passed(
        harness_passed=harness_passed,
        all_metrics_finite=all_metrics_finite,
        improved_from_uniform=improved_from_uniform,
        mean_best_nashconv=arm["mean_best_nashconv"],
        mean_last_nashconv=arm["mean_last_nashconv"],
        baseline_mean_last_nashconv=baseline_mean_last,
    )
    decision = {
        "primary_metric": "exact_open_spiel_nashconv_lower_is_better",
        "harness_passed": harness_passed,
        "all_metrics_finite": all_metrics_finite,
        "small_game_exact_only": True,
        "improved_from_uniform": improved_from_uniform,
        "beats_baseline": last_beats_baseline,
        "best_beats_baseline": best_beats_baseline,
        "last_beats_baseline": last_beats_baseline,
        "candidate_truth_gate_passed": candidate_passed,
        "slumbot_full_hunl_blocked": True,
        "compared_to_baseline": compared_to_baseline,
        "interpretation": (
            "Neural reference-regularized policy gradient is a small-game truth gate "
            "for a NashPG/MMD-style update. Passing requires the deployable last "
            "policy, not only a best training-curve point, to beat the matched "
            "baseline before native mechanism work; it does not promote any HUNL checkpoint."
        ),
    }
    out = {
        "gate": "small_nlhe_neural_nashpg_truth_gate",
        "algorithm": "neural_reference_regularized_policy_gradient",
        "candidate_family": "neural_mmd_nashpg_style_regularized_policy_dynamics",
        "created_at_unix": int(time.time()),
        "seconds": round(time.time() - t0, 3),
        "uses_slumbot_training_data": False,
        "promotion": False,
        "fingerprint": fp,
        "config": {
            "steps": int(args.steps),
            "eval_every": int(args.eval_every),
            "batch_size": int(args.batch_size),
            "seeds": seeds,
            "layers": list(_parse_layers(args.layers)),
            "lr": float(args.lr),
            "reference_kl_weight": float(args.reference_kl_weight),
            "entropy_weight": float(args.entropy_weight),
            "value_weight": float(args.value_weight),
            "reference_update_every": int(args.reference_update_every),
            "advantage_target": str(args.advantage_target),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "decision_weight_mode": str(args.decision_weight_mode),
            "max_decision_weight": float(args.max_decision_weight),
            "collector_mode": str(args.collector_mode),
            "baseline_json": args.baseline_json,
            "baseline_arm": args.baseline_arm,
        },
        "baseline": baseline,
        "arms": {"neural_nashpg": arm},
        "summary": {
            "mean_last_nashconv": arm["mean_last_nashconv"],
            "mean_best_nashconv": arm["mean_best_nashconv"],
            "mean_start_nashconv": float(fmean([float(run["start"]) for run in runs])),
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
        f"[neural-nashpg] steps={args.steps} seeds={seeds} "
        f"mean_best={arm['mean_best_nashconv']:.6f} mean_last={arm['mean_last_nashconv']:.6f}",
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
