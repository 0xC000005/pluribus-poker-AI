#!/usr/bin/env python3
"""Exact counterfactual neural policy-gradient gate on locked small-NLHE.

This is a small-game mechanism gate. It computes exact current-policy
counterfactual action advantages from the local game tree, applies a
mirror-descent-style policy target, and fits a neural stochastic policy to that
update. Exact NashConv is used only for evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import fmean
from typing import Sequence

import numpy as np
import torch

from poker_ai.rnad.network import RNaDNetwork
from run_rnad_tabular_gate import Tree
from run_small_nlhe_baseline_hardening_gate import (
    _fingerprint,
    _nashconv_from_policy_fn,
    _obs_lookup,
    _stats,
)
from run_small_nlhe_mmd_truth_gate import _load_baseline


def _parse_layers(values: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(int(v) for v in values)
    if not layers or any(v <= 0 for v in layers):
        raise ValueError("--layers must contain positive hidden sizes")
    return layers


def _counterfactual_values_and_weights(tree: Tree, pi: list[dict[int, float]]) -> tuple[list[dict[int, float]], list[float]]:
    qnum: list[dict[int, float]] = [dict() for _ in range(tree.n_iset)]
    qden = [0.0] * tree.n_iset

    def rec(node, r0: float, r1: float, rc: float) -> np.ndarray:
        kind = node[0]
        if kind == "term":
            return np.asarray(node[1], dtype=np.float64)
        if kind == "chance":
            ev = np.zeros(2, dtype=np.float64)
            for prob, child in node[1]:
                ev += float(prob) * rec(child, r0, r1, rc * float(prob))
            return ev
        _, player, iid, children = node
        policy = pi[int(iid)]
        cf_reach = (r1 * rc) if int(player) == 0 else (r0 * rc)
        qden[int(iid)] += float(cf_reach)
        ev = np.zeros(2, dtype=np.float64)
        row = qnum[int(iid)]
        for action, child in children:
            prob = float(policy[int(action)])
            if int(player) == 0:
                child_ev = rec(child, r0 * prob, r1, rc)
            else:
                child_ev = rec(child, r0, r1 * prob, rc)
            ev += prob * child_ev
            row[int(action)] = row.get(int(action), 0.0) + float(cf_reach) * float(child_ev[int(player)])
        return ev

    rec(tree.root, 1.0, 1.0, 1.0)
    q: list[dict[int, float]] = []
    for iid, actions in enumerate(tree.iset_actions):
        den = float(qden[iid])
        q.append({int(action): (qnum[iid].get(int(action), 0.0) / den if den > 1e-15 else 0.0) for action in actions})
    return q, qden


def _soft_counterfactual_target(
    *,
    legal: np.ndarray,
    current_policy: np.ndarray,
    advantage: np.ndarray,
    advantage_temperature: float,
) -> np.ndarray:
    if float(advantage_temperature) <= 0.0:
        raise ValueError("advantage_temperature must be positive")
    legal = np.asarray(legal, dtype=np.float64)
    current_policy = np.asarray(current_policy, dtype=np.float64)
    advantage = np.asarray(advantage, dtype=np.float64)
    active = legal > 0.0
    if not bool(active.any()):
        raise ValueError("legal mask has no active actions")
    logits = np.full_like(current_policy, -np.inf, dtype=np.float64)
    logits[active] = (
        np.log(np.maximum(current_policy[active], 1e-12))
        + advantage[active] / float(advantage_temperature)
    )
    logits[active] -= float(np.max(logits[active]))
    probs = np.zeros_like(current_policy, dtype=np.float64)
    probs[active] = np.exp(logits[active])
    total = float(probs[active].sum())
    if total <= 0.0 or not math.isfinite(total):
        probs[active] = 1.0 / float(active.sum())
    else:
        probs[active] /= total
    return probs.astype(np.float32, copy=False)


def _counterfactual_training_rows(
    *,
    game,
    tree: Tree,
    by: dict,
    pi: list[dict[int, float]],
    reference_pi: list[dict[int, float]] | None = None,
    reference_regularization_weight: float = 0.0,
    advantage_temperature: float,
) -> list[dict]:
    if float(reference_regularization_weight) < 0.0:
        raise ValueError("reference_regularization_weight must be non-negative")
    if reference_pi is not None and len(reference_pi) != len(pi):
        raise ValueError("reference_pi must match pi length")
    q, cf_weights = _counterfactual_values_and_weights(tree, pi)
    rows: list[dict] = []
    for key, iid in tree.infosets.items():
        if key not in by:
            continue
        obs, legal = by[key]
        legal_arr = np.asarray(legal, dtype=np.float32)
        current = np.zeros(int(game.num_distinct_actions()), dtype=np.float64)
        advantage = np.zeros_like(current)
        value = 0.0
        for action in tree.iset_actions[iid]:
            value += float(pi[iid][int(action)]) * float(q[iid][int(action)])
        for action in tree.iset_actions[iid]:
            current[int(action)] = float(pi[iid][int(action)])
            advantage[int(action)] = float(q[iid][int(action)] - value)
        regularized_advantage = advantage.copy()
        if reference_pi is not None and float(reference_regularization_weight) > 0.0:
            for action in tree.iset_actions[iid]:
                current_prob = max(float(pi[iid][int(action)]), 1e-12)
                reference_prob = max(float(reference_pi[iid][int(action)]), 1e-12)
                regularized_advantage[int(action)] += -float(reference_regularization_weight) * (
                    math.log(current_prob) - math.log(reference_prob)
                )
        target = _soft_counterfactual_target(
            legal=legal_arr,
            current_policy=current,
            advantage=regularized_advantage,
            advantage_temperature=float(advantage_temperature),
        )
        rows.append(
            {
                "key": str(key),
                "obs": np.asarray(obs, dtype=np.float32),
                "legal": legal_arr,
                "current_policy": current,
                "advantage": advantage,
                "regularized_advantage": regularized_advantage,
                "target_policy": target,
                "cf_reach_weight": float(max(cf_weights[iid], 0.0)),
                "counterfactual_value": float(value),
                "reference_regularization_weight": float(reference_regularization_weight),
            }
        )
    if not rows:
        raise ValueError("counterfactual training row set is empty")
    weights = np.asarray([row["cf_reach_weight"] for row in rows], dtype=np.float64)
    mean_weight = float(weights.mean()) if weights.size else 0.0
    if mean_weight > 1e-15:
        for row in rows:
            row["cf_reach_weight"] = float(row["cf_reach_weight"] / mean_weight)
    return rows


class ExactCounterfactualNeuralPGSolver:
    def __init__(
        self,
        *,
        game,
        tree: Tree,
        by: dict,
        layers: tuple[int, ...],
        lr: float,
        fit_epochs_per_step: int,
        advantage_temperature: float,
        reference_regularization_weight: float,
        reference_update_every: int,
        seed: int,
    ) -> None:
        if int(fit_epochs_per_step) <= 0:
            raise ValueError("fit_epochs_per_step must be positive")
        if float(lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(advantage_temperature) <= 0.0:
            raise ValueError("advantage_temperature must be positive")
        if float(reference_regularization_weight) < 0.0:
            raise ValueError("reference_regularization_weight must be non-negative")
        if int(reference_update_every) <= 0:
            raise ValueError("reference_update_every must be positive")
        self.game = game
        self.tree = tree
        self.by = by
        self.device = torch.device("cpu")
        self.n_actions = int(game.num_distinct_actions())
        self.obs_dim = int(game.information_state_tensor_size())
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self.net = RNaDNetwork(self.obs_dim, self.n_actions, layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        self.fit_epochs_per_step = int(fit_epochs_per_step)
        self.advantage_temperature = float(advantage_temperature)
        self.reference_regularization_weight = float(reference_regularization_weight)
        self.reference_update_every = int(reference_update_every)
        self.reference_pi = self._tabular_policy_dict()
        self.learner_steps = 0

    @torch.no_grad()
    def action_probabilities(self, obs_np, legal_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(legal_np, dtype=torch.float32, device=self.device)
        pi, _value, _log_pi, _logits = self.net(obs, legal)
        return pi.detach().cpu().numpy()

    def _tabular_policy_dict(self) -> list[dict[int, float]]:
        pi: list[dict[int, float]] = []
        for key, iid in sorted(self.tree.infosets.items(), key=lambda item: item[1]):
            obs, legal = self.by[key]
            probs = self.action_probabilities(np.asarray(obs)[None], np.asarray(legal)[None])[0]
            row = {int(action): float(probs[int(action)]) for action in self.tree.iset_actions[iid]}
            total = sum(max(v, 0.0) for v in row.values())
            if total <= 0.0:
                row = {action: 1.0 / len(row) for action in row}
            else:
                row = {action: max(prob, 0.0) / total for action, prob in row.items()}
            pi.append(row)
        return pi

    def step(self) -> dict:
        rows = _counterfactual_training_rows(
            game=self.game,
            tree=self.tree,
            by=self.by,
            pi=self._tabular_policy_dict(),
            reference_pi=self.reference_pi if self.reference_regularization_weight > 0.0 else None,
            reference_regularization_weight=self.reference_regularization_weight,
            advantage_temperature=self.advantage_temperature,
        )
        obs = torch.as_tensor(np.stack([row["obs"] for row in rows]), dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(np.stack([row["legal"] for row in rows]), dtype=torch.float32, device=self.device)
        target = torch.as_tensor(np.stack([row["target_policy"] for row in rows]), dtype=torch.float32, device=self.device)
        weight = torch.as_tensor([row["cf_reach_weight"] for row in rows], dtype=torch.float32, device=self.device)
        denom = weight.sum().clamp_min(1.0)
        last_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for _ in range(self.fit_epochs_per_step):
            pi, _value, _log_pi, _logits = self.net(obs, legal)
            row_ce = -(target * torch.log(pi.clamp_min(1e-9))).sum(dim=1)
            loss = (row_ce * weight).sum() / denom
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.optimizer.step()
            last_loss = loss.detach()
        self.learner_steps += 1
        if (
            self.reference_regularization_weight > 0.0
            and self.learner_steps % self.reference_update_every == 0
        ):
            self.reference_pi = self._tabular_policy_dict()
        target_l1 = [
            float(np.abs(row["target_policy"] - row["current_policy"]).sum())
            for row in rows
        ]
        advantages = np.concatenate([np.asarray(row["advantage"], dtype=np.float64) for row in rows])
        regularized_advantages = np.concatenate(
            [np.asarray(row["regularized_advantage"], dtype=np.float64) for row in rows]
        )
        return {
            "loss": float(last_loss.cpu()),
            "n_information_sets": int(len(rows)),
            "mean_cf_reach_weight": float(np.mean([row["cf_reach_weight"] for row in rows])),
            "max_cf_reach_weight": float(np.max([row["cf_reach_weight"] for row in rows])),
            "mean_target_l1_from_current": float(np.mean(target_l1)),
            "max_abs_advantage": float(np.max(np.abs(advantages))) if advantages.size else 0.0,
            "max_abs_regularized_advantage": (
                float(np.max(np.abs(regularized_advantages))) if regularized_advantages.size else 0.0
            ),
            "fit_epochs_per_step": int(self.fit_epochs_per_step),
            "advantage_temperature": float(self.advantage_temperature),
            "reference_regularization_weight": float(self.reference_regularization_weight),
            "reference_update_every": int(self.reference_update_every),
            "learner_steps": int(self.learner_steps),
        }


def _run_seed(args, seed: int, game, tree: Tree, by, policy_lib, exploitability) -> dict:
    solver = ExactCounterfactualNeuralPGSolver(
        game=game,
        tree=tree,
        by=by,
        layers=_parse_layers(args.layers),
        lr=float(args.lr),
        fit_epochs_per_step=int(args.fit_epochs_per_step),
        advantage_temperature=float(args.advantage_temperature),
        reference_regularization_weight=float(args.reference_regularization_weight),
        reference_update_every=int(args.reference_update_every),
        seed=int(seed),
    )
    history = [(0, _nashconv_from_policy_fn(game, by, solver.action_probabilities, policy_lib, exploitability))]
    losses: list[dict] = []
    t0 = time.time()
    for step in range(1, int(args.steps) + 1):
        losses.append(solver.step())
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            history.append((step, _nashconv_from_policy_fn(game, by, solver.action_probabilities, policy_lib, exploitability)))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fit-epochs-per-step", type=int, default=4)
    parser.add_argument("--advantage-temperature", type=float, default=1.0)
    parser.add_argument("--reference-regularization-weight", type=float, default=0.0)
    parser.add_argument("--reference-update-every", type=int, default=100)
    parser.add_argument("--baseline-json")
    parser.add_argument("--baseline-arm", default="rnad")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be positive")
    if int(args.fit_epochs_per_step) <= 0:
        raise ValueError("--fit-epochs-per-step must be positive")
    if float(args.advantage_temperature) <= 0.0:
        raise ValueError("--advantage-temperature must be positive")
    if float(args.reference_regularization_weight) < 0.0:
        raise ValueError("--reference-regularization-weight must be non-negative")
    if int(args.reference_update_every) <= 0:
        raise ValueError("--reference-update-every must be positive")

    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    game = load_small_nlhe()
    tree = Tree(game)
    by = _obs_lookup(game)
    fp = _fingerprint(game, policy_lib, exploitability)
    baseline = _load_baseline(args.baseline_json, args.baseline_arm)

    t0 = time.time()
    runs = [_run_seed(args, seed, game, tree, by, policy_lib, exploitability) for seed in seeds]
    arm = _summarize("exact_cfpg", runs)
    all_values = [float(v) for run in runs for _, v in run["history"]]
    harness_passed = bool(
        fp["num_players"] == 2
        and fp["num_distinct_actions"] == 4
        and fp["max_game_length"] == 7
        and abs(fp["uniform_nashconv"] - 1.7) < 1e-3
    )
    improved_from_uniform = bool(arm["mean_best_nashconv"] < fp["uniform_nashconv"])
    baseline_mean_last = None
    compared_to_baseline = None
    best_beats_baseline = None
    last_beats_baseline = None
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
            "Exact-CFPG is a small-game sanity gate for a counterfactual-compatible "
            "neural self-play update. Training uses exact current-policy reach and "
            "terminal rewards, not NashConv or solver policy labels. Passing is "
            "required before designing a sampled native approximation."
        ),
    }
    out = {
        "gate": "small_nlhe_exact_counterfactual_neural_pg",
        "algorithm": "exact_counterfactual_neural_policy_gradient",
        "candidate_family": "counterfactual_compatible_neural_policy_gradient",
        "created_at_unix": int(time.time()),
        "seconds": round(time.time() - t0, 3),
        "uses_slumbot_training_data": False,
        "promotion": False,
        "fingerprint": fp,
        "config": {
            "steps": int(args.steps),
            "eval_every": int(args.eval_every),
            "seeds": seeds,
            "layers": list(_parse_layers(args.layers)),
            "lr": float(args.lr),
            "fit_epochs_per_step": int(args.fit_epochs_per_step),
            "advantage_temperature": float(args.advantage_temperature),
            "reference_regularization_weight": float(args.reference_regularization_weight),
            "reference_update_every": int(args.reference_update_every),
            "baseline_json": args.baseline_json,
            "baseline_arm": args.baseline_arm,
        },
        "baseline": baseline,
        "arms": {"exact_cfpg": arm},
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
        f"[exact-cfpg] steps={args.steps} seeds={seeds} "
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
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
