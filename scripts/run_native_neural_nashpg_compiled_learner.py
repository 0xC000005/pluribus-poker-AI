#!/usr/bin/env python3
"""Train a native 9-action neural NashPG/MMD-style learner from local self-play.

This is the native translation of the small-NLHE neural NashPG truth gate:
compiled stochastic self-play, terminal-return policy gradient with a learned
value baseline, entropy, and reference-policy KL. It uses only the local
full-deck simulator and is not Slumbot or solver-label training.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES  # noqa: E402
from poker_ai.research.local_vtrace import masked_log_probs  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.native_ppo_policy import _PolicyMLP, _ValueMLP  # noqa: E402
from poker_ai.research.native_rollout_substrate import (  # noqa: E402
    _collect_compiled_fast_policy_gradient_rollout,
)
from scripts.run_local_vtrace_compiled_native_learner import (  # noqa: E402
    _load_compiled_rollout_opponents,
    _load_reference_policy,
    _summarize_opponent_kinds,
)


def _write_metrics(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _load_checkpoint_in(
    *,
    checkpoint_in: str | Path | None,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    q_net: torch.nn.Module | None = None,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint_in is None:
        return None
    payload = torch.load(str(checkpoint_in), map_location=device, weights_only=False)
    if str(payload.get("environment", "")) != "poker_ai:full_deck_hu_nlhe":
        raise ValueError("checkpoint_in must be trained in the native full-deck environment")
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("checkpoint_in action count does not match native contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("checkpoint_in feature count does not match native contract")
    if "policy_net_state_dict" not in payload:
        raise ValueError("checkpoint_in does not contain a policy_net_state_dict")
    policy_net.load_state_dict(payload["policy_net_state_dict"])
    if "value_net_state_dict" in payload:
        value_net.load_state_dict(payload["value_net_state_dict"])
    payload = dict(payload)
    payload["checkpoint_in_q_loaded"] = False
    if q_net is not None and "q_net_state_dict" in payload:
        q_net.load_state_dict(payload["q_net_state_dict"])
        payload["checkpoint_in_q_loaded"] = True
    return dict(payload)


def _sampled_counterfactual_decision_weights(
    batch: dict[str, np.ndarray],
    *,
    mode: str,
    max_decision_weight: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Approximate counterfactual occupancy by removing prior same-player reach.

    This is still a sampled full-game approximation, not exact tree CFR. It makes
    the native learner's row weights closer to the small-game counterfactual
    signal: a player's earlier sampled action probability should not suppress
    later feedback for that same player's information-set trajectory.
    """
    n_rows = int(np.asarray(batch["old_log_probs"]).shape[0])
    if str(mode) == "uniform":
        return torch.ones(n_rows, dtype=torch.float32, device=device)
    if str(mode) != "inverse-own-reach":
        raise ValueError("decision_weight_mode must be one of: uniform, inverse-own-reach")
    if float(max_decision_weight) <= 0.0:
        raise ValueError("max_decision_weight must be positive")

    old_probs = np.exp(np.asarray(batch["old_log_probs"], dtype=np.float64))
    game_indices = np.asarray(batch["game_indices"], dtype=np.int64)
    players = np.asarray(batch["players"], dtype=np.int64)
    step_indices = np.asarray(batch["step_indices"], dtype=np.int64)
    weights = np.ones(n_rows, dtype=np.float64)
    grouped: dict[tuple[int, int], list[int]] = {}
    for record_i, (game_i, player_i) in enumerate(zip(game_indices, players, strict=False)):
        grouped.setdefault((int(game_i), int(player_i)), []).append(int(record_i))

    for indices in grouped.values():
        own_reach = 1.0
        for record_i in sorted(indices, key=lambda idx: int(step_indices[idx])):
            weights[record_i] = min(
                1.0 / max(float(own_reach), 1e-8),
                float(max_decision_weight),
            )
            own_reach *= max(float(old_probs[record_i]), 1e-8)

    observed_mean = float(weights.mean()) if weights.size else 1.0
    if observed_mean > 0.0:
        weights = weights / observed_mean
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _policy_value_reference_loss(
    *,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    reference_policy: torch.nn.Module | None,
    batch: dict[str, np.ndarray],
    device: torch.device,
    value_weight: float,
    entropy_weight: float,
    reference_kl_weight: float,
    advantage_target: str,
    gamma: float,
    gae_lambda: float,
    decision_weight_mode: str,
    max_decision_weight: float,
) -> tuple[torch.Tensor, dict[str, float], int]:
    if int(batch["actions"].shape[0]) <= 0:
        return torch.zeros((), dtype=torch.float32, device=device), {}, 0

    features = torch.as_tensor(batch["features"], dtype=torch.float32, device=device)
    legal_masks = torch.as_tensor(batch["legal_masks"] > 0, dtype=torch.bool, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    returns = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    decision_weights = _sampled_counterfactual_decision_weights(
        batch,
        mode=str(decision_weight_mode),
        max_decision_weight=float(max_decision_weight),
        device=device,
    )
    weight_denom = decision_weights.sum().clamp_min(1e-6)

    logits = policy_net(features)
    log_probs = masked_log_probs(logits, legal_masks)
    probs = torch.exp(log_probs)
    values = value_net(features).reshape(-1)
    action_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)

    target_mode = str(advantage_target)
    if target_mode == "terminal":
        value_targets = returns
        advantages = value_targets - values.detach()
    elif target_mode == "gae":
        value_targets, advantages = _linked_bootstrap_targets_and_advantages(
            values=values.detach(),
            terminal_returns=returns,
            batch=batch,
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
        )
    elif target_mode == "player-gae":
        value_targets, advantages = _player_perspective_bootstrap_targets_and_advantages(
            values=values.detach(),
            terminal_returns=returns,
            batch=batch,
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
        )
    else:
        raise ValueError("advantage_target must be one of: terminal, gae, player-gae")
    if int(advantages.numel()) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
    policy_loss = -torch.sum(decision_weights * action_log_probs * advantages) / weight_denom
    value_loss = torch.sum(decision_weights * torch.square(values - value_targets)) / weight_denom
    per_row_entropy = -torch.sum(probs * torch.nan_to_num(log_probs, neginf=0.0), dim=1)
    entropy = torch.sum(decision_weights * per_row_entropy) / weight_denom

    reference_kl = torch.zeros((), dtype=torch.float32, device=device)
    if reference_policy is not None and float(reference_kl_weight) > 0.0:
        with torch.no_grad():
            reference_logits = reference_policy(features)
            reference_log_probs = masked_log_probs(reference_logits, legal_masks)
        kl_terms = probs * (
            torch.nan_to_num(log_probs, neginf=0.0)
            - torch.nan_to_num(reference_log_probs, neginf=0.0)
        )
        kl_terms = torch.where(legal_masks, kl_terms, torch.zeros_like(kl_terms))
        per_row_reference_kl = torch.sum(kl_terms, dim=1)
        reference_kl = torch.sum(decision_weights * per_row_reference_kl) / weight_denom

    illegal_probability = torch.where(
        legal_masks,
        torch.zeros_like(probs),
        probs,
    ).sum(dim=1).max()
    loss = (
        policy_loss
        + float(value_weight) * value_loss
        + float(reference_kl_weight) * reference_kl
        - float(entropy_weight) * entropy
    )
    stats = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "reference_kl": float(reference_kl.detach().cpu()),
        "reference_kl_weight": float(reference_kl_weight),
        "illegal_action_probability": float(illegal_probability.detach().cpu()),
        "mean_return": float(returns.detach().mean().cpu()),
        "std_return": float(returns.detach().std(unbiased=False).cpu()),
        "advantage_target": target_mode,
        "mean_value_target": float(value_targets.detach().mean().cpu()),
        "std_value_target": float(value_targets.detach().std(unbiased=False).cpu()),
        "decision_weight_mode": str(decision_weight_mode),
        "max_decision_weight": float(max_decision_weight),
        "mean_decision_weight": float(decision_weights.detach().mean().cpu()),
        "max_observed_decision_weight": float(decision_weights.detach().max().cpu()),
    }
    return loss, stats, int(actions.numel())


def _q_expected_lambda_targets(
    *,
    batch: dict[str, np.ndarray],
    policy_logits: torch.Tensor,
    q_values: torch.Tensor,
    gamma: float,
    trace_lambda: float,
) -> torch.Tensor:
    """Expected-SARSA(lambda) targets over recorded same-player decision links."""
    rewards = torch.as_tensor(
        batch["rewards"],
        dtype=q_values.dtype,
        device=q_values.device,
    )
    if policy_logits.shape != q_values.shape:
        raise ValueError("policy_logits and q_values must have matching shapes")
    if int(rewards.numel()) != int(q_values.shape[0]):
        raise ValueError("rewards must match the number of q rows")
    legal_masks = torch.as_tensor(
        batch["legal_masks"] > 0,
        dtype=torch.bool,
        device=q_values.device,
    )
    next_decision_indices = np.asarray(batch["next_decision_indices"], dtype=np.int64)
    log_probs = masked_log_probs(policy_logits, legal_masks)
    probs = torch.exp(log_probs)
    expected_q = torch.sum(probs * torch.where(legal_masks, q_values, torch.zeros_like(q_values)), dim=1)
    targets = rewards.clone()
    lam = float(trace_lambda)
    for record_i in range(int(rewards.numel()) - 1, -1, -1):
        next_i = int(next_decision_indices[record_i])
        if next_i >= 0:
            targets[record_i] = float(gamma) * (
                (1.0 - lam) * expected_q[next_i] + lam * targets[next_i]
            )
    return targets.detach()


def _prepare_ppo_fixed_rows(
    *,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    q_net: torch.nn.Module | None = None,
    batch: dict[str, np.ndarray],
    device: torch.device,
    advantage_target: str,
    gamma: float,
    gae_lambda: float,
    decision_weight_mode: str,
    max_decision_weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float], int]:
    if int(batch["actions"].shape[0]) <= 0:
        return {}, {}, 0
    features = torch.as_tensor(batch["features"], dtype=torch.float32, device=device)
    legal_masks = torch.as_tensor(batch["legal_masks"] > 0, dtype=torch.bool, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    decision_weights = _sampled_counterfactual_decision_weights(
        batch,
        mode=str(decision_weight_mode),
        max_decision_weight=float(max_decision_weight),
        device=device,
    )
    q_modes = {"q_expected_mc", "q_expected_lambda"}
    with torch.no_grad():
        values = value_net(features).reshape(-1)
        policy_logits = policy_net(features)
        q_values = q_net(features) if q_net is not None else None
    target_mode = str(advantage_target)
    if target_mode == "terminal":
        value_targets = returns
        advantages = value_targets - values
    elif target_mode == "gae":
        value_targets, advantages = _linked_bootstrap_targets_and_advantages(
            values=values,
            terminal_returns=returns,
            batch=batch,
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
        )
    elif target_mode == "player-gae":
        value_targets, advantages = _player_perspective_bootstrap_targets_and_advantages(
            values=values,
            terminal_returns=returns,
            batch=batch,
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
        )
    elif target_mode == "q_expected_mc":
        if q_values is None:
            raise ValueError("q_expected_mc requires q_net")
        legal_q = torch.where(legal_masks, q_values, torch.zeros_like(q_values))
        legal_log_probs = masked_log_probs(policy_logits, legal_masks)
        expected_q = torch.sum(torch.exp(legal_log_probs) * legal_q, dim=1)
        value_targets = returns
        advantages = value_targets - expected_q
    elif target_mode == "q_expected_lambda":
        if q_values is None:
            raise ValueError("q_expected_lambda requires q_net")
        value_targets = _q_expected_lambda_targets(
            batch=batch,
            policy_logits=policy_logits,
            q_values=q_values,
            gamma=float(gamma),
            trace_lambda=float(gae_lambda),
        )
        legal_q = torch.where(legal_masks, q_values, torch.zeros_like(q_values))
        legal_log_probs = masked_log_probs(policy_logits, legal_masks)
        expected_q = torch.sum(torch.exp(legal_log_probs) * legal_q, dim=1)
        advantages = value_targets - expected_q
    else:
        raise ValueError("advantage_target must be one of: terminal, gae, player-gae, q_expected_mc, q_expected_lambda")
    if int(advantages.numel()) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
    rows = {
        "features": features,
        "legal_masks": legal_masks,
        "actions": actions,
        "old_log_probs": old_log_probs.detach(),
        "value_targets": value_targets.detach(),
        "advantages": advantages.detach(),
        "decision_weights": decision_weights.detach(),
        "critic_is_q": torch.as_tensor(target_mode in q_modes, dtype=torch.bool, device=device),
    }
    stats = {
        "advantage_target": target_mode,
        "mean_value_target": float(value_targets.detach().mean().cpu()),
        "std_value_target": float(value_targets.detach().std(unbiased=False).cpu()),
        "mean_return": float(returns.detach().mean().cpu()),
        "std_return": float(returns.detach().std(unbiased=False).cpu()),
        "decision_weight_mode": str(decision_weight_mode),
        "max_decision_weight": float(max_decision_weight),
        "mean_decision_weight": float(decision_weights.detach().mean().cpu()),
        "max_observed_decision_weight": float(decision_weights.detach().max().cpu()),
    }
    return rows, stats, int(actions.numel())


def _compiled_neurd_actor_loss(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    decision_weights: torch.Tensor,
) -> torch.Tensor:
    selected_legal = legal_masks.gather(1, actions.view(-1, 1)).squeeze(1)
    if bool(torch.any(~selected_legal).detach().cpu()):
        raise ValueError("NEURD actor update received an illegal sampled action")
    selected_logits = logits.gather(1, actions.view(-1, 1)).squeeze(1)
    denom = decision_weights.sum().clamp_min(1e-6)
    return -torch.sum(decision_weights * selected_logits * advantages.detach()) / denom


def _policy_value_reference_ppo_loss_on_rows(
    *,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    q_net: torch.nn.Module | None,
    reference_policy: torch.nn.Module | None,
    rows: dict[str, torch.Tensor],
    indices: torch.Tensor,
    value_weight: float,
    entropy_weight: float,
    reference_kl_weight: float,
    clip_coef: float,
    actor_update_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = rows["features"][indices]
    legal_masks = rows["legal_masks"][indices]
    actions = rows["actions"][indices]
    old_log_probs = rows["old_log_probs"][indices]
    value_targets = rows["value_targets"][indices]
    advantages = rows["advantages"][indices]
    decision_weights = rows["decision_weights"][indices]
    weight_denom = decision_weights.sum().clamp_min(1e-6)

    logits = policy_net(features)
    log_probs = masked_log_probs(logits, legal_masks)
    probs = torch.exp(log_probs)
    action_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)
    ratio = torch.exp(action_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_coef), 1.0 + float(clip_coef))
    if str(actor_update_mode) == "ppo":
        per_row_policy_loss = torch.maximum(
            -advantages * ratio,
            -advantages * clipped_ratio,
        )
        policy_loss = torch.sum(decision_weights * per_row_policy_loss) / weight_denom
    elif str(actor_update_mode) == "neurd":
        policy_loss = _compiled_neurd_actor_loss(
            logits,
            legal_masks,
            actions,
            advantages,
            decision_weights,
        )
    else:
        raise ValueError("actor_update_mode must be one of: ppo, neurd")
    critic_is_q = bool(rows.get("critic_is_q", torch.tensor(False)).detach().cpu())
    if critic_is_q:
        if q_net is None:
            raise ValueError("q_expected PPO rows require q_net")
        q_values = q_net(features)
        q_taken = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
        value_loss = torch.sum(decision_weights * torch.square(q_taken - value_targets)) / weight_denom
    else:
        values = value_net(features).reshape(-1)
        value_loss = torch.sum(decision_weights * torch.square(values - value_targets)) / weight_denom
    per_row_entropy = -torch.sum(probs * torch.nan_to_num(log_probs, neginf=0.0), dim=1)
    entropy = torch.sum(decision_weights * per_row_entropy) / weight_denom

    reference_kl = torch.zeros((), dtype=torch.float32, device=features.device)
    if reference_policy is not None and float(reference_kl_weight) > 0.0:
        with torch.no_grad():
            reference_logits = reference_policy(features)
            reference_log_probs = masked_log_probs(reference_logits, legal_masks)
        kl_terms = probs * (
            torch.nan_to_num(log_probs, neginf=0.0)
            - torch.nan_to_num(reference_log_probs, neginf=0.0)
        )
        kl_terms = torch.where(legal_masks, kl_terms, torch.zeros_like(kl_terms))
        per_row_reference_kl = torch.sum(kl_terms, dim=1)
        reference_kl = torch.sum(decision_weights * per_row_reference_kl) / weight_denom

    illegal_probability = torch.where(
        legal_masks,
        torch.zeros_like(probs),
        probs,
    ).sum(dim=1).max()
    loss = (
        policy_loss
        + float(value_weight) * value_loss
        + float(reference_kl_weight) * reference_kl
        - float(entropy_weight) * entropy
    )
    with torch.no_grad():
        per_row_approx_kl = (ratio - 1.0) - torch.log(ratio.clamp_min(1e-9))
        approx_kl = torch.sum(decision_weights * per_row_approx_kl) / weight_denom
        clip_fraction = (
            (torch.abs(ratio - 1.0) > float(clip_coef))
            .to(dtype=ratio.dtype)
            * decision_weights
        ).sum() / weight_denom
    stats = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "reference_kl": float(reference_kl.detach().cpu()),
        "reference_kl_weight": float(reference_kl_weight),
        "illegal_action_probability": float(illegal_probability.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "mean_decision_weight": float(decision_weights.detach().mean().cpu()),
        "max_observed_decision_weight": float(decision_weights.detach().max().cpu()),
        "actor_update_mode": str(actor_update_mode),
        "uses_q_critic": bool(critic_is_q),
    }
    return loss, stats


@torch.no_grad()
def _policy_snapshot_for_kl(
    policy_net: torch.nn.Module,
    features: torch.Tensor,
    legal_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = masked_log_probs(policy_net(features), legal_masks)
    probs = torch.exp(log_probs)
    return probs.detach(), log_probs.detach()


def _masked_mean_policy_kl(
    *,
    old_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    if old_probs.shape != old_log_probs.shape or old_probs.shape != new_log_probs.shape:
        raise ValueError("policy tensors must have matching shapes")
    if old_probs.shape != legal_masks.shape:
        raise ValueError("legal_masks must match policy tensor shape")
    legal = legal_masks.to(dtype=torch.bool, device=old_probs.device)
    kl_terms = old_probs * (
        torch.nan_to_num(old_log_probs, neginf=0.0)
        - torch.nan_to_num(new_log_probs, neginf=0.0)
    )
    kl_terms = torch.where(legal, kl_terms, torch.zeros_like(kl_terms))
    return torch.sum(kl_terms, dim=1).mean()


def _optimizer_step_with_optional_policy_kl_control(
    *,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    q_net: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor | None,
    loss_builder: Callable[[], torch.Tensor] | None = None,
    features: torch.Tensor,
    legal_masks: torch.Tensor,
    max_policy_kl: float,
    max_backtracks: int,
    backtrack_factor: float,
    grad_clip_norm: float = 10.0,
) -> dict[str, float | bool | int | None]:
    """Apply one optimizer step, optionally backtracking on realized policy KL."""
    parameters = list(policy_net.parameters()) + list(value_net.parameters())
    if q_net is not None:
        parameters += list(q_net.parameters())
    if loss is None and loss_builder is None:
        raise ValueError("loss or loss_builder must be provided")
    if float(max_policy_kl) <= 0.0:
        optimizer.zero_grad(set_to_none=True)
        step_loss = loss if loss is not None else loss_builder()
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip_norm))
        optimizer.step()
        return {
            "policy_kl_control_enabled": False,
            "policy_update_kl": None,
            "policy_update_backtracks": 0,
            "policy_update_lr_scale": 1.0,
            "policy_update_accepted": True,
            "policy_update_skipped": False,
        }

    if int(max_backtracks) < 0:
        raise ValueError("max_backtracks must be non-negative")
    if not (0.0 < float(backtrack_factor) < 1.0):
        raise ValueError("backtrack_factor must be in (0, 1)")
    if loss_builder is None and int(max_backtracks) > 0:
        raise ValueError("loss_builder is required when policy-KL backtracking may retry")

    old_probs, old_log_probs = _policy_snapshot_for_kl(policy_net, features, legal_masks)
    policy_state = copy.deepcopy(policy_net.state_dict())
    value_state = copy.deepcopy(value_net.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    accepted = False
    accepted_kl: float | None = None
    accepted_backtracks = 0
    accepted_scale = 1.0

    for backtrack_i in range(int(max_backtracks) + 1):
        scale = float(backtrack_factor) ** int(backtrack_i)
        if loss_builder is not None or backtrack_i > 0:
            policy_net.load_state_dict(policy_state)
            value_net.load_state_dict(value_state)
            optimizer.load_state_dict(optimizer_state)
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * scale
        step_loss = loss_builder() if loss_builder is not None else loss
        if step_loss is None:
            raise ValueError("loss_builder returned None")
        optimizer.zero_grad(set_to_none=True)
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip_norm))
        optimizer.step()
        with torch.no_grad():
            new_log_probs = masked_log_probs(policy_net(features), legal_masks)
            step_kl = _masked_mean_policy_kl(
                old_probs=old_probs,
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                legal_masks=legal_masks,
            )
            step_kl_float = float(step_kl.detach().cpu())
        if step_kl_float <= float(max_policy_kl):
            accepted = True
            accepted_kl = step_kl_float
            accepted_backtracks = int(backtrack_i)
            accepted_scale = float(scale)
            break

    if not accepted:
        policy_net.load_state_dict(policy_state)
        value_net.load_state_dict(value_state)
        optimizer.load_state_dict(optimizer_state)
        accepted_backtracks = int(max_backtracks)
        accepted_scale = float(backtrack_factor) ** int(max_backtracks)

    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = base_lr

    return {
        "policy_kl_control_enabled": True,
        "policy_update_kl": accepted_kl,
        "policy_update_backtracks": accepted_backtracks,
        "policy_update_lr_scale": accepted_scale,
        "policy_update_accepted": bool(accepted),
        "policy_update_skipped": not bool(accepted),
    }


def _linked_bootstrap_targets_and_advantages(
    *,
    values: torch.Tensor,
    terminal_returns: torch.Tensor,
    batch: dict[str, np.ndarray],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE-style targets over same-player decision links."""
    targets = torch.zeros_like(terminal_returns)
    advantages = torch.zeros_like(terminal_returns)
    grouped: dict[tuple[int, int], list[int]] = {}
    for record_i, (game_i, player_i) in enumerate(
        zip(batch["game_indices"], batch["players"], strict=False)
    ):
        grouped.setdefault((int(game_i), int(player_i)), []).append(int(record_i))

    next_decision_indices = batch["next_decision_indices"]
    step_indices = batch["step_indices"]
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda idx: int(step_indices[idx]), reverse=True)
        for record_i in ordered:
            next_i = int(next_decision_indices[record_i])
            if next_i >= 0:
                delta = float(gamma) * values[next_i] - values[record_i]
                advantages[record_i] = (
                    delta
                    + float(gamma)
                    * float(gae_lambda)
                    * advantages[next_i]
                )
                targets[record_i] = advantages[record_i] + values[record_i]
            else:
                advantages[record_i] = terminal_returns[record_i] - values[record_i]
                targets[record_i] = terminal_returns[record_i]
    return targets.detach(), advantages.detach()


def _player_perspective_bootstrap_targets_and_advantages(
    *,
    values: torch.Tensor,
    terminal_returns: torch.Tensor,
    batch: dict[str, np.ndarray],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over alternating turns in each actor's value perspective."""
    targets = torch.zeros_like(terminal_returns)
    advantages = torch.zeros_like(terminal_returns)
    game_indices = np.asarray(batch["game_indices"], dtype=np.int64)
    players = np.asarray(batch["players"], dtype=np.int64)
    step_indices = np.asarray(batch["step_indices"], dtype=np.int64)
    grouped: dict[int, list[int]] = {}
    for record_i, game_i in enumerate(game_indices):
        grouped.setdefault(int(game_i), []).append(int(record_i))

    for indices in grouped.values():
        ordered = sorted(indices, key=lambda idx: int(step_indices[idx]), reverse=True)
        next_i = -1
        for record_i in ordered:
            if next_i >= 0:
                same_player = int(players[next_i]) == int(players[record_i])
                sign = 1.0 if same_player else -1.0
                delta = float(gamma) * sign * values[next_i] - values[record_i]
                advantages[record_i] = (
                    delta
                    + float(gamma)
                    * float(gae_lambda)
                    * sign
                    * advantages[next_i]
                )
                targets[record_i] = advantages[record_i] + values[record_i]
            else:
                advantages[record_i] = terminal_returns[record_i] - values[record_i]
                targets[record_i] = terminal_returns[record_i]
            next_i = int(record_i)
    return targets.detach(), advantages.detach()


def run_learner(
    *,
    train_iterations: int = 8,
    games_per_iteration: int = 512,
    collector_batch_size: int = 128,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    value_weight: float = 0.5,
    entropy_weight: float = 0.02,
    reference_kl_weight: float = 0.05,
    reference_update_every: int = 4,
    inner_update: str = "pg",
    actor_update_mode: str = "ppo",
    ppo_epochs: int = 4,
    ppo_minibatches: int = 4,
    clip_coef: float = 0.1,
    adaptive_policy_kl_target: float = 0.0,
    adaptive_policy_kl_max_backtracks: int = 4,
    adaptive_policy_kl_backtrack_factor: float = 0.5,
    advantage_target: str = "terminal",
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    decision_weight_mode: str = "uniform",
    max_decision_weight: float = 64.0,
    seed: int = 20260661,
    device: str = "auto",
    checkpoint_in: str | Path | None = None,
    opponent_checkpoints: list[str | Path] | None = None,
    reference_policy_checkpoint: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    if int(train_iterations) <= 0:
        raise ValueError("train_iterations must be positive")
    if int(games_per_iteration) <= 0:
        raise ValueError("games_per_iteration must be positive")
    if int(reference_update_every) <= 0:
        raise ValueError("reference_update_every must be positive")
    if float(reference_kl_weight) < 0.0:
        raise ValueError("reference_kl_weight must be non-negative")
    if str(inner_update) not in {"pg", "ppo"}:
        raise ValueError("inner_update must be one of: pg, ppo")
    if str(actor_update_mode) not in {"ppo", "neurd"}:
        raise ValueError("actor_update_mode must be one of: ppo, neurd")
    if str(actor_update_mode) != "ppo" and str(inner_update) != "ppo":
        raise ValueError("actor_update_mode is only used with inner_update=ppo")
    if int(ppo_epochs) <= 0:
        raise ValueError("ppo_epochs must be positive")
    if int(ppo_minibatches) <= 0:
        raise ValueError("ppo_minibatches must be positive")
    if float(clip_coef) <= 0.0:
        raise ValueError("clip_coef must be positive")
    if float(adaptive_policy_kl_target) < 0.0:
        raise ValueError("adaptive_policy_kl_target must be non-negative")
    if str(inner_update) == "ppo" and float(adaptive_policy_kl_target) > 0.0:
        raise ValueError("adaptive_policy_kl_target is supported only for inner_update=pg")
    if int(adaptive_policy_kl_max_backtracks) < 0:
        raise ValueError("adaptive_policy_kl_max_backtracks must be non-negative")
    if not (0.0 < float(adaptive_policy_kl_backtrack_factor) < 1.0):
        raise ValueError("adaptive_policy_kl_backtrack_factor must be in (0, 1)")
    q_advantage_targets = {"q_expected_mc", "q_expected_lambda"}
    if str(advantage_target) not in {"terminal", "gae", "player-gae", *q_advantage_targets}:
        raise ValueError("advantage_target must be one of: terminal, gae, player-gae, q_expected_mc, q_expected_lambda")
    if str(advantage_target) in q_advantage_targets and str(inner_update) != "ppo":
        raise ValueError("q_expected_* advantage targets require inner_update=ppo")
    if not (0.0 <= float(gamma) <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if not (0.0 <= float(gae_lambda) <= 1.0):
        raise ValueError("gae_lambda must be in [0, 1]")
    if str(decision_weight_mode) not in {"uniform", "inverse-own-reach"}:
        raise ValueError("decision_weight_mode must be one of: uniform, inverse-own-reach")
    if float(max_decision_weight) <= 0.0:
        raise ValueError("max_decision_weight must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    policy_net = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    value_net = _ValueMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    q_net = (
        _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
        if str(advantage_target) in q_advantage_targets
        else None
    )
    checkpoint_payload = _load_checkpoint_in(
        checkpoint_in=checkpoint_in,
        policy_net=policy_net,
        value_net=value_net,
        q_net=q_net,
        device=resolved_device,
    )
    opponent_policies, opponent_kinds = _load_compiled_rollout_opponents(
        opponent_checkpoints,
        resolved_device,
    )

    loaded_reference, reference_kind = _load_reference_policy(
        reference_policy_checkpoint,
        resolved_device,
    )
    moving_reference = loaded_reference is None and float(reference_kl_weight) > 0.0
    if moving_reference:
        reference_policy = copy.deepcopy(policy_net).to(resolved_device)
        reference_kind = "moving-self"
    else:
        reference_policy = loaded_reference
    if reference_policy is not None:
        reference_policy.eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)

    optimizer_params = list(policy_net.parameters()) + list(value_net.parameters())
    if q_net is not None:
        optimizer_params += list(q_net.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=float(lr))

    losses: list[float] = []
    entropy_values: list[float] = []
    reference_kls: list[float] = []
    illegal_probabilities: list[float] = []
    policy_update_kls: list[float] = []
    policy_update_backtracks: list[int] = []
    policy_update_lr_scales: list[float] = []
    mean_decision_weights: list[float] = []
    max_observed_decision_weights: list[float] = []
    policy_update_accepted = 0
    policy_update_skipped = 0
    n_samples = 0
    collector_steps = 0
    collector_total_env_steps = 0
    collector_seconds = 0.0
    compiled_needs_python_showdown = 0
    reference_updates = 0
    train_start = time.perf_counter()

    for iteration_i in range(int(train_iterations)):
        batch, collector_metrics = _collect_compiled_fast_policy_gradient_rollout(
            policy_net,
            n_games=int(games_per_iteration),
            batch_size=int(collector_batch_size),
            max_steps_per_game=int(max_steps_per_game),
            initial_chips=int(initial_chips),
            device=resolved_device,
            seed=int(seed) + iteration_i * 10_000,
            opponent_policies=opponent_policies if opponent_policies else None,
        )
        collector_steps += int(collector_metrics.get("steps", 0))
        collector_total_env_steps += int(collector_metrics.get("total_env_steps", 0))
        collector_seconds += float(collector_metrics.get("seconds", 0.0))
        compiled_needs_python_showdown += int(collector_metrics.get("needs_python_showdown", 0))

        if str(inner_update) == "ppo":
            rows, base_stats, batch_samples = _prepare_ppo_fixed_rows(
                policy_net=policy_net,
                value_net=value_net,
                q_net=q_net,
                batch=batch,
                device=resolved_device,
                advantage_target=str(advantage_target),
                gamma=float(gamma),
                gae_lambda=float(gae_lambda),
                decision_weight_mode=str(decision_weight_mode),
                max_decision_weight=float(max_decision_weight),
            )
            if batch_samples <= 0:
                continue
            minibatches = max(1, min(int(ppo_minibatches), int(batch_samples)))
            last_stats: dict[str, float] = {}
            for _epoch_i in range(int(ppo_epochs)):
                order = np.random.permutation(int(batch_samples))
                for mb_np in np.array_split(order, minibatches):
                    if mb_np.size <= 0:
                        continue
                    mb = torch.as_tensor(mb_np, dtype=torch.long, device=resolved_device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, last_stats = _policy_value_reference_ppo_loss_on_rows(
                        policy_net=policy_net,
                        value_net=value_net,
                        q_net=q_net,
                        reference_policy=reference_policy,
                        rows=rows,
                        indices=mb,
                        value_weight=float(value_weight),
                        entropy_weight=float(entropy_weight),
                        reference_kl_weight=float(reference_kl_weight),
                        clip_coef=float(clip_coef),
                        actor_update_mode=str(actor_update_mode),
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(optimizer_params, 10.0)
                    optimizer.step()
            stats = {**base_stats, **last_stats}
            step_stats = {
                "policy_update_kl": None,
                "policy_update_backtracks": 0,
                "policy_update_lr_scale": 1.0,
                "policy_update_accepted": True,
                "policy_update_skipped": False,
            }
        else:
            loss, stats, batch_samples = _policy_value_reference_loss(
                policy_net=policy_net,
                value_net=value_net,
                reference_policy=reference_policy,
                batch=batch,
                device=resolved_device,
                value_weight=float(value_weight),
                entropy_weight=float(entropy_weight),
                reference_kl_weight=float(reference_kl_weight),
                advantage_target=str(advantage_target),
                gamma=float(gamma),
                gae_lambda=float(gae_lambda),
                decision_weight_mode=str(decision_weight_mode),
                max_decision_weight=float(max_decision_weight),
            )
            if batch_samples <= 0:
                continue
            step_features = torch.as_tensor(batch["features"], dtype=torch.float32, device=resolved_device)
            step_legal_masks = torch.as_tensor(
                batch["legal_masks"] > 0,
                dtype=torch.bool,
                device=resolved_device,
            )

            def _rebuild_step_loss() -> torch.Tensor:
                rebuilt_loss, _rebuilt_stats, _rebuilt_samples = _policy_value_reference_loss(
                    policy_net=policy_net,
                    value_net=value_net,
                    reference_policy=reference_policy,
                    batch=batch,
                    device=resolved_device,
                    value_weight=float(value_weight),
                    entropy_weight=float(entropy_weight),
                    reference_kl_weight=float(reference_kl_weight),
                    advantage_target=str(advantage_target),
                    gamma=float(gamma),
                    gae_lambda=float(gae_lambda),
                    decision_weight_mode=str(decision_weight_mode),
                    max_decision_weight=float(max_decision_weight),
                )
                return rebuilt_loss

            step_stats = _optimizer_step_with_optional_policy_kl_control(
                policy_net=policy_net,
                value_net=value_net,
                q_net=q_net,
                optimizer=optimizer,
                loss=loss,
                loss_builder=_rebuild_step_loss,
                features=step_features,
                legal_masks=step_legal_masks,
                max_policy_kl=float(adaptive_policy_kl_target),
                max_backtracks=int(adaptive_policy_kl_max_backtracks),
                backtrack_factor=float(adaptive_policy_kl_backtrack_factor),
            )

        losses.append(float(stats["loss"]))
        entropy_values.append(float(stats["entropy"]))
        reference_kls.append(float(stats["reference_kl"]))
        illegal_probabilities.append(float(stats["illegal_action_probability"]))
        if "mean_decision_weight" in stats:
            mean_decision_weights.append(float(stats["mean_decision_weight"]))
        if "max_observed_decision_weight" in stats:
            max_observed_decision_weights.append(float(stats["max_observed_decision_weight"]))
        if step_stats["policy_update_kl"] is not None:
            policy_update_kls.append(float(step_stats["policy_update_kl"]))
        policy_update_backtracks.append(int(step_stats["policy_update_backtracks"]))
        policy_update_lr_scales.append(float(step_stats["policy_update_lr_scale"]))
        if bool(step_stats["policy_update_accepted"]):
            policy_update_accepted += 1
        if bool(step_stats["policy_update_skipped"]):
            policy_update_skipped += 1
        n_samples += int(batch_samples)

        if moving_reference and (iteration_i + 1) % int(reference_update_every) == 0:
            reference_policy.load_state_dict(policy_net.state_dict())
            reference_updates += 1

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    max_illegal_probability = float(max(illegal_probabilities) if illegal_probabilities else 0.0)
    checkpoint_path = None if checkpoint_out is None else str(Path(checkpoint_out))
    metrics: dict[str, Any] = {
        "algorithm": "native_neural_nashpg_compiled",
        "role": "native_full_deck_reference_regularized_policy_gradient",
        "gate": "native_neural_nashpg_compiled_smoke",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Native 9-action tabula-rasa learner smoke; not Slumbot strength evidence.",
        **device_info,
        "seed": int(seed),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "train_iterations": int(train_iterations),
        "games_per_iteration": int(games_per_iteration),
        "collector_batch_size": int(collector_batch_size),
        "max_steps_per_game": int(max_steps_per_game),
        "initial_chips": int(initial_chips),
        "hidden_dim": int(hidden_dim),
        "lr": float(lr),
        "value_weight": float(value_weight),
        "entropy_weight": float(entropy_weight),
        "reference_kl_weight": float(reference_kl_weight),
        "reference_update_every": int(reference_update_every),
        "inner_update": str(inner_update),
        "actor_update_mode": str(actor_update_mode),
        "ppo_epochs": int(ppo_epochs),
        "ppo_minibatches": int(ppo_minibatches),
        "clip_coef": float(clip_coef),
        "adaptive_policy_kl_enabled": bool(float(adaptive_policy_kl_target) > 0.0),
        "adaptive_policy_kl_target": float(adaptive_policy_kl_target),
        "adaptive_policy_kl_max_backtracks": int(adaptive_policy_kl_max_backtracks),
        "adaptive_policy_kl_backtrack_factor": float(adaptive_policy_kl_backtrack_factor),
        "policy_update_accepted_steps": int(policy_update_accepted),
        "policy_update_skipped_steps": int(policy_update_skipped),
        "mean_policy_update_kl": float(np.mean(policy_update_kls)) if policy_update_kls else None,
        "max_policy_update_kl": float(max(policy_update_kls)) if policy_update_kls else None,
        "mean_policy_update_backtracks": (
            float(np.mean(policy_update_backtracks)) if policy_update_backtracks else None
        ),
        "max_policy_update_backtracks": (
            int(max(policy_update_backtracks)) if policy_update_backtracks else 0
        ),
        "mean_policy_update_lr_scale": (
            float(np.mean(policy_update_lr_scales)) if policy_update_lr_scales else None
        ),
        "advantage_target": str(advantage_target),
        "uses_q_critic": bool(q_net is not None),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "decision_weight_mode": str(decision_weight_mode),
        "max_decision_weight": float(max_decision_weight),
        "mean_decision_weight": (
            float(np.mean(mean_decision_weights)) if mean_decision_weights else None
        ),
        "max_observed_decision_weight": (
            float(max(max_observed_decision_weights)) if max_observed_decision_weights else None
        ),
        "moving_reference": bool(moving_reference),
        "reference_updates": int(reference_updates),
        "reference_policy_checkpoint": (
            str(reference_policy_checkpoint) if reference_policy_checkpoint is not None else None
        ),
        "reference_policy_kind": reference_kind,
        "checkpoint_in": str(checkpoint_in) if checkpoint_in is not None else None,
        "checkpoint_in_algorithm": (
            str(checkpoint_payload.get("algorithm", "")) if checkpoint_payload is not None else None
        ),
        "checkpoint_in_q_loaded": bool(
            checkpoint_payload.get("checkpoint_in_q_loaded", False)
            if checkpoint_payload is not None
            else False
        ),
        "fresh_initial_policy": checkpoint_payload is None,
        "population_training": bool(opponent_policies),
        "opponent_kind": _summarize_opponent_kinds(opponent_kinds),
        "opponent_kinds": list(opponent_kinds),
        "opponent_population_size": int(len(opponent_policies)),
        "opponent_checkpoints": [str(path) for path in (opponent_checkpoints or [])],
        "collector_backend": "compiled-fast-state",
        "collector_steps": int(collector_steps),
        "collector_total_env_steps": int(collector_total_env_steps),
        "collector_seconds": float(collector_seconds),
        "compiled_needs_python_showdown": int(compiled_needs_python_showdown),
        "n_samples": int(n_samples),
        "train_seconds": float(train_seconds),
        "samples_per_second": float(n_samples / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "loss_is_finite": bool(losses and np.isfinite(losses[-1])),
        "mean_entropy": float(np.mean(entropy_values)) if entropy_values else None,
        "last_entropy": float(entropy_values[-1]) if entropy_values else None,
        "mean_reference_kl": float(np.mean(reference_kls)) if reference_kls else None,
        "last_reference_kl": float(reference_kls[-1]) if reference_kls else None,
        "illegal_action_probability": max_illegal_probability,
        "trained_environment_native": True,
        "native_action_projection": False,
        "rlcard_candidate": False,
        "uses_slumbot_data": False,
        "uses_slumbot_training_data": False,
        "uses_alphanlholdem_training_data": False,
        "uses_solver_labels": False,
        "promotion": False,
        "checkpoint_path": checkpoint_path,
    }
    metrics["passed"] = bool(
        losses
        and np.isfinite(losses[-1])
        and max_illegal_probability == 0.0
        and compiled_needs_python_showdown == 0
        and n_samples > 0
    )

    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "native_neural_nashpg_compiled",
                "environment": "poker_ai:full_deck_hu_nlhe",
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "policy_net_state_dict": policy_net.state_dict(),
                "value_net_state_dict": value_net.state_dict(),
                **({"q_net_state_dict": q_net.state_dict()} if q_net is not None else {}),
                "config": {
                    "feature_mode": "flat",
                    "hidden_dim": int(hidden_dim),
                    "initial_chips": int(initial_chips),
                    "max_steps_per_hand": int(max_steps_per_game),
                    "rollout_backend": "compiled-fast-state",
                    "fsp_average_policy": False,
                    "train_environment": "poker_ai:full_deck_hu_nlhe",
                    "reference_policy_checkpoint": (
                        str(reference_policy_checkpoint)
                        if reference_policy_checkpoint is not None
                        else None
                    ),
                    "reference_policy_kind": reference_kind,
                    "reference_kl_weight": float(reference_kl_weight),
                    "reference_update_every": int(reference_update_every),
                    "inner_update": str(inner_update),
                    "actor_update_mode": str(actor_update_mode),
                    "ppo_epochs": int(ppo_epochs),
                    "ppo_minibatches": int(ppo_minibatches),
                    "clip_coef": float(clip_coef),
                    "adaptive_policy_kl_enabled": bool(float(adaptive_policy_kl_target) > 0.0),
                    "adaptive_policy_kl_target": float(adaptive_policy_kl_target),
                    "adaptive_policy_kl_max_backtracks": int(adaptive_policy_kl_max_backtracks),
                    "adaptive_policy_kl_backtrack_factor": float(adaptive_policy_kl_backtrack_factor),
                    "advantage_target": str(advantage_target),
                    "uses_q_critic": bool(q_net is not None),
                    "gamma": float(gamma),
                    "gae_lambda": float(gae_lambda),
                    "decision_weight_mode": str(decision_weight_mode),
                    "max_decision_weight": float(max_decision_weight),
                    "moving_reference": bool(moving_reference),
                    "opponent_population_size": int(len(opponent_policies)),
                    "opponent_kinds": list(opponent_kinds),
                },
                "metrics": metrics,
                "trained_environment_native": True,
                "native_action_projection": False,
                "rlcard_candidate": False,
                "uses_slumbot_data": False,
                "uses_slumbot_training_data": False,
                "uses_alphanlholdem_training_data": False,
                "uses_solver_labels": False,
            },
            path,
        )
    return _write_metrics(metrics, output_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iterations", type=int, default=8)
    parser.add_argument("--games-per-iteration", type=int, default=512)
    parser.add_argument("--collector-batch-size", type=int, default=128)
    parser.add_argument("--max-steps-per-game", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument("--reference-update-every", type=int, default=4)
    parser.add_argument("--inner-update", choices=("pg", "ppo"), default="pg")
    parser.add_argument(
        "--actor-update-mode",
        choices=("ppo", "neurd"),
        default="ppo",
        help="Actor update for inner_update=ppo. NEURD applies a direct legal-logit advantage update.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatches", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument(
        "--adaptive-policy-kl-target",
        type=float,
        default=0.0,
        help="Opt-in realized legal-action policy-KL budget for adaptive proximal update control. 0 disables.",
    )
    parser.add_argument("--adaptive-policy-kl-max-backtracks", type=int, default=4)
    parser.add_argument("--adaptive-policy-kl-backtrack-factor", type=float, default=0.5)
    parser.add_argument(
        "--advantage-target",
        choices=("terminal", "gae", "player-gae", "q_expected_mc", "q_expected_lambda"),
        default="terminal",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--decision-weight-mode",
        choices=("uniform", "inverse-own-reach"),
        default="uniform",
        help=(
            "Opt-in sampled counterfactual row weighting. inverse-own-reach removes "
            "prior same-player sampled action reach from later decision feedback."
        ),
    )
    parser.add_argument("--max-decision-weight", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=20260661)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument(
        "--opponent-checkpoint",
        action="append",
        dest="opponent_checkpoints",
        help="Frozen native-ppo/Rainbow opponent checkpoint. May be supplied multiple times.",
    )
    parser.add_argument(
        "--reference-policy-checkpoint",
        type=Path,
        help="Frozen local policy checkpoint used as the reference policy.",
    )
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_learner(
        train_iterations=args.train_iterations,
        games_per_iteration=args.games_per_iteration,
        collector_batch_size=args.collector_batch_size,
        max_steps_per_game=args.max_steps_per_game,
        initial_chips=args.initial_chips,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        value_weight=args.value_weight,
        entropy_weight=args.entropy_weight,
        reference_kl_weight=args.reference_kl_weight,
        reference_update_every=args.reference_update_every,
        inner_update=args.inner_update,
        actor_update_mode=args.actor_update_mode,
        ppo_epochs=args.ppo_epochs,
        ppo_minibatches=args.ppo_minibatches,
        clip_coef=args.clip_coef,
        adaptive_policy_kl_target=args.adaptive_policy_kl_target,
        adaptive_policy_kl_max_backtracks=args.adaptive_policy_kl_max_backtracks,
        adaptive_policy_kl_backtrack_factor=args.adaptive_policy_kl_backtrack_factor,
        advantage_target=args.advantage_target,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        decision_weight_mode=args.decision_weight_mode,
        max_decision_weight=args.max_decision_weight,
        seed=args.seed,
        device=args.device,
        checkpoint_in=args.checkpoint_in,
        opponent_checkpoints=args.opponent_checkpoints,
        reference_policy_checkpoint=args.reference_policy_checkpoint,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
