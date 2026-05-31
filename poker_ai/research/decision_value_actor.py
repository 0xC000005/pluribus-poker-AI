"""Decision-value-aware actor updates for search-improved policies."""

from __future__ import annotations

from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from poker_ai.deep_cfr.fast_state import FastPokerState
from poker_ai.deep_cfr.policy_targets import (
    PolicyTargetBuffer,
    masked_policy_cross_entropy,
)
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.policy_calibration import train_policy_head_calibration
from poker_ai.research.public_action_rollout_value import (
    PublicActionRolloutConfig,
    _action_name,
    _payoff_after_rollout,
    _policy_for_root,
    build_public_world_state,
    checkpoint_continuation_policy,
    checkpoint_population_continuation_policy,
    evaluate_public_action_rollout_values,
    masked_uniform_policy,
    sample_public_worlds,
    sample_seeded_hero_cards,
    score_first_actions_across_worlds,
    select_deployed_root_action,
)


@dataclass(frozen=True)
class DecisionValueQTargetBatch:
    features: torch.Tensor
    legal_masks: torch.Tensor
    action_values: torch.Tensor
    base_probs: torch.Tensor
    weights: torch.Tensor


class DecisionValueQTargetBuffer:
    """In-memory policy-improvement targets with per-action search values."""

    def __init__(
        self,
        features: np.ndarray,
        legal_masks: np.ndarray,
        action_values: np.ndarray,
        base_probs: np.ndarray,
        weights: np.ndarray | None = None,
    ):
        features = np.asarray(features, dtype=np.float32)
        legal_masks = (np.asarray(legal_masks, dtype=np.float32) > 0).astype(np.float32)
        action_values = np.asarray(action_values, dtype=np.float32)
        base_probs = np.asarray(base_probs, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise ValueError(f"features must have shape (N, {N_FEATURES})")
        if legal_masks.shape != (features.shape[0], N_ACTIONS):
            raise ValueError(f"legal_masks must have shape (N, {N_ACTIONS})")
        if action_values.shape != (features.shape[0], N_ACTIONS):
            raise ValueError(f"action_values must have shape (N, {N_ACTIONS})")
        if base_probs.shape != (features.shape[0], N_ACTIONS):
            raise ValueError(f"base_probs must have shape (N, {N_ACTIONS})")
        if np.any(legal_masks.sum(axis=1) <= 0):
            raise ValueError("each Q target needs at least one legal action")
        legal = legal_masks > 0
        if not np.all(np.isfinite(action_values[legal])):
            raise ValueError("legal action_values must be finite")
        action_values = np.where(legal, action_values, 0.0).astype(np.float32)
        base_probs = np.where(legal, np.clip(base_probs, 1e-8, None), 0.0)
        base_totals = base_probs.sum(axis=1, keepdims=True)
        legal_totals = legal_masks.sum(axis=1, keepdims=True)
        uniform = legal_masks / np.maximum(legal_totals, 1.0)
        base_probs = np.where(base_totals > 1e-8, base_probs / base_totals, uniform)
        if weights is None:
            weights = np.ones(features.shape[0], dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)
            if weights.shape != (features.shape[0],):
                raise ValueError("weights must have shape (N,)")

        self.features = features
        self.legal_masks = legal_masks
        self.action_values = action_values.astype(np.float32)
        self.base_probs = base_probs.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.size = int(features.shape[0])

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> DecisionValueQTargetBatch:
        if self.size <= 0:
            raise ValueError("empty decision-value Q target buffer")
        n = min(int(batch_size), self.size)
        indices = np.random.randint(0, self.size, size=n)
        batch = DecisionValueQTargetBatch(
            features=torch.from_numpy(self.features[indices]),
            legal_masks=torch.from_numpy(self.legal_masks[indices]),
            action_values=torch.from_numpy(self.action_values[indices]),
            base_probs=torch.from_numpy(self.base_probs[indices]),
            weights=torch.from_numpy(self.weights[indices]),
        )
        if device is None:
            return batch
        return DecisionValueQTargetBatch(
            features=batch.features.to(device),
            legal_masks=batch.legal_masks.to(device),
            action_values=batch.action_values.to(device),
            base_probs=batch.base_probs.to(device),
            weights=batch.weights.to(device),
        )


@dataclass(frozen=True)
class DecisionValueTargetConfig:
    checkpoint: str
    strategy_source: str = "average-policy"
    continuation_checkpoints: tuple[str, ...] = ()
    continuation_strategy_source: str | None = None
    greedy_continuation: bool = True
    n_roots: int = 32
    n_worlds: int = 32
    initial_chips: int = 1000
    small_blind: int = 50
    big_blind: int = 100
    max_steps_per_hand: int = 128
    seed: int = 20260525
    eta: float = 1.0
    min_prior: float = 1e-6
    device: str = "auto"


@dataclass(frozen=True)
class OnPolicyDecisionValueTargetConfig:
    checkpoint: str
    strategy_source: str = "average-policy"
    continuation_checkpoints: tuple[str, ...] = ()
    continuation_strategy_source: str | None = None
    greedy_continuation: bool = True
    state_policy_greedy: bool = False
    behavior_policy: str = "base"
    search_improved_behavior_greedy: bool = True
    n_states: int = 64
    n_worlds: int = 32
    target_streets: tuple[int, ...] = (0, 1, 2, 3)
    required_streets: tuple[int, ...] = ()
    min_rows_per_required_street: int = 1
    initial_chips: int = 1000
    small_blind: int = 50
    big_blind: int = 100
    max_steps_per_hand: int = 128
    max_hands: int = 256
    seed: int = 20260525
    eta: float = 1.0
    min_prior: float = 1e-6
    device: str = "auto"


def regularized_action_value_target(
    prior: np.ndarray,
    legal_mask: np.ndarray,
    action_values: np.ndarray,
    *,
    eta: float = 1.0,
    min_prior: float = 1e-6,
) -> np.ndarray:
    """Return a legal mirror-descent target from prior policy and action values."""
    prior_arr = np.asarray(prior, dtype=np.float64)
    legal_arr = np.asarray(legal_mask, dtype=np.float64) > 0
    values = np.asarray(action_values, dtype=np.float64)
    if prior_arr.shape[0] != legal_arr.shape[0] or values.shape[0] != legal_arr.shape[0]:
        raise ValueError("prior, legal_mask, and action_values must have the same length")
    if not np.any(legal_arr):
        raise ValueError("legal_mask must contain at least one legal action")

    legal_values = np.where(legal_arr, values, np.nan)
    finite_legal = legal_arr & np.isfinite(legal_values)
    if not np.any(finite_legal):
        raise ValueError("at least one legal action value must be finite")
    legal_arr = finite_legal

    legal_prior = np.clip(prior_arr[legal_arr], float(min_prior), None)
    legal_prior = legal_prior / legal_prior.sum()
    legal_q = values[legal_arr]
    baseline = float(np.dot(legal_prior, legal_q))
    centered = legal_q - baseline
    scale = float(np.std(legal_q))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = max(float(np.max(np.abs(centered))), 1.0)
    logits = np.log(legal_prior) + float(eta) * centered / scale
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    probs = probs / probs.sum()

    target = np.zeros_like(prior_arr, dtype=np.float32)
    target[legal_arr] = probs.astype(np.float32)
    return target


def select_decision_value_behavior_action(
    prior: np.ndarray,
    target: np.ndarray,
    legal_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    behavior_policy: str = "base",
    greedy: bool = True,
) -> int:
    """Select a behavior action from the base prior or search-improved target."""
    legal = np.flatnonzero(np.asarray(legal_mask, dtype=np.float32) > 0)
    if legal.size <= 0:
        raise ValueError("legal_mask must contain at least one legal action")
    if behavior_policy == "base":
        probs = np.asarray(prior, dtype=np.float64)
    elif behavior_policy == "search-improved":
        probs = np.asarray(target, dtype=np.float64)
    else:
        raise ValueError("behavior_policy must be 'base' or 'search-improved'")
    legal_probs = np.clip(probs[legal], 0.0, None)
    total = float(legal_probs.sum())
    if total <= 0.0:
        legal_probs = np.ones_like(legal_probs) / float(legal_probs.size)
    else:
        legal_probs = legal_probs / total
    if greedy:
        return int(legal[int(np.argmax(legal_probs))])
    return int(rng.choice(legal, p=legal_probs))


def trajectory_return_weights(
    returns: np.ndarray,
    *,
    temperature: float = 1.0,
    max_weight: float = 20.0,
    min_weight: float = 0.0,
) -> np.ndarray:
    """Convert realized trajectory returns into non-negative CE weights.

    This is a small mirror-descent/REPS-style weighting rule: actions from
    above-baseline trajectories receive more imitation weight, while failed
    trajectories are retained but downweighted. The normalization keeps the
    average learning-rate contribution stable across batches.
    """
    values = np.asarray(returns, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if values.size == 0:
        return values.copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must be finite")
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError("temperature must be positive")
    centered = values.astype(np.float64) - float(values.mean())
    scale = float(centered.std())
    if not np.isfinite(scale) or scale < 1e-6:
        scale = max(float(np.max(np.abs(centered))), 1.0)
    logits = centered / (scale * temp)
    logits = logits - float(logits.max())
    weights = np.exp(logits)
    if np.isfinite(float(max_weight)):
        weights = np.minimum(weights, float(max_weight))
    weights = np.maximum(weights, float(min_weight))
    mean = float(weights.mean())
    if mean <= 1e-8:
        return np.ones(values.shape, dtype=np.float32)
    return (weights / mean).astype(np.float32)


def _evaluate_average_policy_loss(
    policy_net: torch.nn.Module,
    targets: PolicyTargetBuffer,
    device: torch.device,
    *,
    batch_size: int,
) -> float:
    losses: list[float] = []
    weights: list[int] = []
    policy_net.eval()
    with torch.no_grad():
        for start in range(0, targets.size, int(batch_size)):
            end = min(start + int(batch_size), targets.size)
            features = torch.from_numpy(targets.features[start:end]).to(device)
            legal_masks = torch.from_numpy(targets.legal_masks[start:end]).to(device)
            target_probs = torch.from_numpy(targets.target_probs[start:end]).to(device)
            target_weights = torch.from_numpy(targets.weights[start:end]).to(device)
            logits = policy_net(features)
            loss = masked_policy_cross_entropy(
                logits,
                legal_masks,
                target_probs,
                weights=target_weights,
            )
            losses.append(float(loss.cpu()))
            weights.append(end - start)
    if not losses:
        return 0.0
    return float(np.average(losses, weights=weights))


def decision_value_eval_gate(
    *,
    before_eval: dict[str, Any],
    after_eval: dict[str, Any],
    require_policy_ev_non_decrease: bool = False,
) -> dict[str, Any]:
    """Return direct fixed-root decision-impact checks for actor updates."""
    checks = {
        "selected_action_value_non_decrease": bool(
            float(after_eval["mean_selected_action_value"])
            >= float(before_eval["mean_selected_action_value"])
        ),
        "oracle_gap_non_increase": bool(
            float(after_eval["mean_oracle_gap"]) <= float(before_eval["mean_oracle_gap"])
        ),
    }
    if require_policy_ev_non_decrease:
        checks["policy_ev_non_decrease"] = bool(
            float(after_eval["mean_policy_ev"]) >= float(before_eval["mean_policy_ev"])
        )
    return {
        "passed": bool(all(checks.values())),
        "require_policy_ev_non_decrease": bool(require_policy_ev_non_decrease),
        "checks": checks,
        "before": {
            "mean_selected_action_value": float(before_eval["mean_selected_action_value"]),
            "mean_policy_ev": float(before_eval["mean_policy_ev"]),
            "mean_oracle_gap": float(before_eval["mean_oracle_gap"]),
        },
        "after": {
            "mean_selected_action_value": float(after_eval["mean_selected_action_value"]),
            "mean_policy_ev": float(after_eval["mean_policy_ev"]),
            "mean_oracle_gap": float(after_eval["mean_oracle_gap"]),
        },
    }


def _target_street_metadata_from_features(features: np.ndarray) -> dict[str, Any]:
    street_one_hot = np.asarray(features, dtype=np.float32)[:, 104:108]
    valid = np.max(street_one_hot, axis=1) > 0
    streets = np.argmax(street_one_hot, axis=1)[valid].astype(np.int64)
    unique, counts = np.unique(streets, return_counts=True)
    return {
        "target_streets": [int(street) for street in unique],
        "target_street_counts": {
            str(int(street)): int(count)
            for street, count in zip(unique, counts, strict=True)
        },
    }


def build_decision_value_q_target_buffer(
    targets: PolicyTargetBuffer,
    *,
    action_values: np.ndarray | None = None,
    base_probs: np.ndarray | None = None,
    target_metadata: dict[str, Any] | None = None,
) -> DecisionValueQTargetBuffer:
    """Build a Q-target buffer from arrays or collector row metadata."""
    if action_values is None or base_probs is None:
        rows = list((target_metadata or {}).get("rows") or [])
        if len(rows) != int(targets.size):
            raise ValueError("target_metadata rows must match target size for Q targets")
        action_rows: list[list[float]] = []
        prior_rows: list[list[float]] = []
        for row in rows:
            if "action_values" not in row or "prior_policy" not in row:
                raise ValueError("collector rows must include action_values and prior_policy")
            action_rows.append(
                [np.nan if value is None else float(value) for value in row["action_values"]]
            )
            prior_rows.append([float(value) for value in row["prior_policy"]])
        action_values = np.asarray(action_rows, dtype=np.float32)
        base_probs = np.asarray(prior_rows, dtype=np.float32)
    return DecisionValueQTargetBuffer(
        targets.features,
        targets.legal_masks,
        np.asarray(action_values, dtype=np.float32),
        np.asarray(base_probs, dtype=np.float32),
        weights=targets.weights,
    )


def _masked_policy_probs(logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(legal_masks <= 0, -1.0e9)
    probs = torch.softmax(masked_logits, dim=-1)
    return probs * legal_masks


def _normalize_q_advantages(
    action_values: torch.Tensor,
    legal_masks: torch.Tensor,
    base_probs: torch.Tensor,
) -> torch.Tensor:
    legal_q = action_values * legal_masks
    baseline = (base_probs * legal_q).sum(dim=-1, keepdim=True)
    centered = (legal_q - baseline) * legal_masks
    legal_count = legal_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = centered.sum(dim=-1, keepdim=True) / legal_count
    var = (((centered - mean) * legal_masks) ** 2).sum(dim=-1, keepdim=True) / legal_count
    scale = torch.sqrt(var).clamp_min(1.0)
    return centered / scale


def _kl_q_actor_loss(
    logits: torch.Tensor,
    batch: DecisionValueQTargetBatch,
    *,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probs = _masked_policy_probs(logits, batch.legal_masks)
    q_advantages = _normalize_q_advantages(
        batch.action_values,
        batch.legal_masks,
        batch.base_probs,
    )
    expected_q = (probs * q_advantages).sum(dim=-1)
    safe_probs = torch.clamp(probs, min=1e-8)
    safe_base = torch.clamp(batch.base_probs, min=1e-8)
    kl = (probs * (torch.log(safe_probs) - torch.log(safe_base))).sum(dim=-1)
    row_loss = -expected_q + float(kl_beta) * kl
    weighted = row_loss * batch.weights
    loss = weighted.sum() / batch.weights.sum().clamp_min(1.0)
    return loss, {
        "expected_q": expected_q.detach(),
        "kl_to_base": kl.detach(),
    }


def _evaluate_policy_head_kl_q_objective(
    value_net: torch.nn.Module,
    targets: DecisionValueQTargetBuffer,
    device: torch.device,
    *,
    batch_size: int,
    kl_beta: float,
) -> dict[str, float]:
    losses: list[float] = []
    expected_qs: list[float] = []
    kls: list[float] = []
    weights: list[int] = []
    value_net.eval()
    with torch.no_grad():
        for start in range(0, targets.size, int(batch_size)):
            end = min(start + int(batch_size), targets.size)
            batch = DecisionValueQTargetBatch(
                features=torch.from_numpy(targets.features[start:end]).to(device),
                legal_masks=torch.from_numpy(targets.legal_masks[start:end]).to(device),
                action_values=torch.from_numpy(targets.action_values[start:end]).to(device),
                base_probs=torch.from_numpy(targets.base_probs[start:end]).to(device),
                weights=torch.from_numpy(targets.weights[start:end]).to(device),
            )
            _, logits = value_net.forward_with_policy(batch.features)
            loss, stats = _kl_q_actor_loss(logits, batch, kl_beta=kl_beta)
            losses.append(float(loss.cpu()))
            expected_qs.append(float(stats["expected_q"].mean().cpu()))
            kls.append(float(stats["kl_to_base"].mean().cpu()))
            weights.append(end - start)
    if not losses:
        return {"loss": 0.0, "expected_q": 0.0, "kl_to_base": 0.0}
    return {
        "loss": float(np.average(losses, weights=weights)),
        "expected_q": float(np.average(expected_qs, weights=weights)),
        "kl_to_base": float(np.average(kls, weights=weights)),
    }


def calibrate_average_policy_net_from_targets(
    checkpoint: str | Path,
    targets: PolicyTargetBuffer,
    output: str | Path,
    *,
    n_steps: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str | torch.device = "auto",
    target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train only the standalone average-policy net against decision-value targets."""
    resolved_device = (
        device if isinstance(device, torch.device) else torch.device(resolve_device(str(device))["resolved_device"])
    )
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    if loaded.average_policy_net is None:
        raise RuntimeError("decision-value actor calibration requires average_policy_net")
    policy_net = loaded.average_policy_net
    before_loss = _evaluate_average_policy_loss(
        policy_net,
        targets,
        resolved_device,
        batch_size=batch_size,
    )

    for param in policy_net.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(policy_net.parameters(), lr=float(lr), weight_decay=0.0)
    policy_net.train()
    losses: list[float] = []
    for _ in range(int(n_steps)):
        batch = targets.sample_batch(batch_size, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        logits = policy_net(batch.features)
        loss = masked_policy_cross_entropy(
            logits,
            batch.legal_masks,
            batch.target_probs,
            weights=batch.weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    after_loss = _evaluate_average_policy_loss(
        policy_net,
        targets,
        resolved_device,
        batch_size=batch_size,
    )
    original = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    original["average_policy_net"] = policy_net.state_dict()
    original["decision_value_actor"] = {
        "mode": "decision_value_average_policy_calibration",
        "source_checkpoint": str(checkpoint),
        "target_size": int(targets.size),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "before_loss": round(float(before_loss), 6),
        "after_loss": round(float(after_loss), 6),
        "target_metadata": dict(target_metadata or {}),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(original, output)
    return {
        "mode": "decision_value_average_policy_calibration",
        "passed": output.exists() and after_loss <= before_loss,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "device": str(resolved_device),
        "target_size": int(targets.size),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "before_loss": round(float(before_loss), 6),
        "after_loss": round(float(after_loss), 6),
        "loss_delta": round(float(after_loss - before_loss), 6),
        "mean_train_loss": round(float(np.mean(losses)), 6) if losses else 0.0,
        "target_metadata": dict(target_metadata or {}),
    }


def calibrate_policy_head_from_decision_value_targets(
    checkpoint: str | Path,
    targets: PolicyTargetBuffer,
    output: str | Path,
    *,
    n_steps: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str | torch.device = "auto",
    target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the checkpoint policy head from decision-value targets."""
    metrics = train_policy_head_calibration(
        checkpoint,
        targets,
        output,
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    resolved_device = (
        device if isinstance(device, torch.device) else torch.device(resolve_device(str(device))["resolved_device"])
    )
    saved = torch.load(output, map_location=resolved_device, weights_only=False)
    calibration = dict(saved.get("policy_calibration") or {})
    calibration.update(
        {
            "mode": "decision_value_policy_head_calibration",
            "source_checkpoint": str(checkpoint),
            "target_metadata": dict(target_metadata or {}),
        }
    )
    saved["policy_calibration"] = calibration
    saved["decision_value_actor"] = {
        "mode": "decision_value_policy_head_calibration",
        "actor_target": "policy-head",
        "source_checkpoint": str(checkpoint),
        "target_size": int(targets.size),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "before_loss": metrics["before_loss"],
        "after_loss": metrics["after_loss"],
        "target_metadata": dict(target_metadata or {}),
    }
    torch.save(saved, output)
    metrics.update(
        {
            "mode": "decision_value_policy_head_calibration",
            "target_metadata": dict(target_metadata or {}),
        }
    )
    return metrics


def calibrate_policy_head_with_kl_q_targets(
    checkpoint: str | Path,
    targets: DecisionValueQTargetBuffer,
    output: str | Path,
    *,
    n_steps: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    kl_beta: float = 1.0,
    device: str | torch.device = "auto",
    target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the policy head by KL-anchored expected search-value improvement."""
    resolved_device = (
        device if isinstance(device, torch.device) else torch.device(resolve_device(str(device))["resolved_device"])
    )
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    value_net = loaded.value_net
    before = _evaluate_policy_head_kl_q_objective(
        value_net,
        targets,
        resolved_device,
        batch_size=batch_size,
        kl_beta=kl_beta,
    )

    for param in value_net.parameters():
        param.requires_grad = False
    for param in value_net.policy_head.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(value_net.policy_head.parameters(), lr=float(lr), weight_decay=0.0)
    value_net.train()
    losses: list[float] = []
    for _ in range(int(n_steps)):
        batch = targets.sample_batch(batch_size, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        _, logits = value_net.forward_with_policy(batch.features)
        loss, _stats = _kl_q_actor_loss(logits, batch, kl_beta=kl_beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.policy_head.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    after = _evaluate_policy_head_kl_q_objective(
        value_net,
        targets,
        resolved_device,
        batch_size=batch_size,
        kl_beta=kl_beta,
    )
    original = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    original["value_net"] = value_net.state_dict()
    street_metadata = _target_street_metadata_from_features(targets.features)
    original["policy_calibration"] = {
        "mode": "decision_value_policy_head_kl_q_calibration",
        "source_checkpoint": str(checkpoint),
        "target_size": int(targets.size),
        **street_metadata,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "kl_beta": float(kl_beta),
        "before_loss": round(float(before["loss"]), 6),
        "after_loss": round(float(after["loss"]), 6),
        "before_expected_q": round(float(before["expected_q"]), 6),
        "after_expected_q": round(float(after["expected_q"]), 6),
        "before_kl_to_base": round(float(before["kl_to_base"]), 6),
        "after_kl_to_base": round(float(after["kl_to_base"]), 6),
        "target_metadata": dict(target_metadata or {}),
    }
    original["decision_value_actor"] = {
        "mode": "decision_value_policy_head_kl_q_calibration",
        "actor_target": "policy-head",
        "source_checkpoint": str(checkpoint),
        "target_size": int(targets.size),
        **street_metadata,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "kl_beta": float(kl_beta),
        "before_loss": round(float(before["loss"]), 6),
        "after_loss": round(float(after["loss"]), 6),
        "before_expected_q": round(float(before["expected_q"]), 6),
        "after_expected_q": round(float(after["expected_q"]), 6),
        "target_metadata": dict(target_metadata or {}),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(original, output)
    return {
        "mode": "decision_value_policy_head_kl_q_calibration",
        "passed": bool(output.exists() and after["expected_q"] >= before["expected_q"]),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "device": str(resolved_device),
        "target_size": int(targets.size),
        **street_metadata,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "kl_beta": float(kl_beta),
        "before_loss": round(float(before["loss"]), 6),
        "after_loss": round(float(after["loss"]), 6),
        "loss_delta": round(float(after["loss"] - before["loss"]), 6),
        "before_expected_q": round(float(before["expected_q"]), 6),
        "after_expected_q": round(float(after["expected_q"]), 6),
        "expected_q_delta": round(float(after["expected_q"] - before["expected_q"]), 6),
        "before_kl_to_base": round(float(before["kl_to_base"]), 6),
        "after_kl_to_base": round(float(after["kl_to_base"]), 6),
        "mean_train_loss": round(float(np.mean(losses)), 6) if losses else 0.0,
        "target_metadata": dict(target_metadata or {}),
    }


def _build_continuation_policy(
    cfg: DecisionValueTargetConfig,
    device: torch.device,
) -> tuple[Any, list[dict[str, Any]] | dict[str, Any]]:
    continuation_source = cfg.continuation_strategy_source or cfg.strategy_source
    loaded_population = []
    for path in cfg.continuation_checkpoints:
        loaded = load_value_network_checkpoint(path, device)
        assert_strategy_source_supported(loaded, continuation_source)
        loaded_population.append(loaded)
    if len(loaded_population) > 1:
        return (
            checkpoint_population_continuation_policy(
                loaded_population,
                device,
                strategy_source=continuation_source,
                greedy=cfg.greedy_continuation,
            ),
            [dict(item.metadata) for item in loaded_population],
        )
    if len(loaded_population) == 1:
        return (
            checkpoint_continuation_policy(
                loaded_population[0],
                device,
                strategy_source=continuation_source,
                greedy=cfg.greedy_continuation,
            ),
            dict(loaded_population[0].metadata),
        )
    return masked_uniform_policy, {}


def _build_continuation_policy_from_parts(
    *,
    checkpoint: str,
    strategy_source: str,
    continuation_checkpoints: tuple[str, ...],
    continuation_strategy_source: str | None,
    greedy_continuation: bool,
    device: torch.device,
) -> tuple[Any, list[dict[str, Any]] | dict[str, Any]]:
    source = continuation_strategy_source or strategy_source
    paths = tuple(continuation_checkpoints) or (checkpoint,)
    loaded_population = []
    for path in paths:
        loaded = load_value_network_checkpoint(path, device)
        assert_strategy_source_supported(loaded, source)
        loaded_population.append(loaded)
    if len(loaded_population) > 1:
        return (
            checkpoint_population_continuation_policy(
                loaded_population,
                device,
                strategy_source=source,
                greedy=greedy_continuation,
            ),
            [dict(item.metadata) for item in loaded_population],
        )
    return (
        checkpoint_continuation_policy(
            loaded_population[0],
            device,
            strategy_source=source,
            greedy=greedy_continuation,
        ),
        dict(loaded_population[0].metadata),
    )


def resample_hidden_world_state(
    public_state: FastPokerState,
    rng: np.random.Generator,
    *,
    hero_player: int | None = None,
) -> FastPokerState:
    """Resample hidden opponent cards and future deck while preserving public state."""
    if public_state.is_terminal:
        raise ValueError("cannot resample a terminal state")
    state = public_state.copy()
    hero = int(public_state.current_player_i if hero_player is None else hero_player)
    known_cards = {
        int(card)
        for card in state.hole_cards[hero]
        if int(card) >= 0
    }
    known_cards.update(int(card) for card in state.community if int(card) >= 0)
    unknown = np.asarray(
        [card for card in range(52) if card not in known_cards],
        dtype=np.int16,
    )
    permuted = rng.permutation(unknown)
    cursor = 0
    for player in range(state.n_players):
        if player == hero:
            continue
        state.hole_cards[player, :] = permuted[cursor : cursor + 2].astype(np.int8)
        cursor += 2
    prefix: list[int] = []
    for card in state.hole_cards[hero]:
        if int(card) >= 0:
            prefix.append(int(card))
    for player in range(state.n_players):
        if player == hero:
            continue
        prefix.extend(int(card) for card in state.hole_cards[player])
    prefix.extend(int(card) for card in state.community if int(card) >= 0)
    future = [int(card) for card in permuted[cursor:] if int(card) not in set(prefix)]
    deck_order = np.asarray(prefix + future, dtype=np.int8)
    if deck_order.shape != (52,):
        raise ValueError("resampled deck_order must contain 52 cards")
    state.deck_order = deck_order
    return state


def score_first_actions_for_public_state(
    public_state: FastPokerState,
    *,
    n_worlds: int,
    continuation_policy: Any,
    seed: int,
    max_steps_per_hand: int = 128,
) -> Any:
    """Score legal first actions from an arbitrary public information state."""
    if int(n_worlds) <= 0:
        raise ValueError("n_worlds must be positive")
    acting_player = int(public_state.current_player_i)
    legal_actions = tuple(
        int(action) for action in np.flatnonzero(public_state.get_legal_mask() > 0)
    )
    action_values: dict[int, float] = {}
    action_standard_errors: dict[int, float] = {}
    n_truncated = 0
    for action in legal_actions:
        payoffs: list[float] = []
        for world_idx in range(int(n_worlds)):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(action), int(world_idx)])
            )
            world_state = resample_hidden_world_state(public_state, rng)
            rollout_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(action), int(world_idx), 17])
            )
            payoff, truncated = _payoff_after_rollout(
                world_state,
                action=action,
                acting_player=acting_player,
                continuation_policy=continuation_policy,
                rng=rollout_rng,
                max_steps_per_hand=max_steps_per_hand,
            )
            payoffs.append(float(payoff))
            n_truncated += int(truncated)
        payoff_arr = np.asarray(payoffs, dtype=np.float64)
        action_values[int(action)] = float(payoff_arr.mean())
        action_standard_errors[int(action)] = (
            float(payoff_arr.std(ddof=1) / np.sqrt(payoff_arr.size))
            if payoff_arr.size > 1
            else 0.0
        )
    best_action = max(action_values, key=action_values.__getitem__)
    from poker_ai.research.public_action_rollout_value import FirstActionRolloutResult

    return FirstActionRolloutResult(
        legal_actions=legal_actions,
        action_values=action_values,
        action_standard_errors=action_standard_errors,
        best_action=int(best_action),
        best_action_value=float(action_values[best_action]),
        n_worlds=int(n_worlds),
        n_truncated_rollouts=int(n_truncated),
    )


def collect_on_policy_decision_value_policy_targets(
    cfg: OnPolicyDecisionValueTargetConfig,
) -> tuple[PolicyTargetBuffer, dict[str, Any]]:
    """Collect value-aware actor targets from states visited by the current actor."""
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    loaded = load_value_network_checkpoint(cfg.checkpoint, device)
    assert_strategy_source_supported(loaded, cfg.strategy_source)
    continuation, continuation_metadata = _build_continuation_policy_from_parts(
        checkpoint=cfg.checkpoint,
        strategy_source=cfg.strategy_source,
        continuation_checkpoints=cfg.continuation_checkpoints,
        continuation_strategy_source=cfg.continuation_strategy_source,
        greedy_continuation=cfg.greedy_continuation,
        device=device,
    )
    state_policy = checkpoint_continuation_policy(
        loaded,
        device,
        strategy_source=cfg.strategy_source,
        greedy=cfg.state_policy_greedy,
    )
    rng = np.random.default_rng(cfg.seed)
    target_streets = {int(street) for street in cfg.target_streets}
    required_streets = {int(street) for street in cfg.required_streets}
    min_rows_per_required_street = max(1, int(cfg.min_rows_per_required_street))
    if cfg.behavior_policy not in ("base", "search-improved"):
        raise ValueError("behavior_policy must be 'base' or 'search-improved'")
    if not required_streets.issubset(target_streets):
        raise ValueError("required_streets must be a subset of target_streets")
    features_out: list[np.ndarray] = []
    masks_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    street_counts: dict[str, int] = {}
    n_truncated = 0
    attempted_hands = 0

    def missing_required_streets() -> list[int]:
        return [
            int(street)
            for street in sorted(required_streets)
            if street_counts.get(str(int(street)), 0) < min_rows_per_required_street
        ]

    def collection_done() -> bool:
        return len(features_out) >= int(cfg.n_states) and not missing_required_streets()

    while not collection_done() and attempted_hands < int(cfg.max_hands):
        attempted_hands += 1
        state = FastPokerState(
            n_players=2,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            initial_chips=cfg.initial_chips,
        )
        for step in range(int(cfg.max_steps_per_hand)):
            if state.is_terminal:
                break
            street = int(state.stage)
            behavior_action: int | None = None
            row_idx_for_state: int | None = None
            needs_more_total = len(features_out) < int(cfg.n_states)
            needs_required_street = (
                street in required_streets
                and street_counts.get(str(street), 0) < min_rows_per_required_street
            )
            if street in target_streets and (needs_more_total or needs_required_street):
                advantages, prior = _policy_for_root(
                    loaded,
                    state,
                    device,
                    strategy_source=cfg.strategy_source,
                )
                legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
                selected_action = select_deployed_root_action(
                    advantages=advantages,
                    strategy=prior,
                    legal_mask=legal_mask,
                    strategy_source=cfg.strategy_source,
                    greedy=cfg.greedy_continuation,
                )
                result = score_first_actions_for_public_state(
                    state,
                    n_worlds=cfg.n_worlds,
                    continuation_policy=continuation,
                    seed=int(cfg.seed) + 4099 * len(features_out),
                    max_steps_per_hand=cfg.max_steps_per_hand,
                )
                action_values = np.full(N_ACTIONS, np.nan, dtype=np.float32)
                for action, value in result.action_values.items():
                    action_values[int(action)] = float(value)
                target = regularized_action_value_target(
                    prior,
                    legal_mask,
                    action_values,
                    eta=cfg.eta,
                    min_prior=cfg.min_prior,
                )
                if cfg.behavior_policy == "search-improved":
                    behavior_action = select_decision_value_behavior_action(
                        prior,
                        target,
                        legal_mask,
                        rng,
                        behavior_policy=cfg.behavior_policy,
                        greedy=cfg.search_improved_behavior_greedy,
                    )
                target_ev = float(
                    sum(
                        float(target[action]) * float(result.action_values[action])
                        for action in result.legal_actions
                    )
                )
                features_out.append(state.to_feature_vector().astype(np.float32, copy=False))
                masks_out.append(legal_mask)
                targets_out.append(target)
                street_counts[str(street)] = street_counts.get(str(street), 0) + 1
                n_truncated += int(result.n_truncated_rollouts)
                row_idx_for_state = len(rows)
                rows.append(
                    {
                        "row_idx": len(rows),
                        "hand_idx": int(attempted_hands - 1),
                        "step": int(step),
                        "street": int(street),
                        "acting_player": int(state.current_player_i),
                        "selected_action": _action_name(selected_action),
                        "oracle_action": _action_name(result.best_action),
                        "selected_action_value": float(result.action_values[selected_action]),
                        "behavior_action": None,
                        "target_policy_ev": target_ev,
                        "oracle_action_value": float(result.best_action_value),
                        "target_oracle_gap": float(result.best_action_value - target_ev),
                        "n_truncated_rollouts": int(result.n_truncated_rollouts),
                        "action_values": [
                            None if not np.isfinite(value) else float(value)
                            for value in action_values
                        ],
                        "prior_policy": [float(value) for value in prior],
                    }
                )
            if behavior_action is None:
                action = int(state_policy(state, rng))
            else:
                action = int(behavior_action)
            mask = state.get_legal_mask()
            if action < 0 or action >= N_ACTIONS or mask[action] <= 0:
                raise RuntimeError(f"state policy selected illegal action {action}")
            if row_idx_for_state is not None:
                rows[row_idx_for_state]["behavior_action"] = _action_name(action)
            state.apply_action(action)

    if len(features_out) != int(cfg.n_states):
        if len(features_out) > int(cfg.n_states) and not missing_required_streets():
            pass
        else:
            raise RuntimeError(
                f"collected {len(features_out)} targets out of requested {cfg.n_states} "
                f"after {attempted_hands} hands"
            )
    missing_streets = missing_required_streets()
    if missing_streets:
        raise RuntimeError(
            "missing required street coverage "
            f"{missing_streets} after collecting {len(features_out)} targets "
            f"over {attempted_hands} hands"
        )
    targets = PolicyTargetBuffer(
        np.asarray(features_out, dtype=np.float32),
        np.asarray(masks_out, dtype=np.float32),
        np.asarray(targets_out, dtype=np.float32),
    )
    target_evs = np.asarray([row["target_policy_ev"] for row in rows], dtype=np.float64)
    selected_values = np.asarray(
        [row["selected_action_value"] for row in rows],
        dtype=np.float64,
    )
    oracle_values = np.asarray(
        [row["oracle_action_value"] for row in rows],
        dtype=np.float64,
    )
    metadata = {
        "mode": "on_policy_decision_value_policy_targets",
        "checkpoint": cfg.checkpoint,
        "strategy_source": cfg.strategy_source,
        "continuation_checkpoints": list(cfg.continuation_checkpoints) or [cfg.checkpoint],
        "continuation_strategy_source": cfg.continuation_strategy_source or cfg.strategy_source,
        "continuation_metadata": continuation_metadata,
        "behavior_policy": cfg.behavior_policy,
        "search_improved_behavior_greedy": bool(cfg.search_improved_behavior_greedy),
        "device": device_info,
        "n_states": int(cfg.n_states),
        "n_targets": int(len(features_out)),
        "n_worlds": int(cfg.n_worlds),
        "attempted_hands": int(attempted_hands),
        "target_streets": [int(street) for street in cfg.target_streets],
        "required_streets": [int(street) for street in cfg.required_streets],
        "min_rows_per_required_street": int(min_rows_per_required_street),
        "street_counts": street_counts,
        "coverage_gate": {
            "passed": True,
            "required_streets": [int(street) for street in sorted(required_streets)],
            "missing_streets": [],
            "min_rows_per_required_street": int(min_rows_per_required_street),
        },
        "seed": int(cfg.seed),
        "eta": float(cfg.eta),
        "mean_selected_action_value": float(selected_values.mean()),
        "mean_target_policy_ev": float(target_evs.mean()),
        "mean_target_improvement_over_selected": float((target_evs - selected_values).mean()),
        "mean_oracle_action_value": float(oracle_values.mean()),
        "mean_target_oracle_gap": float((oracle_values - target_evs).mean()),
        "n_truncated_rollouts": int(n_truncated),
        "rows": rows,
    }
    return targets, metadata


def build_on_policy_average_strategy_target_artifact(
    cfg: OnPolicyDecisionValueTargetConfig,
    output_npz: str | Path,
    *,
    output_json: str | Path | None = None,
    recommended_average_strategy_weight: float = 1.0,
) -> dict[str, Any]:
    """Write on-policy decision-value targets for GPU average-policy training."""
    targets, target_metadata = collect_on_policy_decision_value_policy_targets(cfg)
    output_path = Path(output_npz)
    targets.save_npz(output_path)
    metrics: dict[str, Any] = {
        "passed": bool(output_path.exists() and int(targets.size) > 0),
        "mode": "on_policy_average_strategy_target_artifact",
        "average_strategy_targets": str(output_path),
        "target_size": int(targets.size),
        "recommended_gpu_flags": {
            "average_strategy_targets": str(output_path),
            "average_strategy_weight": float(recommended_average_strategy_weight),
        },
        "target_metadata": target_metadata,
    }
    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        metrics["output_json"] = str(json_path)
    return metrics


def _action_idx_from_name(name: str) -> int:
    for action in range(N_ACTIONS):
        if _action_name(action) == name:
            return int(action)
    raise ValueError(f"unknown action name: {name}")


def collect_on_policy_behavior_average_strategy_targets(
    cfg: OnPolicyDecisionValueTargetConfig,
) -> tuple[PolicyTargetBuffer, dict[str, Any]]:
    """Collect one-hot targets for the actions actually used during collection."""
    targets, metadata = collect_on_policy_decision_value_policy_targets(cfg)
    behavior_probs = np.zeros_like(targets.target_probs, dtype=np.float32)
    rows = list(metadata.get("rows") or [])
    if len(rows) != int(targets.size):
        raise ValueError("metadata rows must match target size")
    for row_idx, row in enumerate(rows):
        action_idx = _action_idx_from_name(str(row["behavior_action"]))
        if targets.legal_masks[row_idx, action_idx] <= 0:
            raise ValueError(f"behavior action is illegal for row {row_idx}")
        behavior_probs[row_idx, action_idx] = 1.0
        row["behavior_action_idx"] = int(action_idx)
    behavior_targets = PolicyTargetBuffer(
        targets.features,
        targets.legal_masks,
        behavior_probs,
        weights=targets.weights,
    )
    metadata = dict(metadata)
    metadata["mode"] = "on_policy_behavior_average_strategy_targets"
    metadata["target_policy"] = "one_hot_behavior_action"
    metadata["rows"] = rows
    return behavior_targets, metadata


def build_on_policy_behavior_average_strategy_target_artifact(
    cfg: OnPolicyDecisionValueTargetConfig,
    output_npz: str | Path,
    *,
    output_json: str | Path | None = None,
    recommended_average_strategy_weight: float = 1.0,
) -> dict[str, Any]:
    """Write one-hot behavior targets for GPU average-policy training."""
    targets, target_metadata = collect_on_policy_behavior_average_strategy_targets(cfg)
    output_path = Path(output_npz)
    targets.save_npz(output_path)
    metrics: dict[str, Any] = {
        "passed": bool(output_path.exists() and int(targets.size) > 0),
        "mode": "on_policy_behavior_average_strategy_target_artifact",
        "average_strategy_targets": str(output_path),
        "target_size": int(targets.size),
        "recommended_gpu_flags": {
            "average_strategy_targets": str(output_path),
            "average_strategy_weight": float(recommended_average_strategy_weight),
            "seed_average_strategy_memory_from_targets": True,
        },
        "target_metadata": target_metadata,
    }
    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        metrics["output_json"] = str(json_path)
    return metrics


def collect_on_policy_trajectory_return_average_strategy_targets(
    cfg: OnPolicyDecisionValueTargetConfig,
    *,
    return_temperature: float = 1.0,
    max_return_weight: float = 20.0,
) -> tuple[PolicyTargetBuffer, dict[str, Any]]:
    """Collect one-hot behavior targets weighted by realized trajectory returns."""
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    loaded = load_value_network_checkpoint(cfg.checkpoint, device)
    assert_strategy_source_supported(loaded, cfg.strategy_source)
    continuation, continuation_metadata = _build_continuation_policy_from_parts(
        checkpoint=cfg.checkpoint,
        strategy_source=cfg.strategy_source,
        continuation_checkpoints=cfg.continuation_checkpoints,
        continuation_strategy_source=cfg.continuation_strategy_source,
        greedy_continuation=cfg.greedy_continuation,
        device=device,
    )
    state_policy = checkpoint_continuation_policy(
        loaded,
        device,
        strategy_source=cfg.strategy_source,
        greedy=cfg.state_policy_greedy,
    )
    if cfg.behavior_policy not in ("base", "search-improved"):
        raise ValueError("behavior_policy must be 'base' or 'search-improved'")
    rng = np.random.default_rng(cfg.seed)
    target_streets = {int(street) for street in cfg.target_streets}
    required_streets = {int(street) for street in cfg.required_streets}
    min_rows_per_required_street = max(1, int(cfg.min_rows_per_required_street))
    if not required_streets.issubset(target_streets):
        raise ValueError("required_streets must be a subset of target_streets")

    features_out: list[np.ndarray] = []
    masks_out: list[np.ndarray] = []
    target_probs_out: list[np.ndarray] = []
    returns_out: list[float] = []
    rows: list[dict[str, Any]] = []
    street_counts: dict[str, int] = {}
    attempted_hands = 0
    n_terminal_hands = 0
    n_truncated_hands = 0
    n_truncated_rollouts = 0

    def missing_required_streets() -> list[int]:
        return [
            int(street)
            for street in sorted(required_streets)
            if street_counts.get(str(int(street)), 0) < min_rows_per_required_street
        ]

    def collection_done() -> bool:
        return len(features_out) >= int(cfg.n_states) and not missing_required_streets()

    while not collection_done() and attempted_hands < int(cfg.max_hands):
        hand_idx = int(attempted_hands)
        attempted_hands += 1
        state = FastPokerState(
            n_players=2,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            initial_chips=cfg.initial_chips,
        )
        hand_records: list[dict[str, Any]] = []
        for step in range(int(cfg.max_steps_per_hand)):
            if state.is_terminal:
                break
            street = int(state.stage)
            acting_player = int(state.current_player_i)
            legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
            behavior_action: int | None = None
            record_state = False
            needs_more_total = (
                len(features_out) + len(hand_records) < int(cfg.n_states)
            )
            needs_required_street = (
                street in required_streets
                and street_counts.get(str(street), 0) < min_rows_per_required_street
            )
            if street in target_streets and (needs_more_total or needs_required_street):
                record_state = True
                row_extra: dict[str, Any] = {}
                if cfg.behavior_policy == "search-improved":
                    advantages, prior = _policy_for_root(
                        loaded,
                        state,
                        device,
                        strategy_source=cfg.strategy_source,
                    )
                    result = score_first_actions_for_public_state(
                        state,
                        n_worlds=cfg.n_worlds,
                        continuation_policy=continuation,
                        seed=int(cfg.seed) + 6151 * (len(features_out) + len(hand_records)),
                        max_steps_per_hand=cfg.max_steps_per_hand,
                    )
                    action_values = np.full(N_ACTIONS, np.nan, dtype=np.float32)
                    for action, value in result.action_values.items():
                        action_values[int(action)] = float(value)
                    target = regularized_action_value_target(
                        prior,
                        legal_mask,
                        action_values,
                        eta=cfg.eta,
                        min_prior=cfg.min_prior,
                    )
                    behavior_action = select_decision_value_behavior_action(
                        prior,
                        target,
                        legal_mask,
                        rng,
                        behavior_policy="search-improved",
                        greedy=cfg.search_improved_behavior_greedy,
                    )
                    n_truncated_rollouts += int(result.n_truncated_rollouts)
                    row_extra = {
                        "search_target_policy_ev": float(
                            sum(
                                float(target[action]) * float(result.action_values[action])
                                for action in result.legal_actions
                            )
                        ),
                        "search_oracle_action": _action_name(result.best_action),
                        "search_oracle_action_value": float(result.best_action_value),
                        "search_n_truncated_rollouts": int(result.n_truncated_rollouts),
                    }
            if behavior_action is None:
                action = int(state_policy(state, rng))
            else:
                action = int(behavior_action)
            if action < 0 or action >= N_ACTIONS or legal_mask[action] <= 0:
                raise RuntimeError(f"state policy selected illegal action {action}")
            if record_state:
                target_probs = np.zeros(N_ACTIONS, dtype=np.float32)
                target_probs[action] = 1.0
                hand_records.append(
                    {
                        "feature": state.to_feature_vector().astype(np.float32, copy=False),
                        "legal_mask": legal_mask.copy(),
                        "target_probs": target_probs,
                        "hand_idx": hand_idx,
                        "step": int(step),
                        "street": int(street),
                        "acting_player": acting_player,
                        "behavior_action": _action_name(action),
                        "behavior_action_idx": int(action),
                        **row_extra,
                    }
                )
            state.apply_action(action)

        if state.is_terminal:
            n_terminal_hands += 1
            payouts = state.payout
            for record in hand_records:
                actor = int(record["acting_player"])
                realized_return = float(payouts[actor])
                row = {
                    "row_idx": len(rows),
                    **{
                        key: value
                        for key, value in record.items()
                        if key not in {"feature", "legal_mask", "target_probs"}
                    },
                    "terminal_payoff_for_actor": realized_return,
                    "terminal_payouts": {
                        str(player): float(value) for player, value in payouts.items()
                    },
                }
                features_out.append(record["feature"])
                masks_out.append(record["legal_mask"])
                target_probs_out.append(record["target_probs"])
                returns_out.append(realized_return)
                rows.append(row)
                street_key = str(int(record["street"]))
                street_counts[street_key] = street_counts.get(street_key, 0) + 1
        elif hand_records:
            n_truncated_hands += 1

    if len(features_out) != int(cfg.n_states):
        if len(features_out) > int(cfg.n_states) and not missing_required_streets():
            pass
        else:
            raise RuntimeError(
                f"collected {len(features_out)} trajectory targets out of requested "
                f"{cfg.n_states} after {attempted_hands} hands"
            )
    missing_streets = missing_required_streets()
    if missing_streets:
        raise RuntimeError(
            "missing required street coverage "
            f"{missing_streets} after collecting {len(features_out)} trajectory targets "
            f"over {attempted_hands} hands"
        )
    returns_arr = np.asarray(returns_out, dtype=np.float32)
    weights = trajectory_return_weights(
        returns_arr,
        temperature=return_temperature,
        max_weight=max_return_weight,
    )
    for row, weight in zip(rows, weights, strict=True):
        row["trajectory_return_weight"] = float(weight)
    targets = PolicyTargetBuffer(
        np.asarray(features_out, dtype=np.float32),
        np.asarray(masks_out, dtype=np.float32),
        np.asarray(target_probs_out, dtype=np.float32),
        weights=weights,
    )
    metadata = {
        "mode": "on_policy_trajectory_return_average_strategy_targets",
        "target_policy": "one_hot_behavior_action",
        "return_weight_mode": "centered_exp",
        "return_temperature": float(return_temperature),
        "max_return_weight": float(max_return_weight),
        "checkpoint": cfg.checkpoint,
        "strategy_source": cfg.strategy_source,
        "continuation_checkpoints": list(cfg.continuation_checkpoints) or [cfg.checkpoint],
        "continuation_strategy_source": cfg.continuation_strategy_source or cfg.strategy_source,
        "continuation_metadata": continuation_metadata,
        "behavior_policy": cfg.behavior_policy,
        "search_improved_behavior_greedy": bool(cfg.search_improved_behavior_greedy),
        "device": device_info,
        "n_states": int(cfg.n_states),
        "n_targets": int(targets.size),
        "n_worlds": int(cfg.n_worlds),
        "attempted_hands": int(attempted_hands),
        "n_terminal_hands": int(n_terminal_hands),
        "n_truncated_hands": int(n_truncated_hands),
        "n_truncated_rollouts": int(n_truncated_rollouts),
        "target_streets": [int(street) for street in cfg.target_streets],
        "required_streets": [int(street) for street in cfg.required_streets],
        "min_rows_per_required_street": int(min_rows_per_required_street),
        "street_counts": street_counts,
        "coverage_gate": {
            "passed": True,
            "required_streets": [int(street) for street in sorted(required_streets)],
            "missing_streets": [],
            "min_rows_per_required_street": int(min_rows_per_required_street),
        },
        "seed": int(cfg.seed),
        "eta": float(cfg.eta),
        "mean_realized_return": float(returns_arr.mean()) if returns_arr.size else 0.0,
        "min_realized_return": float(returns_arr.min()) if returns_arr.size else 0.0,
        "max_realized_return": float(returns_arr.max()) if returns_arr.size else 0.0,
        "mean_return_weight": float(weights.mean()) if weights.size else 0.0,
        "max_observed_return_weight": float(weights.max()) if weights.size else 0.0,
        "rows": rows,
    }
    return targets, metadata


def build_on_policy_trajectory_return_average_strategy_target_artifact(
    cfg: OnPolicyDecisionValueTargetConfig,
    output_npz: str | Path,
    *,
    output_json: str | Path | None = None,
    recommended_average_strategy_weight: float = 1.0,
    return_temperature: float = 1.0,
    max_return_weight: float = 20.0,
) -> dict[str, Any]:
    """Write realized-return-weighted behavior targets for GPU average-policy training."""
    targets, target_metadata = collect_on_policy_trajectory_return_average_strategy_targets(
        cfg,
        return_temperature=return_temperature,
        max_return_weight=max_return_weight,
    )
    output_path = Path(output_npz)
    targets.save_npz(output_path)
    metrics: dict[str, Any] = {
        "passed": bool(output_path.exists() and int(targets.size) > 0),
        "mode": "on_policy_trajectory_return_average_strategy_target_artifact",
        "average_strategy_targets": str(output_path),
        "target_size": int(targets.size),
        "recommended_gpu_flags": {
            "average_strategy_targets": str(output_path),
            "average_strategy_weight": float(recommended_average_strategy_weight),
            "seed_average_strategy_memory_from_targets": True,
        },
        "target_metadata": target_metadata,
    }
    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        metrics["output_json"] = str(json_path)
    return metrics


def collect_decision_value_policy_targets(
    cfg: DecisionValueTargetConfig,
) -> tuple[PolicyTargetBuffer, dict[str, Any]]:
    """Collect mirror-descent policy targets from public rollout action values."""
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    loaded = load_value_network_checkpoint(cfg.checkpoint, device)
    assert_strategy_source_supported(loaded, cfg.strategy_source)
    continuation, continuation_metadata = _build_continuation_policy(cfg, device)

    features_out: list[np.ndarray] = []
    legal_masks_out: list[np.ndarray] = []
    target_probs_out: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    selected_values: list[float] = []
    target_selected_values: list[float] = []
    target_evs: list[float] = []
    oracle_values: list[float] = []
    n_truncated = 0
    for root_idx in range(int(cfg.n_roots)):
        hero_cards = sample_seeded_hero_cards(seed=cfg.seed, root_idx=root_idx)
        worlds = sample_public_worlds(
            hero_cards=hero_cards,
            n_worlds=cfg.n_worlds,
            seed=int(cfg.seed) + 1009 * (root_idx + 1),
        )
        root = build_public_world_state(
            hero_cards=worlds[0].hero_cards,
            opponent_cards=worlds[0].opponent_cards,
            deck_tail=worlds[0].deck_tail,
            initial_chips=cfg.initial_chips,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
        )
        advantages, prior = _policy_for_root(
            loaded,
            root,
            device,
            strategy_source=cfg.strategy_source,
        )
        legal_mask = root.get_legal_mask().astype(np.float32, copy=False)
        selected_action = select_deployed_root_action(
            advantages=advantages,
            strategy=prior,
            legal_mask=legal_mask,
            strategy_source=cfg.strategy_source,
            greedy=cfg.greedy_continuation,
        )
        result = score_first_actions_across_worlds(
            worlds=worlds,
            continuation_policy=continuation,
            seed=int(cfg.seed) + 2003 * (root_idx + 1),
            max_steps_per_hand=cfg.max_steps_per_hand,
            initial_chips=cfg.initial_chips,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
        )
        action_values = np.full(N_ACTIONS, np.nan, dtype=np.float32)
        for action, value in result.action_values.items():
            action_values[int(action)] = float(value)
        target = regularized_action_value_target(
            prior,
            legal_mask,
            action_values,
            eta=cfg.eta,
            min_prior=cfg.min_prior,
        )
        target_selected_action = select_decision_value_behavior_action(
            prior,
            target,
            legal_mask,
            np.random.default_rng(np.random.SeedSequence([int(cfg.seed), int(root_idx)])),
            behavior_policy="search-improved",
            greedy=True,
        )
        target_ev = float(
            sum(float(target[action]) * float(result.action_values[action]) for action in result.legal_actions)
        )
        selected_value = float(result.action_values[selected_action])
        target_selected_value = float(result.action_values[target_selected_action])
        selected_values.append(selected_value)
        target_selected_values.append(target_selected_value)
        target_evs.append(target_ev)
        oracle_values.append(float(result.best_action_value))
        n_truncated += int(result.n_truncated_rollouts)
        features_out.append(root.to_feature_vector().astype(np.float32, copy=False))
        legal_masks_out.append(legal_mask)
        target_probs_out.append(target)
        rows.append(
            {
                "root_idx": int(root_idx),
                "hero_cards": [int(c) for c in hero_cards],
                "selected_action": _action_name(selected_action),
                "target_selected_action": _action_name(target_selected_action),
                "oracle_action": _action_name(result.best_action),
                "selected_action_value": selected_value,
                "target_selected_action_value": target_selected_value,
                "target_policy_ev": target_ev,
                "oracle_action_value": float(result.best_action_value),
                "target_selected_oracle_gap": float(
                    result.best_action_value - target_selected_value
                ),
                "target_oracle_gap": float(result.best_action_value - target_ev),
                "action_values": {
                    _action_name(action): float(value)
                    for action, value in sorted(result.action_values.items())
                },
                "prior_strategy": {
                    _action_name(action): float(prior[action])
                    for action in result.legal_actions
                },
                "target_strategy": {
                    _action_name(action): float(target[action])
                    for action in result.legal_actions
                },
                "n_truncated_rollouts": int(result.n_truncated_rollouts),
            }
        )

    targets = PolicyTargetBuffer(
        np.asarray(features_out, dtype=np.float32),
        np.asarray(legal_masks_out, dtype=np.float32),
        np.asarray(target_probs_out, dtype=np.float32),
    )
    selected_arr = np.asarray(selected_values, dtype=np.float64)
    target_selected_arr = np.asarray(target_selected_values, dtype=np.float64)
    target_arr = np.asarray(target_evs, dtype=np.float64)
    oracle_arr = np.asarray(oracle_values, dtype=np.float64)
    selected_counter = Counter(row["selected_action"] for row in rows)
    target_selected_counter = Counter(row["target_selected_action"] for row in rows)
    oracle_counter = Counter(row["oracle_action"] for row in rows)
    metadata = {
        "mode": "decision_value_policy_targets",
        "checkpoint": cfg.checkpoint,
        "strategy_source": cfg.strategy_source,
        "continuation_checkpoints": list(cfg.continuation_checkpoints),
        "continuation_strategy_source": cfg.continuation_strategy_source or cfg.strategy_source,
        "continuation_metadata": continuation_metadata,
        "device": device_info,
        "n_roots": int(cfg.n_roots),
        "n_worlds": int(cfg.n_worlds),
        "seed": int(cfg.seed),
        "eta": float(cfg.eta),
        "mean_selected_action_value": float(selected_arr.mean()) if selected_arr.size else 0.0,
        "mean_target_selected_action_value": (
            float(target_selected_arr.mean()) if target_selected_arr.size else 0.0
        ),
        "mean_target_policy_ev": float(target_arr.mean()) if target_arr.size else 0.0,
        "mean_oracle_action_value": float(oracle_arr.mean()) if oracle_arr.size else 0.0,
        "mean_target_selected_oracle_gap": (
            float((oracle_arr - target_selected_arr).mean())
            if target_selected_arr.size
            else 0.0
        ),
        "mean_target_oracle_gap": float((oracle_arr - target_arr).mean()) if target_arr.size else 0.0,
        "mean_target_improvement_over_selected": float((target_arr - selected_arr).mean()) if target_arr.size else 0.0,
        "mean_target_selected_improvement_over_selected": (
            float((target_selected_arr - selected_arr).mean())
            if target_selected_arr.size
            else 0.0
        ),
        "selected_action_counts": dict(sorted(selected_counter.items())),
        "target_selected_action_counts": dict(sorted(target_selected_counter.items())),
        "oracle_action_counts": dict(sorted(oracle_counter.items())),
        "n_truncated_rollouts": int(n_truncated),
        "rows": rows,
    }
    return targets, metadata


def run_decision_value_actor_pilot(
    *,
    train_cfg: DecisionValueTargetConfig,
    output_checkpoint: str | Path,
    actor_target: str = "average-policy",
    n_steps: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    eval_roots: int = 0,
    eval_worlds: int | None = None,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    """Collect targets, train the average-policy actor, and optionally evaluate."""
    targets, target_metadata = collect_decision_value_policy_targets(train_cfg)
    if actor_target == "average-policy":
        calibration = calibrate_average_policy_net_from_targets(
            train_cfg.checkpoint,
            targets,
            output_checkpoint,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            device=train_cfg.device,
            target_metadata=target_metadata,
        )
        after_strategy_source = "average-policy"
    elif actor_target == "policy-head":
        calibration = calibrate_policy_head_from_decision_value_targets(
            train_cfg.checkpoint,
            targets,
            output_checkpoint,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            device=train_cfg.device,
            target_metadata=target_metadata,
        )
        after_strategy_source = "policy-head-covered"
    else:
        raise ValueError("actor_target must be 'average-policy' or 'policy-head'")
    before_eval = None
    after_eval = None
    if int(eval_roots) > 0:
        eval_cfg = PublicActionRolloutConfig(
            n_roots=int(eval_roots),
            n_worlds=int(eval_worlds or train_cfg.n_worlds),
            initial_chips=train_cfg.initial_chips,
            small_blind=train_cfg.small_blind,
            big_blind=train_cfg.big_blind,
            max_steps_per_hand=train_cfg.max_steps_per_hand,
            seed=int(train_cfg.seed + 7919 if eval_seed is None else eval_seed),
            checkpoint=train_cfg.checkpoint,
            strategy_source=train_cfg.strategy_source,
            continuation_checkpoints=train_cfg.continuation_checkpoints,
            continuation_strategy_source=train_cfg.continuation_strategy_source,
            greedy_continuation=train_cfg.greedy_continuation,
            device=train_cfg.device,
            include_positive_controls=False,
        )
        before_eval = evaluate_public_action_rollout_values(eval_cfg)
        after_cfg = PublicActionRolloutConfig(
            **{
                **eval_cfg.__dict__,
                "checkpoint": str(output_checkpoint),
                "strategy_source": after_strategy_source,
            }
        )
        after_eval = evaluate_public_action_rollout_values(after_cfg)
    eval_gate = None
    passed = bool(calibration["passed"])
    if before_eval is not None and after_eval is not None:
        eval_gate = decision_value_eval_gate(
            before_eval=before_eval,
            after_eval=after_eval,
            require_policy_ev_non_decrease=False,
        )
        passed = bool(passed and eval_gate["passed"])
    return {
        "mode": "decision_value_actor_pilot",
        "passed": passed,
        "actor_target": actor_target,
        "after_strategy_source": after_strategy_source,
        "checkpoint": train_cfg.checkpoint,
        "output_checkpoint": str(output_checkpoint),
        "target_metadata": target_metadata,
        "calibration": calibration,
        "eval_gate": eval_gate,
        "before_eval": before_eval,
        "after_eval": after_eval,
    }


def run_on_policy_decision_value_actor_pilot(
    *,
    train_cfg: OnPolicyDecisionValueTargetConfig,
    output_checkpoint: str | Path,
    actor_target: str = "policy-head",
    actor_update: str = "mirror-ce",
    n_steps: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    kl_beta: float = 1.0,
    eval_roots: int = 0,
    eval_worlds: int | None = None,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    """Collect on-policy multi-street targets, train an actor, and optionally evaluate."""
    targets, target_metadata = collect_on_policy_decision_value_policy_targets(train_cfg)
    if actor_update not in ("mirror-ce", "kl-q"):
        raise ValueError("actor_update must be 'mirror-ce' or 'kl-q'")
    if actor_update == "kl-q":
        if actor_target != "policy-head":
            raise ValueError("actor_update='kl-q' currently supports actor_target='policy-head'")
        q_targets = build_decision_value_q_target_buffer(
            targets,
            target_metadata=target_metadata,
        )
        calibration = calibrate_policy_head_with_kl_q_targets(
            train_cfg.checkpoint,
            q_targets,
            output_checkpoint,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            kl_beta=kl_beta,
            device=train_cfg.device,
            target_metadata=target_metadata,
        )
        after_strategy_source = "policy-head-covered"
    elif actor_target == "average-policy":
        calibration = calibrate_average_policy_net_from_targets(
            train_cfg.checkpoint,
            targets,
            output_checkpoint,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            device=train_cfg.device,
            target_metadata=target_metadata,
        )
        after_strategy_source = "average-policy"
    elif actor_target == "policy-head":
        calibration = calibrate_policy_head_from_decision_value_targets(
            train_cfg.checkpoint,
            targets,
            output_checkpoint,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            device=train_cfg.device,
            target_metadata=target_metadata,
        )
        after_strategy_source = "policy-head-covered"
    else:
        raise ValueError("actor_target must be 'average-policy' or 'policy-head'")

    before_eval = None
    after_eval = None
    if int(eval_roots) > 0:
        eval_cfg = PublicActionRolloutConfig(
            n_roots=int(eval_roots),
            n_worlds=int(eval_worlds or train_cfg.n_worlds),
            initial_chips=train_cfg.initial_chips,
            small_blind=train_cfg.small_blind,
            big_blind=train_cfg.big_blind,
            max_steps_per_hand=train_cfg.max_steps_per_hand,
            seed=int(train_cfg.seed + 7919 if eval_seed is None else eval_seed),
            checkpoint=train_cfg.checkpoint,
            strategy_source=train_cfg.strategy_source,
            continuation_checkpoints=train_cfg.continuation_checkpoints or (train_cfg.checkpoint,),
            continuation_strategy_source=train_cfg.continuation_strategy_source,
            greedy_continuation=train_cfg.greedy_continuation,
            device=train_cfg.device,
            include_positive_controls=False,
        )
        before_eval = evaluate_public_action_rollout_values(eval_cfg)
        after_cfg = PublicActionRolloutConfig(
            **{
                **eval_cfg.__dict__,
                "checkpoint": str(output_checkpoint),
                "strategy_source": after_strategy_source,
            }
        )
        after_eval = evaluate_public_action_rollout_values(after_cfg)
    eval_gate = None
    passed = bool(calibration["passed"])
    if before_eval is not None and after_eval is not None:
        eval_gate = decision_value_eval_gate(
            before_eval=before_eval,
            after_eval=after_eval,
            require_policy_ev_non_decrease=True,
        )
        passed = bool(passed and eval_gate["passed"])
    return {
        "mode": "on_policy_decision_value_actor_pilot",
        "passed": passed,
        "actor_target": actor_target,
        "actor_update": actor_update,
        "after_strategy_source": after_strategy_source,
        "checkpoint": train_cfg.checkpoint,
        "output_checkpoint": str(output_checkpoint),
        "target_metadata": target_metadata,
        "calibration": calibration,
        "eval_gate": eval_gate,
        "before_eval": before_eval,
        "after_eval": after_eval,
    }
