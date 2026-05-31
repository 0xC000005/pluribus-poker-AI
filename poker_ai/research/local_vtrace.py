"""Small V-trace policy/value loss helpers for local poker RL pilots.

This module is intentionally narrow. It provides the standard trajectory update
needed by local actor-learner experiments; poker-specific logic should stay in
environment adapters, legal masks, trajectory packing, and evaluation gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VTraceReturns:
    vs: torch.Tensor
    pg_advantages: torch.Tensor
    rhos: torch.Tensor
    clipped_rhos: torch.Tensor


def masked_log_probs(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Return log-probabilities with illegal actions assigned `-inf`."""
    mask = legal_mask.to(dtype=torch.bool, device=logits.device)
    if logits.shape != mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every policy row must have at least one legal action")
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    return F.log_softmax(masked_logits, dim=-1)


def vtrace_from_importance_weights(
    *,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    target_action_log_probs: torch.Tensor,
    behavior_action_log_probs: torch.Tensor,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> VTraceReturns:
    """Compute V-trace state values and policy-gradient advantages.

    Tensors are shaped `[time, batch]`, except `bootstrap_value` which is
    `[batch]`. The formula follows the IMPALA V-trace recurrence.
    """
    for name, tensor in {
        "rewards": rewards,
        "discounts": discounts,
        "values": values,
        "target_action_log_probs": target_action_log_probs,
        "behavior_action_log_probs": behavior_action_log_probs,
    }.items():
        if tensor.shape != values.shape:
            raise ValueError(f"{name} must have shape [time, batch]")
    if bootstrap_value.shape != values.shape[1:]:
        raise ValueError("bootstrap_value must have shape [batch]")

    if valid_mask is None:
        valid = torch.ones_like(values, dtype=values.dtype)
    else:
        if valid_mask.shape != values.shape:
            raise ValueError("valid_mask must have shape [time, batch]")
        valid = valid_mask.to(device=values.device, dtype=values.dtype)

    rhos = torch.exp(target_action_log_probs - behavior_action_log_probs) * valid
    clipped_rhos = torch.clamp(rhos, max=float(clip_rho_threshold))
    clipped_pg_rhos = torch.clamp(rhos, max=float(clip_pg_rho_threshold))
    cs = torch.clamp(rhos, max=float(clip_c_threshold))

    values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values) * valid

    acc = torch.zeros_like(bootstrap_value)
    vs_minus_v = torch.zeros_like(values)
    for t in range(values.shape[0] - 1, -1, -1):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        vs_minus_v[t] = acc
    vs = values + vs_minus_v

    vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values) * valid
    return VTraceReturns(
        vs=vs,
        pg_advantages=pg_advantages,
        rhos=rhos,
        clipped_rhos=clipped_rhos,
    )


def vtrace_policy_value_loss(
    *,
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    behavior_action_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    bootstrap_value: torch.Tensor,
    value_loss_weight: float = 0.5,
    entropy_weight: float = 0.0,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a masked discrete-action V-trace actor-critic loss."""
    log_probs = masked_log_probs(logits, legal_mask)
    action_log_probs = log_probs.gather(-1, actions.to(device=logits.device).unsqueeze(-1)).squeeze(-1)
    if valid_mask is None:
        valid = torch.ones_like(values, dtype=values.dtype, device=logits.device)
    else:
        if valid_mask.shape != values.shape:
            raise ValueError("valid_mask must have shape [time, batch]")
        valid = valid_mask.to(device=logits.device, dtype=values.dtype)
    valid_count = valid.sum().clamp_min(1.0)
    vtrace = vtrace_from_importance_weights(
        rewards=rewards.to(device=logits.device, dtype=values.dtype),
        discounts=discounts.to(device=logits.device, dtype=values.dtype),
        values=values,
        bootstrap_value=bootstrap_value.to(device=logits.device, dtype=values.dtype),
        target_action_log_probs=action_log_probs,
        behavior_action_log_probs=behavior_action_log_probs.to(device=logits.device, dtype=values.dtype),
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        clip_c_threshold=clip_c_threshold,
        valid_mask=valid,
    )
    policy_loss = -((action_log_probs * vtrace.pg_advantages.detach()) * valid).sum() / valid_count
    value_loss = 0.5 * (torch.square(values - vtrace.vs.detach()) * valid).sum() / valid_count
    probs = torch.exp(log_probs)
    safe_log_probs = torch.nan_to_num(log_probs, neginf=0.0)
    entropy = -((probs * safe_log_probs).sum(dim=-1) * valid).sum() / valid_count
    loss = policy_loss + float(value_loss_weight) * value_loss - float(entropy_weight) * entropy
    illegal_probability = probs.masked_fill(legal_mask.to(dtype=torch.bool, device=logits.device), 0.0).sum(dim=-1).max()
    stats = {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "mean_rho": float(vtrace.rhos.detach().mean().cpu()),
        "illegal_action_probability": float(illegal_probability.detach().cpu()),
        "valid_samples": float(valid_count.detach().cpu()),
    }
    return loss, stats
