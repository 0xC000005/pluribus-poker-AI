"""Train a neural policy consumer for XDO-lite target artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.neural_policy_iteration import (
    _ValueNet,
    _batched_network_policies,
    _PolicyNet,
)


@dataclass(frozen=True)
class XDOPolicyConsumerConfig:
    targets: str
    output: str
    hidden_dim: int = 256
    n_steps: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "auto"
    seed: int = 20260527
    min_training_action_agreement: float = 0.0
    parent_checkpoint: str = ""
    parent_kind: str = ""


def _seed_all(seed: int) -> np.random.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return np.random.default_rng(int(seed))


def _weighted_policy_loss(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    target_probs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(legal_masks <= 0, -1e4)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    per_row = -(target_probs * log_probs).sum(dim=1)
    weight_sum = torch.clamp(weights.sum(), min=1e-8)
    return (per_row * weights).sum() / weight_sum


def _fit_metrics(
    policy_net: torch.nn.Module,
    targets: PolicyTargetBuffer,
    device: torch.device,
) -> dict[str, float]:
    probs = _batched_network_policies(
        policy_net,
        targets.features,
        targets.legal_masks,
        device,
    )
    target_actions = np.argmax(targets.target_probs, axis=1)
    policy_actions = np.argmax(probs, axis=1)
    target_conf = np.max(targets.target_probs, axis=1)
    ce = -np.sum(
        targets.target_probs * np.log(np.clip(probs, 1e-8, 1.0)),
        axis=1,
    )
    return {
        "training_policy_cross_entropy": float(np.mean(ce)) if ce.size else 0.0,
        "training_action_agreement": float(np.mean(policy_actions == target_actions))
        if target_actions.size else 0.0,
        "mean_target_confidence": float(np.mean(target_conf)) if target_conf.size else 0.0,
    }


def train_xdo_policy_consumer(
    targets_npz: str | Path,
    output_checkpoint: str | Path,
    *,
    hidden_dim: int = 256,
    n_steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "auto",
    seed: int = 20260527,
    min_training_action_agreement: float = 0.0,
    parent_checkpoint: str = "",
    parent_kind: str = "",
    metrics_output: str | Path | None = None,
) -> dict:
    """Fit an NPI-compatible policy network to XDO target distributions."""
    started = time.perf_counter()
    rng = _seed_all(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    targets = PolicyTargetBuffer.from_npz(targets_npz)
    policy_net = _PolicyNet(int(hidden_dim), N_FEATURES).to(resolved_device)
    value_net = _ValueNet(int(hidden_dim), N_FEATURES).to(resolved_device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=float(lr))

    initial = _fit_metrics(policy_net, targets, resolved_device)
    losses: list[float] = []
    for _ in range(int(n_steps)):
        count = min(int(batch_size), int(targets.size))
        idx = rng.integers(0, int(targets.size), size=count)
        features = torch.from_numpy(targets.features[idx]).to(resolved_device)
        legal_masks = torch.from_numpy(targets.legal_masks[idx]).to(resolved_device)
        target_probs = torch.from_numpy(targets.target_probs[idx]).to(resolved_device)
        weights = torch.from_numpy(targets.weights[idx]).to(resolved_device)
        optimizer.zero_grad(set_to_none=True)
        loss = _weighted_policy_loss(
            policy_net(features),
            legal_masks,
            target_probs,
            weights,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    final = _fit_metrics(policy_net, targets, resolved_device)
    gate_failures: list[str] = []
    if not np.isfinite(np.asarray(losses, dtype=np.float64)).all():
        gate_failures.append("nonfinite_training_loss")
    if final["training_action_agreement"] < float(min_training_action_agreement):
        gate_failures.append("training_action_agreement_below_threshold")

    cfg = XDOPolicyConsumerConfig(
        targets=str(targets_npz),
        output=str(output_checkpoint),
        hidden_dim=int(hidden_dim),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        lr=float(lr),
        device=str(device),
        seed=int(seed),
        min_training_action_agreement=float(min_training_action_agreement),
        parent_checkpoint=str(parent_checkpoint),
        parent_kind=str(parent_kind),
    )
    metrics = {
        "algorithm": "xdo_lite_policy_target_consumer",
        "role": "information_state_response_policy_consumer",
        "passed": bool(not gate_failures),
        "promotion": False,
        "promotable": False,
        "promotion_blockers": [
            "consumer_fit_only",
            "requires_parent_population_h2h",
            "slumbot_heldout_confirmation_required",
        ],
        "gate_failures": gate_failures,
        "checkpoint": str(output_checkpoint),
        "targets": str(targets_npz),
        "target_size": int(targets.size),
        "feature_dim": N_FEATURES,
        "num_actions": N_ACTIONS,
        "hidden_dim": int(hidden_dim),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_kind": str(parent_kind),
        "initial_training_policy_cross_entropy": initial["training_policy_cross_entropy"],
        "initial_training_action_agreement": initial["training_action_agreement"],
        "final_loss": float(losses[-1]) if losses else 0.0,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "elapsed_seconds": float(time.perf_counter() - started),
        "uses_slumbot_training_data": False,
        **final,
        **device_info,
    }
    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": metrics["algorithm"],
            "role": metrics["role"],
            "config": asdict(cfg),
            "metrics": metrics,
            "parent_checkpoint_path": str(parent_checkpoint),
            "policy_net_state_dict": policy_net.state_dict(),
            "value_net_state_dict": value_net.state_dict(),
        },
        output_path,
    )
    if metrics_output is not None:
        metrics_path = Path(metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
