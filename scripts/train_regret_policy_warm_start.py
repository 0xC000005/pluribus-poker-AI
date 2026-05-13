#!/usr/bin/env python3
"""Train a neural solver-state initializer for CFR+ warm starts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.games.full_deck.state import N_FEATURES
from poker_ai.research.belief_probe import BELIEF_DIM, _normalize_targets, _policy_metrics, _resolve_device
from poker_ai.research.belief_value_probe import _apply_standardization, _standardization_stats, save_metrics

from build_regret_policy_warm_start_targets import normalize_action_rows  # noqa: E402

_MAX_DECODED_LOG_FIELD = 12.0


@dataclass(frozen=True)
class RegretPolicyWarmStartDataset:
    features: np.ndarray
    policy_features: np.ndarray
    belief: np.ndarray
    legal_masks: np.ndarray
    target_regret_sum: np.ndarray
    target_strategy_sum: np.ndarray
    target_probs: np.ndarray
    low_regret_sum: np.ndarray
    low_strategy_sum: np.ndarray
    labels: tuple[str, ...]
    root_labels: tuple[str, ...]


class RegretPolicyWarmStartNet(nn.Module):
    """Predict selected-node regret/policy distributions plus their masses."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        if int(n_layers) < 1:
            raise ValueError("n_layers must be positive")
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 2 * N_ACTIONS + 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(x)
        regret_logits = out[:, :N_ACTIONS]
        strategy_logits = out[:, N_ACTIONS : 2 * N_ACTIONS]
        regret_log_mass = out[:, 2 * N_ACTIONS]
        strategy_log_mass = out[:, 2 * N_ACTIONS + 1]
        return regret_logits, strategy_logits, regret_log_mass, strategy_log_mass


def load_regret_policy_dataset(path: str | Path) -> RegretPolicyWarmStartDataset:
    data = np.load(Path(path), allow_pickle=False)
    required = (
        "features",
        "policy_features",
        "belief",
        "legal_masks",
        "target_regret_sum",
        "target_strategy_sum",
        "target_probs",
        "low_regret_sum",
        "low_strategy_sum",
        "labels",
        "root_labels",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    labels = tuple(str(item) for item in data["labels"].tolist())
    root_labels = tuple(str(item) for item in data["root_labels"].tolist())
    return RegretPolicyWarmStartDataset(
        features=data["features"].astype(np.float32, copy=False),
        policy_features=data["policy_features"].astype(np.float32, copy=False),
        belief=data["belief"].astype(np.float32, copy=False),
        legal_masks=data["legal_masks"].astype(np.float32, copy=False),
        target_regret_sum=data["target_regret_sum"].astype(np.float32, copy=False),
        target_strategy_sum=data["target_strategy_sum"].astype(np.float32, copy=False),
        target_probs=_normalize_targets(data["target_probs"], data["legal_masks"]),
        low_regret_sum=data["low_regret_sum"].astype(np.float32, copy=False),
        low_strategy_sum=data["low_strategy_sum"].astype(np.float32, copy=False),
        labels=labels,
        root_labels=root_labels,
    )


def root_disjoint_audit(
    train: RegretPolicyWarmStartDataset,
    holdout: RegretPolicyWarmStartDataset,
) -> dict[str, Any]:
    train_roots = set(train.root_labels)
    holdout_roots = set(holdout.root_labels)
    overlap = sorted(train_roots & holdout_roots)
    return {
        "train_root_count": int(len(train_roots)),
        "holdout_root_count": int(len(holdout_roots)),
        "overlap_count": int(len(overlap)),
        "overlap_sample": overlap[:10],
        "passed": bool(train_roots and holdout_roots and not overlap),
    }


def _dataset_validation(dataset: RegretPolicyWarmStartDataset) -> None:
    n = dataset.features.shape[0]
    expected = {
        "features": (n, N_FEATURES),
        "policy_features": (n, N_FEATURES),
        "belief": (n, BELIEF_DIM),
        "legal_masks": (n, N_ACTIONS),
        "target_regret_sum": (n, N_ACTIONS),
        "target_strategy_sum": (n, N_ACTIONS),
        "target_probs": (n, N_ACTIONS),
        "low_regret_sum": (n, N_ACTIONS),
        "low_strategy_sum": (n, N_ACTIONS),
    }
    for key, shape in expected.items():
        arr = getattr(dataset, key)
        if arr.shape != shape:
            raise ValueError(f"{key} shape {arr.shape} != {shape}")
    if len(dataset.labels) != n or len(dataset.root_labels) != n:
        raise ValueError("label counts must match row count")
    if np.any(dataset.legal_masks.sum(axis=1) <= 0):
        raise ValueError("each row needs at least one legal action")
    for key in ("target_regret_sum", "target_strategy_sum", "low_regret_sum", "low_strategy_sum"):
        arr = getattr(dataset, key)
        if not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError(f"{key} must be finite and nonnegative")


def make_model_inputs(
    dataset: RegretPolicyWarmStartDataset,
    *,
    public_mean: np.ndarray,
    public_std: np.ndarray,
    belief_mean: np.ndarray,
    belief_std: np.ndarray,
    include_low_state: bool = False,
) -> np.ndarray:
    public = _apply_standardization(dataset.features, public_mean, public_std)
    belief = _apply_standardization(dataset.belief, belief_mean, belief_std)
    parts = [
        public.astype(np.float32, copy=False),
        dataset.policy_features[:, :52].astype(np.float32, copy=False),
        belief.astype(np.float32, copy=False),
        dataset.legal_masks.astype(np.float32, copy=False),
    ]
    if include_low_state:
        low_regret_policy = fields_to_policy(dataset.low_regret_sum, dataset.legal_masks)
        low_strategy_policy = fields_to_policy(dataset.low_strategy_sum, dataset.legal_masks)
        low_regret_mass = np.log1p(dataset.low_regret_sum.sum(axis=1, keepdims=True)).astype(
            np.float32,
            copy=False,
        )
        low_strategy_mass = np.log1p(dataset.low_strategy_sum.sum(axis=1, keepdims=True)).astype(
            np.float32,
            copy=False,
        )
        parts.extend([low_regret_policy, low_strategy_policy, low_regret_mass, low_strategy_mass])
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def fields_to_policy(strategy_sum: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(strategy_sum, dtype=np.float32))
    legal_masks = np.asarray(legal_masks, dtype=np.float32)
    for idx in range(out.shape[0]):
        out[idx] = normalize_action_rows(strategy_sum[idx : idx + 1], legal_masks[idx])[0]
    return out


def logs_to_fields(log_values: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    legal = (np.asarray(legal_masks, dtype=np.float32) > 0).astype(np.float32)
    clipped = np.clip(np.asarray(log_values, dtype=np.float32), 0.0, _MAX_DECODED_LOG_FIELD)
    return (np.expm1(clipped) * legal).astype(np.float32, copy=False)


def _masked_cross_entropy(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    legal = (legal_masks > 0).to(dtype=logits.dtype)
    targets = target_probs.to(dtype=logits.dtype) * legal
    totals = targets.sum(dim=1, keepdim=True)
    fallback = legal / legal.sum(dim=1, keepdim=True).clamp(min=1.0)
    targets = torch.where(totals > 1e-8, targets / totals.clamp(min=1e-8), fallback)
    logits = logits.masked_fill(legal <= 0, -1e4)
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _predict_components(
    model: RegretPolicyWarmStartNet,
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    regret_logits_out: list[np.ndarray] = []
    strategy_logits_out: list[np.ndarray] = []
    regret_mass_out: list[np.ndarray] = []
    strategy_mass_out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], max(1, int(batch_size))):
            x = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            regret_logits, strategy_logits, regret_log_mass, strategy_log_mass = model(x)
            regret_logits_out.append(regret_logits.cpu().numpy().astype(np.float32))
            strategy_logits_out.append(strategy_logits.cpu().numpy().astype(np.float32))
            regret_mass_out.append(regret_log_mass.cpu().numpy().astype(np.float32))
            strategy_mass_out.append(strategy_log_mass.cpu().numpy().astype(np.float32))
    if not regret_logits_out:
        empty = np.zeros((0, N_ACTIONS), dtype=np.float32)
        empty_mass = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty_mass, empty_mass
    return (
        np.concatenate(regret_logits_out, axis=0),
        np.concatenate(strategy_logits_out, axis=0),
        np.concatenate(regret_mass_out, axis=0),
        np.concatenate(strategy_mass_out, axis=0),
    )


def logits_to_policy(logits: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    legal = (np.asarray(legal_masks, dtype=np.float32) > 0).astype(np.float32)
    masked = np.where(legal > 0, logits, -1e9)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    probs = np.exp(shifted) * legal
    return _normalize_targets(probs, legal)


def components_to_fields(
    regret_logits: np.ndarray,
    strategy_logits: np.ndarray,
    regret_log_mass: np.ndarray,
    strategy_log_mass: np.ndarray,
    legal_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    regret_probs = logits_to_policy(regret_logits, legal_masks)
    strategy_probs = logits_to_policy(strategy_logits, legal_masks)
    regret_mass = np.expm1(np.clip(np.asarray(regret_log_mass, dtype=np.float32), 0.0, _MAX_DECODED_LOG_FIELD))
    strategy_mass = np.expm1(
        np.clip(np.asarray(strategy_log_mass, dtype=np.float32), 0.0, _MAX_DECODED_LOG_FIELD)
    )
    return (
        (regret_probs * regret_mass.reshape(-1, 1)).astype(np.float32, copy=False),
        (strategy_probs * strategy_mass.reshape(-1, 1)).astype(np.float32, copy=False),
    )


def predict_regret_policy_fields(
    checkpoint: str | Path,
    *,
    features: np.ndarray,
    policy_features: np.ndarray,
    belief: np.ndarray,
    legal_masks: np.ndarray,
    low_regret_sum: np.ndarray | None = None,
    low_strategy_sum: np.ndarray | None = None,
    device: str | torch.device = "auto",
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_device = _resolve_device(device)
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    if payload.get("mode") != "regret_policy_warm_start_checkpoint":
        raise ValueError(f"{checkpoint} is not a regret-policy warm-start checkpoint")
    model = RegretPolicyWarmStartNet(
        int(payload["input_dim"]),
        int(payload["hidden_dim"]),
        int(payload["n_layers"]),
    ).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    include_low_state = bool(payload.get("include_low_state", False))
    low_regret_arr = (
        np.asarray(low_regret_sum, dtype=np.float32)
        if low_regret_sum is not None
        else np.zeros_like(np.asarray(legal_masks, dtype=np.float32))
    )
    low_strategy_arr = (
        np.asarray(low_strategy_sum, dtype=np.float32)
        if low_strategy_sum is not None
        else np.zeros_like(np.asarray(legal_masks, dtype=np.float32))
    )
    if include_low_state and (low_regret_sum is None or low_strategy_sum is None):
        raise ValueError("checkpoint requires low_regret_sum and low_strategy_sum inputs")
    dataset = RegretPolicyWarmStartDataset(
        features=np.asarray(features, dtype=np.float32),
        policy_features=np.asarray(policy_features, dtype=np.float32),
        belief=np.asarray(belief, dtype=np.float32),
        legal_masks=np.asarray(legal_masks, dtype=np.float32),
        target_regret_sum=np.zeros_like(np.asarray(legal_masks, dtype=np.float32)),
        target_strategy_sum=np.zeros_like(np.asarray(legal_masks, dtype=np.float32)),
        target_probs=np.asarray(legal_masks, dtype=np.float32),
        low_regret_sum=low_regret_arr,
        low_strategy_sum=low_strategy_arr,
        labels=tuple(str(i) for i in range(np.asarray(features).shape[0])),
        root_labels=tuple(str(i) for i in range(np.asarray(features).shape[0])),
    )
    x = make_model_inputs(
        dataset,
        public_mean=np.asarray(payload["public_mean"], dtype=np.float32),
        public_std=np.asarray(payload["public_std"], dtype=np.float32),
        belief_mean=np.asarray(payload["belief_mean"], dtype=np.float32),
        belief_std=np.asarray(payload["belief_std"], dtype=np.float32),
        include_low_state=include_low_state,
    )
    regret_logits, strategy_logits, regret_log_mass, strategy_log_mass = _predict_components(
        model,
        x,
        batch_size=batch_size,
        device=resolved_device,
    )
    return components_to_fields(
        regret_logits,
        strategy_logits,
        regret_log_mass,
        strategy_log_mass,
        legal_masks,
    )


def _baseline_metrics(
    dataset: RegretPolicyWarmStartDataset,
    pred_regret: np.ndarray,
    pred_strategy: np.ndarray,
) -> dict[str, Any]:
    legal = (dataset.legal_masks > 0).astype(np.float32)
    target_regret_log = np.log1p(dataset.target_regret_sum)
    target_strategy_log = np.log1p(dataset.target_strategy_sum)
    pred_regret_log = np.log1p(np.maximum(pred_regret, 0.0))
    pred_strategy_log = np.log1p(np.maximum(pred_strategy, 0.0))
    low_regret_log = np.log1p(dataset.low_regret_sum)
    low_strategy_log = np.log1p(dataset.low_strategy_sum)
    mask_count = float(max(legal.sum(), 1.0))
    pred_policy = fields_to_policy(pred_strategy, dataset.legal_masks)
    low_policy = fields_to_policy(dataset.low_strategy_sum, dataset.legal_masks)
    pred_regret_policy = fields_to_policy(pred_regret, dataset.legal_masks)
    target_regret_policy = fields_to_policy(dataset.target_regret_sum, dataset.legal_masks)
    low_regret_policy = fields_to_policy(dataset.low_regret_sum, dataset.legal_masks)
    pred_regret_mass_log = np.log1p(np.maximum(pred_regret.sum(axis=1), 0.0))
    pred_strategy_mass_log = np.log1p(np.maximum(pred_strategy.sum(axis=1), 0.0))
    target_regret_mass_log = np.log1p(np.maximum(dataset.target_regret_sum.sum(axis=1), 0.0))
    target_strategy_mass_log = np.log1p(np.maximum(dataset.target_strategy_sum.sum(axis=1), 0.0))
    low_regret_mass_log = np.log1p(np.maximum(dataset.low_regret_sum.sum(axis=1), 0.0))
    low_strategy_mass_log = np.log1p(np.maximum(dataset.low_strategy_sum.sum(axis=1), 0.0))
    return {
        "pred_regret_log_mse": round(float((((pred_regret_log - target_regret_log) ** 2) * legal).sum() / mask_count), 8),
        "pred_strategy_log_mse": round(float((((pred_strategy_log - target_strategy_log) ** 2) * legal).sum() / mask_count), 8),
        "low_regret_log_mse": round(float((((low_regret_log - target_regret_log) ** 2) * legal).sum() / mask_count), 8),
        "low_strategy_log_mse": round(float((((low_strategy_log - target_strategy_log) ** 2) * legal).sum() / mask_count), 8),
        "pred_regret_mass_log_mse": round(float(np.mean((pred_regret_mass_log - target_regret_mass_log) ** 2)), 8),
        "pred_strategy_mass_log_mse": round(float(np.mean((pred_strategy_mass_log - target_strategy_mass_log) ** 2)), 8),
        "low_regret_mass_log_mse": round(float(np.mean((low_regret_mass_log - target_regret_mass_log) ** 2)), 8),
        "low_strategy_mass_log_mse": round(float(np.mean((low_strategy_mass_log - target_strategy_mass_log) ** 2)), 8),
        "pred_regret_policy": _policy_metrics(pred_regret_policy, target_regret_policy, dataset.legal_masks),
        "low_regret_policy": _policy_metrics(low_regret_policy, target_regret_policy, dataset.legal_masks),
        "pred_policy": _policy_metrics(pred_policy, dataset.target_probs, dataset.legal_masks),
        "low_policy": _policy_metrics(low_policy, dataset.target_probs, dataset.legal_masks),
        "legal_uniform_policy": _policy_metrics(
            _normalize_targets(dataset.legal_masks, dataset.legal_masks),
            dataset.target_probs,
            dataset.legal_masks,
        ),
    }


def train_regret_policy_warm_start(
    *,
    train_npz: str | Path,
    holdout_npz: str | Path,
    output_checkpoint: str | Path | None = None,
    device: str | torch.device = "auto",
    hidden_dim: int = 256,
    n_layers: int = 2,
    epochs: int = 12,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    amp_dtype: str = "bf16",
    include_low_state: bool = False,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train = load_regret_policy_dataset(train_npz)
    holdout = load_regret_policy_dataset(holdout_npz)
    _dataset_validation(train)
    _dataset_validation(holdout)
    root_audit = root_disjoint_audit(train, holdout)
    public_mean, public_std = _standardization_stats(train.features)
    belief_mean, belief_std = _standardization_stats(train.belief)
    train_x = make_model_inputs(
        train,
        public_mean=public_mean,
        public_std=public_std,
        belief_mean=belief_mean,
        belief_std=belief_std,
        include_low_state=include_low_state,
    )
    holdout_x = make_model_inputs(
        holdout,
        public_mean=public_mean,
        public_std=public_std,
        belief_mean=belief_mean,
        belief_std=belief_std,
        include_low_state=include_low_state,
    )
    torch.manual_seed(int(seed))
    model = RegretPolicyWarmStartNet(train_x.shape[1], int(hidden_dim), int(n_layers)).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    x_t = torch.from_numpy(train_x).to(resolved_device)
    legal_t = torch.from_numpy(train.legal_masks).to(resolved_device)
    target_regret_probs_t = torch.from_numpy(fields_to_policy(train.target_regret_sum, train.legal_masks)).to(
        resolved_device
    )
    target_probs_t = torch.from_numpy(train.target_probs).to(resolved_device)
    target_regret_mass_t = torch.from_numpy(np.log1p(train.target_regret_sum.sum(axis=1))).to(resolved_device)
    target_strategy_mass_t = torch.from_numpy(np.log1p(train.target_strategy_sum.sum(axis=1))).to(
        resolved_device
    )
    n = int(x_t.shape[0])
    batch_size = max(1, min(int(batch_size), n))
    use_amp = resolved_device.type == "cuda" and str(amp_dtype) != "none"
    amp_torch_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == "fp16")
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    losses: list[float] = []
    model.train()
    for _epoch in range(max(1, int(epochs))):
        perm = torch.randperm(n, generator=generator, device=resolved_device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=resolved_device.type, dtype=amp_torch_dtype, enabled=use_amp):
                regret_logits, strategy_logits, regret_log_mass, strategy_log_mass = model(
                    x_t.index_select(0, idx)
                )
                regret_loss = _masked_cross_entropy(
                    regret_logits,
                    legal_t.index_select(0, idx),
                    target_regret_probs_t.index_select(0, idx),
                )
                policy_loss = _masked_cross_entropy(
                    strategy_logits,
                    legal_t.index_select(0, idx),
                    target_probs_t.index_select(0, idx),
                )
                regret_mass_loss = nn.functional.mse_loss(
                    regret_log_mass,
                    target_regret_mass_t.index_select(0, idx).to(dtype=regret_log_mass.dtype),
                )
                strategy_mass_loss = nn.functional.mse_loss(
                    strategy_log_mass,
                    target_strategy_mass_t.index_select(0, idx).to(dtype=strategy_log_mass.dtype),
                )
                loss = regret_loss + policy_loss + regret_mass_loss + strategy_mass_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

    train_regret_logits, train_strategy_logits, train_regret_log_mass, train_strategy_log_mass = _predict_components(
        model,
        train_x,
        batch_size=batch_size,
        device=resolved_device,
    )
    (
        holdout_regret_logits,
        holdout_strategy_logits,
        holdout_regret_log_mass,
        holdout_strategy_log_mass,
    ) = _predict_components(
        model,
        holdout_x,
        batch_size=batch_size,
        device=resolved_device,
    )
    train_pred_regret, train_pred_strategy = components_to_fields(
        train_regret_logits,
        train_strategy_logits,
        train_regret_log_mass,
        train_strategy_log_mass,
        train.legal_masks,
    )
    holdout_pred_regret, holdout_pred_strategy = components_to_fields(
        holdout_regret_logits,
        holdout_strategy_logits,
        holdout_regret_log_mass,
        holdout_strategy_log_mass,
        holdout.legal_masks,
    )
    train_metrics = _baseline_metrics(train, train_pred_regret, train_pred_strategy)
    holdout_metrics = _baseline_metrics(holdout, holdout_pred_regret, holdout_pred_strategy)
    pred_policy = holdout_metrics["pred_policy"]
    low_policy = holdout_metrics["low_policy"]
    beats_low = bool(
        pred_policy["mean_l1"] < low_policy["mean_l1"]
        and pred_policy["mean_kl"] < low_policy["mean_kl"]
        and pred_policy["top1_match_rate"] >= low_policy["top1_match_rate"]
    )
    checkpoint_path = None
    if output_checkpoint is not None:
        path = Path(output_checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(path)
        torch.save(
            {
                "mode": "regret_policy_warm_start_checkpoint",
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "input_dim": int(train_x.shape[1]),
                "hidden_dim": int(hidden_dim),
                "n_layers": int(n_layers),
                "public_mean": public_mean,
                "public_std": public_std,
                "belief_mean": belief_mean,
                "belief_std": belief_std,
                "train_npz": str(train_npz),
                "holdout_npz": str(holdout_npz),
                "seed": int(seed),
                "include_low_state": bool(include_low_state),
            },
            path,
        )
    return {
        "mode": "regret_policy_warm_start_training",
        "passed": bool(root_audit["passed"] and beats_low),
        "pass_criteria": "root-disjoint holdout and predicted strategy policy beats low-solver policy on L1/KL without lower top-1 agreement",
        "device": str(resolved_device),
        "train_npz": str(train_npz),
        "holdout_npz": str(holdout_npz),
        "checkpoint": checkpoint_path,
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "amp_dtype": str(amp_dtype),
        "include_low_state": bool(include_low_state),
        "root_disjoint_audit": root_audit,
        "residual_beats_low_solver": bool(beats_low),
        "final_train_loss": round(float(losses[-1]), 8) if losses else math.nan,
        "mean_train_loss": round(float(np.mean(losses)), 8) if losses else math.nan,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a regret/policy solver-state warm-start predictor."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--output-json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--include-low-state-input",
        action="store_true",
        help="Condition on low-budget solver regret/policy fields as a residual diagnostic.",
    )
    args = parser.parse_args(argv)
    metrics = train_regret_policy_warm_start(
        train_npz=args.train,
        holdout_npz=args.holdout,
        output_checkpoint=args.output_checkpoint,
        device=args.device,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        amp_dtype=args.amp_dtype,
        include_low_state=args.include_low_state_input,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
