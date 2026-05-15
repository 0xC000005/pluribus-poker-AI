"""Capacity probe for restricted root action-value targets."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import ACTION_TO_INDEX, N_ACTIONS, N_FEATURES, new_game
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.restricted_action_value import (
    _advantage_value_corr,
    _set_private_cards_for_control,
    sample_seeded_hole_cards,
    score_legal_actions_by_showdown_equity,
)


@dataclass(frozen=True)
class RestrictedValueDataset:
    features: np.ndarray
    targets: np.ndarray
    legal_masks: np.ndarray
    best_actions: np.ndarray


@dataclass(frozen=True)
class RestrictedValueProbeConfig:
    train_roots: int = 512
    holdout_roots: int = 128
    n_equity_samples: int = 512
    initial_chips: int = 1000
    hidden_dim: int = 128
    n_layers: int = 2
    epochs: int = 80
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 20260517
    device: str = "auto"
    output_checkpoint: str | None = None


def build_restricted_value_dataset(
    *,
    n_roots: int,
    n_equity_samples: int,
    initial_chips: int,
    seed: int,
) -> RestrictedValueDataset:
    features = np.zeros((int(n_roots), N_FEATURES), dtype=np.float32)
    targets = np.zeros((int(n_roots), N_ACTIONS), dtype=np.float32)
    legal_masks = np.zeros((int(n_roots), N_ACTIONS), dtype=np.float32)
    best_actions = np.zeros(int(n_roots), dtype=np.int64)
    scale = float(max(int(initial_chips), 1))

    for root_idx in range(int(n_roots)):
        state = new_game(2, initial_chips=int(initial_chips))
        _set_private_cards_for_control(
            state,
            int(state.player_i),
            list(sample_seeded_hole_cards(seed=int(seed), root_idx=root_idx)),
        )
        scored = score_legal_actions_by_showdown_equity(
            state,
            player_i=int(state.player_i),
            n_equity_samples=int(n_equity_samples),
            seed=int(seed) + 10_000 + root_idx,
        )
        features[root_idx] = state.to_feature_vector()
        for action, value in scored["action_values"].items():
            action_idx = ACTION_TO_INDEX[action]
            legal_masks[root_idx, action_idx] = 1.0
            targets[root_idx, action_idx] = float(value) / scale
        best_actions[root_idx] = ACTION_TO_INDEX[str(scored["best_action"])]

    return RestrictedValueDataset(
        features=features,
        targets=targets,
        legal_masks=legal_masks,
        best_actions=best_actions,
    )


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    sq = ((pred - target) ** 2) * mask
    return sq.sum() / torch.clamp(mask.sum(), min=1.0)


def _full_loss(
    model: nn.Module,
    dataset: RestrictedValueDataset,
    device: torch.device,
) -> float:
    with torch.no_grad():
        x = torch.from_numpy(dataset.features).to(device)
        y = torch.from_numpy(dataset.targets).to(device)
        mask = torch.from_numpy(dataset.legal_masks).to(device)
        pred = model(x)
        return float(_masked_mse(pred, y, mask).detach().cpu().item())


def _evaluate_probe(
    model: nn.Module,
    dataset: RestrictedValueDataset,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        pred = model(torch.from_numpy(dataset.features).to(device)).detach().cpu().numpy()
    masked_pred = np.where(dataset.legal_masks > 0, pred, -1e9)
    top_actions = np.argmax(masked_pred, axis=1)
    matches = top_actions == dataset.best_actions
    corrs = []
    for row_idx in range(dataset.features.shape[0]):
        action_values = {
            action: float(dataset.targets[row_idx, action_idx])
            for action, action_idx in ACTION_TO_INDEX.items()
            if dataset.legal_masks[row_idx, action_idx] > 0
        }
        corrs.append(_advantage_value_corr(pred[row_idx], action_values))
    return {
        "top_action_match": float(np.mean(matches)) if matches.size else 0.0,
        "mean_action_corr": float(np.mean(corrs)) if corrs else 0.0,
    }


def train_restricted_value_probe(cfg: RestrictedValueProbeConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    train = build_restricted_value_dataset(
        n_roots=int(cfg.train_roots),
        n_equity_samples=int(cfg.n_equity_samples),
        initial_chips=int(cfg.initial_chips),
        seed=int(cfg.seed),
    )
    holdout = build_restricted_value_dataset(
        n_roots=int(cfg.holdout_roots),
        n_equity_samples=int(cfg.n_equity_samples),
        initial_chips=int(cfg.initial_chips),
        seed=int(cfg.seed) + 1_000_000,
    )
    model = ValueNetwork(
        N_FEATURES,
        hidden_dim=int(cfg.hidden_dim),
        output_dim=N_ACTIONS,
        n_layers=int(cfg.n_layers),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.lr))
    rng = np.random.default_rng(int(cfg.seed))
    train_loss_initial = _full_loss(model, train, device)
    started = time.perf_counter()

    n_train = int(train.features.shape[0])
    batch_size = max(1, min(int(cfg.batch_size), n_train))
    for _epoch in range(int(cfg.epochs)):
        order = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            x = torch.from_numpy(train.features[idx]).to(device)
            y = torch.from_numpy(train.targets[idx]).to(device)
            mask = torch.from_numpy(train.legal_masks[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_mse(model(x), y, mask)
            loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    train_loss_final = _full_loss(model, train, device)
    holdout_loss = _full_loss(model, holdout, device)
    holdout_metrics = _evaluate_probe(model, holdout, device)
    checkpoint_path = str(cfg.output_checkpoint) if cfg.output_checkpoint else ""
    if checkpoint_path:
        torch.save(
            {
                "value_net": model.state_dict(),
                "hidden_dim": int(cfg.hidden_dim),
                "n_layers": int(cfg.n_layers),
                "input_dim": N_FEATURES,
                "output_dim": N_ACTIONS,
                "initial_chips": int(cfg.initial_chips),
                "uses_betting_history": True,
                "metadata": {
                    "algorithm": "restricted_value_probe",
                    "warning": "Diagnostic action-value baseline checkpoint; not a poker policy.",
                    "train_roots": int(cfg.train_roots),
                    "holdout_roots": int(cfg.holdout_roots),
                    "n_equity_samples": int(cfg.n_equity_samples),
                    "seed": int(cfg.seed),
                },
            },
            checkpoint_path,
        )

    return {
        "algorithm": "restricted_value_probe",
        "role": "architecture_capacity_diagnostic_not_strategy",
        "warning": "Restricted target capacity probe only; not a poker policy.",
        **device_info,
        "train_roots": int(cfg.train_roots),
        "holdout_roots": int(cfg.holdout_roots),
        "n_equity_samples": int(cfg.n_equity_samples),
        "hidden_dim": int(cfg.hidden_dim),
        "n_layers": int(cfg.n_layers),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "seconds": float(seconds),
        "train_loss_initial": float(train_loss_initial),
        "train_loss_final": float(train_loss_final),
        "holdout_loss": float(holdout_loss),
        "holdout_top_action_match": holdout_metrics["top_action_match"],
        "holdout_mean_action_corr": holdout_metrics["mean_action_corr"],
        "checkpoint": checkpoint_path,
        "promotion": False,
    }
