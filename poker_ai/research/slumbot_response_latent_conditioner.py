"""Latent diagnostic selector for response-conditioned resolving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.research.slumbot_response_decision_gate import (
    _feature_names,
    _load_ev_records,
    _matrix,
    _pearson,
    _selection_metrics,
    _standardize,
)


class _LatentSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.strategy_head = nn.Linear(dim, 1)
        self.selected_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.strategy_head(latent).squeeze(-1), self.selected_head(latent).squeeze(-1)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _target_stats(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def _standardize_target(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def _predict(
    model: _LatentSelector,
    x: np.ndarray,
    *,
    device: torch.device,
    strategy_mean: float,
    strategy_std: float,
    selected_mean: float,
    selected_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        strategy, selected = model(torch.from_numpy(x.astype(np.float32)).to(device))
    strategy_np = strategy.cpu().numpy().astype(np.float64) * strategy_std + strategy_mean
    selected_np = selected.cpu().numpy().astype(np.float64) * selected_std + selected_mean
    return strategy_np, selected_np


def evaluate_latent_response_conditioner(
    train_ev_gate_json: str | Path,
    eval_ev_gate_json: str | Path,
    *,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    threshold: float = 0.0,
    device: str | torch.device = "auto",
    seed: int = 20260515,
) -> dict[str, Any]:
    """Train a latent EV selector on one replay artifact and evaluate on another.

    The feature set is intentionally inherited from the shallow decision gate:
    it excludes revealed-hand likelihood, true-hand ranks, realized winnings,
    and counterfactual action values. This is a diagnostic falsifier for a
    learned response/public-belief state object, not a live Slumbot policy.
    """

    train_records = _load_ev_records(train_ev_gate_json)
    eval_records = _load_ev_records(eval_ev_gate_json)
    train_x, train_strategy_y, train_selected_y = _matrix(train_records)
    eval_x, eval_strategy_y, eval_selected_y = _matrix(eval_records)
    train_x_std, eval_x_std = _standardize(train_x, eval_x)

    strategy_mean, strategy_std = _target_stats(train_strategy_y)
    selected_mean, selected_std = _target_stats(train_selected_y)
    train_strategy_target = _standardize_target(train_strategy_y, strategy_mean, strategy_std)
    train_selected_target = _standardize_target(train_selected_y, selected_mean, selected_std)

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _LatentSelector(train_x_std.shape[1], hidden_dim, n_layers).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.from_numpy(train_x_std.astype(np.float32)).to(resolved_device)
    y_strategy_t = torch.from_numpy(train_strategy_target).to(resolved_device)
    y_selected_t = torch.from_numpy(train_selected_target).to(resolved_device)
    indices = np.arange(train_x_std.shape[0])

    for _ in range(max(1, int(epochs))):
        perm = np.random.permutation(indices)
        for start in range(0, len(perm), max(1, int(batch_size))):
            batch = perm[start : start + max(1, int(batch_size))]
            optimizer.zero_grad(set_to_none=True)
            pred_strategy, pred_selected = model(x_t[batch])
            loss = nn.functional.mse_loss(pred_strategy, y_strategy_t[batch])
            loss = loss + nn.functional.mse_loss(pred_selected, y_selected_t[batch])
            loss.backward()
            optimizer.step()

    train_strategy_pred, train_selected_pred = _predict(
        model,
        train_x_std,
        device=resolved_device,
        strategy_mean=strategy_mean,
        strategy_std=strategy_std,
        selected_mean=selected_mean,
        selected_std=selected_std,
    )
    eval_strategy_pred, eval_selected_pred = _predict(
        model,
        eval_x_std,
        device=resolved_device,
        strategy_mean=strategy_mean,
        strategy_std=strategy_std,
        selected_mean=selected_mean,
        selected_std=selected_std,
    )
    train_metrics = _selection_metrics(
        strategy_delta=train_strategy_y,
        selected_delta=train_selected_y,
        predictions=train_strategy_pred,
        threshold=threshold,
    )
    eval_metrics = _selection_metrics(
        strategy_delta=eval_strategy_y,
        selected_delta=eval_selected_y,
        predictions=eval_strategy_pred,
        threshold=threshold,
    )
    decision_passed = bool(
        eval_metrics["selected_count"] > 0
        and eval_metrics["selective_mean_strategy_ev_delta"]
        > max(0.0, eval_metrics["direct_mean_strategy_ev_delta"])
        and eval_metrics["selective_mean_selected_action_ev_delta"]
        > max(0.0, eval_metrics["direct_mean_selected_action_ev_delta"])
        and eval_metrics["selective_response_beats_baseline_rate"]
        >= eval_metrics["direct_response_beats_baseline_rate"]
    )
    return {
        "mode": "slumbot_response_latent_conditioner",
        "train_source": str(train_ev_gate_json),
        "eval_source": str(eval_ev_gate_json),
        "feature_names": _feature_names(),
        "device": str(resolved_device),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "threshold": float(threshold),
        "train_n": int(train_strategy_y.size),
        "eval_n": int(eval_strategy_y.size),
        "train_prediction_mae": float(np.abs(train_strategy_pred - train_strategy_y).mean()),
        "eval_prediction_mae": float(np.abs(eval_strategy_pred - eval_strategy_y).mean()),
        "train_prediction_pearson": _pearson(train_strategy_pred, train_strategy_y),
        "eval_prediction_pearson": _pearson(eval_strategy_pred, eval_strategy_y),
        "train_selected_prediction_mae": float(
            np.abs(train_selected_pred - train_selected_y).mean()
        ),
        "eval_selected_prediction_mae": float(
            np.abs(eval_selected_pred - eval_selected_y).mean()
        ),
        "train_selected_prediction_pearson": _pearson(
            train_selected_pred,
            train_selected_y,
        ),
        "eval_selected_prediction_pearson": _pearson(eval_selected_pred, eval_selected_y),
        "train": train_metrics,
        "eval": eval_metrics,
        **eval_metrics,
        "decision_gate_passed": decision_passed,
        "passed": True,
    }


def write_metrics(metrics: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
