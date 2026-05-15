"""Train a diagnostic opponent-response probe from Slumbot trace actions."""

from __future__ import annotations

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


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import card_str_to_index  # noqa: E402
from range_tracker import (  # noqa: E402
    N_ACTIONS,
    _build_features_batch,
    _get_legal_mask,
    _parse_action,
)


@dataclass(frozen=True)
class OpponentResponseDataset:
    features: np.ndarray
    legal_masks: np.ndarray
    target_probs: np.ndarray
    model_probs: np.ndarray
    uniform_probs: np.ndarray
    hand_indices: np.ndarray
    streets: tuple[str, ...]
    action_chars: tuple[str, ...]


class _ProbeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, N_ACTIONS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _target_vector(record: dict[str, Any]) -> np.ndarray:
    target = np.zeros(N_ACTIONS, dtype=np.float32)
    for item in record.get("mapped_actions") or []:
        try:
            action_idx = int(item["action_idx"])
            weight = float(item.get("weight", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= action_idx < N_ACTIONS and weight > 0.0:
            target[action_idx] += weight
    total = float(target.sum())
    if total > 0.0:
        target /= total
    return target


def _standardize_pair(
    train_x: np.ndarray,
    holdout_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((holdout_x - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def load_opponent_response_dataset(action_likelihood_json: str | Path) -> OpponentResponseDataset:
    """Load revealed-hand opponent-action records into model features."""
    payload = _read_json(action_likelihood_json)
    features: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model_probs: list[float] = []
    uniform_probs: list[float] = []
    hand_indices: list[int] = []
    streets: list[str] = []
    action_chars: list[str] = []

    for record in payload.get("records") or []:
        bot_cards = list(record.get("bot_hole_cards") or [])
        if len(bot_cards) != 2:
            continue
        before = str(record.get("action_str_before") or "")
        parsed = _parse_action(before)
        if "error" in parsed:
            continue
        try:
            client_pos = int(record["client_pos"])
            hand_index = int(record["hand_index"])
        except (KeyError, TypeError, ValueError):
            continue
        target = _target_vector(record)
        if float(target.sum()) <= 0.0:
            continue
        opp_pos = 1 - client_pos
        legal = _get_legal_mask(parsed, before, opp_pos).astype(np.float32)
        if float(legal.sum()) <= 0.0:
            continue
        visible_board = list(record.get("visible_board") or [])
        board_idx = [card_str_to_index(card) for card in visible_board]
        bot_hand = tuple(sorted(card_str_to_index(card) for card in bot_cards))
        row = _build_features_batch([bot_hand], board_idx, before, opp_pos, parsed)[0]
        features.append(row.astype(np.float32, copy=False))
        legal_masks.append(legal)
        targets.append(target)
        model_probs.append(float(record.get("action_prob", 0.0)))
        uniform_probs.append(float(np.dot(target, legal / max(float(legal.sum()), 1.0))))
        hand_indices.append(hand_index)
        streets.append(str(record.get("street") or "unknown"))
        action_chars.append(str(record.get("action_char") or "?"))

    if not features:
        raise ValueError(f"no usable opponent-response records in {action_likelihood_json}")
    return OpponentResponseDataset(
        features=np.stack(features).astype(np.float32),
        legal_masks=np.stack(legal_masks).astype(np.float32),
        target_probs=np.stack(targets).astype(np.float32),
        model_probs=np.asarray(model_probs, dtype=np.float32),
        uniform_probs=np.asarray(uniform_probs, dtype=np.float32),
        hand_indices=np.asarray(hand_indices, dtype=np.int64),
        streets=tuple(streets),
        action_chars=tuple(action_chars),
    )


def _masked_cross_entropy(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    legal = (legal_masks > 0).to(dtype=logits.dtype)
    targets = target_probs.to(dtype=logits.dtype) * legal
    totals = targets.sum(dim=1, keepdim=True)
    legal_totals = legal.sum(dim=1, keepdim=True).clamp(min=1.0)
    fallback = legal / legal_totals
    targets = torch.where(totals > 1e-8, targets / totals.clamp(min=1e-8), fallback)
    masked_logits = logits.masked_fill(legal <= 0, -1e4)
    return -(targets * torch.log_softmax(masked_logits, dim=1)).sum(dim=1).mean()


def _predict_probs(
    model: nn.Module,
    x: np.ndarray,
    legal_masks: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        legal = torch.from_numpy(legal_masks).to(device)
        logits = logits.masked_fill(legal <= 0, -1e4)
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)


def _split_by_hand(
    hand_indices: np.ndarray,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted(set(int(x) for x in hand_indices)), dtype=np.int64)
    n_holdout = max(1, int(round(len(unique) * float(holdout_fraction))))
    n_holdout = min(n_holdout, max(1, len(unique) - 1))
    holdout_hands = set(int(x) for x in unique[-n_holdout:])
    holdout_mask = np.array([int(x) in holdout_hands for x in hand_indices], dtype=bool)
    train_mask = ~holdout_mask
    if not train_mask.any() or not holdout_mask.any():
        raise ValueError("hand split produced an empty train or holdout set")
    return train_mask, holdout_mask


def _metrics(
    *,
    probs: np.ndarray,
    dataset: OpponentResponseDataset,
    mask: np.ndarray,
) -> dict[str, Any]:
    target = dataset.target_probs[mask]
    uniform = np.maximum(dataset.uniform_probs[mask], 1e-12)
    model_prob = np.maximum(dataset.model_probs[mask], 1e-12)
    probe_prob = np.maximum(np.sum(probs[mask] * target, axis=1), 1e-12)
    eps = 1e-12
    ce = -np.sum(target * np.log(np.maximum(probs[mask], eps)), axis=1)
    return {
        "n": int(mask.sum()),
        "probe_mean_prob": float(np.mean(probe_prob)),
        "model_mean_prob": float(np.mean(model_prob)),
        "uniform_mean_prob": float(np.mean(uniform)),
        "probe_mean_cross_entropy": float(np.mean(ce)),
        "probe_mean_log_lift_vs_uniform": float(
            np.mean(np.log(probe_prob) - np.log(uniform))
        ),
        "model_mean_log_lift_vs_uniform": float(
            np.mean(np.log(model_prob) - np.log(uniform))
        ),
        "probe_minus_model_log_lift": float(
            np.mean(np.log(probe_prob) - np.log(model_prob))
        ),
    }


def train_opponent_response_probe(
    action_likelihood_json: str | Path,
    *,
    holdout_fraction: float = 0.3,
    hidden_dim: int = 128,
    n_layers: int = 2,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "auto",
    seed: int = 20260515,
) -> dict[str, Any]:
    """Train a hand-held-out diagnostic action-likelihood probe."""
    dataset = load_opponent_response_dataset(action_likelihood_json)
    train_mask, holdout_mask = _split_by_hand(dataset.hand_indices, holdout_fraction)
    x_train, x_holdout, _, _ = _standardize_pair(
        dataset.features[train_mask],
        dataset.features[holdout_mask],
    )
    x_all = np.zeros_like(dataset.features, dtype=np.float32)
    x_all[train_mask] = x_train
    x_all[holdout_mask] = x_holdout

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _ProbeNet(dataset.features.shape[1], hidden_dim, n_layers).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = np.flatnonzero(train_mask)
    x_t = torch.from_numpy(x_all).to(resolved_device)
    legal_t = torch.from_numpy(dataset.legal_masks).to(resolved_device)
    target_t = torch.from_numpy(dataset.target_probs).to(resolved_device)

    for _ in range(max(1, int(epochs))):
        perm = np.random.permutation(train_idx)
        for start in range(0, len(perm), max(1, int(batch_size))):
            batch = perm[start : start + max(1, int(batch_size))]
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_cross_entropy(
                model(x_t[batch]),
                legal_t[batch],
                target_t[batch],
            )
            loss.backward()
            optimizer.step()

    probs = _predict_probs(model, x_all, dataset.legal_masks, resolved_device)
    train_metrics = _metrics(probs=probs, dataset=dataset, mask=train_mask)
    holdout_metrics = _metrics(probs=probs, dataset=dataset, mask=holdout_mask)
    return {
        "mode": "slumbot_opponent_response_probe",
        "source": str(action_likelihood_json),
        "device": str(resolved_device),
        "n_records": int(dataset.features.shape[0]),
        "n_hands": int(len(set(int(x) for x in dataset.hand_indices))),
        "holdout_fraction": float(holdout_fraction),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "train": train_metrics,
        "holdout": holdout_metrics,
        "probe_beats_model_on_holdout": bool(
            holdout_metrics["probe_minus_model_log_lift"] > 0.0
        ),
        "passed": True,
    }
