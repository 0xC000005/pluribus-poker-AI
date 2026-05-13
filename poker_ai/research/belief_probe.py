"""Probe whether public-belief range inputs explain search targets.

This is a diagnostic layer, not a promoted playing policy. It compares a
feature-only supervised probe against the same probe with raw learned
hero/villain range distributions appended. The goal is to test whether the
current search-target failure is a representation problem before changing the
main Deep CFR trainer.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.resolver_benchmark import (
    ResolverBenchmarkCase,
    load_cases_json,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import card_str_to_index, parse_action  # noqa: E402
from range_tracker import RangeTracker, update_tracker_from_actions  # noqa: E402


N_HANDS = 52 * 51 // 2
BELIEF_DIM = N_HANDS * 2
_ALL_HANDS = tuple(itertools.combinations(range(52), 2))
_HAND_TO_INDEX = {hand: i for i, hand in enumerate(_ALL_HANDS)}


@dataclass(frozen=True)
class BeliefProbeDataset:
    features: np.ndarray
    belief: np.ndarray
    legal_masks: np.ndarray
    target_probs: np.ndarray
    labels: tuple[str, ...]


class _ProbeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _normalize_targets(target_probs: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    legal = (np.asarray(legal_masks, dtype=np.float32) > 0).astype(np.float32)
    targets = np.asarray(target_probs, dtype=np.float32) * legal
    totals = targets.sum(axis=1, keepdims=True)
    legal_totals = legal.sum(axis=1, keepdims=True)
    fallback = legal / np.maximum(legal_totals, 1.0)
    return np.where(totals > 1e-8, targets / np.maximum(totals, 1e-8), fallback)


def _case_range_vectors(
    case: ResolverBenchmarkCase,
    *,
    value_net: Any,
    device: torch.device,
    strategy_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        raise ValueError(f"{case.label}: parse error: {parsed['error']}")
    street = int(parsed.get("st", -1))
    if street not in (2, 3):
        raise ValueError(f"{case.label}: expected turn/river street, got {street}")

    n_board = 4 if street == 2 else 5
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]

    tracker = RangeTracker(
        our_cards_idx,
        value_net,
        device,
        strategy_source=strategy_source,
    )
    update_tracker_from_actions(tracker, case.action_str, case.client_pos, board_idx)

    remaining = sorted(set(range(52)) - set(board_idx))
    solver_hands = list(itertools.combinations(remaining, 2))
    solver_hand_to_idx = {hand: i for i, hand in enumerate(solver_hands)}
    hero_range, villain_range = tracker.get_solver_ranges(solver_hands, solver_hand_to_idx)

    hero_full = np.zeros(N_HANDS, dtype=np.float32)
    villain_full = np.zeros(N_HANDS, dtype=np.float32)
    for local_idx, hand in enumerate(solver_hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        hero_full[global_idx] = float(hero_range[local_idx])
        villain_full[global_idx] = float(villain_range[local_idx])

    for vec in (hero_full, villain_full):
        total = float(vec.sum())
        if total > 0:
            vec /= total
    return hero_full, villain_full


def load_belief_probe_dataset(
    *,
    targets_npz: str | Path,
    cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
) -> BeliefProbeDataset:
    """Load policy targets and append raw learned range distributions."""
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(range_checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, range_strategy_source)
    targets = PolicyTargetBuffer.from_npz(targets_npz)
    cases = load_cases_json(cases_json)
    if len(cases) != targets.size:
        raise ValueError(
            f"case count {len(cases)} does not match target count {targets.size}"
        )

    belief_rows = []
    labels = []
    for case in cases:
        hero_range, villain_range = _case_range_vectors(
            case,
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=range_strategy_source,
        )
        belief_rows.append(np.concatenate([hero_range, villain_range], axis=0))
        labels.append(case.label)

    return BeliefProbeDataset(
        features=targets.features.astype(np.float32, copy=False),
        belief=np.stack(belief_rows, axis=0).astype(np.float32, copy=False),
        legal_masks=targets.legal_masks.astype(np.float32, copy=False),
        target_probs=targets.target_probs.astype(np.float32, copy=False),
        labels=tuple(labels),
    )


def _standardize_pair(
    train_x: np.ndarray,
    holdout_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((holdout_x - mean) / std).astype(np.float32),
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
        x_t = torch.from_numpy(x).to(device)
        legal_t = torch.from_numpy(legal_masks).to(device)
        logits = model(x_t)
        logits = logits.masked_fill(legal_t <= 0, -1e4)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return _normalize_targets(probs, legal_masks)


def _policy_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    legal_masks: np.ndarray,
) -> dict[str, float]:
    probs = _normalize_targets(probs, legal_masks)
    targets = _normalize_targets(targets, legal_masks)
    eps = 1e-8
    kl = targets * (np.log(np.maximum(targets, eps)) - np.log(np.maximum(probs, eps)))
    top_pred = np.argmax(probs, axis=1)
    top_target = np.argmax(targets, axis=1)
    return {
        "mean_l1": round(float(np.abs(probs - targets).sum(axis=1).mean()), 6),
        "mean_kl": round(float(kl.sum(axis=1).mean()), 6),
        "top1_match_rate": round(float(np.mean(top_pred == top_target)), 6),
        "policy_allin_rate": round(float(np.mean(top_pred == 8)), 6),
        "target_allin_rate": round(float(np.mean(top_target == 8)), 6),
    }


def _fit_probe(
    train_x: np.ndarray,
    train_legal: np.ndarray,
    train_targets: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> _ProbeNet:
    torch.manual_seed(seed)
    model = _ProbeNet(train_x.shape[1], hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.from_numpy(train_x).to(device)
    legal_t = torch.from_numpy(train_legal).to(device)
    targets_t = torch.from_numpy(train_targets).to(device)
    model.train()
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        loss = _masked_cross_entropy(model(x_t), legal_t, targets_t)
        loss.backward()
        optimizer.step()
    return model


def run_public_belief_probe(
    *,
    train_targets_npz: str | Path,
    train_cases_json: str | Path,
    holdout_targets_npz: str | Path,
    holdout_cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train = load_belief_probe_dataset(
        targets_npz=train_targets_npz,
        cases_json=train_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
    )
    holdout = load_belief_probe_dataset(
        targets_npz=holdout_targets_npz,
        cases_json=holdout_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
    )

    train_base, holdout_base = _standardize_pair(train.features, holdout.features)
    train_belief_input, holdout_belief_input = _standardize_pair(
        np.concatenate([train.features, train.belief], axis=1),
        np.concatenate([holdout.features, holdout.belief], axis=1),
    )

    base_model = _fit_probe(
        train_base,
        train.legal_masks,
        train.target_probs,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
    )
    belief_model = _fit_probe(
        train_belief_input,
        train.legal_masks,
        train.target_probs,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
    )

    base_train_probs = _predict_probs(
        base_model, train_base, train.legal_masks, resolved_device
    )
    base_holdout_probs = _predict_probs(
        base_model, holdout_base, holdout.legal_masks, resolved_device
    )
    belief_train_probs = _predict_probs(
        belief_model, train_belief_input, train.legal_masks, resolved_device
    )
    belief_holdout_probs = _predict_probs(
        belief_model, holdout_belief_input, holdout.legal_masks, resolved_device
    )

    base_holdout = _policy_metrics(
        base_holdout_probs, holdout.target_probs, holdout.legal_masks
    )
    belief_holdout = _policy_metrics(
        belief_holdout_probs, holdout.target_probs, holdout.legal_masks
    )
    holdout_l1_delta = round(
        float(base_holdout["mean_l1"] - belief_holdout["mean_l1"]),
        6,
    )
    holdout_kl_delta = round(
        float(base_holdout["mean_kl"] - belief_holdout["mean_kl"]),
        6,
    )
    return {
        "mode": "public_belief_probe",
        "passed": bool(holdout_l1_delta > 0 and holdout_kl_delta >= 0),
        "pass_criteria": "belief_holdout must improve mean_l1 and not worsen mean_kl",
        "device": str(resolved_device),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "range_checkpoint": str(range_checkpoint),
        "range_strategy_source": range_strategy_source,
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "feature_dim": int(N_FEATURES),
        "belief_dim": int(BELIEF_DIM),
        "base_train": _policy_metrics(
            base_train_probs, train.target_probs, train.legal_masks
        ),
        "base_holdout": base_holdout,
        "belief_train": _policy_metrics(
            belief_train_probs, train.target_probs, train.legal_masks
        ),
        "belief_holdout": belief_holdout,
        "holdout_l1_delta": holdout_l1_delta,
        "holdout_kl_delta": holdout_kl_delta,
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
