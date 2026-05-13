#!/usr/bin/env python3
"""Probe public-belief inputs for dual-player hand-CFV labels."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
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

from poker_ai.games.full_deck.state import N_FEATURES
from poker_ai.research.belief_probe import BELIEF_DIM, N_HANDS, _HAND_TO_INDEX, _resolve_device
from poker_ai.research.belief_value_probe import (
    _HAND_FEATURES,
    _apply_standardization,
    _standardization_stats,
    compute_hero_cfv_vector,
    compute_villain_cfv_vector,
    load_public_belief_cfv_dataset_cache,
    save_metrics,
)
from poker_ai.research.resolver_benchmark import load_cases_json

from play_slumbot import _compute_bets_before_street, card_str_to_index, parse_action
from solver import StreetSolver, _parse_nav, resolve_solver_backend


@dataclass(frozen=True)
class DualCFVDataset:
    features: np.ndarray
    belief: np.ndarray
    hero_values: np.ndarray
    villain_values: np.ndarray
    hero_masks: np.ndarray
    villain_masks: np.ndarray
    labels: tuple[str, ...]


class _DualHandCFVProbeNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        use_belief: bool,
        head_mode: str = "shared",
        belief_bottleneck_dim: int = 0,
    ):
        super().__init__()
        self.use_belief = bool(use_belief)
        if head_mode not in ("shared", "separate"):
            raise ValueError(f"unknown head_mode: {head_mode}")
        if belief_bottleneck_dim < 0:
            raise ValueError("belief_bottleneck_dim must be non-negative")
        self.head_mode = head_mode
        self.public = nn.Linear(N_FEATURES, hidden_dim)
        self.hand = nn.Linear(52, hidden_dim)
        self.player = nn.Linear(2, hidden_dim)
        if use_belief and belief_bottleneck_dim > 0:
            self.belief = nn.Sequential(
                nn.Linear(BELIEF_DIM, int(belief_bottleneck_dim)),
                nn.ReLU(),
                nn.Linear(int(belief_bottleneck_dim), hidden_dim),
            )
        elif use_belief:
            self.belief = nn.Linear(BELIEF_DIM, hidden_dim)
        else:
            self.belief = None
        self.body = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.hero_out = nn.Linear(hidden_dim, 1)
        self.villain_out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        public_x: torch.Tensor,
        hand_x: torch.Tensor,
        player_x: torch.Tensor,
        belief_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.public(public_x) + self.hand(hand_x) + self.player(player_x)
        if self.belief is not None:
            if belief_x is None:
                raise ValueError("belief_x is required when use_belief=True")
            hidden = hidden + self.belief(belief_x)
        hidden = self.body(hidden)
        if self.head_mode == "shared":
            return self.out(hidden).squeeze(-1)
        hero = self.hero_out(hidden).squeeze(-1)
        villain = self.villain_out(hidden).squeeze(-1)
        return torch.where(player_x[:, 0] > 0.5, hero, villain)


def _normalize(values: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    total = float(out.sum())
    if total <= 1e-12:
        out = np.ones_like(out, dtype=np.float32)
        total = float(out.sum())
    return out / max(total, 1e-12)


def _case_dual_cfv_target(
    case,
    belief_row: np.ndarray,
    *,
    solver_iterations: int,
    solver_backend: str,
    value_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        raise ValueError(f"{case.label}: parse error: {parsed['error']}")
    street = int(parsed.get("st", -1))
    if street != 3:
        raise ValueError(f"{case.label}: expected river state, got street {street}")

    board_idx = [card_str_to_index(card) for card in case.board[:5]]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=street,
    )
    pot = our_bet_pre + opp_bet_pre
    hero_stack = 20000 - our_bet_pre
    villain_stack = 20000 - opp_bet_pre
    hero_first = case.client_pos == 0
    street_parts = case.action_str.split("/")
    street_action = street_parts[street] if len(street_parts) > street else ""

    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range = np.zeros(len(full_hands), dtype=np.float32)
    villain_range = np.zeros(len(full_hands), dtype=np.float32)
    for local_idx, hand in enumerate(full_hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        hero_range[local_idx] = float(belief_row[global_idx])
        villain_range[local_idx] = float(belief_row[N_HANDS + global_idx])
    hero_range = _normalize(hero_range)
    villain_range = _normalize(villain_range)

    backend, backend_device = resolve_solver_backend(solver_backend)
    started = time.perf_counter()
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    solver.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
    )
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None:
        raise ValueError(f"{case.label}: solver could not navigate street action")
    hero_local, hero_local_mask = compute_hero_cfv_vector(solver, node, villain_range)
    villain_local, villain_local_mask = compute_villain_cfv_vector(solver, node, hero_range)
    hero_values = np.zeros(N_HANDS, dtype=np.float32)
    villain_values = np.zeros(N_HANDS, dtype=np.float32)
    hero_masks = np.zeros(N_HANDS, dtype=np.float32)
    villain_masks = np.zeros(N_HANDS, dtype=np.float32)
    for local_idx, hand in enumerate(solver.hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        hero_values[global_idx] = float(hero_local[local_idx]) / float(value_scale)
        villain_values[global_idx] = float(villain_local[local_idx]) / float(value_scale)
        hero_masks[global_idx] = float(hero_local_mask[local_idx])
        villain_masks[global_idx] = float(villain_local_mask[local_idx])
    latency_ms = (time.perf_counter() - started) * 1000.0
    return hero_values, villain_values, hero_masks, villain_masks, {
        "label": case.label,
        "solver_latency_ms": round(float(latency_ms), 3),
        "solver_n_hands": int(solver.n),
        "hero_mask_count": int(hero_masks.sum()),
        "villain_mask_count": int(villain_masks.sum()),
    }


def _save_dual_cache(dataset: DualCFVDataset, records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=dataset.features.astype(np.float32, copy=False),
        belief=dataset.belief.astype(np.float32, copy=False),
        hero_values=dataset.hero_values.astype(np.float32, copy=False),
        villain_values=dataset.villain_values.astype(np.float32, copy=False),
        hero_masks=dataset.hero_masks.astype(np.float32, copy=False),
        villain_masks=dataset.villain_masks.astype(np.float32, copy=False),
        labels=np.asarray(dataset.labels),
        records_json=json.dumps(records),
    )


def _load_dual_cache(path: str | Path) -> tuple[DualCFVDataset, list[dict[str, Any]]]:
    data = np.load(Path(path), allow_pickle=False)
    return (
        DualCFVDataset(
            features=data["features"].astype(np.float32, copy=False),
            belief=data["belief"].astype(np.float32, copy=False),
            hero_values=data["hero_values"].astype(np.float32, copy=False),
            villain_values=data["villain_values"].astype(np.float32, copy=False),
            hero_masks=data["hero_masks"].astype(np.float32, copy=False),
            villain_masks=data["villain_masks"].astype(np.float32, copy=False),
            labels=tuple(str(item) for item in data["labels"].tolist()),
        ),
        json.loads(str(data["records_json"].item())),
    )


def _load_or_build_dual_dataset(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    limit: int,
    solver_iterations: int,
    solver_backend: str,
    value_scale: float,
    dual_cache: str | Path | None,
) -> tuple[DualCFVDataset, list[dict[str, Any]], bool]:
    if dual_cache is not None and Path(dual_cache).exists():
        dataset, records = _load_dual_cache(dual_cache)
        return dataset, records, True
    base, _ = load_public_belief_cfv_dataset_cache(cfv_cache)
    cases = load_cases_json(cases_json)
    if len(cases) != base.features.shape[0]:
        raise ValueError("case count does not match base CFV cache")
    n = min(max(1, int(limit)), len(cases))
    hero_values = []
    villain_values = []
    hero_masks = []
    villain_masks = []
    records = []
    for idx, case in enumerate(cases[:n]):
        hv, vv, hm, vm, record = _case_dual_cfv_target(
            case,
            base.belief[idx],
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            value_scale=value_scale,
        )
        hero_values.append(hv)
        villain_values.append(vv)
        hero_masks.append(hm)
        villain_masks.append(vm)
        records.append(record)
    dataset = DualCFVDataset(
        features=base.features[:n].astype(np.float32, copy=False),
        belief=base.belief[:n].astype(np.float32, copy=False),
        hero_values=np.stack(hero_values).astype(np.float32, copy=False),
        villain_values=np.stack(villain_values).astype(np.float32, copy=False),
        hero_masks=np.stack(hero_masks).astype(np.float32, copy=False),
        villain_masks=np.stack(villain_masks).astype(np.float32, copy=False),
        labels=tuple(str(case.label) for case in cases[:n]),
    )
    if dual_cache is not None:
        _save_dual_cache(dataset, records, dual_cache)
    return dataset, records, False


def _standardize(train_x: np.ndarray, holdout_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((holdout_x - mean) / std).astype(np.float32),
    )


def _standardize_targets(train: DualCFVDataset) -> tuple[float, float]:
    selected = np.concatenate(
        [
            train.hero_values[train.hero_masks > 0],
            train.villain_values[train.villain_masks > 0],
        ]
    )
    mean = float(selected.mean()) if selected.size else 0.0
    std = float(selected.std()) if selected.size else 1.0
    return mean, std if std > 1e-6 else 1.0


def _pair_indices(dataset: DualCFVDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hero_case, hero_hand = np.nonzero(dataset.hero_masks > 0)
    villain_case, villain_hand = np.nonzero(dataset.villain_masks > 0)
    case_idx = np.concatenate([hero_case, villain_case]).astype(np.int64)
    hand_idx = np.concatenate([hero_hand, villain_hand]).astype(np.int64)
    player_idx = np.concatenate(
        [
            np.zeros(hero_case.shape[0], dtype=np.int64),
            np.ones(villain_case.shape[0], dtype=np.int64),
        ]
    )
    values = np.concatenate(
        [
            dataset.hero_values[hero_case, hero_hand],
            dataset.villain_values[villain_case, villain_hand],
        ]
    ).astype(np.float32)
    return case_idx, hand_idx, player_idx, values


def _fit_model(
    dataset: DualCFVDataset,
    *,
    target_mean: float,
    target_std: float,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    use_belief: bool,
    head_mode: str,
    belief_bottleneck_dim: int,
) -> _DualHandCFVProbeNet:
    torch.manual_seed(seed)
    case_idx, hand_idx, player_idx, values = _pair_indices(dataset)
    target = ((values - target_mean) / target_std).astype(np.float32)
    case_t = torch.from_numpy(case_idx).to(device)
    hand_t = torch.from_numpy(hand_idx).to(device)
    player_t = torch.from_numpy(player_idx).to(device)
    target_t = torch.from_numpy(target).to(device)
    public_t = torch.from_numpy(dataset.features).to(device)
    belief_t = torch.from_numpy(dataset.belief).to(device)
    hand_feat_t = torch.from_numpy(_HAND_FEATURES).to(device)
    player_feat_t = torch.eye(2, dtype=torch.float32, device=device)
    model = _DualHandCFVProbeNet(
        hidden_dim,
        use_belief=use_belief,
        head_mode=head_mode,
        belief_bottleneck_dim=belief_bottleneck_dim if use_belief else 0,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = int(case_t.numel())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    model.train()
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(n, generator=generator, device=device)
        for start in range(0, n, max(1, min(int(batch_size), n))):
            batch = perm[start : start + batch_size]
            c = case_t.index_select(0, batch)
            h = hand_t.index_select(0, batch)
            p = player_t.index_select(0, batch)
            belief_batch = belief_t.index_select(0, c) if use_belief else None
            pred = model(
                public_t.index_select(0, c),
                hand_feat_t.index_select(0, h),
                player_feat_t.index_select(0, p),
                belief_batch,
            )
            loss = torch.mean((pred - target_t.index_select(0, batch)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def _predict(
    model: _DualHandCFVProbeNet,
    dataset: DualCFVDataset,
    *,
    target_mean: float,
    target_std: float,
    batch_size: int,
    device: torch.device,
    use_belief: bool,
) -> np.ndarray:
    case_idx, hand_idx, player_idx, _ = _pair_indices(dataset)
    pred = np.zeros((2, dataset.features.shape[0], N_HANDS), dtype=np.float32)
    if case_idx.size == 0:
        return pred
    case_t = torch.from_numpy(case_idx).to(device)
    hand_t = torch.from_numpy(hand_idx).to(device)
    player_t = torch.from_numpy(player_idx).to(device)
    public_t = torch.from_numpy(dataset.features).to(device)
    belief_t = torch.from_numpy(dataset.belief).to(device)
    hand_feat_t = torch.from_numpy(_HAND_FEATURES).to(device)
    player_feat_t = torch.eye(2, dtype=torch.float32, device=device)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(case_t.numel()), max(1, int(batch_size))):
            c = case_t[start : start + batch_size]
            h = hand_t[start : start + batch_size]
            p = player_t[start : start + batch_size]
            belief_batch = belief_t.index_select(0, c) if use_belief else None
            out = model(
                public_t.index_select(0, c),
                hand_feat_t.index_select(0, h),
                player_feat_t.index_select(0, p),
                belief_batch,
            )
            outputs.append(out.cpu().numpy().astype(np.float32))
    values = np.concatenate(outputs, axis=0) * float(target_std) + float(target_mean)
    pred[player_idx, case_idx, hand_idx] = values
    return pred


def _metrics(pred: np.ndarray, dataset: DualCFVDataset) -> dict[str, float]:
    target = np.stack([dataset.hero_values, dataset.villain_values], axis=0)
    mask = np.stack([dataset.hero_masks, dataset.villain_masks], axis=0)
    selected = mask > 0
    err = pred[selected].astype(np.float64) - target[selected].astype(np.float64)
    return {
        "mae": round(float(np.mean(np.abs(err))), 8),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 8),
        "bias": round(float(np.mean(err)), 8),
    }


def _zero_dual_prediction(dataset: DualCFVDataset) -> np.ndarray:
    return np.zeros((2, dataset.features.shape[0], N_HANDS), dtype=np.float32)


def _standardize_dual_datasets(
    train_raw: DualCFVDataset,
    holdout_raw: DualCFVDataset,
) -> tuple[DualCFVDataset, DualCFVDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    public_mean, public_std = _standardization_stats(train_raw.features)
    belief_mean, belief_std = _standardization_stats(train_raw.belief)
    train = DualCFVDataset(
        features=_apply_standardization(train_raw.features, public_mean, public_std),
        belief=_apply_standardization(train_raw.belief, belief_mean, belief_std),
        hero_values=train_raw.hero_values,
        villain_values=train_raw.villain_values,
        hero_masks=train_raw.hero_masks,
        villain_masks=train_raw.villain_masks,
        labels=train_raw.labels,
    )
    holdout = DualCFVDataset(
        features=_apply_standardization(holdout_raw.features, public_mean, public_std),
        belief=_apply_standardization(holdout_raw.belief, belief_mean, belief_std),
        hero_values=holdout_raw.hero_values,
        villain_values=holdout_raw.villain_values,
        hero_masks=holdout_raw.hero_masks,
        villain_masks=holdout_raw.villain_masks,
        labels=holdout_raw.labels,
    )
    return train, holdout, public_mean, public_std, belief_mean, belief_std


def train_public_belief_dual_hand_cfv_checkpoint(
    *,
    train_cases_json: str | Path,
    train_cfv_cache: str | Path,
    holdout_cases_json: str | Path,
    holdout_cfv_cache: str | Path,
    output_checkpoint: str | Path,
    train_limit: int = 128,
    holdout_limit: int = 64,
    train_dual_cache: str | Path | None = None,
    holdout_dual_cache: str | Path | None = None,
    device: str | torch.device = "auto",
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    value_scale: float = 20000.0,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    head_mode: str = "separate",
    belief_bottleneck_dim: int = 32,
) -> dict[str, Any]:
    """Train and save the belief-conditioned dual-player hand-CFV model."""
    resolved_device = _resolve_device(device)
    train_raw, train_records, train_loaded = _load_or_build_dual_dataset(
        cases_json=train_cases_json,
        cfv_cache=train_cfv_cache,
        limit=train_limit,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        value_scale=value_scale,
        dual_cache=train_dual_cache,
    )
    holdout_raw, holdout_records, holdout_loaded = _load_or_build_dual_dataset(
        cases_json=holdout_cases_json,
        cfv_cache=holdout_cfv_cache,
        limit=holdout_limit,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        value_scale=value_scale,
        dual_cache=holdout_dual_cache,
    )
    train, holdout, public_mean, public_std, belief_mean, belief_std = (
        _standardize_dual_datasets(train_raw, holdout_raw)
    )
    target_mean, target_std = _standardize_targets(train)
    model = _fit_model(
        train,
        target_mean=target_mean,
        target_std=target_std,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
        use_belief=True,
        head_mode=head_mode,
        belief_bottleneck_dim=belief_bottleneck_dim,
    )
    holdout_pred = _predict(
        model,
        holdout,
        target_mean=target_mean,
        target_std=target_std,
        batch_size=batch_size,
        device=resolved_device,
        use_belief=True,
    )
    holdout_metrics = _metrics(holdout_pred, holdout)
    zero_metrics = _metrics(_zero_dual_prediction(holdout), holdout)
    beats_zero = (
        holdout_metrics["mae"] < zero_metrics["mae"]
        and holdout_metrics["rmse"] <= zero_metrics["rmse"]
    )

    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": "public_belief_dual_hand_cfv_checkpoint",
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "hidden_dim": int(hidden_dim),
            "head_mode": head_mode,
            "belief_bottleneck_dim": int(belief_bottleneck_dim),
            "use_belief": True,
            "feature_dim": int(N_FEATURES),
            "belief_dim": int(BELIEF_DIM),
            "hand_feature_dim": 52,
            "player_feature_dim": 2,
            "target_dim": int(N_HANDS),
            "public_mean": public_mean,
            "public_std": public_std,
            "belief_mean": belief_mean,
            "belief_std": belief_std,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "value_scale": float(value_scale),
            "solver_iterations": int(solver_iterations),
            "solver_backend": solver_backend,
            "train_cases_json": str(train_cases_json),
            "train_cfv_cache": str(train_cfv_cache),
            "train_dual_cache": str(train_dual_cache) if train_dual_cache else None,
            "seed": int(seed),
        },
        output_path,
    )
    return {
        "mode": "public_belief_dual_hand_cfv_checkpoint_train",
        "passed": bool(np.isfinite(holdout_metrics["mae"]) and beats_zero),
        "pass_criteria": "belief checkpoint must beat zero-CFV MAE and not worsen zero-CFV RMSE",
        "checkpoint": str(output_path),
        "device": str(resolved_device),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "value_scale": float(value_scale),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "head_mode": head_mode,
        "belief_bottleneck_dim": int(belief_bottleneck_dim),
        "train_cfv_cache": str(train_cfv_cache),
        "holdout_cfv_cache": str(holdout_cfv_cache),
        "train_dual_cache": str(train_dual_cache) if train_dual_cache else None,
        "holdout_dual_cache": str(holdout_dual_cache) if holdout_dual_cache else None,
        "train_loaded_from_cache": bool(train_loaded),
        "holdout_loaded_from_cache": bool(holdout_loaded),
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "feature_dim": int(N_FEATURES),
        "belief_dim": int(BELIEF_DIM),
        "hand_feature_dim": 52,
        "player_feature_dim": 2,
        "target_dim": int(N_HANDS),
        "train_label_count": int(train.hero_masks.sum() + train.villain_masks.sum()),
        "holdout_label_count": int(
            holdout.hero_masks.sum() + holdout.villain_masks.sum()
        ),
        "belief_holdout": holdout_metrics,
        "zero_baseline": zero_metrics,
        "train_dual_record_count": len(train_records),
        "holdout_dual_record_count": len(holdout_records),
    }


def load_public_belief_dual_hand_cfv_checkpoint(
    checkpoint: str | Path,
    device: str | torch.device = "auto",
) -> tuple[_DualHandCFVProbeNet, dict[str, Any]]:
    """Load a saved dual-player public-belief hand-CFV checkpoint."""
    resolved_device = _resolve_device(device)
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    if payload.get("mode") != "public_belief_dual_hand_cfv_checkpoint":
        raise ValueError(f"{checkpoint} is not a dual hand-CFV checkpoint")
    model = _DualHandCFVProbeNet(
        int(payload["hidden_dim"]),
        use_belief=True,
        head_mode=str(payload.get("head_mode", "separate")),
        belief_bottleneck_dim=int(payload.get("belief_bottleneck_dim", 0)),
    ).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def predict_public_belief_dual_hand_cfv_model(
    model: _DualHandCFVProbeNet,
    payload: dict[str, Any],
    features: np.ndarray,
    belief: np.ndarray,
    hero_masks: np.ndarray | None = None,
    villain_masks: np.ndarray | None = None,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 8192,
) -> np.ndarray:
    """Predict both players' global hand CFVs from a loaded dual-CFV model."""
    features = np.asarray(features, dtype=np.float32)
    belief = np.asarray(belief, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if belief.ndim == 1:
        belief = belief.reshape(1, -1)
    if features.shape[0] != belief.shape[0]:
        raise ValueError("features and belief must have the same number of rows")
    n = int(features.shape[0])
    if hero_masks is None:
        hero_masks = np.ones((n, N_HANDS), dtype=np.float32)
    else:
        hero_masks = np.asarray(hero_masks, dtype=np.float32)
    if villain_masks is None:
        villain_masks = np.ones((n, N_HANDS), dtype=np.float32)
    else:
        villain_masks = np.asarray(villain_masks, dtype=np.float32)
    dataset = DualCFVDataset(
        features=_apply_standardization(
            features,
            payload["public_mean"],
            payload["public_std"],
        ),
        belief=_apply_standardization(
            belief,
            payload["belief_mean"],
            payload["belief_std"],
        ),
        hero_values=np.zeros((n, N_HANDS), dtype=np.float32),
        villain_values=np.zeros((n, N_HANDS), dtype=np.float32),
        hero_masks=hero_masks,
        villain_masks=villain_masks,
        labels=tuple(f"predict-{idx}" for idx in range(n)),
    )
    return _predict(
        model,
        dataset,
        target_mean=float(payload["target_mean"]),
        target_std=float(payload["target_std"]),
        batch_size=batch_size,
        device=_resolve_device(device),
        use_belief=True,
    )


def predict_public_belief_dual_hand_cfv_checkpoint(
    checkpoint: str | Path,
    features: np.ndarray,
    belief: np.ndarray,
    hero_masks: np.ndarray | None = None,
    villain_masks: np.ndarray | None = None,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 8192,
) -> np.ndarray:
    """Predict both players' global hand CFVs from a saved dual-CFV checkpoint."""
    model, payload = load_public_belief_dual_hand_cfv_checkpoint(checkpoint, device=device)
    return predict_public_belief_dual_hand_cfv_model(
        model,
        payload,
        features,
        belief,
        hero_masks,
        villain_masks,
        device=device,
        batch_size=batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare dual-player public+hand and public+hand+belief CFV probes."
    )
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--train-cfv-cache", required=True)
    parser.add_argument("--holdout-cases", required=True)
    parser.add_argument("--holdout-cfv-cache", required=True)
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--holdout-limit", type=int, default=8)
    parser.add_argument("--train-dual-cache")
    parser.add_argument("--holdout-dual-cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--solver-iterations", type=int, default=25)
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "cpu", "torch-cuda", "torch-cpu"),
        default="auto",
    )
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--head-mode",
        choices=("shared", "separate"),
        default="shared",
        help="Use one scalar output head or separate hero/villain output heads.",
    )
    parser.add_argument(
        "--belief-bottleneck-dim",
        type=int,
        default=0,
        help="Optional learned bottleneck for the public belief vector; 0 keeps the linear baseline.",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    train_raw, train_records, train_loaded = _load_or_build_dual_dataset(
        cases_json=args.train_cases,
        cfv_cache=args.train_cfv_cache,
        limit=args.train_limit,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        dual_cache=args.train_dual_cache,
    )
    holdout_raw, holdout_records, holdout_loaded = _load_or_build_dual_dataset(
        cases_json=args.holdout_cases,
        cfv_cache=args.holdout_cfv_cache,
        limit=args.holdout_limit,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        dual_cache=args.holdout_dual_cache,
    )
    train_features, holdout_features = _standardize(train_raw.features, holdout_raw.features)
    train_belief, holdout_belief = _standardize(train_raw.belief, holdout_raw.belief)
    train = DualCFVDataset(
        features=train_features,
        belief=train_belief,
        hero_values=train_raw.hero_values,
        villain_values=train_raw.villain_values,
        hero_masks=train_raw.hero_masks,
        villain_masks=train_raw.villain_masks,
        labels=train_raw.labels,
    )
    holdout = DualCFVDataset(
        features=holdout_features,
        belief=holdout_belief,
        hero_values=holdout_raw.hero_values,
        villain_values=holdout_raw.villain_values,
        hero_masks=holdout_raw.hero_masks,
        villain_masks=holdout_raw.villain_masks,
        labels=holdout_raw.labels,
    )
    target_mean, target_std = _standardize_targets(train)
    base = _fit_model(
        train,
        target_mean=target_mean,
        target_std=target_std,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        use_belief=False,
        head_mode=args.head_mode,
        belief_bottleneck_dim=0,
    )
    belief = _fit_model(
        train,
        target_mean=target_mean,
        target_std=target_std,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        use_belief=True,
        head_mode=args.head_mode,
        belief_bottleneck_dim=args.belief_bottleneck_dim,
    )
    base_metrics = _metrics(
        _predict(
            base,
            holdout,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=args.batch_size,
            device=device,
            use_belief=False,
        ),
        holdout,
    )
    belief_metrics = _metrics(
        _predict(
            belief,
            holdout,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=args.batch_size,
            device=device,
            use_belief=True,
        ),
        holdout,
    )
    zero_metrics = _metrics(_zero_dual_prediction(holdout), holdout)
    mae_delta = round(float(base_metrics["mae"] - belief_metrics["mae"]), 8)
    rmse_delta = round(float(base_metrics["rmse"] - belief_metrics["rmse"]), 8)
    zero_mae_delta = round(float(zero_metrics["mae"] - belief_metrics["mae"]), 8)
    zero_rmse_delta = round(float(zero_metrics["rmse"] - belief_metrics["rmse"]), 8)
    metrics = {
        "mode": "public_belief_dual_hand_cfv_probe",
        "passed": bool(
            mae_delta > 0
            and rmse_delta >= 0
            and zero_mae_delta > 0
            and zero_rmse_delta >= 0
        ),
        "pass_criteria": (
            "belief_holdout must improve feature baseline and beat zero-CFV MAE "
            "without worsening feature or zero-CFV RMSE"
        ),
        "device": str(device),
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "train_loaded_from_cache": bool(train_loaded),
        "holdout_loaded_from_cache": bool(holdout_loaded),
        "train_dual_cache": str(args.train_dual_cache) if args.train_dual_cache else None,
        "holdout_dual_cache": str(args.holdout_dual_cache) if args.holdout_dual_cache else None,
        "solver_iterations": int(args.solver_iterations),
        "solver_backend": args.solver_backend,
        "value_scale": float(args.value_scale),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "head_mode": args.head_mode,
        "belief_bottleneck_dim": int(args.belief_bottleneck_dim),
        "target_dim": int(N_HANDS),
        "train_label_count": int(train.hero_masks.sum() + train.villain_masks.sum()),
        "holdout_label_count": int(holdout.hero_masks.sum() + holdout.villain_masks.sum()),
        "base_holdout": base_metrics,
        "belief_holdout": belief_metrics,
        "zero_baseline": zero_metrics,
        "holdout_mae_delta": mae_delta,
        "holdout_rmse_delta": rmse_delta,
        "holdout_zero_mae_delta": zero_mae_delta,
        "holdout_zero_rmse_delta": zero_rmse_delta,
        "train_solver_mean_ms": round(
            float(np.mean([record["solver_latency_ms"] for record in train_records])), 3
        ),
        "holdout_solver_mean_ms": round(
            float(np.mean([record["solver_latency_ms"] for record in holdout_records])), 3
        ),
    }
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
