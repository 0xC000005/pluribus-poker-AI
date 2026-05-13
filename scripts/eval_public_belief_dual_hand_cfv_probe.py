#!/usr/bin/env python3
"""Probe public-belief inputs for dual-player hand-CFV labels."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
        card_encoder: str = "flat",
    ):
        super().__init__()
        self.use_belief = bool(use_belief)
        if head_mode not in ("shared", "separate"):
            raise ValueError(f"unknown head_mode: {head_mode}")
        if card_encoder not in ("flat", "deepset"):
            raise ValueError(f"unknown card_encoder: {card_encoder}")
        if belief_bottleneck_dim < 0:
            raise ValueError("belief_bottleneck_dim must be non-negative")
        self.head_mode = head_mode
        self.card_encoder = card_encoder
        if card_encoder == "flat":
            self.public = nn.Linear(N_FEATURES, hidden_dim)
            self.hand = nn.Linear(52, hidden_dim)
            self.board = None
            self.public_misc = None
            self.card_interaction = None
        else:
            self.public = None
            self.hand = nn.Linear(52, hidden_dim, bias=False)
            self.board = nn.Linear(52, hidden_dim, bias=False)
            self.public_misc = nn.Linear(N_FEATURES - 104, hidden_dim)
            self.card_interaction = nn.Linear(hidden_dim, hidden_dim)
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
        if self.card_encoder == "flat":
            hidden = self.public(public_x) + self.hand(hand_x)
        else:
            hand_hidden = self.hand(hand_x)
            board_hidden = self.board(public_x[:, 52:104])
            hidden = (
                hand_hidden
                + board_hidden
                + self.public_misc(public_x[:, 104:])
                + self.card_interaction(hand_hidden * board_hidden)
            )
        hidden = hidden + self.player(player_x)
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
    if street not in (2, 3):
        raise ValueError(f"{case.label}: expected turn/river state, got street {street}")

    n_board = 4 if street == 2 else 5
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
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
        "street": street,
        "solver_latency_ms": round(float(latency_ms), 3),
        "solver_n_hands": int(solver.n),
        "hero_mask_count": int(hero_masks.sum()),
        "villain_mask_count": int(villain_masks.sum()),
    }


def _dual_target_worker(args: tuple[Any, np.ndarray, int, str, float, bool]):
    case, belief_row, solver_iterations, solver_backend, value_scale, limit_threads = args
    if limit_threads:
        try:
            from threadpoolctl import threadpool_limits
            limit_context = threadpool_limits(limits=1)
        except Exception:
            limit_context = contextlib.nullcontext()
    else:
        limit_context = contextlib.nullcontext()
    with limit_context:
        return _case_dual_cfv_target(
            case,
            belief_row,
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            value_scale=value_scale,
        )


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
    label_jobs: int = 1,
) -> tuple[DualCFVDataset, list[dict[str, Any]], bool]:
    if dual_cache is not None and Path(dual_cache).exists():
        dataset, records = _load_dual_cache(dual_cache)
        return dataset, records, True
    base, _ = load_public_belief_cfv_dataset_cache(cfv_cache)
    cases = load_cases_json(cases_json)
    if len(cases) != base.features.shape[0]:
        raise ValueError("case count does not match base CFV cache")
    n = min(max(1, int(limit)), len(cases))
    jobs = max(1, int(label_jobs))
    if jobs > 1 and solver_backend == "torch-cuda":
        raise ValueError("parallel label generation is not supported with torch-cuda")
    worker_args = [
        (
            case,
            base.belief[idx],
            int(solver_iterations),
            solver_backend,
            float(value_scale),
            jobs > 1,
        )
        for idx, case in enumerate(cases[:n])
    ]
    hero_values = []
    villain_values = []
    hero_masks = []
    villain_masks = []
    records = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(_dual_target_worker, worker_args))
    else:
        results = [_dual_target_worker(item) for item in worker_args]
    for hv, vv, hm, vm, record in results:
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
    card_encoder: str,
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
        card_encoder=card_encoder,
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


def _constant_dual_prediction(dataset: DualCFVDataset, value: float) -> np.ndarray:
    return np.full(
        (2, dataset.features.shape[0], N_HANDS),
        float(value),
        dtype=np.float32,
    )


def _train_target_values(dataset: DualCFVDataset) -> np.ndarray:
    _, _, _, values = _pair_indices(dataset)
    return values.astype(np.float32, copy=False)


def _constant_baselines(
    holdout: DualCFVDataset,
    *,
    target_mean: float,
    target_median: float,
) -> dict[str, dict[str, float]]:
    zero = _metrics(_zero_dual_prediction(holdout), holdout)
    mean = _metrics(_constant_dual_prediction(holdout, target_mean), holdout)
    median = _metrics(_constant_dual_prediction(holdout, target_median), holdout)
    return {"zero": zero, "train_mean": mean, "train_median": median}


def _best_constant_metric(baselines: dict[str, dict[str, float]], metric: str) -> float:
    return min(float(item[metric]) for item in baselines.values())


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
    card_encoder: str = "flat",
    label_jobs: int = 1,
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
        label_jobs=label_jobs,
    )
    holdout_raw, holdout_records, holdout_loaded = _load_or_build_dual_dataset(
        cases_json=holdout_cases_json,
        cfv_cache=holdout_cfv_cache,
        limit=holdout_limit,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        value_scale=value_scale,
        dual_cache=holdout_dual_cache,
        label_jobs=label_jobs,
    )
    train, holdout, public_mean, public_std, belief_mean, belief_std = (
        _standardize_dual_datasets(train_raw, holdout_raw)
    )
    target_mean, target_std = _standardize_targets(train)
    target_median = float(np.median(_train_target_values(train)))
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
        card_encoder=card_encoder,
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
    constant_baselines = _constant_baselines(
        holdout,
        target_mean=target_mean,
        target_median=target_median,
    )
    zero_metrics = constant_baselines["zero"]
    best_constant_mae = _best_constant_metric(constant_baselines, "mae")
    best_constant_rmse = _best_constant_metric(constant_baselines, "rmse")
    beats_zero = (
        holdout_metrics["mae"] < zero_metrics["mae"]
        and holdout_metrics["rmse"] <= zero_metrics["rmse"]
    )
    beats_constants = (
        holdout_metrics["mae"] < best_constant_mae
        and holdout_metrics["rmse"] <= best_constant_rmse
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
            "card_encoder": card_encoder,
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
            "target_median": float(target_median),
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
        "passed": bool(
            np.isfinite(holdout_metrics["mae"])
            and beats_zero
            and beats_constants
        ),
        "pass_criteria": (
            "belief checkpoint must beat zero-CFV and train-constant MAE "
            "without worsening zero-CFV or train-constant RMSE"
        ),
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
        "card_encoder": card_encoder,
        "label_jobs": int(label_jobs),
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
        "constant_baselines": constant_baselines,
        "best_constant_mae": round(float(best_constant_mae), 8),
        "best_constant_rmse": round(float(best_constant_rmse), 8),
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
        card_encoder=str(payload.get("card_encoder", "flat")),
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


def predict_public_belief_dual_hand_cfv_model_vectorized(
    model: _DualHandCFVProbeNet,
    payload: dict[str, Any],
    features: np.ndarray,
    belief: np.ndarray,
    hero_masks: np.ndarray | None = None,
    villain_masks: np.ndarray | None = None,
    *,
    device: str | torch.device = "auto",
    state_batch_size: int = 64,
    hand_batch_size: int = N_HANDS,
) -> np.ndarray:
    """Predict both players' global hand CFVs with state/hand batching.

    The pairwise predictor is convenient for sparse supervised datasets, but
    learned-leaf search asks for nearly all hands at many public states.  This
    path computes public/belief embeddings once per state and broadcasts them
    over hand chunks, avoiding repeated 5k-dim belief copies for every hand.
    """
    resolved_device = _resolve_device(device)
    features = np.asarray(features, dtype=np.float32)
    belief = np.asarray(belief, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if belief.ndim == 1:
        belief = belief.reshape(1, -1)
    if features.shape[0] != belief.shape[0]:
        raise ValueError("features and belief must have the same number of rows")
    n_states = int(features.shape[0])
    if hero_masks is None:
        hero_masks = np.ones((n_states, N_HANDS), dtype=np.float32)
    else:
        hero_masks = np.asarray(hero_masks, dtype=np.float32)
    if villain_masks is None:
        villain_masks = np.ones((n_states, N_HANDS), dtype=np.float32)
    else:
        villain_masks = np.asarray(villain_masks, dtype=np.float32)

    public = _apply_standardization(features, payload["public_mean"], payload["public_std"])
    belief_std = _apply_standardization(belief, payload["belief_mean"], payload["belief_std"])
    public_t = torch.from_numpy(public).to(resolved_device)
    belief_t = torch.from_numpy(belief_std).to(resolved_device)
    hand_feat_t = torch.from_numpy(_HAND_FEATURES).to(resolved_device)
    masks_t = torch.from_numpy(
        np.stack([hero_masks, villain_masks], axis=0).astype(np.float32, copy=False)
    ).to(resolved_device)
    player_t = torch.eye(2, dtype=torch.float32, device=resolved_device)
    pred = np.zeros((2, n_states, N_HANDS), dtype=np.float32)
    target_mean = float(payload["target_mean"])
    target_std = float(payload["target_std"])
    state_step = max(1, int(state_batch_size))
    hand_step = max(1, min(int(hand_batch_size), N_HANDS))

    model.eval()
    with torch.no_grad():
        for state_start in range(0, n_states, state_step):
            state_end = min(state_start + state_step, n_states)
            public_batch = public_t[state_start:state_end]
            belief_batch = belief_t[state_start:state_end]
            if model.card_encoder == "flat":
                state_base = model.public(public_batch)
                board_hidden = None
            else:
                board_hidden = model.board(public_batch[:, 52:104])
                state_base = board_hidden + model.public_misc(public_batch[:, 104:])
            if model.belief is not None:
                state_base = state_base + model.belief(belief_batch)

            for hand_start in range(0, N_HANDS, hand_step):
                hand_end = min(hand_start + hand_step, N_HANDS)
                hand_hidden = model.hand(hand_feat_t[hand_start:hand_end])
                if model.card_encoder == "flat":
                    hidden_base = state_base[:, None, :] + hand_hidden[None, :, :]
                else:
                    assert board_hidden is not None
                    interaction = model.card_interaction(
                        hand_hidden[None, :, :] * board_hidden[:, None, :]
                    )
                    hidden_base = (
                        state_base[:, None, :]
                        + hand_hidden[None, :, :]
                        + interaction
                    )
                flat_size = int((state_end - state_start) * (hand_end - hand_start))
                for player_idx in (0, 1):
                    hidden = hidden_base + model.player(player_t[player_idx]).view(1, 1, -1)
                    body = model.body(hidden.reshape(flat_size, -1))
                    if model.head_mode == "shared":
                        out = model.out(body).squeeze(-1)
                    elif player_idx == 0:
                        out = model.hero_out(body).squeeze(-1)
                    else:
                        out = model.villain_out(body).squeeze(-1)
                    values = out.reshape(state_end - state_start, hand_end - hand_start)
                    values = values * target_std + target_mean
                    values = values * masks_t[
                        player_idx,
                        state_start:state_end,
                        hand_start:hand_end,
                    ]
                    pred[
                        player_idx,
                        state_start:state_end,
                        hand_start:hand_end,
                    ] = values.cpu().numpy().astype(np.float32)
    return pred


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


def load_public_belief_dual_hand_cfv_ensemble(
    checkpoints: list[str | Path] | tuple[str | Path, ...],
    device: str | torch.device = "auto",
) -> list[tuple[_DualHandCFVProbeNet, dict[str, Any]]]:
    """Load one or more dual-player hand-CFV checkpoints for prediction averaging."""
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    return [
        load_public_belief_dual_hand_cfv_checkpoint(checkpoint, device=device)
        for checkpoint in checkpoints
    ]


def predict_public_belief_dual_hand_cfv_ensemble(
    checkpoints: list[str | Path] | tuple[str | Path, ...],
    features: np.ndarray,
    belief: np.ndarray,
    hero_masks: np.ndarray | None = None,
    villain_masks: np.ndarray | None = None,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 8192,
) -> np.ndarray:
    """Average predictions from multiple saved dual-player hand-CFV checkpoints."""
    loaded = load_public_belief_dual_hand_cfv_ensemble(checkpoints, device=device)
    preds = [
        predict_public_belief_dual_hand_cfv_model(
            model,
            payload,
            features,
            belief,
            hero_masks,
            villain_masks,
            device=device,
            batch_size=batch_size,
        )
        for model, payload in loaded
    ]
    return np.mean(np.stack(preds, axis=0), axis=0, dtype=np.float32)


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
        "--label-jobs",
        type=int,
        default=1,
        help="Parallel CPU workers for building dual-CFV labels; not supported with torch-cuda.",
    )
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
    parser.add_argument(
        "--card-encoder",
        choices=("flat", "deepset"),
        default="flat",
        help="Use flat feature projections or learned hand/board card-set interactions.",
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
        label_jobs=args.label_jobs,
    )
    holdout_raw, holdout_records, holdout_loaded = _load_or_build_dual_dataset(
        cases_json=args.holdout_cases,
        cfv_cache=args.holdout_cfv_cache,
        limit=args.holdout_limit,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        dual_cache=args.holdout_dual_cache,
        label_jobs=args.label_jobs,
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
        card_encoder=args.card_encoder,
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
        card_encoder=args.card_encoder,
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
    target_values = _train_target_values(train)
    target_median = float(np.median(target_values))
    constant_baselines = _constant_baselines(
        holdout,
        target_mean=target_mean,
        target_median=target_median,
    )
    zero_metrics = constant_baselines["zero"]
    best_constant_mae = _best_constant_metric(constant_baselines, "mae")
    best_constant_rmse = _best_constant_metric(constant_baselines, "rmse")
    mae_delta = round(float(base_metrics["mae"] - belief_metrics["mae"]), 8)
    rmse_delta = round(float(base_metrics["rmse"] - belief_metrics["rmse"]), 8)
    zero_mae_delta = round(float(zero_metrics["mae"] - belief_metrics["mae"]), 8)
    zero_rmse_delta = round(float(zero_metrics["rmse"] - belief_metrics["rmse"]), 8)
    best_constant_mae_delta = round(float(best_constant_mae - belief_metrics["mae"]), 8)
    best_constant_rmse_delta = round(float(best_constant_rmse - belief_metrics["rmse"]), 8)
    metrics = {
        "mode": "public_belief_dual_hand_cfv_probe",
        "passed": bool(
            mae_delta > 0
            and rmse_delta >= 0
            and zero_mae_delta > 0
            and zero_rmse_delta >= 0
            and best_constant_mae_delta > 0
            and best_constant_rmse_delta >= 0
        ),
        "pass_criteria": (
            "belief_holdout must improve feature baseline and beat zero-CFV and "
            "train-constant MAE without worsening feature, zero-CFV, or train-constant RMSE"
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
        "card_encoder": args.card_encoder,
        "label_jobs": int(args.label_jobs),
        "target_mean": round(float(target_mean), 8),
        "target_median": round(float(target_median), 8),
        "target_dim": int(N_HANDS),
        "train_label_count": int(train.hero_masks.sum() + train.villain_masks.sum()),
        "holdout_label_count": int(holdout.hero_masks.sum() + holdout.villain_masks.sum()),
        "base_holdout": base_metrics,
        "belief_holdout": belief_metrics,
        "zero_baseline": zero_metrics,
        "constant_baselines": constant_baselines,
        "holdout_mae_delta": mae_delta,
        "holdout_rmse_delta": rmse_delta,
        "holdout_zero_mae_delta": zero_mae_delta,
        "holdout_zero_rmse_delta": zero_rmse_delta,
        "holdout_best_constant_mae_delta": best_constant_mae_delta,
        "holdout_best_constant_rmse_delta": best_constant_rmse_delta,
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
