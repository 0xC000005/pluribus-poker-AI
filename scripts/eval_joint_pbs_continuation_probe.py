#!/usr/bin/env python3
"""Evaluate a joint public-belief policy/value continuation probe."""

from __future__ import annotations

import argparse
import json
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

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.belief_probe import (
    BELIEF_DIM,
    N_HANDS,
    _normalize_targets,
    _policy_metrics,
    _resolve_device,
)
from poker_ai.research.belief_value_probe import (
    _HAND_FEATURES,
    _apply_standardization,
    _standardization_stats,
    save_metrics,
)


@dataclass(frozen=True)
class JointPBSDataset:
    features: np.ndarray
    policy_features: np.ndarray
    belief: np.ndarray
    legal_masks: np.ndarray
    target_probs: np.ndarray
    policy_weights: np.ndarray
    hero_values: np.ndarray
    villain_values: np.ndarray
    hero_masks: np.ndarray
    villain_masks: np.ndarray
    labels: tuple[str, ...]


class _JointPBSContinuationNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        belief_bottleneck_dim: int = 32,
        card_encoder: str = "deepset",
    ):
        super().__init__()
        if belief_bottleneck_dim < 0:
            raise ValueError("belief_bottleneck_dim must be non-negative")
        if card_encoder not in ("flat", "deepset"):
            raise ValueError(f"unknown card_encoder: {card_encoder}")
        self.card_encoder = card_encoder
        if card_encoder == "flat":
            self.public = nn.Linear(N_FEATURES, hidden_dim)
            self.board = None
            self.public_misc = None
            self.card_interaction = None
        else:
            self.public = None
            self.board = nn.Linear(52, hidden_dim, bias=False)
            self.public_misc = nn.Linear(N_FEATURES - 104, hidden_dim)
            self.card_interaction = nn.Linear(hidden_dim, hidden_dim)
        self.private_cards = nn.Linear(52, hidden_dim, bias=False)
        self.hand = nn.Linear(52, hidden_dim, bias=False)
        self.player = nn.Linear(2, hidden_dim)
        if belief_bottleneck_dim > 0:
            self.belief = nn.Sequential(
                nn.Linear(BELIEF_DIM, int(belief_bottleneck_dim)),
                nn.ReLU(),
                nn.Linear(int(belief_bottleneck_dim), hidden_dim),
            )
        else:
            self.belief = nn.Linear(BELIEF_DIM, hidden_dim)
        self.trunk_body = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hero_value = nn.Linear(hidden_dim, 1)
        self.villain_value = nn.Linear(hidden_dim, 1)
        self.policy_body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )

    def _public_hidden(
        self,
        public_x: torch.Tensor,
        belief_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.card_encoder == "flat":
            return self.public(public_x) + self.belief(belief_x), None
        board_hidden = self.board(public_x[:, 52:104])
        public_hidden = (
            board_hidden
            + self.public_misc(public_x[:, 104:])
            + self.belief(belief_x)
        )
        return public_hidden, board_hidden

    def trunk(
        self,
        public_x: torch.Tensor,
        belief_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        public_hidden, board_hidden = self._public_hidden(public_x, belief_x)
        return self.trunk_body(public_hidden), board_hidden

    def value(
        self,
        public_x: torch.Tensor,
        hand_x: torch.Tensor,
        player_x: torch.Tensor,
        belief_x: torch.Tensor,
    ) -> torch.Tensor:
        trunk_hidden, board_hidden = self.trunk(public_x, belief_x)
        hand_hidden = self.hand(hand_x)
        hidden = trunk_hidden + hand_hidden + self.player(player_x)
        if board_hidden is not None and self.card_interaction is not None:
            hidden = hidden + self.card_interaction(hand_hidden * board_hidden)
        hidden = self.value_body(hidden)
        hero = self.hero_value(hidden).squeeze(-1)
        villain = self.villain_value(hidden).squeeze(-1)
        return torch.where(player_x[:, 0] > 0.5, hero, villain)

    def policy(
        self,
        public_x: torch.Tensor,
        private_x: torch.Tensor,
        belief_x: torch.Tensor,
    ) -> torch.Tensor:
        trunk_hidden, board_hidden = self.trunk(public_x, belief_x)
        private_hidden = self.private_cards(private_x)
        hidden = trunk_hidden + private_hidden
        if board_hidden is not None and self.card_interaction is not None:
            hidden = hidden + self.card_interaction(private_hidden * board_hidden)
        return self.policy_body(hidden)


def load_joint_pbs_dataset(path: str | Path) -> JointPBSDataset:
    data = np.load(Path(path), allow_pickle=False)
    policy_features = data["policy_features"] if "policy_features" in data else data["features"]
    policy_weights = (
        data["policy_weights"]
        if "policy_weights" in data
        else np.ones(data["features"].shape[0], dtype=np.float32)
    )
    return JointPBSDataset(
        features=data["features"].astype(np.float32, copy=False),
        policy_features=policy_features.astype(np.float32, copy=False),
        belief=data["belief"].astype(np.float32, copy=False),
        legal_masks=data["legal_masks"].astype(np.float32, copy=False),
        target_probs=_normalize_targets(data["target_probs"], data["legal_masks"]),
        policy_weights=policy_weights.astype(np.float32, copy=False),
        hero_values=data["hero_values"].astype(np.float32, copy=False),
        villain_values=data["villain_values"].astype(np.float32, copy=False),
        hero_masks=data["hero_masks"].astype(np.float32, copy=False),
        villain_masks=data["villain_masks"].astype(np.float32, copy=False),
        labels=tuple(str(item) for item in data["labels"].tolist()),
    )


def _standardize_joint_pair(
    train: JointPBSDataset,
    holdout: JointPBSDataset,
) -> tuple[JointPBSDataset, JointPBSDataset]:
    public_mean, public_std = _standardization_stats(train.features)
    belief_mean, belief_std = _standardization_stats(train.belief)
    train_std = JointPBSDataset(
        features=_apply_standardization(train.features, public_mean, public_std),
        policy_features=train.policy_features,
        belief=_apply_standardization(train.belief, belief_mean, belief_std),
        legal_masks=train.legal_masks,
        target_probs=train.target_probs,
        policy_weights=train.policy_weights,
        hero_values=train.hero_values,
        villain_values=train.villain_values,
        hero_masks=train.hero_masks,
        villain_masks=train.villain_masks,
        labels=train.labels,
    )
    holdout_std = JointPBSDataset(
        features=_apply_standardization(holdout.features, public_mean, public_std),
        policy_features=holdout.policy_features,
        belief=_apply_standardization(holdout.belief, belief_mean, belief_std),
        legal_masks=holdout.legal_masks,
        target_probs=holdout.target_probs,
        policy_weights=holdout.policy_weights,
        hero_values=holdout.hero_values,
        villain_values=holdout.villain_values,
        hero_masks=holdout.hero_masks,
        villain_masks=holdout.villain_masks,
        labels=holdout.labels,
    )
    return train_std, holdout_std


def _pair_indices(dataset: JointPBSDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _target_stats(dataset: JointPBSDataset) -> tuple[float, float, float]:
    _case_idx, _hand_idx, _player_idx, values = _pair_indices(dataset)
    mean = float(values.mean()) if values.size else 0.0
    median = float(np.median(values)) if values.size else 0.0
    std = float(values.std()) if values.size else 1.0
    return mean, median, std if std > 1e-6 else 1.0


def _masked_policy_loss(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    target_probs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    legal = (legal_masks > 0).to(dtype=logits.dtype)
    targets = target_probs.to(dtype=logits.dtype) * legal
    totals = targets.sum(dim=1, keepdim=True)
    legal_totals = legal.sum(dim=1, keepdim=True).clamp(min=1.0)
    fallback = legal / legal_totals
    targets = torch.where(totals > 1e-8, targets / totals.clamp(min=1e-8), fallback)
    masked_logits = logits.masked_fill(legal <= 0, -1e4)
    per_sample = -(targets * torch.log_softmax(masked_logits, dim=1)).sum(dim=1)
    weights = weights.to(dtype=per_sample.dtype).clamp(min=0.0)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1.0)


def _fit_joint_model(
    dataset: JointPBSDataset,
    *,
    target_mean: float,
    target_std: float,
    hidden_dim: int,
    belief_bottleneck_dim: int,
    card_encoder: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> _JointPBSContinuationNet:
    torch.manual_seed(int(seed))
    case_idx, hand_idx, player_idx, values = _pair_indices(dataset)
    if case_idx.size == 0:
        raise ValueError("cannot train joint PBS probe without value labels")
    value_target = ((values - float(target_mean)) / float(target_std)).astype(np.float32)
    case_t = torch.from_numpy(case_idx).to(device)
    hand_t = torch.from_numpy(hand_idx).to(device)
    player_t = torch.from_numpy(player_idx).to(device)
    value_target_t = torch.from_numpy(value_target).to(device)
    public_t = torch.from_numpy(dataset.features).to(device)
    belief_t = torch.from_numpy(dataset.belief).to(device)
    private_t = torch.from_numpy(dataset.policy_features[:, :52]).to(device)
    legal_t = torch.from_numpy(dataset.legal_masks).to(device)
    policy_target_t = torch.from_numpy(dataset.target_probs).to(device)
    policy_weight_t = torch.from_numpy(dataset.policy_weights).to(device)
    hand_feat_t = torch.from_numpy(_HAND_FEATURES).to(device)
    player_feat_t = torch.eye(2, dtype=torch.float32, device=device)

    model = _JointPBSContinuationNet(
        hidden_dim,
        belief_bottleneck_dim=belief_bottleneck_dim,
        card_encoder=card_encoder,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_value = int(case_t.numel())
    batch_size = max(1, min(int(batch_size), n_value))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    model.train()
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(n_value, generator=generator, device=device)
        for start in range(0, n_value, batch_size):
            batch = perm[start : start + batch_size]
            c = case_t.index_select(0, batch)
            h = hand_t.index_select(0, batch)
            p = player_t.index_select(0, batch)
            pred = model.value(
                public_t.index_select(0, c),
                hand_feat_t.index_select(0, h),
                player_feat_t.index_select(0, p),
                belief_t.index_select(0, c),
            )
            loss = torch.mean((pred - value_target_t.index_select(0, batch)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        logits = model.policy(public_t, private_t, belief_t)
        policy_loss = _masked_policy_loss(
            logits,
            legal_t,
            policy_target_t,
            policy_weight_t,
        )
        optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        optimizer.step()
    return model


def _predict_values(
    model: _JointPBSContinuationNet,
    dataset: JointPBSDataset,
    *,
    target_mean: float,
    target_std: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    case_idx, hand_idx, player_idx, _values = _pair_indices(dataset)
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
            out = model.value(
                public_t.index_select(0, c),
                hand_feat_t.index_select(0, h),
                player_feat_t.index_select(0, p),
                belief_t.index_select(0, c),
            )
            outputs.append(out.cpu().numpy().astype(np.float32))
    values = np.concatenate(outputs, axis=0) * float(target_std) + float(target_mean)
    pred[player_idx, case_idx, hand_idx] = values
    return pred


def _predict_policy(
    model: _JointPBSContinuationNet,
    dataset: JointPBSDataset,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        public_t = torch.from_numpy(dataset.features).to(device)
        belief_t = torch.from_numpy(dataset.belief).to(device)
        private_t = torch.from_numpy(dataset.policy_features[:, :52]).to(device)
        legal_t = torch.from_numpy(dataset.legal_masks).to(device)
        logits = model.policy(public_t, private_t, belief_t).masked_fill(legal_t <= 0, -1e4)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return _normalize_targets(probs, dataset.legal_masks)


def _value_metrics(pred: np.ndarray, dataset: JointPBSDataset) -> dict[str, float]:
    target = np.stack([dataset.hero_values, dataset.villain_values], axis=0)
    mask = np.stack([dataset.hero_masks, dataset.villain_masks], axis=0)
    selected = mask > 0
    if not np.any(selected):
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0}
    err = pred[selected].astype(np.float64) - target[selected].astype(np.float64)
    return {
        "mae": round(float(np.mean(np.abs(err))), 8),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 8),
        "bias": round(float(np.mean(err)), 8),
    }


def _constant_value_metrics(
    dataset: JointPBSDataset,
    *,
    target_mean: float,
    target_median: float,
) -> dict[str, dict[str, float]]:
    shape = (2, dataset.features.shape[0], N_HANDS)
    return {
        "zero": _value_metrics(np.zeros(shape, dtype=np.float32), dataset),
        "train_mean": _value_metrics(
            np.full(shape, float(target_mean), dtype=np.float32),
            dataset,
        ),
        "train_median": _value_metrics(
            np.full(shape, float(target_median), dtype=np.float32),
            dataset,
        ),
    }


def _best_constant_metric(baselines: dict[str, dict[str, float]], metric: str) -> float:
    return min(float(item[metric]) for item in baselines.values())


def _legal_uniform_policy(dataset: JointPBSDataset) -> np.ndarray:
    legal = (dataset.legal_masks > 0).astype(np.float32)
    totals = legal.sum(axis=1, keepdims=True)
    return legal / np.maximum(totals, 1.0)


def run_joint_pbs_continuation_probe(
    *,
    train_joint_npz: str | Path,
    holdout_joint_npz: str | Path,
    device: str | torch.device = "auto",
    hidden_dim: int = 64,
    belief_bottleneck_dim: int = 32,
    card_encoder: str = "deepset",
    epochs: int = 30,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train_raw = load_joint_pbs_dataset(train_joint_npz)
    holdout_raw = load_joint_pbs_dataset(holdout_joint_npz)
    train, holdout = _standardize_joint_pair(train_raw, holdout_raw)
    target_mean, target_median, target_std = _target_stats(train)
    model = _fit_joint_model(
        train,
        target_mean=target_mean,
        target_std=target_std,
        hidden_dim=hidden_dim,
        belief_bottleneck_dim=belief_bottleneck_dim,
        card_encoder=card_encoder,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
    )
    value_pred = _predict_values(
        model,
        holdout,
        target_mean=target_mean,
        target_std=target_std,
        batch_size=batch_size,
        device=resolved_device,
    )
    value_holdout = _value_metrics(value_pred, holdout)
    constant_baselines = _constant_value_metrics(
        holdout,
        target_mean=target_mean,
        target_median=target_median,
    )
    policy_probs = _predict_policy(model, holdout, device=resolved_device)
    policy_holdout = _policy_metrics(
        policy_probs,
        holdout.target_probs,
        holdout.legal_masks,
    )
    uniform_policy = _policy_metrics(
        _legal_uniform_policy(holdout),
        holdout.target_probs,
        holdout.legal_masks,
    )
    best_constant_mae = _best_constant_metric(constant_baselines, "mae")
    best_constant_rmse = _best_constant_metric(constant_baselines, "rmse")
    beats_value_baselines = (
        value_holdout["mae"] < constant_baselines["zero"]["mae"]
        and value_holdout["rmse"] <= constant_baselines["zero"]["rmse"]
        and value_holdout["mae"] < best_constant_mae
        and value_holdout["rmse"] <= best_constant_rmse
    )
    beats_policy_baseline = (
        policy_holdout["mean_l1"] < uniform_policy["mean_l1"]
        and policy_holdout["mean_kl"] < uniform_policy["mean_kl"]
    )
    return {
        "mode": "joint_pbs_continuation_probe",
        "passed": bool(beats_value_baselines and beats_policy_baseline),
        "pass_criteria": (
            "joint model must beat zero/train-constant value baselines and "
            "legal-uniform policy L1/KL on held-out PBS states"
        ),
        "device": str(resolved_device),
        "train_joint_npz": str(train_joint_npz),
        "holdout_joint_npz": str(holdout_joint_npz),
        "hidden_dim": int(hidden_dim),
        "belief_bottleneck_dim": int(belief_bottleneck_dim),
        "card_encoder": card_encoder,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "feature_dim": int(N_FEATURES),
        "belief_dim": int(BELIEF_DIM),
        "hand_feature_dim": 52,
        "target_dim": int(N_HANDS),
        "train_value_label_count": int(train.hero_masks.sum() + train.villain_masks.sum()),
        "holdout_value_label_count": int(
            holdout.hero_masks.sum() + holdout.villain_masks.sum()
        ),
        "target_mean": round(float(target_mean), 8),
        "target_median": round(float(target_median), 8),
        "target_std": round(float(target_std), 8),
        "value_holdout": value_holdout,
        "constant_baselines": constant_baselines,
        "best_constant_mae": round(float(best_constant_mae), 8),
        "best_constant_rmse": round(float(best_constant_rmse), 8),
        "policy_holdout": policy_holdout,
        "policy_uniform_baseline": uniform_policy,
        "value_beats_baselines": bool(beats_value_baselines),
        "policy_beats_uniform": bool(beats_policy_baseline),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a joint PBS policy/value continuation probe."
    )
    parser.add_argument("--train-joint", required=True)
    parser.add_argument("--holdout-joint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--belief-bottleneck-dim", type=int, default=32)
    parser.add_argument(
        "--card-encoder",
        choices=("flat", "deepset"),
        default="deepset",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_joint_pbs_continuation_probe(
        train_joint_npz=args.train_joint,
        holdout_joint_npz=args.holdout_joint,
        device=args.device,
        hidden_dim=args.hidden_dim,
        belief_bottleneck_dim=args.belief_bottleneck_dim,
        card_encoder=args.card_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
