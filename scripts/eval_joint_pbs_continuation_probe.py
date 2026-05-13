#!/usr/bin/env python3
"""Evaluate a joint public-belief policy/value continuation probe."""

from __future__ import annotations

import argparse
import json
import math
import re
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

_ACTION_TOKEN_TO_ID = {"<pad>": 0, "/": 1, "k": 2, "c": 3, "f": 4, "b": 5}
_ACTION_TOKEN_RE = re.compile(r"b\d+|[kcf/]")
_DEFAULT_MAX_ACTION_TOKENS = 32


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
    hero_value_weights: np.ndarray
    villain_value_weights: np.ndarray
    action_tokens: np.ndarray
    action_amounts: np.ndarray
    labels: tuple[str, ...]


class _JointPBSContinuationNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        belief_bottleneck_dim: int = 32,
        card_encoder: str = "deepset",
        action_encoder: str = "none",
        max_action_tokens: int = _DEFAULT_MAX_ACTION_TOKENS,
    ):
        super().__init__()
        if belief_bottleneck_dim < 0:
            raise ValueError("belief_bottleneck_dim must be non-negative")
        if card_encoder not in ("flat", "deepset"):
            raise ValueError(f"unknown card_encoder: {card_encoder}")
        if action_encoder not in ("none", "gru"):
            raise ValueError(f"unknown action_encoder: {action_encoder}")
        self.card_encoder = card_encoder
        self.action_encoder = action_encoder
        self.max_action_tokens = int(max_action_tokens)
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
        if action_encoder == "gru":
            self.action_token = nn.Embedding(
                len(_ACTION_TOKEN_TO_ID),
                hidden_dim,
                padding_idx=0,
            )
            self.action_amount = nn.Linear(1, hidden_dim, bias=False)
            self.action_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.action_token = None
            self.action_amount = None
            self.action_gru = None
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

    def _action_hidden(
        self,
        action_tokens: torch.Tensor | None,
        action_amounts: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.action_encoder == "none":
            return None
        if action_tokens is None or action_amounts is None:
            return None
        if self.action_token is None or self.action_amount is None or self.action_gru is None:
            return None
        tokens = action_tokens.to(dtype=torch.long)
        amounts = action_amounts.to(dtype=self.action_token.weight.dtype)
        embedded = self.action_token(tokens) + self.action_amount(amounts.unsqueeze(-1))
        output, _hidden = self.action_gru(embedded)
        mask = tokens > 0
        lengths = mask.sum(dim=1).clamp(min=1)
        gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
        selected = output.gather(1, gather_idx).squeeze(1)
        has_tokens = mask.any(dim=1).to(dtype=selected.dtype).unsqueeze(-1)
        return selected * has_tokens

    def trunk(
        self,
        public_x: torch.Tensor,
        belief_x: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
        action_amounts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        public_hidden, board_hidden = self._public_hidden(public_x, belief_x)
        action_hidden = self._action_hidden(action_tokens, action_amounts)
        if action_hidden is not None:
            public_hidden = public_hidden + action_hidden
        return self.trunk_body(public_hidden), board_hidden

    def value(
        self,
        public_x: torch.Tensor,
        hand_x: torch.Tensor,
        player_x: torch.Tensor,
        belief_x: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
        action_amounts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trunk_hidden, board_hidden = self.trunk(
            public_x,
            belief_x,
            action_tokens,
            action_amounts,
        )
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
        action_tokens: torch.Tensor | None = None,
        action_amounts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trunk_hidden, board_hidden = self.trunk(
            public_x,
            belief_x,
            action_tokens,
            action_amounts,
        )
        private_hidden = self.private_cards(private_x)
        hidden = trunk_hidden + private_hidden
        if board_hidden is not None and self.card_interaction is not None:
            hidden = hidden + self.card_interaction(private_hidden * board_hidden)
        return self.policy_body(hidden)


def _encode_action_sequence(
    action_str: str,
    *,
    max_tokens: int = _DEFAULT_MAX_ACTION_TOKENS,
) -> tuple[np.ndarray, np.ndarray]:
    token_ids: list[int] = []
    amounts: list[float] = []
    amount_scale = math.log1p(20000.0)
    for match in _ACTION_TOKEN_RE.finditer(str(action_str)):
        token = match.group(0)
        if token.startswith("b"):
            token_ids.append(_ACTION_TOKEN_TO_ID["b"])
            amounts.append(math.log1p(float(token[1:])) / amount_scale)
        else:
            token_ids.append(_ACTION_TOKEN_TO_ID.get(token, 0))
            amounts.append(0.0)
    if max_tokens > 0:
        token_ids = token_ids[-int(max_tokens) :]
        amounts = amounts[-int(max_tokens) :]
    tokens_out = np.zeros(int(max_tokens), dtype=np.int64)
    amounts_out = np.zeros(int(max_tokens), dtype=np.float32)
    n = min(len(token_ids), int(max_tokens))
    if n > 0:
        tokens_out[:n] = np.asarray(token_ids[:n], dtype=np.int64)
        amounts_out[:n] = np.asarray(amounts[:n], dtype=np.float32)
    return tokens_out, amounts_out


def _load_action_sequences(
    labels: tuple[str, ...],
    *,
    metadata_json: str | Path | None,
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    action_by_label: dict[str, str] = {}
    metadata_path = Path(metadata_json) if metadata_json is not None else None
    if metadata_path is not None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for field in ("cut_records", "records"):
            for item in metadata.get(field, []):
                label = item.get("label")
                action = item.get("action_str")
                if label is not None and action is not None:
                    action_by_label[str(label)] = str(action)
    encoded = [
        _encode_action_sequence(action_by_label.get(label, ""), max_tokens=max_tokens)
        for label in labels
    ]
    if not encoded:
        return (
            np.zeros((0, int(max_tokens)), dtype=np.int64),
            np.zeros((0, int(max_tokens)), dtype=np.float32),
        )
    tokens, amounts = zip(*encoded, strict=True)
    return np.stack(tokens), np.stack(amounts)


def load_joint_pbs_dataset(
    path: str | Path,
    *,
    metadata_json: str | Path | None = None,
    max_action_tokens: int = _DEFAULT_MAX_ACTION_TOKENS,
) -> JointPBSDataset:
    data = np.load(Path(path), allow_pickle=False)
    policy_features = data["policy_features"] if "policy_features" in data else data["features"]
    policy_weights = (
        data["policy_weights"]
        if "policy_weights" in data
        else np.ones(data["features"].shape[0], dtype=np.float32)
    )
    hero_masks = data["hero_masks"].astype(np.float32, copy=False)
    villain_masks = data["villain_masks"].astype(np.float32, copy=False)
    hero_value_weights = (
        data["hero_value_weights"].astype(np.float32, copy=False)
        if "hero_value_weights" in data
        else hero_masks
    )
    villain_value_weights = (
        data["villain_value_weights"].astype(np.float32, copy=False)
        if "villain_value_weights" in data
        else villain_masks
    )
    labels = tuple(str(item) for item in data["labels"].tolist())
    if "action_tokens" in data and "action_amounts" in data:
        action_tokens = data["action_tokens"].astype(np.int64, copy=False)
        action_amounts = data["action_amounts"].astype(np.float32, copy=False)
    else:
        inferred_metadata = metadata_json
        if inferred_metadata is None:
            candidate = Path(path).with_suffix(".json")
            inferred_metadata = candidate if candidate.exists() else None
        action_tokens, action_amounts = _load_action_sequences(
            labels,
            metadata_json=inferred_metadata,
            max_tokens=max_action_tokens,
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
        hero_masks=hero_masks,
        villain_masks=villain_masks,
        hero_value_weights=hero_value_weights,
        villain_value_weights=villain_value_weights,
        action_tokens=action_tokens,
        action_amounts=action_amounts,
        labels=labels,
    )


def _standardize_joint_pair(
    train: JointPBSDataset,
    holdout: JointPBSDataset,
) -> tuple[JointPBSDataset, JointPBSDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        hero_value_weights=train.hero_value_weights,
        villain_value_weights=train.villain_value_weights,
        action_tokens=train.action_tokens,
        action_amounts=train.action_amounts,
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
        hero_value_weights=holdout.hero_value_weights,
        villain_value_weights=holdout.villain_value_weights,
        action_tokens=holdout.action_tokens,
        action_amounts=holdout.action_amounts,
        labels=holdout.labels,
    )
    return train_std, holdout_std, public_mean, public_std, belief_mean, belief_std


def _pair_indices(
    dataset: JointPBSDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    weights = np.concatenate(
        [
            dataset.hero_value_weights[hero_case, hero_hand],
            dataset.villain_value_weights[villain_case, villain_hand],
        ]
    ).astype(np.float32)
    weights = np.maximum(weights, 0.0)
    return case_idx, hand_idx, player_idx, values, weights


def _target_stats(dataset: JointPBSDataset) -> tuple[float, float, float]:
    _case_idx, _hand_idx, _player_idx, values, weights = _pair_indices(dataset)
    if values.size and float(weights.sum()) > 1e-8:
        mean = float(np.average(values, weights=weights))
        var = float(np.average((values - mean) ** 2, weights=weights))
        std = math.sqrt(max(var, 0.0))
    else:
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 1.0
    median = float(np.median(values)) if values.size else 0.0
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
    action_encoder: str,
    max_action_tokens: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> _JointPBSContinuationNet:
    torch.manual_seed(int(seed))
    case_idx, hand_idx, player_idx, values, value_weights = _pair_indices(dataset)
    policy_idx_np = np.flatnonzero(np.asarray(dataset.policy_weights, dtype=np.float32) > 0.0).astype(
        np.int64,
        copy=False,
    )
    if case_idx.size == 0 and policy_idx_np.size == 0:
        raise ValueError("cannot train joint PBS probe without value or policy labels")
    value_target = ((values - float(target_mean)) / float(target_std)).astype(np.float32)
    case_t = torch.from_numpy(case_idx).to(device)
    hand_t = torch.from_numpy(hand_idx).to(device)
    player_t = torch.from_numpy(player_idx).to(device)
    value_target_t = torch.from_numpy(value_target).to(device)
    value_weight_t = torch.from_numpy(value_weights).to(device)
    public_t = torch.from_numpy(dataset.features).to(device)
    belief_t = torch.from_numpy(dataset.belief).to(device)
    private_t = torch.from_numpy(dataset.policy_features[:, :52]).to(device)
    legal_t = torch.from_numpy(dataset.legal_masks).to(device)
    policy_target_t = torch.from_numpy(dataset.target_probs).to(device)
    policy_weight_t = torch.from_numpy(dataset.policy_weights).to(device)
    policy_idx_t = torch.from_numpy(policy_idx_np).to(device)
    action_token_t = torch.from_numpy(dataset.action_tokens).to(device)
    action_amount_t = torch.from_numpy(dataset.action_amounts).to(device)
    hand_feat_t = torch.from_numpy(_HAND_FEATURES).to(device)
    player_feat_t = torch.eye(2, dtype=torch.float32, device=device)

    model = _JointPBSContinuationNet(
        hidden_dim,
        belief_bottleneck_dim=belief_bottleneck_dim,
        card_encoder=card_encoder,
        action_encoder=action_encoder,
        max_action_tokens=max_action_tokens,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_value = int(case_t.numel())
    batch_size = max(1, int(batch_size))
    value_batch_size = max(1, min(batch_size, n_value)) if n_value else batch_size
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    model.train()
    for _ in range(max(1, int(epochs))):
        if n_value:
            perm = torch.randperm(n_value, generator=generator, device=device)
            for start in range(0, n_value, value_batch_size):
                batch = perm[start : start + value_batch_size]
                c = case_t.index_select(0, batch)
                h = hand_t.index_select(0, batch)
                p = player_t.index_select(0, batch)
                pred = model.value(
                    public_t.index_select(0, c),
                    hand_feat_t.index_select(0, h),
                    player_feat_t.index_select(0, p),
                    belief_t.index_select(0, c),
                    action_token_t.index_select(0, c),
                    action_amount_t.index_select(0, c),
                )
                batch_weights = value_weight_t.index_select(0, batch).to(dtype=pred.dtype)
                squared_error = (pred - value_target_t.index_select(0, batch)) ** 2
                loss = (squared_error * batch_weights).sum() / batch_weights.sum().clamp(min=1e-8)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        if policy_idx_t.numel() > 0:
            policy_perm = policy_idx_t.index_select(
                0,
                torch.randperm(policy_idx_t.numel(), generator=generator, device=device),
            )
            for start in range(0, int(policy_perm.numel()), batch_size):
                p_idx = policy_perm[start : start + batch_size]
                logits = model.policy(
                    public_t.index_select(0, p_idx),
                    private_t.index_select(0, p_idx),
                    belief_t.index_select(0, p_idx),
                    action_token_t.index_select(0, p_idx),
                    action_amount_t.index_select(0, p_idx),
                )
                policy_loss = _masked_policy_loss(
                    logits,
                    legal_t.index_select(0, p_idx),
                    policy_target_t.index_select(0, p_idx),
                    policy_weight_t.index_select(0, p_idx),
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
    case_idx, hand_idx, player_idx, _values, _weights = _pair_indices(dataset)
    pred = np.zeros((2, dataset.features.shape[0], N_HANDS), dtype=np.float32)
    if case_idx.size == 0:
        return pred
    case_t = torch.from_numpy(case_idx).to(device)
    hand_t = torch.from_numpy(hand_idx).to(device)
    player_t = torch.from_numpy(player_idx).to(device)
    public_t = torch.from_numpy(dataset.features).to(device)
    belief_t = torch.from_numpy(dataset.belief).to(device)
    action_token_t = torch.from_numpy(dataset.action_tokens).to(device)
    action_amount_t = torch.from_numpy(dataset.action_amounts).to(device)
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
                action_token_t.index_select(0, c),
                action_amount_t.index_select(0, c),
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
    batch_size: int = 8192,
) -> np.ndarray:
    batch_size = max(1, int(batch_size))
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        public_t = torch.from_numpy(dataset.features).to(device)
        belief_t = torch.from_numpy(dataset.belief).to(device)
        private_t = torch.from_numpy(dataset.policy_features[:, :52]).to(device)
        legal_t = torch.from_numpy(dataset.legal_masks).to(device)
        action_token_t = torch.from_numpy(dataset.action_tokens).to(device)
        action_amount_t = torch.from_numpy(dataset.action_amounts).to(device)
        for start in range(0, int(dataset.features.shape[0]), batch_size):
            sl = slice(start, start + batch_size)
            logits = model.policy(
                public_t[sl],
                private_t[sl],
                belief_t[sl],
                action_token_t[sl],
                action_amount_t[sl],
            ).masked_fill(legal_t[sl] <= 0, -1e4)
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
    probs = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, N_ACTIONS), dtype=np.float32)
    return _normalize_targets(probs, dataset.legal_masks)


def _save_checkpoint(
    *,
    model: _JointPBSContinuationNet,
    path: str | Path,
    hidden_dim: int,
    belief_bottleneck_dim: int,
    card_encoder: str,
    action_encoder: str,
    max_action_tokens: int,
    public_mean: np.ndarray,
    public_std: np.ndarray,
    belief_mean: np.ndarray,
    belief_std: np.ndarray,
    target_mean: float,
    target_median: float,
    target_std: float,
    train_joint_npz: str | Path,
    holdout_joint_npz: str | Path,
    seed: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": "joint_pbs_continuation_checkpoint",
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "hidden_dim": int(hidden_dim),
            "belief_bottleneck_dim": int(belief_bottleneck_dim),
            "card_encoder": card_encoder,
            "action_encoder": action_encoder,
            "max_action_tokens": int(max_action_tokens),
            "action_token_vocab_size": int(len(_ACTION_TOKEN_TO_ID)),
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
            "train_joint_npz": str(train_joint_npz),
            "holdout_joint_npz": str(holdout_joint_npz),
            "seed": int(seed),
        },
        output_path,
    )


def load_joint_pbs_continuation_checkpoint(
    checkpoint: str | Path,
    device: str | torch.device = "auto",
) -> tuple[_JointPBSContinuationNet, dict[str, Any]]:
    resolved_device = _resolve_device(device)
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    if payload.get("mode") != "joint_pbs_continuation_checkpoint":
        raise ValueError(f"{checkpoint} is not a joint PBS continuation checkpoint")
    model = _JointPBSContinuationNet(
        int(payload["hidden_dim"]),
        belief_bottleneck_dim=int(payload.get("belief_bottleneck_dim", 0)),
        card_encoder=str(payload.get("card_encoder", "deepset")),
        action_encoder=str(payload.get("action_encoder", "none")),
        max_action_tokens=int(payload.get("max_action_tokens", _DEFAULT_MAX_ACTION_TOKENS)),
    ).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def predict_joint_pbs_cfv_model(
    model: _JointPBSContinuationNet,
    payload: dict[str, Any],
    features: np.ndarray,
    belief: np.ndarray,
    hero_masks: np.ndarray,
    villain_masks: np.ndarray,
    *,
    action_tokens: np.ndarray | None = None,
    action_amounts: np.ndarray | None = None,
    device: str | torch.device = "auto",
    batch_size: int = 8192,
) -> np.ndarray:
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    features_std = _apply_standardization(
        np.asarray(features, dtype=np.float32),
        payload["public_mean"],
        payload["public_std"],
    )
    belief_std = _apply_standardization(
        np.asarray(belief, dtype=np.float32),
        payload["belief_mean"],
        payload["belief_std"],
    )
    n_states = int(features_std.shape[0])
    max_tokens = int(payload.get("max_action_tokens", _DEFAULT_MAX_ACTION_TOKENS))
    if action_tokens is None:
        action_tokens = np.zeros((n_states, max_tokens), dtype=np.int64)
    if action_amounts is None:
        action_amounts = np.zeros((n_states, max_tokens), dtype=np.float32)
    dataset = JointPBSDataset(
        features=features_std,
        policy_features=features_std,
        belief=belief_std,
        legal_masks=np.ones((n_states, N_ACTIONS), dtype=np.float32),
        target_probs=np.ones((n_states, N_ACTIONS), dtype=np.float32) / float(N_ACTIONS),
        policy_weights=np.ones(n_states, dtype=np.float32),
        hero_values=np.zeros((n_states, N_HANDS), dtype=np.float32),
        villain_values=np.zeros((n_states, N_HANDS), dtype=np.float32),
        hero_masks=np.asarray(hero_masks, dtype=np.float32),
        villain_masks=np.asarray(villain_masks, dtype=np.float32),
        hero_value_weights=np.asarray(hero_masks, dtype=np.float32),
        villain_value_weights=np.asarray(villain_masks, dtype=np.float32),
        action_tokens=np.asarray(action_tokens, dtype=np.int64),
        action_amounts=np.asarray(action_amounts, dtype=np.float32),
        labels=tuple(f"predict-{idx}" for idx in range(n_states)),
    )
    return _predict_values(
        model,
        dataset,
        target_mean=float(payload["target_mean"]),
        target_std=float(payload["target_std"]),
        batch_size=batch_size,
        device=resolved_device,
    )


def predict_joint_pbs_policy_model(
    model: _JointPBSContinuationNet,
    payload: dict[str, Any],
    features: np.ndarray,
    policy_features: np.ndarray,
    belief: np.ndarray,
    legal_masks: np.ndarray,
    *,
    action_tokens: np.ndarray | None = None,
    action_amounts: np.ndarray | None = None,
    device: str | torch.device = "auto",
) -> np.ndarray:
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    features_std = _apply_standardization(
        np.asarray(features, dtype=np.float32),
        payload["public_mean"],
        payload["public_std"],
    )
    belief_std = _apply_standardization(
        np.asarray(belief, dtype=np.float32),
        payload["belief_mean"],
        payload["belief_std"],
    )
    n_states = int(features_std.shape[0])
    max_tokens = int(payload.get("max_action_tokens", _DEFAULT_MAX_ACTION_TOKENS))
    if action_tokens is None:
        action_tokens = np.zeros((n_states, max_tokens), dtype=np.int64)
    if action_amounts is None:
        action_amounts = np.zeros((n_states, max_tokens), dtype=np.float32)
    dataset = JointPBSDataset(
        features=features_std,
        policy_features=np.asarray(policy_features, dtype=np.float32),
        belief=belief_std,
        legal_masks=np.asarray(legal_masks, dtype=np.float32),
        target_probs=np.ones((n_states, N_ACTIONS), dtype=np.float32) / float(N_ACTIONS),
        policy_weights=np.ones(n_states, dtype=np.float32),
        hero_values=np.zeros((n_states, N_HANDS), dtype=np.float32),
        villain_values=np.zeros((n_states, N_HANDS), dtype=np.float32),
        hero_masks=np.zeros((n_states, N_HANDS), dtype=np.float32),
        villain_masks=np.zeros((n_states, N_HANDS), dtype=np.float32),
        hero_value_weights=np.zeros((n_states, N_HANDS), dtype=np.float32),
        villain_value_weights=np.zeros((n_states, N_HANDS), dtype=np.float32),
        action_tokens=np.asarray(action_tokens, dtype=np.int64),
        action_amounts=np.asarray(action_amounts, dtype=np.float32),
        labels=tuple(f"policy-{idx}" for idx in range(n_states)),
    )
    return _predict_policy(model, dataset, device=resolved_device)


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
    train_metadata_json: str | Path | None = None,
    holdout_metadata_json: str | Path | None = None,
    device: str | torch.device = "auto",
    hidden_dim: int = 64,
    belief_bottleneck_dim: int = 32,
    card_encoder: str = "deepset",
    action_encoder: str = "none",
    max_action_tokens: int = _DEFAULT_MAX_ACTION_TOKENS,
    epochs: int = 30,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    output_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train_raw = load_joint_pbs_dataset(
        train_joint_npz,
        metadata_json=train_metadata_json,
        max_action_tokens=max_action_tokens,
    )
    holdout_raw = load_joint_pbs_dataset(
        holdout_joint_npz,
        metadata_json=holdout_metadata_json,
        max_action_tokens=max_action_tokens,
    )
    train, holdout, public_mean, public_std, belief_mean, belief_std = (
        _standardize_joint_pair(train_raw, holdout_raw)
    )
    target_mean, target_median, target_std = _target_stats(train)
    model = _fit_joint_model(
        train,
        target_mean=target_mean,
        target_std=target_std,
        hidden_dim=hidden_dim,
        belief_bottleneck_dim=belief_bottleneck_dim,
        card_encoder=card_encoder,
        action_encoder=action_encoder,
        max_action_tokens=max_action_tokens,
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
    policy_probs = _predict_policy(model, holdout, device=resolved_device, batch_size=batch_size)
    policy_mask = holdout.policy_weights > 0
    train_value_label_count = int(train.hero_masks.sum() + train.villain_masks.sum())
    holdout_value_label_count = int(holdout.hero_masks.sum() + holdout.villain_masks.sum())
    value_gate_active = bool(train_value_label_count > 0 and holdout_value_label_count > 0)
    train_policy_label_count = int(np.count_nonzero(train.policy_weights > 0))
    holdout_policy_label_count = int(np.count_nonzero(policy_mask))
    policy_gate_active = bool(train_policy_label_count > 0 and holdout_policy_label_count > 0)
    if policy_gate_active:
        policy_holdout = _policy_metrics(
            policy_probs[policy_mask],
            holdout.target_probs[policy_mask],
            holdout.legal_masks[policy_mask],
        )
        uniform_policy = _policy_metrics(
            _legal_uniform_policy(holdout)[policy_mask],
            holdout.target_probs[policy_mask],
            holdout.legal_masks[policy_mask],
        )
    else:
        policy_holdout = {"mean_l1": 0.0, "mean_kl": 0.0}
        uniform_policy = {"mean_l1": 0.0, "mean_kl": 0.0}
    best_constant_mae = _best_constant_metric(constant_baselines, "mae")
    best_constant_rmse = _best_constant_metric(constant_baselines, "rmse")
    beats_value_baselines = (
        True
        if not value_gate_active
        else (
            value_holdout["mae"] < constant_baselines["zero"]["mae"]
            and value_holdout["rmse"] <= constant_baselines["zero"]["rmse"]
            and value_holdout["mae"] < best_constant_mae
            and value_holdout["rmse"] <= best_constant_rmse
        )
    )
    beats_policy_baseline = (
        True
        if not policy_gate_active
        else (
            policy_holdout["mean_l1"] < uniform_policy["mean_l1"]
            and policy_holdout["mean_kl"] < uniform_policy["mean_kl"]
        )
    )
    if output_checkpoint is not None:
        _save_checkpoint(
            model=model,
            path=output_checkpoint,
            hidden_dim=hidden_dim,
            belief_bottleneck_dim=belief_bottleneck_dim,
            card_encoder=card_encoder,
            action_encoder=action_encoder,
            max_action_tokens=max_action_tokens,
            public_mean=public_mean,
            public_std=public_std,
            belief_mean=belief_mean,
            belief_std=belief_std,
            target_mean=target_mean,
            target_median=target_median,
            target_std=target_std,
            train_joint_npz=train_joint_npz,
            holdout_joint_npz=holdout_joint_npz,
            seed=seed,
        )
    return {
        "mode": "joint_pbs_continuation_probe",
        "passed": bool(beats_value_baselines and beats_policy_baseline),
        "pass_criteria": (
            "joint model must beat zero/train-constant value baselines when "
            "value targets are present and legal-uniform policy L1/KL when "
            "policy targets are present"
        ),
        "device": str(resolved_device),
        "train_joint_npz": str(train_joint_npz),
        "holdout_joint_npz": str(holdout_joint_npz),
        "train_metadata_json": str(train_metadata_json) if train_metadata_json else None,
        "holdout_metadata_json": str(holdout_metadata_json) if holdout_metadata_json else None,
        "hidden_dim": int(hidden_dim),
        "belief_bottleneck_dim": int(belief_bottleneck_dim),
        "card_encoder": card_encoder,
        "action_encoder": action_encoder,
        "max_action_tokens": int(max_action_tokens),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "checkpoint": str(output_checkpoint) if output_checkpoint is not None else None,
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "feature_dim": int(N_FEATURES),
        "belief_dim": int(BELIEF_DIM),
        "hand_feature_dim": 52,
        "target_dim": int(N_HANDS),
        "train_value_label_count": train_value_label_count,
        "holdout_value_label_count": holdout_value_label_count,
        "value_gate_active": bool(value_gate_active),
        "train_policy_label_count": train_policy_label_count,
        "holdout_policy_label_count": holdout_policy_label_count,
        "policy_gate_active": bool(policy_gate_active),
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
    parser.add_argument("--train-metadata-json")
    parser.add_argument("--holdout-metadata-json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--belief-bottleneck-dim", type=int, default=32)
    parser.add_argument(
        "--card-encoder",
        choices=("flat", "deepset"),
        default="deepset",
    )
    parser.add_argument(
        "--action-encoder",
        choices=("none", "gru"),
        default="none",
    )
    parser.add_argument("--max-action-tokens", type=int, default=_DEFAULT_MAX_ACTION_TOKENS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_joint_pbs_continuation_probe(
        train_joint_npz=args.train_joint,
        holdout_joint_npz=args.holdout_joint,
        train_metadata_json=args.train_metadata_json,
        holdout_metadata_json=args.holdout_metadata_json,
        device=args.device,
        hidden_dim=args.hidden_dim,
        belief_bottleneck_dim=args.belief_bottleneck_dim,
        card_encoder=args.card_encoder,
        action_encoder=args.action_encoder,
        max_action_tokens=args.max_action_tokens,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        output_checkpoint=args.output_checkpoint,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
