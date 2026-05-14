"""Native full-deck 9-action NFSP-style pilot.

This is intentionally a small autoresearch pilot, not a promoted agent. It
keeps the repository's full-deck state/action contract and uses Monte-Carlo
terminal returns as the first best-response learner target.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.games.full_deck.state import (
    ACTION_TO_INDEX,
    INDEX_TO_ACTION,
    N_ACTIONS,
    N_FEATURES,
    PokerState,
    new_game,
)
from poker_ai.research.game_theoretic_rl import (
    epsilon_greedy_distribution,
    legal_softmax,
)


@dataclass(frozen=True)
class NativeNFSPConfig:
    train_episodes: int = 100
    eval_games: int = 100
    hidden_dim: int = 64
    batch_size: int = 128
    min_buffer_size_to_learn: int = 32
    anticipatory_param: float = 0.1
    epsilon: float = 0.06
    lr: float = 1e-3
    initial_chips: int = 1000
    max_steps_per_hand: int = 256
    seed: int = 20260514
    device: str = "auto"


class _MLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class _SampleBuffer:
    def __init__(self, capacity: int = 20_000):
        self.capacity = int(capacity)
        self.items: list[tuple[np.ndarray, np.ndarray, int, float]] = []

    def add(
        self,
        feature: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        value: float = 0.0,
    ) -> None:
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(
            (
                np.asarray(feature, dtype=np.float32),
                np.asarray(legal_mask, dtype=np.float32),
                int(action),
                float(value),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = min(int(batch_size), len(self.items))
        indices = rng.choice(len(self.items), size=n, replace=False)
        return [self.items[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self.items)


def resolve_device(requested_device: str = "auto") -> dict:
    requested = requested_device.strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
        policy = "cuda_when_available_for_native_network_training"
    elif requested == "cuda":
        if not cuda_available:
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        resolved = "cuda"
        policy = "explicit_cuda"
    elif requested == "cpu":
        resolved = "cpu"
        policy = "explicit_cpu"
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return {
        "requested_device": requested,
        "resolved_device": resolved,
        "device_policy": policy,
        "torch_cuda_available": cuda_available,
        "torch_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "torch_device_name": torch.cuda.get_device_name(0) if cuda_available else "",
    }


def get_legal_mask(state: PokerState) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for action in state.legal_actions:
        if action is not None and action in ACTION_TO_INDEX:
            mask[ACTION_TO_INDEX[action]] = 1.0
    if mask.sum() <= 0:
        raise ValueError("state has no legal indexed actions")
    return mask


def masked_uniform(legal_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(legal_mask, dtype=np.float32)
    total = float(mask.sum())
    if total <= 0:
        raise ValueError("legal_mask must contain at least one legal action")
    return mask / total


def select_action(
    probs: np.ndarray,
    legal_mask: np.ndarray,
    *,
    rng: np.random.Generator,
) -> int:
    masked = np.asarray(probs, dtype=np.float32) * np.asarray(legal_mask, dtype=np.float32)
    total = float(masked.sum())
    if total <= 1e-8:
        masked = masked_uniform(legal_mask)
    else:
        masked = masked / total
    return int(rng.choice(np.arange(masked.shape[0]), p=masked))


def _network_probs(net: nn.Module, features: np.ndarray, legal_mask: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = net(x).detach().cpu().numpy().reshape(-1)
    return legal_softmax(logits, legal_mask)


def _q_values(net: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        return net(x).detach().cpu().numpy().reshape(-1)


def _train_q(
    q_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: _SampleBuffer,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float | None:
    if len(buffer) <= 0:
        return None
    batch = buffer.sample(batch_size, rng)
    features = torch.tensor(np.stack([b[0] for b in batch]), device=device)
    actions = torch.tensor([int(b[2]) for b in batch], device=device, dtype=torch.long)
    targets = torch.tensor([float(b[3]) for b in batch], device=device, dtype=torch.float32)
    pred = q_net(features).gather(1, actions[:, None]).squeeze(1)
    loss = ((pred - targets) ** 2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _train_avg_policy(
    avg_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: _SampleBuffer,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float | None:
    if len(buffer) <= 0:
        return None
    batch = buffer.sample(batch_size, rng)
    features = torch.tensor(np.stack([b[0] for b in batch]), device=device)
    legal_masks = torch.tensor(np.stack([b[1] for b in batch]), device=device)
    actions = torch.tensor([int(b[2]) for b in batch], device=device, dtype=torch.long)
    logits = avg_net(features).masked_fill(legal_masks <= 0, -1e4)
    loss = nn.functional.cross_entropy(logits, actions)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _play_hand(
    q_net: nn.Module,
    avg_net: nn.Module,
    cfg: NativeNFSPConfig,
    rng: np.random.Generator,
    device: torch.device,
    *,
    training: bool,
    opponent_random: bool = False,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray, int, bool]], list[float], int]:
    state = new_game(2, initial_chips=cfg.initial_chips)
    records: list[tuple[int, np.ndarray, np.ndarray, int, bool]] = []
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = state.player_i
        features = state.to_feature_vector()
        legal_mask = get_legal_mask(state)
        if opponent_random and player == 1:
            probs = masked_uniform(legal_mask)
            best_response_mode = False
        else:
            q_values = _q_values(q_net, features, device)
            br_probs = epsilon_greedy_distribution(
                q_values,
                legal_mask,
                epsilon=cfg.epsilon if training else 0.0,
            )
            avg_probs = _network_probs(avg_net, features, legal_mask, device)
            if training:
                best_response_mode = bool(rng.random() < cfg.anticipatory_param)
                probs = br_probs if best_response_mode else avg_probs
            else:
                best_response_mode = False
                probs = avg_probs
        action_idx = select_action(probs, legal_mask, rng=rng)
        records.append((player, features, legal_mask, action_idx, best_response_mode))
        state = state.apply_action(INDEX_TO_ACTION[action_idx])
        n_steps += 1
    payouts = [float(state.payout.get(i, 0)) / float(cfg.initial_chips) for i in range(2)]
    return records, payouts, n_steps


def run_native_nfsp_pilot(cfg: NativeNFSPConfig | None = None) -> dict:
    cfg = cfg or NativeNFSPConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])

    q_net = _MLP(cfg.hidden_dim).to(device)
    avg_net = _MLP(cfg.hidden_dim).to(device)
    q_opt = optim.Adam(q_net.parameters(), lr=cfg.lr)
    avg_opt = optim.Adam(avg_net.parameters(), lr=cfg.lr)
    q_buffer = _SampleBuffer()
    sl_buffer = _SampleBuffer()

    train_start = time.perf_counter()
    total_steps = 0
    last_q_loss = None
    last_sl_loss = None
    payoffs: list[float] = []
    for _ in range(int(cfg.train_episodes)):
        records, payouts, steps = _play_hand(
            q_net,
            avg_net,
            cfg,
            rng,
            device,
            training=True,
        )
        total_steps += steps
        payoffs.append(payouts[0])
        for player, features, legal_mask, action_idx, best_response_mode in records:
            payoff = payouts[player]
            q_buffer.add(features, legal_mask, action_idx, payoff)
            if best_response_mode:
                sl_buffer.add(features, legal_mask, action_idx)
        if len(q_buffer) >= cfg.min_buffer_size_to_learn:
            last_q_loss = _train_q(q_net, q_opt, q_buffer, cfg.batch_size, rng, device)
        if len(sl_buffer) >= cfg.min_buffer_size_to_learn:
            last_sl_loss = _train_avg_policy(avg_net, avg_opt, sl_buffer, cfg.batch_size, rng, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    eval_payoffs = []
    eval_steps = 0
    for _ in range(int(cfg.eval_games)):
        _, payouts, steps = _play_hand(
            q_net,
            avg_net,
            cfg,
            rng,
            device,
            training=False,
            opponent_random=True,
        )
        eval_payoffs.append(payouts[0])
        eval_steps += steps
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    return {
        "algorithm": "native_nfsp_mc",
        "role": "native_game_theoretic_rl_pilot",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Monte-Carlo NFSP pilot for plumbing and compute diagnostics; not a promoted poker agent.",
        **device_info,
        "num_actions": N_ACTIONS,
        "train_episodes": int(cfg.train_episodes),
        "eval_games": int(cfg.eval_games),
        "train_steps": int(total_steps),
        "eval_steps": int(eval_steps),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "episodes_per_second": float(cfg.train_episodes / max(train_seconds, 1e-9)),
        "train_steps_per_second": float(total_steps / max(train_seconds, 1e-9)),
        "mean_train_payoff_p0": float(np.mean(payoffs)) if payoffs else 0.0,
        "mean_eval_payoff_p0_vs_random": float(np.mean(eval_payoffs)) if eval_payoffs else 0.0,
        "q_buffer_size": len(q_buffer),
        "sl_buffer_size": len(sl_buffer),
        "last_q_loss": last_q_loss,
        "last_sl_loss": last_sl_loss,
        "promotion": False,
    }
