"""Native full-deck 9-action NFSP-style pilot.

This is intentionally a small autoresearch pilot, not a promoted agent. It
keeps the repository's full-deck state/action contract and uses same-player
next-decision transitions for the first DQN-style best-response learner target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class BestResponseTransition:
    features: np.ndarray
    legal_mask: np.ndarray
    action: int
    reward: float
    next_features: np.ndarray
    next_legal_mask: np.ndarray
    done: bool


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


class ReservoirPolicyBuffer:
    def __init__(self, capacity: int = 20_000):
        self.capacity = int(capacity)
        self.items: list[tuple[np.ndarray, np.ndarray, int, float]] = []
        self.n_seen = 0

    def add(
        self,
        feature: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        value: float = 0.0,
        *,
        rng: np.random.Generator,
    ) -> None:
        item = (
            np.asarray(feature, dtype=np.float32),
            np.asarray(legal_mask, dtype=np.float32),
            int(action),
            float(value),
        )
        self.n_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replacement_idx = int(rng.integers(self.n_seen))
        if replacement_idx < self.capacity:
            self.items[replacement_idx] = item

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = min(int(batch_size), len(self.items))
        indices = rng.choice(len(self.items), size=n, replace=False)
        return [self.items[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self.items)


class _TransitionBuffer:
    def __init__(self, capacity: int = 20_000):
        self.capacity = int(capacity)
        self.items: list[BestResponseTransition] = []

    def add(self, transition: BestResponseTransition) -> None:
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[BestResponseTransition]:
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


def sample_episode_policy_modes(
    *,
    n_players: int,
    anticipatory_param: float,
    rng: np.random.Generator,
) -> tuple[bool, ...]:
    """Sample NFSP best-response modes once per player for the whole hand."""
    eta = float(np.clip(anticipatory_param, 0.0, 1.0))
    return tuple(bool(rng.random() < eta) for _ in range(int(n_players)))


def build_player_transitions(
    records: list[tuple[int, np.ndarray, np.ndarray, int, bool]],
    payouts: list[float],
) -> list[BestResponseTransition]:
    """Build same-player next-decision transitions from one completed hand."""
    transitions: list[BestResponseTransition] = []
    for i, (player, features, legal_mask, action_idx, _best_response_mode) in enumerate(records):
        next_record = next(
            (record for record in records[i + 1 :] if record[0] == player),
            None,
        )
        if next_record is None:
            next_features = np.zeros_like(features, dtype=np.float32)
            next_legal_mask = np.zeros_like(legal_mask, dtype=np.float32)
            reward = float(payouts[player])
            done = True
        else:
            next_features = np.asarray(next_record[1], dtype=np.float32)
            next_legal_mask = np.asarray(next_record[2], dtype=np.float32)
            reward = 0.0
            done = False
        transitions.append(
            BestResponseTransition(
                features=np.asarray(features, dtype=np.float32),
                legal_mask=np.asarray(legal_mask, dtype=np.float32),
                action=int(action_idx),
                reward=reward,
                next_features=next_features,
                next_legal_mask=next_legal_mask,
                done=done,
            )
        )
    return transitions


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
    buffer: _TransitionBuffer,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float | None:
    if len(buffer) <= 0:
        return None
    batch = buffer.sample(batch_size, rng)
    features = torch.tensor(np.stack([b.features for b in batch]), device=device)
    actions = torch.tensor([int(b.action) for b in batch], device=device, dtype=torch.long)
    rewards = torch.tensor([float(b.reward) for b in batch], device=device, dtype=torch.float32)
    next_features = torch.tensor(np.stack([b.next_features for b in batch]), device=device)
    next_legal_masks = torch.tensor(np.stack([b.next_legal_mask for b in batch]), device=device)
    dones = torch.tensor([bool(b.done) for b in batch], device=device, dtype=torch.bool)
    with torch.no_grad():
        next_values = q_net(next_features).masked_fill(next_legal_masks <= 0, -1e4).max(dim=1).values
        targets = torch.where(dones, rewards, rewards + next_values)
    pred = q_net(features).gather(1, actions[:, None]).squeeze(1)
    loss = ((pred - targets) ** 2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _train_avg_policy(
    avg_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: ReservoirPolicyBuffer,
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
    episode_best_response_modes = sample_episode_policy_modes(
        n_players=len(state.players),
        anticipatory_param=cfg.anticipatory_param,
        rng=rng,
    )
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = state.player_i
        features = state.to_feature_vector()
        legal_mask = get_legal_mask(state)
        if opponent_random and player == 1:
            probs = masked_uniform(legal_mask)
            best_response_mode = False
        else:
            avg_probs = _network_probs(avg_net, features, legal_mask, device)
            if training:
                q_values = _q_values(q_net, features, device)
                br_probs = epsilon_greedy_distribution(
                    q_values,
                    legal_mask,
                    epsilon=cfg.epsilon,
                )
                best_response_mode = episode_best_response_modes[player]
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
    q_buffer = _TransitionBuffer()
    sl_buffer = ReservoirPolicyBuffer()

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
        for transition in build_player_transitions(records, payouts):
            q_buffer.add(transition)
        for _player, features, legal_mask, action_idx, best_response_mode in records:
            if best_response_mode:
                sl_buffer.add(features, legal_mask, action_idx, rng=rng)
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

    metrics = {
        "algorithm": "native_nfsp_dqn",
        "role": "native_game_theoretic_rl_pilot",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "DQN-style NFSP pilot for plumbing and compute diagnostics; not a promoted poker agent.",
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
    if cfg.checkpoint_path:
        path = Path(cfg.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["checkpoint_path"] = str(path)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "role": metrics["role"],
                "environment": metrics["environment"],
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(cfg.hidden_dim),
                "q_net_state_dict": q_net.state_dict(),
                "avg_net_state_dict": avg_net.state_dict(),
                "config": {
                    "train_episodes": int(cfg.train_episodes),
                    "eval_games": int(cfg.eval_games),
                    "hidden_dim": int(cfg.hidden_dim),
                    "batch_size": int(cfg.batch_size),
                    "min_buffer_size_to_learn": int(cfg.min_buffer_size_to_learn),
                    "anticipatory_param": float(cfg.anticipatory_param),
                    "epsilon": float(cfg.epsilon),
                    "lr": float(cfg.lr),
                    "initial_chips": int(cfg.initial_chips),
                    "max_steps_per_hand": int(cfg.max_steps_per_hand),
                    "seed": int(cfg.seed),
                    "device": str(cfg.device),
                },
                "metrics": metrics,
            },
            path,
        )
    return metrics


def _load_native_checkpoint_networks(
    checkpoint_path: str,
    resolved_device: torch.device,
) -> tuple[dict, nn.Module, nn.Module]:
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("checkpoint action count does not match native full-deck action contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("checkpoint feature count does not match native full-deck feature contract")

    config_payload = payload.get("config", {})
    hidden_dim = int(payload.get("hidden_dim", config_payload.get("hidden_dim", 64)))
    q_net = _MLP(hidden_dim).to(resolved_device)
    avg_net = _MLP(hidden_dim).to(resolved_device)
    q_net.load_state_dict(payload["q_net_state_dict"])
    avg_net.load_state_dict(payload["avg_net_state_dict"])
    q_net.eval()
    avg_net.eval()
    return payload, q_net, avg_net


def evaluate_native_nfsp_checkpoint(
    checkpoint_path: str,
    *,
    eval_games: int = 100,
    device: str = "auto",
    seed: int = 20260514,
) -> dict:
    """Evaluate a saved native NFSP average policy against a random player."""
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    payload, q_net, avg_net = _load_native_checkpoint_networks(checkpoint_path, resolved_device)
    config_payload = payload.get("config", {})
    cfg = NativeNFSPConfig(
        train_episodes=0,
        eval_games=int(eval_games),
        hidden_dim=int(payload.get("hidden_dim", config_payload.get("hidden_dim", 64))),
        initial_chips=int(config_payload.get("initial_chips", 1000)),
        max_steps_per_hand=int(config_payload.get("max_steps_per_hand", 256)),
        seed=int(seed),
        device=device,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    eval_start = time.perf_counter()
    eval_payoffs = []
    eval_steps = 0
    for _ in range(int(eval_games)):
        _, payouts, steps = _play_hand(
            q_net,
            avg_net,
            cfg,
            rng,
            resolved_device,
            training=False,
            opponent_random=True,
        )
        eval_payoffs.append(payouts[0])
        eval_steps += steps
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    return {
        "algorithm": str(payload.get("algorithm", "native_nfsp_dqn")),
        "role": "native_game_theoretic_rl_checkpoint_eval",
        "environment": str(payload.get("environment", "poker_ai:full_deck_hu_nlhe")),
        "source_checkpoint": str(checkpoint_path),
        **device_info,
        "num_actions": N_ACTIONS,
        "eval_games": int(eval_games),
        "eval_steps": int(eval_steps),
        "eval_seconds": float(eval_seconds),
        "eval_games_per_second": float(eval_games / max(eval_seconds, 1e-9)),
        "eval_steps_per_second": float(eval_steps / max(eval_seconds, 1e-9)),
        "mean_eval_payoff_p0_vs_random": float(np.mean(eval_payoffs)) if eval_payoffs else 0.0,
        "promotion": False,
    }


def evaluate_native_nfsp_head_to_head(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260514,
) -> dict:
    """Evaluate two native NFSP average policies in alternating seats."""
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_payload, _candidate_q, candidate_avg = _load_native_checkpoint_networks(
        candidate_checkpoint,
        resolved_device,
    )
    baseline_payload, _baseline_q, baseline_avg = _load_native_checkpoint_networks(
        baseline_checkpoint,
        resolved_device,
    )
    config_payload = candidate_payload.get("config", {})
    initial_chips = int(config_payload.get("initial_chips", 1000))
    max_steps_per_hand = int(config_payload.get("max_steps_per_hand", 256))
    candidate_payoffs: list[float] = []
    total_steps = 0

    eval_start = time.perf_counter()
    pair_i = 0
    while len(candidate_payoffs) < int(n_games):
        game_seed = int(seed) + pair_i
        action_seed = int(seed) + 1_000_000 + pair_i
        for candidate_seat in (0, 1):
            if len(candidate_payoffs) >= int(n_games):
                break
            random.seed(game_seed)
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)
            rng = np.random.default_rng(action_seed)
            baseline_seat = 1 - candidate_seat
            policies = {
                candidate_seat: candidate_avg,
                baseline_seat: baseline_avg,
            }
            state = new_game(2, initial_chips=initial_chips)
            n_steps = 0
            while not state.is_terminal and n_steps < max_steps_per_hand:
                features = state.to_feature_vector()
                legal_mask = get_legal_mask(state)
                probs = _network_probs(policies[state.player_i], features, legal_mask, resolved_device)
                action_idx = select_action(probs, legal_mask, rng=rng)
                state = state.apply_action(INDEX_TO_ACTION[action_idx])
                n_steps += 1
            total_steps += n_steps
            candidate_payoffs.append(float(state.payout.get(candidate_seat, 0)) / float(initial_chips))
        pair_i += 1
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    return {
        "algorithm": "native_nfsp_dqn_h2h",
        "role": "native_game_theoretic_rl_checkpoint_head_to_head",
        "environment": str(candidate_payload.get("environment", "poker_ai:full_deck_hu_nlhe")),
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_algorithm": str(baseline_payload.get("algorithm", "")),
        **device_info,
        "num_actions": N_ACTIONS,
        "n_games": int(n_games),
        "eval_seconds": float(eval_seconds),
        "eval_steps": int(total_steps),
        "eval_games_per_second": float(n_games / max(eval_seconds, 1e-9)),
        "eval_steps_per_second": float(total_steps / max(eval_seconds, 1e-9)),
        "mean_candidate_payoff": float(np.mean(candidate_payoffs)) if candidate_payoffs else 0.0,
        "promotion": False,
    }
