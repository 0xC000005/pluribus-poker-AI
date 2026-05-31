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
from typing import Any, Sequence

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
from poker_ai.deep_cfr.fast_state import new_fast_game
from poker_ai.research.game_theoretic_rl import (
    epsilon_greedy_distribution,
    legal_softmax,
)


@dataclass(frozen=True)
class NativeNFSPConfig:
    train_episodes: int = 100
    eval_games: int = 100
    hidden_dim: int = 64
    q_network_arch: str = "mlp"
    batch_size: int = 128
    min_buffer_size_to_learn: int = 32
    anticipatory_param: float = 0.1
    epsilon: float = 0.06
    lr: float = 1e-3
    q_discount: float = 0.99
    q_target_sync_interval: int = 1000
    initial_chips: int = 1000
    max_steps_per_hand: int = 256
    seed: int = 20260514
    device: str = "auto"
    checkpoint_path: str | None = None
    opponent_kind: str = "self"
    opponent_checkpoint: str | Sequence[str] | None = None
    opponent_device: str = "same"
    state_backend: str = "full-deck"


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


class _DuelingQNetwork(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        hidden = self.trunk(x)
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def _normalize_q_network_arch(q_network_arch: str) -> str:
    arch = str(q_network_arch).strip().lower()
    if arch not in {"mlp", "dueling"}:
        raise ValueError("q_network_arch must be one of: mlp, dueling")
    return arch


def _build_q_network(hidden_dim: int, q_network_arch: str) -> nn.Module:
    arch = _normalize_q_network_arch(q_network_arch)
    if arch == "dueling":
        return _DuelingQNetwork(hidden_dim)
    return _MLP(hidden_dim)


def _native_nfsp_algorithm_name(q_network_arch: str) -> str:
    arch = _normalize_q_network_arch(q_network_arch)
    if arch == "dueling":
        return "native_nfsp_dueling_ddqn"
    return "native_nfsp_dqn"


def _normalize_opponent_checkpoints(
    opponent_checkpoint: str | Path | Sequence[str | Path] | None,
) -> list[str]:
    if opponent_checkpoint is None:
        return []
    if isinstance(opponent_checkpoint, (str, Path)):
        return [str(opponent_checkpoint)]
    return [str(path) for path in opponent_checkpoint if str(path)]


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


def _normalize_state_backend(state_backend: str) -> str:
    backend = str(state_backend).strip().lower()
    if backend not in {"full-deck", "fast-state"}:
        raise ValueError("state_backend must be one of: full-deck, fast-state")
    return backend


def _new_training_state(cfg: NativeNFSPConfig):
    if _normalize_state_backend(cfg.state_backend) == "fast-state":
        return new_fast_game(2, initial_chips=cfg.initial_chips)
    return new_game(2, initial_chips=cfg.initial_chips)


def _state_player_i(state) -> int:
    return int(getattr(state, "player_i", getattr(state, "current_player_i", 0)))


def _state_n_players(state) -> int:
    if hasattr(state, "players"):
        return len(state.players)
    return int(getattr(state, "n_players", 2))


def _state_legal_mask(state) -> np.ndarray:
    if hasattr(state, "get_legal_mask"):
        return state.get_legal_mask().astype(np.float32, copy=False)
    return get_legal_mask(state)


def _state_apply_action(state, action_idx: int):
    if hasattr(state, "get_legal_mask"):
        state.apply_action(int(action_idx))
        return state
    return state.apply_action(INDEX_TO_ACTION[int(action_idx)])


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
    q_target_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: _TransitionBuffer,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    discount: float,
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
        next_online_values = q_net(next_features).masked_fill(next_legal_masks <= 0, -1e4)
        best_next_actions = next_online_values.argmax(dim=1)
        next_target_values = q_target_net(next_features).gather(1, best_next_actions[:, None]).squeeze(1)
        targets = torch.where(dones, rewards, rewards + float(discount) * next_target_values)
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
    opponent_adapter: Any | None = None,
    opponent_device: torch.device | None = None,
    learner_seat: int | None = None,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray, int, bool]], list[float], int]:
    state = _new_training_state(cfg)
    records: list[tuple[int, np.ndarray, np.ndarray, int, bool]] = []
    episode_best_response_modes = sample_episode_policy_modes(
        n_players=_state_n_players(state),
        anticipatory_param=cfg.anticipatory_param,
        rng=rng,
    )
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = _state_player_i(state)
        features = state.to_feature_vector()
        legal_mask = _state_legal_mask(state)
        if opponent_adapter is not None and learner_seat is not None and player != learner_seat:
            action_idx = int(
                opponent_adapter.select_action(
                    state=state,
                    features=features,
                    legal_mask=legal_mask,
                    device=opponent_device or device,
                    rng=rng,
                )
            )
            best_response_mode = False
        elif opponent_random and player == 1:
            probs = masked_uniform(legal_mask)
            best_response_mode = False
            action_idx = select_action(probs, legal_mask, rng=rng)
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
        state = _state_apply_action(state, action_idx)
        n_steps += 1
    payouts = [float(state.payout.get(i, 0)) / float(cfg.initial_chips) for i in range(2)]
    return records, payouts, n_steps


def run_native_nfsp_pilot(cfg: NativeNFSPConfig | None = None) -> dict:
    cfg = cfg or NativeNFSPConfig()
    state_backend = _normalize_state_backend(cfg.state_backend)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    opponent_kind = str(cfg.opponent_kind).strip().lower()
    opponent_adapter = None
    opponent_adapters: list[Any] = []
    opponent_device_info: dict | None = None
    opponent_device = device
    opponent_checkpoints = _normalize_opponent_checkpoints(cfg.opponent_checkpoint)
    if opponent_kind not in {"self", "self-play", "self_play"}:
        if opponent_kind == "random":
            pass
        else:
            if not opponent_checkpoints:
                raise ValueError("opponent_checkpoint is required for learned fixed opponents")
            opponent_device_request = (
                device_info["resolved_device"]
                if str(cfg.opponent_device).strip().lower() == "same"
                else str(cfg.opponent_device)
            )
            opponent_device_info = resolve_device(opponent_device_request)
            opponent_device = torch.device(opponent_device_info["resolved_device"])
            from poker_ai.research.mixed_policy_h2h import load_policy_adapter  # noqa: PLC0415

            opponent_adapters = [
                load_policy_adapter(
                    checkpoint,
                    kind=opponent_kind,
                    device=opponent_device,
                )
                for checkpoint in opponent_checkpoints
            ]
            opponent_adapter = opponent_adapters[0]

    q_network_arch = _normalize_q_network_arch(cfg.q_network_arch)
    q_net = _build_q_network(cfg.hidden_dim, q_network_arch).to(device)
    q_target_net = _build_q_network(cfg.hidden_dim, q_network_arch).to(device)
    q_target_net.load_state_dict(q_net.state_dict())
    q_target_net.eval()
    avg_net = _MLP(cfg.hidden_dim).to(device)
    q_opt = optim.Adam(q_net.parameters(), lr=cfg.lr)
    avg_opt = optim.Adam(avg_net.parameters(), lr=cfg.lr)
    q_buffer = _TransitionBuffer()
    sl_buffer = ReservoirPolicyBuffer()

    train_start = time.perf_counter()
    total_steps = 0
    last_q_loss = None
    last_sl_loss = None
    q_updates = 0
    q_target_syncs = 0
    payoffs: list[float] = []
    fixed_opponent_learning_seats: set[int] = set()
    fixed_opponent_training = opponent_adapter is not None
    opponent_sample_counts: dict[str, int] = {checkpoint: 0 for checkpoint in opponent_checkpoints}
    for episode_idx in range(int(cfg.train_episodes)):
        learner_seat = None
        hand_opponent_adapter = opponent_adapter
        if fixed_opponent_training:
            learner_seat = int(episode_idx % 2)
            fixed_opponent_learning_seats.add(learner_seat)
            if opponent_adapters:
                opponent_i = int(episode_idx % len(opponent_adapters))
                hand_opponent_adapter = opponent_adapters[opponent_i]
                opponent_sample_counts[opponent_checkpoints[opponent_i]] += 1
        records, payouts, steps = _play_hand(
            q_net,
            avg_net,
            cfg,
            rng,
            device,
            training=True,
            opponent_adapter=hand_opponent_adapter,
            opponent_device=opponent_device,
            learner_seat=learner_seat,
        )
        learner_records = (
            records
            if learner_seat is None
            else [record for record in records if int(record[0]) == int(learner_seat)]
        )
        total_steps += steps
        payoffs.append(payouts[0 if learner_seat is None else learner_seat])
        for transition in build_player_transitions(learner_records, payouts):
            q_buffer.add(transition)
        for _player, features, legal_mask, action_idx, best_response_mode in learner_records:
            if best_response_mode:
                sl_buffer.add(features, legal_mask, action_idx, rng=rng)
        if len(q_buffer) >= cfg.min_buffer_size_to_learn:
            last_q_loss = _train_q(
                q_net,
                q_target_net,
                q_opt,
                q_buffer,
                cfg.batch_size,
                rng,
                device,
                cfg.q_discount,
            )
            q_updates += 1
            if q_updates % max(int(cfg.q_target_sync_interval), 1) == 0:
                q_target_net.load_state_dict(q_net.state_dict())
                q_target_syncs += 1
        if len(sl_buffer) >= cfg.min_buffer_size_to_learn:
            last_sl_loss = _train_avg_policy(avg_net, avg_opt, sl_buffer, cfg.batch_size, rng, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    eval_payoffs = []
    eval_steps = 0
    for game_idx in range(int(cfg.eval_games)):
        eval_learner_seat = int(game_idx % 2) if fixed_opponent_training else None
        eval_opponent_adapter = opponent_adapter
        if fixed_opponent_training and opponent_adapters:
            eval_opponent_adapter = opponent_adapters[int(game_idx % len(opponent_adapters))]
        _, payouts, steps = _play_hand(
            q_net,
            avg_net,
            cfg,
            rng,
            device,
            training=False,
            opponent_random=not fixed_opponent_training,
            opponent_adapter=eval_opponent_adapter,
            opponent_device=opponent_device,
            learner_seat=eval_learner_seat,
        )
        eval_payoffs.append(payouts[0 if eval_learner_seat is None else eval_learner_seat])
        eval_steps += steps
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    metrics = {
        "algorithm": _native_nfsp_algorithm_name(q_network_arch),
        "role": "native_game_theoretic_rl_pilot",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "DQN-style NFSP pilot for plumbing and compute diagnostics; not a promoted poker agent.",
        "uses_slumbot_training_data": False,
        **device_info,
        "num_actions": N_ACTIONS,
        "state_backend": state_backend,
        "train_episodes": int(cfg.train_episodes),
        "eval_games": int(cfg.eval_games),
        "q_network_arch": q_network_arch,
        "q_learning_target": "double_dqn",
        "train_steps": int(total_steps),
        "eval_steps": int(eval_steps),
        "train_opponent_mode": (
            "fixed_policy_population"
            if len(opponent_checkpoints) > 1
            else "fixed_policy"
            if fixed_opponent_training
            else "self_play"
        ),
        "opponent_kind": opponent_kind,
        "opponent_checkpoint": opponent_checkpoints[0] if len(opponent_checkpoints) == 1 else None,
        "opponent_checkpoints": opponent_checkpoints,
        "opponent_device": str(opponent_device),
        "opponent_algorithm": (
            getattr(opponent_adapter, "algorithm", None) if opponent_adapter is not None else None
        ),
        "fixed_opponent_population_size": len(opponent_checkpoints),
        "opponent_sample_counts": opponent_sample_counts,
        "fixed_opponent_learning_seats": sorted(fixed_opponent_learning_seats),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "episodes_per_second": float(cfg.train_episodes / max(train_seconds, 1e-9)),
        "train_steps_per_second": float(total_steps / max(train_seconds, 1e-9)),
        "mean_train_payoff_p0": float(np.mean(payoffs)) if payoffs else 0.0,
        "mean_eval_payoff_p0_vs_random": float(np.mean(eval_payoffs)) if eval_payoffs else 0.0,
        "mean_eval_payoff_learner_vs_opponent": (
            float(np.mean(eval_payoffs)) if fixed_opponent_training and eval_payoffs else None
        ),
        "q_buffer_size": len(q_buffer),
        "sl_buffer_size": len(sl_buffer),
        "q_updates": int(q_updates),
        "q_discount": float(cfg.q_discount),
        "q_target_sync_interval": int(cfg.q_target_sync_interval),
        "q_target_syncs": int(q_target_syncs),
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
                "q_network_arch": q_network_arch,
                "q_net_state_dict": q_net.state_dict(),
                "avg_net_state_dict": avg_net.state_dict(),
                "config": {
                    "train_episodes": int(cfg.train_episodes),
                    "eval_games": int(cfg.eval_games),
                    "hidden_dim": int(cfg.hidden_dim),
                    "q_network_arch": q_network_arch,
                    "batch_size": int(cfg.batch_size),
                    "min_buffer_size_to_learn": int(cfg.min_buffer_size_to_learn),
                    "anticipatory_param": float(cfg.anticipatory_param),
                    "epsilon": float(cfg.epsilon),
                    "lr": float(cfg.lr),
                    "q_discount": float(cfg.q_discount),
                    "q_target_sync_interval": int(cfg.q_target_sync_interval),
                    "initial_chips": int(cfg.initial_chips),
                    "max_steps_per_hand": int(cfg.max_steps_per_hand),
                    "seed": int(cfg.seed),
                    "device": str(cfg.device),
                    "opponent_kind": opponent_kind,
                    "opponent_checkpoint": opponent_checkpoints[0] if len(opponent_checkpoints) == 1 else None,
                    "opponent_checkpoints": opponent_checkpoints,
                    "opponent_device": str(cfg.opponent_device),
                    "state_backend": state_backend,
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
    q_network_arch = _normalize_q_network_arch(
        payload.get("q_network_arch", config_payload.get("q_network_arch", "mlp"))
    )
    q_net = _build_q_network(hidden_dim, q_network_arch).to(resolved_device)
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
        "q_network_arch": _normalize_q_network_arch(
            payload.get("q_network_arch", config_payload.get("q_network_arch", "mlp"))
        ),
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
    candidate_pair_payoffs: list[float] = []
    total_steps = 0

    eval_start = time.perf_counter()
    pair_i = 0
    while len(candidate_payoffs) < int(n_games):
        game_seed = int(seed) + pair_i
        action_seed = int(seed) + 1_000_000 + pair_i
        pair_payoffs: list[float] = []
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
            payoff = float(state.payout.get(candidate_seat, 0)) / float(initial_chips)
            candidate_payoffs.append(payoff)
            pair_payoffs.append(payoff)
        if pair_payoffs:
            candidate_pair_payoffs.append(float(np.mean(pair_payoffs)))
        pair_i += 1
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start
    payoff_mean = float(np.mean(candidate_pair_payoffs)) if candidate_pair_payoffs else 0.0
    payoff_std = float(np.std(candidate_pair_payoffs, ddof=1)) if len(candidate_pair_payoffs) > 1 else 0.0
    payoff_se = payoff_std / float(np.sqrt(len(candidate_pair_payoffs))) if candidate_pair_payoffs else 0.0

    return {
        "algorithm": "native_nfsp_dqn_h2h",
        "role": "native_game_theoretic_rl_checkpoint_head_to_head",
        "environment": str(candidate_payload.get("environment", "poker_ai:full_deck_hu_nlhe")),
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_algorithm": str(baseline_payload.get("algorithm", "")),
        "candidate_q_network_arch": _normalize_q_network_arch(
            candidate_payload.get(
                "q_network_arch",
                candidate_payload.get("config", {}).get("q_network_arch", "mlp"),
            )
        ),
        "baseline_q_network_arch": _normalize_q_network_arch(
            baseline_payload.get(
                "q_network_arch",
                baseline_payload.get("config", {}).get("q_network_arch", "mlp"),
            )
        ),
        **device_info,
        "num_actions": N_ACTIONS,
        "n_games": int(n_games),
        "n_pairs": int(len(candidate_pair_payoffs)),
        "eval_seconds": float(eval_seconds),
        "eval_steps": int(total_steps),
        "eval_games_per_second": float(n_games / max(eval_seconds, 1e-9)),
        "eval_steps_per_second": float(total_steps / max(eval_seconds, 1e-9)),
        "mean_candidate_payoff": payoff_mean,
        "std_candidate_payoff": payoff_std,
        "lower95_candidate_payoff": payoff_mean - 1.96 * payoff_se,
        "upper95_candidate_payoff": payoff_mean + 1.96 * payoff_se,
        "promotion": False,
    }
