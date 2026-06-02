"""Native full-deck policy-only PPO pilot.

This is a bounded control branch for the autoresearch workflow. It intentionally
uses the repo's full-deck 9-action environment without resolver calls, explicit
opponent ranges, or Slumbot-specific supervision so it can falsify whether a
plain frozen neural policy is enough before more belief/search work is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, card_to_index, new_game
from poker_ai.research.game_theoretic_rl import legal_softmax
from poker_ai.research.native_nfsp import (
    ReservoirPolicyBuffer,
    get_legal_mask,
    resolve_device,
    select_action,
)
from poker_ai.research.structured_observation import (
    RAW_SEQUENCE_N_FEATURES,
    encode_raw_sequence_observation,
)


@dataclass(frozen=True)
class NativePPOConfig:
    train_episodes: int = 100
    eval_games: int = 100
    hidden_dim: int = 64
    batch_size: int = 128
    rollout_episodes_per_update: int = 1
    ppo_epochs: int = 4
    lr: float = 3e-4
    clip_epsilon: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    entropy_anneal_to: float | None = None
    advantage_mode: str = "value"
    centralized_q_critic: bool = False
    fsp_average_policy: bool = False
    average_policy_batch_size: int = 512
    average_policy_memory_capacity: int = 20_000
    trace_lambda: float = 0.9
    discount: float = 1.0
    feature_mode: str = "flat"
    historical_opponent_interval: int = 0
    historical_opponent_capacity: int = 8
    historical_opponent_selection: str = "fifo"
    external_opponent_checkpoint: str | None = None
    external_opponent_kind: str = "auto"
    ppo_loss_mode: str = "standard"
    actor_update_mode: str = "ppo"
    behavior_mode: str = "policy"
    q_boost_beta: float = 1.0
    q_boost_min_prior: float = 1e-6
    initial_chips: int = 1000
    max_steps_per_hand: int = 256
    seed: int = 20260515
    device: str = "auto"
    checkpoint_path: str | None = None


class _ValueMLP(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


CENTRALIZED_Q_FEATURES = N_FEATURES + 52 + 8


class _PolicyMLP(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class _QMLP(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class _RainbowDistributionNet(nn.Module):
    def __init__(self, *, hidden_dim: int, num_atoms: int, device: torch.device) -> None:
        super().__init__()
        self.num_atoms = int(num_atoms)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), N_ACTIONS * int(num_atoms)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.net(obs).view(-1, N_ACTIONS, self.num_atoms)
        return torch.softmax(logits, dim=-1)


class _RainbowQPolicy(nn.Module):
    def __init__(self, model: nn.Module, *, num_atoms: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "support",
            torch.linspace(-1.0, 1.0, int(num_atoms), dtype=torch.float32),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        distribution = self.model(features)
        return torch.sum(distribution * self.support.view(1, 1, -1), dim=-1)


def _feature_dim_for_mode(feature_mode: str) -> int:
    mode = str(feature_mode)
    if mode == "flat":
        return N_FEATURES
    if mode == "raw_sequence":
        return RAW_SEQUENCE_N_FEATURES
    raise ValueError("feature_mode must be one of: flat, raw_sequence")


def _policy_feature_vector(state, feature_mode: str) -> np.ndarray:
    mode = str(feature_mode)
    if mode == "flat":
        return state.to_feature_vector().astype(np.float32, copy=False)
    if mode == "raw_sequence":
        return encode_raw_sequence_observation(state)
    raise ValueError("feature_mode must be one of: flat, raw_sequence")


def _centralized_q_feature_dim(policy_feature_dim: int) -> int:
    return int(policy_feature_dim) + 52 + 8


@dataclass(frozen=True)
class _DecisionRecord:
    player: int
    features: np.ndarray
    critic_features: np.ndarray
    legal_mask: np.ndarray
    action: int
    old_log_prob: float
    value: float
    expected_q: float | None = None


@dataclass(frozen=True)
class _ExternalOpponentPolicy:
    checkpoint_path: str
    kind: str
    algorithm: str
    action_fn: Callable[[object, np.ndarray, torch.device, np.random.Generator], int]


def _centralized_q_feature_vector(state, observation: np.ndarray) -> np.ndarray:
    """Training-only CTDE critic features; the deployed actor never sees these."""
    observation = np.asarray(observation, dtype=np.float32)
    features = np.zeros(_centralized_q_feature_dim(observation.shape[0]), dtype=np.float32)
    features[: observation.shape[0]] = observation
    offset = observation.shape[0]
    for player_idx, player in enumerate(state.players):
        if int(player_idx) == int(state.player_i):
            continue
        for card in player.cards:
            features[offset + card_to_index(card)] = 1.0
    offset += 52
    scale = max(float(getattr(state, "_initial_n_chips", 1)), 1.0)
    for player in state.players[:2]:
        features[offset] = float(player.n_chips) / scale
        features[offset + 1] = float(player.n_bet_chips) / scale
        features[offset + 2] = 1.0 if player.is_active else 0.0
        features[offset + 3] = 1.0 if player.is_all_in else 0.0
        offset += 4
    return features


def _critic_feature_vector(state, cfg: NativePPOConfig, observation: np.ndarray) -> np.ndarray:
    if bool(cfg.centralized_q_critic):
        return _centralized_q_feature_vector(state, observation)
    return np.asarray(observation, dtype=np.float32)


def _policy_step(
    policy_net: nn.Module,
    value_net: nn.Module,
    features: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        mask = torch.from_numpy(legal_mask.astype(np.float32)).to(device)
        logits = policy_net(x).reshape(-1).masked_fill(mask <= 0, -1e4)
        log_probs = nn.functional.log_softmax(logits, dim=0)
        probs = torch.exp(log_probs).detach().cpu().numpy().astype(np.float32)
        value = float(value_net(x).detach().cpu().reshape(-1)[0])
    return probs, float(log_probs.detach().cpu().numpy().max()), value


def q_boosted_behavior_policy(
    policy_probs: torch.Tensor,
    q_values: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    beta: float = 1.0,
    min_prior: float = 1e-6,
) -> torch.Tensor:
    """Mirror-descent behavior policy that boosts the actor prior with Q values."""
    legal = legal_mask.to(dtype=torch.bool)
    if legal.sum().item() <= 0:
        raise ValueError("legal_mask must contain at least one legal action")
    prior = torch.clamp(policy_probs.to(dtype=torch.float32), min=0.0)
    prior = torch.where(legal, prior, torch.zeros_like(prior))
    total = prior.sum()
    uniform = legal.to(dtype=torch.float32) / legal.to(dtype=torch.float32).sum().clamp_min(1.0)
    prior = torch.where(total > 1e-8, prior / total.clamp_min(1e-8), uniform)
    legal_prior = torch.clamp(prior[legal], min=float(min_prior))
    legal_prior = legal_prior / legal_prior.sum().clamp_min(1e-8)
    legal_q = q_values.to(dtype=torch.float32)[legal]
    baseline = torch.sum(legal_prior * legal_q)
    centered = legal_q - baseline
    scale = torch.std(legal_q, unbiased=False)
    if not torch.isfinite(scale) or float(scale.detach().cpu()) < 1e-6:
        scale = torch.clamp(torch.max(torch.abs(centered)), min=1.0)
    logits = torch.log(legal_prior) + float(beta) * centered / scale
    probs = torch.softmax(logits, dim=0)
    boosted = torch.zeros_like(prior)
    boosted[legal] = probs
    return boosted


def _behavior_step(
    policy_net: nn.Module,
    value_net: nn.Module,
    q_net: nn.Module | None,
    features: np.ndarray,
    critic_features: np.ndarray,
    legal_mask: np.ndarray,
    cfg: NativePPOConfig,
    device: torch.device,
) -> tuple[np.ndarray, float, float | None]:
    """Return behavior probabilities and current value estimate for one state."""
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        mask = torch.from_numpy(legal_mask.astype(np.float32)).to(device)
        logits = policy_net(x).reshape(-1).masked_fill(mask <= 0, -1e4)
        log_probs = nn.functional.log_softmax(logits, dim=0)
        policy_probs = torch.exp(log_probs)
        value = float(value_net(x).detach().cpu().reshape(-1)[0])
        q_values = None
        expected_q = None
        if q_net is not None:
            qx = torch.from_numpy(critic_features.astype(np.float32)).to(device)
            q_values = q_net(qx).reshape(-1).masked_fill(mask <= 0, 0.0)
            expected_q = float((policy_probs * q_values).sum().detach().cpu())
        if str(cfg.behavior_mode) == "policy":
            behavior = policy_probs
        elif str(cfg.behavior_mode) == "q_boosted":
            if q_values is None:
                raise ValueError("behavior_mode='q_boosted' requires q_net")
            behavior = q_boosted_behavior_policy(
                policy_probs,
                q_values,
                mask,
                beta=float(cfg.q_boost_beta),
                min_prior=float(cfg.q_boost_min_prior),
            )
        else:
            raise ValueError("behavior_mode must be one of: policy, q_boosted")
    return behavior.detach().cpu().numpy().astype(np.float32), value, expected_q


def _action_log_prob(
    policy_net: nn.Module,
    features: np.ndarray,
    legal_mask: np.ndarray,
    action: int,
    device: torch.device,
) -> float:
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        mask = torch.from_numpy(legal_mask.astype(np.float32)).to(device)
        logits = policy_net(x).reshape(-1).masked_fill(mask <= 0, -1e4)
        return float(nn.functional.log_softmax(logits, dim=0)[int(action)].detach().cpu())


def _play_hand(
    policy_net: nn.Module,
    value_net: nn.Module,
    q_net: nn.Module | None,
    cfg: NativePPOConfig,
    rng: np.random.Generator,
    device: torch.device,
    *,
    opponent_random: bool = False,
) -> tuple[list[_DecisionRecord], list[float], int]:
    state = new_game(2, initial_chips=cfg.initial_chips)
    records: list[_DecisionRecord] = []
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = int(state.player_i)
        features = _policy_feature_vector(state, cfg.feature_mode)
        critic_features = _critic_feature_vector(state, cfg, features)
        legal_mask = get_legal_mask(state)
        if opponent_random and player == 1:
            probs = np.asarray(legal_mask, dtype=np.float32) / float(np.sum(legal_mask))
            value = 0.0
            expected_q = None
        else:
            probs, value, expected_q = _behavior_step(
                policy_net,
                value_net,
                q_net,
                features,
                critic_features,
                legal_mask,
                cfg,
                device,
            )
        action_idx = select_action(probs, legal_mask, rng=rng)
        old_log_prob = float(np.log(max(float(probs[action_idx]), 1e-8)))
        records.append(
            _DecisionRecord(
                player=player,
                features=np.asarray(features, dtype=np.float32),
                critic_features=np.asarray(critic_features, dtype=np.float32),
                legal_mask=np.asarray(legal_mask, dtype=np.float32),
                action=int(action_idx),
                old_log_prob=float(old_log_prob),
                value=float(value),
                expected_q=expected_q,
            )
        )
        state = state.apply_action(INDEX_TO_ACTION[action_idx])
        n_steps += 1
    payouts = [float(state.payout.get(i, 0)) / float(cfg.initial_chips) for i in range(2)]
    return records, payouts, n_steps


def _play_fsp_hand(
    policy_net: nn.Module,
    value_net: nn.Module,
    q_net: nn.Module | None,
    avg_net: nn.Module,
    cfg: NativePPOConfig,
    rng: np.random.Generator,
    device: torch.device,
    *,
    learner_seat: int,
) -> tuple[list[_DecisionRecord], list[float], int]:
    state = new_game(2, initial_chips=cfg.initial_chips)
    records: list[_DecisionRecord] = []
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = int(state.player_i)
        features = _policy_feature_vector(state, cfg.feature_mode)
        critic_features = _critic_feature_vector(state, cfg, features)
        legal_mask = get_legal_mask(state)
        if player == int(learner_seat):
            probs, value, expected_q = _behavior_step(
                policy_net,
                value_net,
                q_net,
                features,
                critic_features,
                legal_mask,
                cfg,
                device,
            )
            action_idx = select_action(probs, legal_mask, rng=rng)
            old_log_prob = float(np.log(max(float(probs[action_idx]), 1e-8)))
            records.append(
                _DecisionRecord(
                    player=player,
                    features=np.asarray(features, dtype=np.float32),
                    critic_features=np.asarray(critic_features, dtype=np.float32),
                    legal_mask=np.asarray(legal_mask, dtype=np.float32),
                    action=int(action_idx),
                    old_log_prob=float(old_log_prob),
                    value=float(value),
                    expected_q=expected_q,
                )
            )
        else:
            probs = _network_probs(avg_net, features, legal_mask, device)
            action_idx = select_action(probs, legal_mask, rng=rng)
        state = state.apply_action(INDEX_TO_ACTION[action_idx])
        n_steps += 1
    payouts = [float(state.payout.get(i, 0)) / float(cfg.initial_chips) for i in range(2)]
    return records, payouts, n_steps


def _clone_policy_net(policy_net: nn.Module, hidden_dim: int, feature_dim: int, device: torch.device) -> nn.Module:
    snapshot = _PolicyMLP(hidden_dim, input_dim=feature_dim).to(device)
    snapshot.load_state_dict(policy_net.state_dict())
    snapshot.eval()
    for param in snapshot.parameters():
        param.requires_grad_(False)
    return snapshot


def _play_historical_opponent_hand(
    policy_net: nn.Module,
    value_net: nn.Module,
    q_net: nn.Module | None,
    opponent_net: nn.Module,
    cfg: NativePPOConfig,
    rng: np.random.Generator,
    device: torch.device,
    *,
    learner_seat: int,
) -> tuple[list[_DecisionRecord], list[float], int]:
    state = new_game(2, initial_chips=cfg.initial_chips)
    records: list[_DecisionRecord] = []
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = int(state.player_i)
        features = _policy_feature_vector(state, cfg.feature_mode)
        legal_mask = get_legal_mask(state)
        if player == int(learner_seat):
            critic_features = _critic_feature_vector(state, cfg, features)
            probs, value, expected_q = _behavior_step(
                policy_net,
                value_net,
                q_net,
                features,
                critic_features,
                legal_mask,
                cfg,
                device,
            )
            action_idx = select_action(probs, legal_mask, rng=rng)
            old_log_prob = float(np.log(max(float(probs[action_idx]), 1e-8)))
            records.append(
                _DecisionRecord(
                    player=player,
                    features=np.asarray(features, dtype=np.float32),
                    critic_features=np.asarray(critic_features, dtype=np.float32),
                    legal_mask=np.asarray(legal_mask, dtype=np.float32),
                    action=int(action_idx),
                    old_log_prob=float(old_log_prob),
                    value=float(value),
                    expected_q=expected_q,
                )
            )
        else:
            probs = _network_probs(opponent_net, features, legal_mask, device)
            action_idx = select_action(probs, legal_mask, rng=rng)
        state = state.apply_action(INDEX_TO_ACTION[action_idx])
        n_steps += 1
    payouts = [float(state.payout.get(i, 0)) / float(cfg.initial_chips) for i in range(2)]
    return records, payouts, n_steps


def _play_external_opponent_hand(
    policy_net: nn.Module,
    value_net: nn.Module,
    q_net: nn.Module | None,
    opponent: _ExternalOpponentPolicy,
    cfg: NativePPOConfig,
    rng: np.random.Generator,
    device: torch.device,
    *,
    learner_seat: int,
) -> tuple[list[_DecisionRecord], list[float], int]:
    state = new_game(2, initial_chips=cfg.initial_chips)
    records: list[_DecisionRecord] = []
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = int(state.player_i)
        legal_mask = get_legal_mask(state)
        if player == int(learner_seat):
            features = _policy_feature_vector(state, cfg.feature_mode)
            critic_features = _critic_feature_vector(state, cfg, features)
            probs, value, expected_q = _behavior_step(
                policy_net,
                value_net,
                q_net,
                features,
                critic_features,
                legal_mask,
                cfg,
                device,
            )
            action_idx = select_action(probs, legal_mask, rng=rng)
            old_log_prob = float(np.log(max(float(probs[action_idx]), 1e-8)))
            records.append(
                _DecisionRecord(
                    player=player,
                    features=np.asarray(features, dtype=np.float32),
                    critic_features=np.asarray(critic_features, dtype=np.float32),
                    legal_mask=np.asarray(legal_mask, dtype=np.float32),
                    action=int(action_idx),
                    old_log_prob=float(old_log_prob),
                    value=float(value),
                    expected_q=expected_q,
                )
            )
        else:
            action_idx = int(opponent.action_fn(state, legal_mask, device, rng))
        state = state.apply_action(INDEX_TO_ACTION[action_idx])
        n_steps += 1
    payouts = [float(state.payout.get(i, 0)) / float(cfg.initial_chips) for i in range(2)]
    return records, payouts, n_steps


def _train_average_policy_batch(
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


def _policy_surrogate_loss(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    loss_mode: str,
) -> torch.Tensor:
    unclipped = ratios * advantages
    clipped = torch.clamp(
        ratios,
        1.0 - float(clip_epsilon),
        1.0 + float(clip_epsilon),
    ) * advantages
    surrogate = torch.min(unclipped, clipped)
    mode = str(loss_mode)
    if mode == "trinal_clip":
        dual_clip = 3.0 * advantages
        surrogate = torch.where(advantages < 0.0, torch.max(surrogate, dual_clip), surrogate)
    elif mode != "standard":
        raise ValueError("ppo_loss_mode must be one of: standard, trinal_clip")
    return -surrogate.mean()


def _neurd_actor_loss(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    selected_legal = legal_masks.gather(1, actions.unsqueeze(1)).squeeze(1)
    if bool(torch.any(selected_legal <= 0).detach().cpu()):
        raise ValueError("neurd actor update received an illegal sampled action")
    selected_logits = logits.gather(1, actions.unsqueeze(1)).squeeze(1)
    return -(selected_logits * advantages.detach()).mean()


def _train_ppo_batch(
    policy_net: nn.Module,
    value_net: nn.Module,
    q_net: nn.Module | None,
    optimizer: optim.Optimizer,
    records: list[_DecisionRecord],
    returns: np.ndarray,
    cfg: NativePPOConfig,
    rng: np.random.Generator,
    device: torch.device,
    entropy_weight: float | None = None,
) -> tuple[int, float | None]:
    if not records:
        return 0, None

    features = torch.tensor(np.stack([r.features for r in records]), device=device)
    critic_features = torch.tensor(np.stack([r.critic_features for r in records]), device=device)
    legal_masks = torch.tensor(np.stack([r.legal_mask for r in records]), device=device)
    actions = torch.tensor([r.action for r in records], device=device, dtype=torch.long)
    old_log_probs = torch.tensor([r.old_log_prob for r in records], device=device)
    returns_t = torch.tensor(returns.astype(np.float32), device=device)
    mode = str(cfg.advantage_mode)
    q_modes = {"q_expected_mc", "q_expected_lambda", "q_expected_lambda_target"}
    if mode not in {"value", *q_modes}:
        raise ValueError(
            "advantage_mode must be 'value', 'q_expected_mc', "
            "'q_expected_lambda', or 'q_expected_lambda_target'"
        )
    if mode in q_modes and q_net is None:
        raise ValueError(f"{mode} requires q_net")
    if mode in q_modes:
        if any(record.expected_q is None for record in records):
            raise ValueError(f"{mode} requires rollout records with frozen expected_q")
        rollout_expected_q = torch.tensor(
            [float(record.expected_q) for record in records],
            device=device,
        )
    else:
        rollout_expected_q = None
    old_values = torch.tensor([r.value for r in records], device=device)
    value_advantages = returns_t - old_values
    if value_advantages.numel() > 1:
        value_advantages = (
            (value_advantages - value_advantages.mean())
            / value_advantages.std(unbiased=False).clamp_min(1e-6)
        )

    updates = 0
    last_loss: float | None = None
    n = len(records)
    batch_size = max(1, min(int(cfg.batch_size), n))
    for _ in range(int(cfg.ppo_epochs)):
        order = np.arange(n)
        rng.shuffle(order)
        for start in range(0, n, batch_size):
            idx = torch.tensor(order[start : start + batch_size], device=device, dtype=torch.long)
            logits = policy_net(features[idx]).masked_fill(legal_masks[idx] <= 0, -1e4)
            log_probs = nn.functional.log_softmax(logits, dim=1)
            action_log_probs = log_probs.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
            ratios = torch.exp(action_log_probs - old_log_probs[idx])
            if mode in q_modes:
                q_values = q_net(critic_features[idx])
                q_values = q_values.masked_fill(legal_masks[idx] <= 0, 0.0)
                q_taken = q_values.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                if rollout_expected_q is None:
                    raise ValueError(f"{mode} requires rollout records with frozen expected_q")
                advantages = returns_t[idx] - rollout_expected_q[idx]
                if advantages.numel() > 1:
                    advantages = (
                        (advantages - advantages.mean())
                        / advantages.std(unbiased=False).clamp_min(1e-6)
                    )
                if str(cfg.ppo_loss_mode) == "trinal_clip":
                    critic_loss = nn.functional.smooth_l1_loss(q_taken, returns_t[idx])
                else:
                    critic_loss = nn.functional.mse_loss(q_taken, returns_t[idx])
            else:
                advantages = value_advantages[idx]
                values = value_net(features[idx])
                if str(cfg.ppo_loss_mode) == "trinal_clip":
                    critic_loss = nn.functional.smooth_l1_loss(values, returns_t[idx])
                else:
                    critic_loss = nn.functional.mse_loss(values, returns_t[idx])
            actor_update_mode = str(cfg.actor_update_mode)
            if actor_update_mode == "ppo":
                policy_loss = _policy_surrogate_loss(
                    ratios,
                    advantages,
                    clip_epsilon=float(cfg.clip_epsilon),
                    loss_mode=str(cfg.ppo_loss_mode),
                )
            elif actor_update_mode == "neurd":
                policy_loss = _neurd_actor_loss(
                    logits,
                    legal_masks[idx],
                    actions[idx],
                    advantages,
                )
            else:
                raise ValueError("actor_update_mode must be one of: ppo, neurd")
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            effective_entropy_weight = (
                float(cfg.entropy_weight) if entropy_weight is None else float(entropy_weight)
            )
            loss = (
                policy_loss
                + float(cfg.value_loss_weight) * critic_loss
                - effective_entropy_weight * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = list(policy_net.parameters()) + list(value_net.parameters())
            if q_net is not None:
                params += list(q_net.parameters())
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            updates += 1
            last_loss = float(loss.detach().cpu())
    return updates, last_loss


def _returns_for_records(records: list[_DecisionRecord], payouts: list[float]) -> np.ndarray:
    return np.asarray([float(payouts[record.player]) for record in records], dtype=np.float32)


def _expected_q_value(
    policy_net: nn.Module,
    q_net: nn.Module,
    record: _DecisionRecord,
    device: torch.device,
) -> float:
    with torch.no_grad():
        x = torch.from_numpy(record.features.astype(np.float32)).to(device)
        qx = torch.from_numpy(record.critic_features.astype(np.float32)).to(device)
        mask = torch.from_numpy(record.legal_mask.astype(np.float32)).to(device)
        logits = policy_net(x).reshape(-1).masked_fill(mask <= 0, -1e4)
        probs = torch.softmax(logits, dim=0)
        q_values = q_net(qx).reshape(-1).masked_fill(mask <= 0, 0.0)
        return float((probs * q_values).sum().detach().cpu())


def _expected_sarsa_lambda_returns(
    records: list[_DecisionRecord],
    payouts: list[float],
    *,
    policy_net: nn.Module,
    q_net: nn.Module,
    cfg: NativePPOConfig,
    device: torch.device,
) -> np.ndarray:
    """Build same-player Expected-SARSA(lambda) targets from one hand trajectory."""
    targets = np.zeros(len(records), dtype=np.float32)
    next_record_by_player: dict[int, _DecisionRecord] = {}
    next_target_by_player: dict[int, float] = {}
    lam = float(np.clip(cfg.trace_lambda, 0.0, 1.0))
    gamma = float(cfg.discount)
    for idx in range(len(records) - 1, -1, -1):
        record = records[idx]
        player = int(record.player)
        if player in next_record_by_player:
            expected_next = _expected_q_value(
                policy_net,
                q_net,
                next_record_by_player[player],
                device,
            )
            target = gamma * (
                (1.0 - lam) * expected_next
                + lam * float(next_target_by_player[player])
            )
        else:
            target = float(payouts[player])
        targets[idx] = float(target)
        next_record_by_player[player] = record
        next_target_by_player[player] = float(target)
    return targets


def _historical_average_score(item: dict) -> float:
    return float(item.get("score", 0.0)) / float(max(int(item.get("games", 0)), 1))


def _trim_historical_policy_pool(pool: list[dict], capacity: int, selection: str) -> None:
    capacity = max(1, int(capacity))
    if len(pool) <= capacity:
        return
    if selection == "fifo":
        del pool[: len(pool) - capacity]
        return
    if selection == "k_best":
        pool.sort(key=_historical_average_score, reverse=True)
        del pool[capacity:]
        return
    raise ValueError("historical_opponent_selection must be one of: fifo, k_best")


def _is_external_rainbow_payload(payload: dict[str, Any]) -> bool:
    return str(payload.get("algorithm", "")) in {
        "tianshou_rainbow_dqn",
        "tianshou_marl_rainbow_dqn",
        "compiled_tianshou_rainbow_response_oracle",
    }


def _rainbow_state_dicts_by_seat_from_payload(payload: dict[str, Any]) -> dict[int, dict]:
    if payload.get("shared_model_state_dict") is not None:
        shared = payload["shared_model_state_dict"]
        return {0: shared, 1: shared}
    if "agent_model_state_dicts" in payload:
        agent_state_dicts = payload["agent_model_state_dicts"]
        fallback_key = "player_0" if "player_0" in agent_state_dicts else sorted(agent_state_dicts)[0]
        return {
            seat: agent_state_dicts.get(f"player_{seat}", agent_state_dicts[fallback_key])
            for seat in (0, 1)
        }
    if "model_state_dict" in payload:
        shared = payload["model_state_dict"]
        return {0: shared, 1: shared}
    raise ValueError("Rainbow checkpoint is missing a loadable model state dict")


def _load_external_rainbow_opponent(
    checkpoint_path: str | Path,
    payload: dict[str, Any],
    device: torch.device,
) -> _ExternalOpponentPolicy:
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("external Rainbow opponent action count does not match native contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("external Rainbow opponent feature count does not match native contract")
    hidden_dim = int(payload.get("hidden_dim", 128))
    num_atoms = int(payload.get("num_atoms", 51))
    model = _RainbowDistributionNet(
        hidden_dim=hidden_dim,
        num_atoms=num_atoms,
        device=device,
    ).to(device)
    state_dicts = _rainbow_state_dicts_by_seat_from_payload(payload)
    model.load_state_dict(state_dicts.get(0, state_dicts[sorted(state_dicts)[0]]))
    policy = _RainbowQPolicy(model, num_atoms=num_atoms).to(device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    def _act(
        state: object,
        legal_mask: np.ndarray,
        resolved_device: torch.device,
        _rng: np.random.Generator,
    ) -> int:
        features = state.to_feature_vector().astype(np.float32, copy=False)
        legal = np.asarray(legal_mask, dtype=np.float32)
        with torch.no_grad():
            q_values = policy(
                torch.as_tensor(features, dtype=torch.float32, device=resolved_device).unsqueeze(0)
            ).detach().reshape(-1)
            mask = torch.as_tensor(legal > 0, dtype=torch.bool, device=q_values.device)
            q_values = q_values.masked_fill(~mask, -1.0e30)
            return int(torch.argmax(q_values).detach().cpu())

    return _ExternalOpponentPolicy(
        checkpoint_path=str(checkpoint_path),
        kind="tianshou-rainbow",
        algorithm=str(payload.get("algorithm", "tianshou_rainbow_dqn")),
        action_fn=_act,
    )


def _load_external_opponent_policy(
    checkpoint_path: str | Path,
    *,
    kind: str,
    device: torch.device,
) -> _ExternalOpponentPolicy:
    requested_kind = str(kind)
    if requested_kind not in {"auto", "native-ppo", "tianshou-rainbow"}:
        raise ValueError("external_opponent_kind must be one of: auto, native-ppo, tianshou-rainbow")
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    resolved_kind = requested_kind
    if resolved_kind == "auto":
        resolved_kind = "tianshou-rainbow" if _is_external_rainbow_payload(payload) else "native-ppo"
    if resolved_kind == "tianshou-rainbow":
        if not _is_external_rainbow_payload(payload):
            raise ValueError("external_opponent_kind='tianshou-rainbow' requires a Rainbow payload")
        return _load_external_rainbow_opponent(checkpoint_path, payload, device)

    native_payload, policy_net, feature_mode = _load_policy_network(
        str(checkpoint_path),
        device,
        strategy_source="auto",
    )

    def _act(
        state: object,
        legal_mask: np.ndarray,
        resolved_device: torch.device,
        rng: np.random.Generator,
    ) -> int:
        features = _policy_feature_vector(state, feature_mode)
        probs = _network_probs(policy_net, features, legal_mask, resolved_device)
        return select_action(probs, legal_mask, rng=rng)

    return _ExternalOpponentPolicy(
        checkpoint_path=str(checkpoint_path),
        kind="native-ppo",
        algorithm=str(native_payload.get("algorithm", "native_ppo_policy")),
        action_fn=_act,
    )


def run_native_ppo_policy_pilot(cfg: NativePPOConfig | None = None) -> dict:
    cfg = cfg or NativePPOConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    feature_dim = _feature_dim_for_mode(cfg.feature_mode)
    historical_enabled = int(cfg.historical_opponent_interval) > 0
    external_enabled = cfg.external_opponent_checkpoint is not None
    historical_selection = str(cfg.historical_opponent_selection)
    if historical_selection not in {"fifo", "k_best"}:
        raise ValueError("historical_opponent_selection must be one of: fifo, k_best")
    loss_mode = str(cfg.ppo_loss_mode)
    if loss_mode not in {"standard", "trinal_clip"}:
        raise ValueError("ppo_loss_mode must be one of: standard, trinal_clip")
    actor_update_mode = str(cfg.actor_update_mode)
    if actor_update_mode not in {"ppo", "neurd"}:
        raise ValueError("actor_update_mode must be one of: ppo, neurd")
    behavior_mode = str(cfg.behavior_mode)
    if behavior_mode not in {"policy", "q_boosted"}:
        raise ValueError("behavior_mode must be one of: policy, q_boosted")
    if historical_enabled and cfg.fsp_average_policy:
        raise ValueError("historical_opponent_interval cannot be combined with fsp_average_policy")
    if external_enabled and cfg.fsp_average_policy:
        raise ValueError("external_opponent_checkpoint cannot be combined with fsp_average_policy")
    if external_enabled and historical_enabled:
        raise ValueError("external_opponent_checkpoint cannot be combined with historical_opponent_interval")

    policy_net = _PolicyMLP(cfg.hidden_dim, input_dim=feature_dim).to(device)
    value_net = _ValueMLP(cfg.hidden_dim, input_dim=feature_dim).to(device)
    avg_net = (
        _PolicyMLP(cfg.hidden_dim, input_dim=feature_dim).to(device)
        if cfg.fsp_average_policy
        else None
    )
    avg_buffer = (
        ReservoirPolicyBuffer(capacity=cfg.average_policy_memory_capacity)
        if cfg.fsp_average_policy
        else None
    )
    q_critic_feature_dim = (
        _centralized_q_feature_dim(feature_dim)
        if cfg.centralized_q_critic
        else feature_dim
    )
    q_net = (
        _QMLP(cfg.hidden_dim, input_dim=q_critic_feature_dim).to(device)
        if cfg.advantage_mode
        in {"q_expected_mc", "q_expected_lambda", "q_expected_lambda_target"}
        else None
    )
    target_q_net = (
        _QMLP(cfg.hidden_dim, input_dim=q_critic_feature_dim).to(device)
        if cfg.advantage_mode == "q_expected_lambda_target"
        else None
    )
    if target_q_net is not None:
        if q_net is None:
            raise ValueError("q_expected_lambda_target requires q_net")
        target_q_net.load_state_dict(q_net.state_dict())
        target_q_net.eval()
    if behavior_mode == "q_boosted" and q_net is None:
        raise ValueError("behavior_mode='q_boosted' requires a q_expected_* advantage mode")
    external_opponent = (
        _load_external_opponent_policy(
            cfg.external_opponent_checkpoint,
            kind=cfg.external_opponent_kind,
            device=device,
        )
        if external_enabled
        else None
    )
    optimizer_params = list(policy_net.parameters()) + list(value_net.parameters())
    if q_net is not None:
        optimizer_params += list(q_net.parameters())
    optimizer = optim.Adam(optimizer_params, lr=float(cfg.lr))
    avg_optimizer = optim.Adam(avg_net.parameters(), lr=float(cfg.lr)) if avg_net is not None else None

    train_start = time.perf_counter()
    total_steps = 0
    total_records = 0
    policy_updates = 0
    rollout_updates = 0
    last_loss = None
    last_avg_loss = None
    avg_policy_updates = 0
    train_payoffs: list[float] = []
    rollout_records: list[_DecisionRecord] = []
    rollout_returns: list[np.ndarray] = []
    rollout_episode_count = 0
    historical_policy_pool: list[dict] = []
    historical_snapshots_created = 0
    for episode_idx in range(int(cfg.train_episodes)):
        if historical_enabled and (
            not historical_policy_pool
            or episode_idx % max(1, int(cfg.historical_opponent_interval)) == 0
        ):
            historical_policy_pool.append(
                {
                    "net": _clone_policy_net(policy_net, cfg.hidden_dim, feature_dim, device),
                    "score": 0.0,
                    "games": 0,
                }
            )
            historical_snapshots_created += 1
            _trim_historical_policy_pool(
                historical_policy_pool,
                int(cfg.historical_opponent_capacity),
                historical_selection,
            )

        if external_opponent is not None:
            learner_seat = episode_idx % 2
            records, payouts, steps = _play_external_opponent_hand(
                policy_net,
                value_net,
                q_net,
                external_opponent,
                cfg,
                rng,
                device,
                learner_seat=learner_seat,
            )
        elif historical_enabled:
            opponent_idx = int(rng.integers(len(historical_policy_pool)))
            opponent_item = historical_policy_pool[opponent_idx]
            records, payouts, steps = _play_historical_opponent_hand(
                policy_net,
                value_net,
                q_net,
                opponent_item["net"],
                cfg,
                rng,
                device,
                learner_seat=episode_idx % 2,
            )
            learner_seat = episode_idx % 2
            opponent_item["score"] = float(opponent_item.get("score", 0.0)) + float(
                payouts[1 - learner_seat]
            )
            opponent_item["games"] = int(opponent_item.get("games", 0)) + 1
        elif cfg.fsp_average_policy:
            if avg_net is None or avg_buffer is None:
                raise ValueError("fsp_average_policy requires avg_net and avg_buffer")
            records, payouts, steps = _play_fsp_hand(
                policy_net,
                value_net,
                q_net,
                avg_net,
                cfg,
                rng,
                device,
                learner_seat=episode_idx % 2,
            )
        else:
            records, payouts, steps = _play_hand(policy_net, value_net, q_net, cfg, rng, device)
        total_steps += steps
        total_records += len(records)
        train_payoffs.append(float(payouts[0]))
        rollout_records.extend(records)
        if avg_buffer is not None:
            for record in records:
                avg_buffer.add(
                    record.features,
                    record.legal_mask,
                    record.action,
                    rng=rng,
                )
        if records:
            if cfg.advantage_mode in {"q_expected_lambda", "q_expected_lambda_target"}:
                if q_net is None:
                    raise ValueError(f"{cfg.advantage_mode} requires q_net")
                target_for_returns = target_q_net if target_q_net is not None else q_net
                rollout_returns.append(
                    _expected_sarsa_lambda_returns(
                        records,
                        payouts,
                        policy_net=policy_net,
                        q_net=target_for_returns,
                        cfg=cfg,
                        device=device,
                    )
                )
            else:
                rollout_returns.append(_returns_for_records(records, payouts))
        rollout_episode_count += 1
        flush_rollout = (
            rollout_episode_count >= max(1, int(cfg.rollout_episodes_per_update))
            or episode_idx == int(cfg.train_episodes) - 1
        )
        if flush_rollout and rollout_records:
            returns = np.concatenate(rollout_returns).astype(np.float32, copy=False)
            if cfg.entropy_anneal_to is None:
                effective_ew = float(cfg.entropy_weight)
            else:
                frac = episode_idx / max(1, int(cfg.train_episodes) - 1)
                effective_ew = float(cfg.entropy_weight) + (
                    float(cfg.entropy_anneal_to) - float(cfg.entropy_weight)
                ) * frac
            updates, last_loss = _train_ppo_batch(
                policy_net,
                value_net,
                q_net,
                optimizer,
                rollout_records,
                returns,
                cfg,
                rng,
                device,
                entropy_weight=effective_ew,
            )
            policy_updates += updates
            rollout_updates += 1
            if target_q_net is not None and q_net is not None:
                target_q_net.load_state_dict(q_net.state_dict())
                target_q_net.eval()
            if avg_net is not None and avg_optimizer is not None and avg_buffer is not None:
                last_avg_loss = _train_average_policy_batch(
                    avg_net,
                    avg_optimizer,
                    avg_buffer,
                    cfg.average_policy_batch_size,
                    rng,
                    device,
                )
                avg_policy_updates += 1 if last_avg_loss is not None else 0
            rollout_records = []
            rollout_returns = []
            rollout_episode_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    eval_payoffs = []
    eval_steps = 0
    for _ in range(int(cfg.eval_games)):
        _, payouts, steps = _play_hand(
            policy_net,
            value_net,
            q_net,
            cfg,
            rng,
            device,
            opponent_random=True,
        )
        eval_payoffs.append(float(payouts[0]))
        eval_steps += steps
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    metrics = {
        "algorithm": "native_ppo_policy",
        "role": "policy_only_self_play_falsifier",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Policy-only PPO control; no resolver, no explicit belief, not a promoted poker agent.",
        **device_info,
        "num_actions": N_ACTIONS,
        "num_features": int(feature_dim),
        "uses_resolver": False,
        "uses_explicit_belief": False,
        "train_episodes": int(cfg.train_episodes),
        "eval_games": int(cfg.eval_games),
        "train_steps": int(total_steps),
        "train_records": int(total_records),
        "rollout_episodes_per_update": int(cfg.rollout_episodes_per_update),
        "rollout_updates": int(rollout_updates),
        "eval_steps": int(eval_steps),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "episodes_per_second": float(cfg.train_episodes / max(train_seconds, 1e-9)),
        "train_steps_per_second": float(total_steps / max(train_seconds, 1e-9)),
        "mean_train_payoff_p0": float(np.mean(train_payoffs)) if train_payoffs else 0.0,
        "mean_eval_payoff_p0_vs_random": float(np.mean(eval_payoffs)) if eval_payoffs else 0.0,
        "advantage_mode": str(cfg.advantage_mode),
        "ppo_loss_mode": loss_mode,
        "actor_update_mode": actor_update_mode,
        "behavior_mode": behavior_mode,
        "q_boost_beta": float(cfg.q_boost_beta),
        "q_boost_min_prior": float(cfg.q_boost_min_prior),
        "feature_mode": str(cfg.feature_mode),
        "uses_external_opponent": bool(external_opponent is not None),
        "external_opponent_checkpoint": (
            external_opponent.checkpoint_path if external_opponent is not None else None
        ),
        "external_opponent_kind": (
            external_opponent.kind if external_opponent is not None else None
        ),
        "external_opponent_algorithm": (
            external_opponent.algorithm if external_opponent is not None else None
        ),
        "uses_historical_opponents": bool(historical_enabled),
        "historical_opponent_interval": int(cfg.historical_opponent_interval),
        "historical_opponent_capacity": int(cfg.historical_opponent_capacity),
        "historical_opponent_selection": historical_selection,
        "historical_policy_pool_size": int(len(historical_policy_pool)),
        "historical_snapshots_created": int(historical_snapshots_created),
        "uses_centralized_q_critic": bool(cfg.centralized_q_critic),
        "q_critic_feature_dim": int(q_critic_feature_dim),
        "fsp_average_policy": bool(cfg.fsp_average_policy),
        "average_policy_buffer_size": int(len(avg_buffer)) if avg_buffer is not None else 0,
        "average_policy_updates": int(avg_policy_updates),
        "last_average_policy_loss": last_avg_loss,
        "uses_target_q": bool(target_q_net is not None),
        "trace_lambda": float(cfg.trace_lambda),
        "discount": float(cfg.discount),
        "policy_updates": int(policy_updates),
        "last_loss": last_loss,
        "promotion": False,
        "passed": True,
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
                "num_features": int(feature_dim),
                "hidden_dim": int(cfg.hidden_dim),
                "policy_net_state_dict": policy_net.state_dict(),
                "value_net_state_dict": value_net.state_dict(),
                "config": {
                    "train_episodes": int(cfg.train_episodes),
                    "eval_games": int(cfg.eval_games),
                    "hidden_dim": int(cfg.hidden_dim),
                    "batch_size": int(cfg.batch_size),
                    "rollout_episodes_per_update": int(cfg.rollout_episodes_per_update),
                    "ppo_epochs": int(cfg.ppo_epochs),
                    "lr": float(cfg.lr),
                    "clip_epsilon": float(cfg.clip_epsilon),
                    "value_loss_weight": float(cfg.value_loss_weight),
                    "entropy_weight": float(cfg.entropy_weight),
                    "advantage_mode": str(cfg.advantage_mode),
                    "ppo_loss_mode": loss_mode,
                    "actor_update_mode": actor_update_mode,
                    "behavior_mode": behavior_mode,
                    "q_boost_beta": float(cfg.q_boost_beta),
                    "q_boost_min_prior": float(cfg.q_boost_min_prior),
                    "feature_mode": str(cfg.feature_mode),
                    "uses_external_opponent": bool(external_opponent is not None),
                    "external_opponent_checkpoint": (
                        external_opponent.checkpoint_path if external_opponent is not None else None
                    ),
                    "external_opponent_kind": (
                        external_opponent.kind if external_opponent is not None else None
                    ),
                    "external_opponent_algorithm": (
                        external_opponent.algorithm if external_opponent is not None else None
                    ),
                    "historical_opponent_interval": int(cfg.historical_opponent_interval),
                    "historical_opponent_capacity": int(cfg.historical_opponent_capacity),
                    "historical_opponent_selection": historical_selection,
                    "centralized_q_critic": bool(cfg.centralized_q_critic),
                    "q_critic_feature_dim": int(q_critic_feature_dim),
                    "fsp_average_policy": bool(cfg.fsp_average_policy),
                    "average_policy_batch_size": int(cfg.average_policy_batch_size),
                    "average_policy_memory_capacity": int(cfg.average_policy_memory_capacity),
                    "uses_target_q": bool(target_q_net is not None),
                    "trace_lambda": float(cfg.trace_lambda),
                    "discount": float(cfg.discount),
                    "initial_chips": int(cfg.initial_chips),
                    "max_steps_per_hand": int(cfg.max_steps_per_hand),
                    "seed": int(cfg.seed),
                    "device": str(cfg.device),
                },
                "metrics": metrics,
                **({"avg_net_state_dict": avg_net.state_dict()} if avg_net is not None else {}),
                **({"q_net_state_dict": q_net.state_dict()} if q_net is not None else {}),
                **(
                    {"target_q_net_state_dict": target_q_net.state_dict()}
                    if target_q_net is not None
                    else {}
                ),
            },
            path,
        )
    return metrics


def _load_policy_network(
    checkpoint_path: str,
    resolved_device: torch.device,
    *,
    strategy_source: str = "auto",
) -> tuple[dict, nn.Module, str]:
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("checkpoint action count does not match native full-deck action contract")
    config_payload = payload.get("config", {})
    feature_mode = str(config_payload.get("feature_mode", "flat"))
    feature_dim = _feature_dim_for_mode(feature_mode)
    if int(payload.get("num_features", -1)) != feature_dim:
        raise ValueError("checkpoint feature count does not match native full-deck feature contract")
    hidden_dim = int(payload.get("hidden_dim", config_payload.get("hidden_dim", 64)))
    policy_net = _PolicyMLP(hidden_dim, input_dim=feature_dim).to(resolved_device)
    source = strategy_source
    if source == "auto":
        source = (
            "average"
            if bool(config_payload.get("fsp_average_policy", False))
            and "avg_net_state_dict" in payload
            else "actor"
        )
    if source not in {"actor", "average"}:
        raise ValueError("strategy_source must be one of: auto, actor, average")
    if source == "average" and "avg_net_state_dict" in payload:
        policy_net.load_state_dict(payload["avg_net_state_dict"])
    elif source == "actor" and "policy_net_state_dict" in payload:
        policy_net.load_state_dict(payload["policy_net_state_dict"])
    elif "avg_net_state_dict" in payload:
        policy_net.load_state_dict(payload["avg_net_state_dict"])
    else:
        raise ValueError("checkpoint does not contain a loadable policy network")
    policy_net.eval()
    return payload, policy_net, feature_mode


def _network_probs(net: nn.Module, features: np.ndarray, legal_mask: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = net(x).detach().cpu().numpy().reshape(-1)
    return legal_softmax(logits, legal_mask)


def evaluate_native_ppo_policy_head_to_head(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260515,
    min_lower95_candidate_payoff: float | None = None,
    strategy_source: str = "auto",
) -> dict:
    """Evaluate two native policy checkpoints in paired duplicate-swapped hands."""
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_payload, candidate_policy, candidate_feature_mode = _load_policy_network(
        candidate_checkpoint,
        resolved_device,
        strategy_source=strategy_source,
    )
    baseline_payload, baseline_policy, baseline_feature_mode = _load_policy_network(
        baseline_checkpoint,
        resolved_device,
        strategy_source="auto",
    )
    config_payload = candidate_payload.get("config", {})
    initial_chips = int(config_payload.get("initial_chips", 1000))
    max_steps_per_hand = int(config_payload.get("max_steps_per_hand", 256))
    candidate_pair_payoffs: list[float] = []
    candidate_payoffs: list[float] = []
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
                candidate_seat: (candidate_policy, candidate_feature_mode),
                baseline_seat: (baseline_policy, baseline_feature_mode),
            }
            state = new_game(2, initial_chips=initial_chips)
            n_steps = 0
            while not state.is_terminal and n_steps < max_steps_per_hand:
                policy_net, feature_mode = policies[state.player_i]
                features = _policy_feature_vector(state, feature_mode)
                legal_mask = get_legal_mask(state)
                probs = _network_probs(policy_net, features, legal_mask, resolved_device)
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
    lower95 = payoff_mean - 1.96 * payoff_se
    passed = True
    if min_lower95_candidate_payoff is not None:
        passed = lower95 >= float(min_lower95_candidate_payoff)

    return {
        "algorithm": "native_policy_h2h",
        "role": "policy_only_checkpoint_head_to_head",
        "environment": str(candidate_payload.get("environment", "poker_ai:full_deck_hu_nlhe")),
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_algorithm": str(candidate_payload.get("algorithm", "")),
        "baseline_algorithm": str(baseline_payload.get("algorithm", "")),
        "candidate_strategy_source": str(strategy_source),
        **device_info,
        "num_actions": N_ACTIONS,
        "uses_resolver": False,
        "uses_explicit_belief": False,
        "n_games": int(n_games),
        "n_pairs": int(len(candidate_pair_payoffs)),
        "eval_seconds": float(eval_seconds),
        "eval_steps": int(total_steps),
        "eval_games_per_second": float(n_games / max(eval_seconds, 1e-9)),
        "eval_steps_per_second": float(total_steps / max(eval_seconds, 1e-9)),
        "mean_candidate_payoff": payoff_mean,
        "std_candidate_payoff": payoff_std,
        "lower95_candidate_payoff": lower95,
        "upper95_candidate_payoff": payoff_mean + 1.96 * payoff_se,
        "min_lower95_candidate_payoff": min_lower95_candidate_payoff,
        "passed": passed,
        "promotion": False,
    }
