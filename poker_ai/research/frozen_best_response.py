"""Approximate best-response pilot against a frozen checkpoint policy."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, new_game
from poker_ai.research.evaluation import (
    _strategies_from_network,
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.game_theoretic_rl import epsilon_greedy_distribution
from poker_ai.research.native_nfsp import (
    BestResponseTransition,
    _MLP,
    _TransitionBuffer,
    _q_values,
    _train_q,
    get_legal_mask,
    resolve_device,
    select_action,
)


@dataclass(frozen=True)
class FrozenBestResponseConfig:
    checkpoint: str
    train_episodes: int = 100
    eval_games: int = 100
    br_player: int = 1
    hidden_dim: int = 128
    batch_size: int = 256
    min_buffer_size_to_learn: int = 64
    epsilon: float = 0.08
    lr: float = 1e-3
    q_discount: float = 0.99
    q_target_sync_interval: int = 200
    initial_chips: int = 1000
    max_steps_per_hand: int = 256
    seed: int = 20260515
    device: str = "auto"
    strategy_source: str = "regret"


def build_monte_carlo_transitions(
    records: list[tuple[int, np.ndarray, np.ndarray, int, bool]],
    payoff: float,
) -> list[BestResponseTransition]:
    """Assign the final hand payoff directly to each BR decision."""
    transitions: list[BestResponseTransition] = []
    for _player, features, legal_mask, action_idx, _mode in records:
        transitions.append(
            BestResponseTransition(
                features=np.asarray(features, dtype=np.float32),
                legal_mask=np.asarray(legal_mask, dtype=np.float32),
                action=int(action_idx),
                reward=float(payoff),
                next_features=np.zeros_like(features, dtype=np.float32),
                next_legal_mask=np.zeros_like(legal_mask, dtype=np.float32),
                done=True,
            )
        )
    return transitions


def _checkpoint_policy_probs(
    loaded: Any,
    state,
    device: torch.device,
    *,
    strategy_source: str,
) -> np.ndarray:
    features = state.to_feature_vector().reshape(1, N_FEATURES)
    legal_mask = get_legal_mask(state)
    strategy = _strategies_from_network(
        loaded.value_net,
        features,
        [legal_mask],
        device,
        strategy_source=strategy_source,
        checkpoint_metadata=loaded.metadata,
    )[0]
    return np.asarray(strategy, dtype=np.float32)


def _greedy_q_distribution(
    q_net: nn.Module,
    features: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    q_values = _q_values(q_net, features, device)
    masked = np.where(legal_mask > 0, q_values, -1e9)
    probs = np.zeros_like(legal_mask, dtype=np.float32)
    probs[int(np.argmax(masked))] = 1.0
    return probs


def _play_hand_against_frozen(
    q_net: nn.Module,
    loaded: Any,
    cfg: FrozenBestResponseConfig,
    rng: np.random.Generator,
    device: torch.device,
    *,
    training: bool,
) -> tuple[list[BestResponseTransition], float, int]:
    state = new_game(2, initial_chips=cfg.initial_chips)
    records: list[tuple[int, np.ndarray, np.ndarray, int, bool]] = []
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = int(state.player_i)
        features = state.to_feature_vector()
        legal_mask = get_legal_mask(state)
        if player == int(cfg.br_player):
            if training:
                q_values = _q_values(q_net, features, device)
                probs = epsilon_greedy_distribution(q_values, legal_mask, epsilon=cfg.epsilon)
            else:
                probs = _greedy_q_distribution(q_net, features, legal_mask, device)
            action_idx = select_action(probs, legal_mask, rng=rng)
            records.append((player, features, legal_mask, action_idx, True))
        else:
            probs = _checkpoint_policy_probs(
                loaded,
                state,
                device,
                strategy_source=cfg.strategy_source,
            )
            action_idx = select_action(probs, legal_mask, rng=rng)
        state = state.apply_action(INDEX_TO_ACTION[action_idx])
        n_steps += 1
    payouts = [float(state.payout.get(i, 0)) / float(cfg.initial_chips) for i in range(2)]
    payoff = float(payouts[int(cfg.br_player)])
    transitions = build_monte_carlo_transitions(records, payoff)
    return transitions, payoff, n_steps


def run_frozen_best_response(cfg: FrozenBestResponseConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    loaded = load_value_network_checkpoint(cfg.checkpoint, device)
    assert_strategy_source_supported(loaded, cfg.strategy_source)

    q_net = _MLP(cfg.hidden_dim).to(device)
    q_target_net = _MLP(cfg.hidden_dim).to(device)
    q_target_net.load_state_dict(q_net.state_dict())
    q_target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=float(cfg.lr))
    buffer = _TransitionBuffer(capacity=50_000)

    train_payoffs: list[float] = []
    train_steps = 0
    q_updates = 0
    q_target_syncs = 0
    last_q_loss = None
    train_start = time.perf_counter()
    for _ in range(int(cfg.train_episodes)):
        transitions, payoff, steps = _play_hand_against_frozen(
            q_net,
            loaded,
            cfg,
            rng,
            device,
            training=True,
        )
        train_payoffs.append(payoff)
        train_steps += steps
        for transition in transitions:
            buffer.add(transition)
        if len(buffer) >= int(cfg.min_buffer_size_to_learn):
            last_q_loss = _train_q(
                q_net,
                q_target_net,
                optimizer,
                buffer,
                int(cfg.batch_size),
                rng,
                device,
                float(cfg.q_discount),
            )
            q_updates += 1
            if q_updates % max(int(cfg.q_target_sync_interval), 1) == 0:
                q_target_net.load_state_dict(q_net.state_dict())
                q_target_syncs += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    eval_payoffs: list[float] = []
    eval_steps = 0
    eval_start = time.perf_counter()
    for _ in range(int(cfg.eval_games)):
        _transitions, payoff, steps = _play_hand_against_frozen(
            q_net,
            loaded,
            cfg,
            rng,
            device,
            training=False,
        )
        eval_payoffs.append(payoff)
        eval_steps += steps
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    eval_arr = np.asarray(eval_payoffs, dtype=np.float64)
    mean_eval = float(eval_arr.mean()) if eval_arr.size else 0.0
    std_eval = float(eval_arr.std(ddof=1)) if eval_arr.size > 1 else 0.0
    ci95_eval = float(1.96 * std_eval / np.sqrt(eval_arr.size)) if eval_arr.size > 1 else 0.0
    return {
        "algorithm": "frozen_checkpoint_best_response",
        "role": "approximate_exploitability_diagnostic",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Bounded DQN-style best-response pilot; diagnostic only, not exact exploitability.",
        **device_info,
        "source_checkpoint": str(cfg.checkpoint),
        "checkpoint_iteration": loaded.metadata.get("checkpoint_iteration"),
        "strategy_source": cfg.strategy_source,
        "num_actions": N_ACTIONS,
        "br_player": int(cfg.br_player),
        "train_episodes": int(cfg.train_episodes),
        "eval_games": int(cfg.eval_games),
        "train_steps": int(train_steps),
        "eval_steps": int(eval_steps),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "train_steps_per_second": float(train_steps / max(train_seconds, 1e-9)),
        "eval_steps_per_second": float(eval_steps / max(eval_seconds, 1e-9)),
        "mean_train_br_payoff": float(np.mean(train_payoffs)) if train_payoffs else 0.0,
        "mean_eval_br_payoff": mean_eval,
        "ci95_eval_br_payoff": ci95_eval,
        "lower95_eval_br_payoff": mean_eval - ci95_eval,
        "q_buffer_size": len(buffer),
        "q_updates": int(q_updates),
        "q_target_syncs": int(q_target_syncs),
        "last_q_loss": last_q_loss,
        "promotion": False,
    }
