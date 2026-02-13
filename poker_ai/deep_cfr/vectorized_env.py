"""Vectorized poker environment for fast evaluation.

Optimization #3: Run N games simultaneously with batched neural net inference.
Primary use case: evaluate_vs_random goes from ~5s to ~50ms.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from poker_ai.deep_cfr.deep_cfr import regret_match
from poker_ai.deep_cfr.fast_state import FastPokerState, N_ACTIONS, new_fast_game
from poker_ai.deep_cfr.networks import ValueNetwork


class VectorizedPokerEnv:
    """N parallel poker games backed by FastPokerState instances."""

    def __init__(self, num_envs: int, n_players: int = 2, **game_kwargs):
        self.num_envs = num_envs
        self.n_players = n_players
        self.game_kwargs = game_kwargs
        self.states: List[FastPokerState] = []
        self.done = np.zeros(num_envs, dtype=np.bool_)

    def reset(self):
        """Create num_envs fresh games."""
        self.states = [
            new_fast_game(self.n_players, **self.game_kwargs)
            for _ in range(self.num_envs)
        ]
        self.done[:] = False

    def get_features(self) -> np.ndarray:
        """Return (num_envs, 126) feature matrix."""
        return np.stack([s.to_feature_vector() for s in self.states])

    def get_legal_masks(self) -> np.ndarray:
        """Return (num_envs, 3) legal action masks."""
        return np.stack([s.get_legal_mask() for s in self.states])

    def get_current_players(self) -> np.ndarray:
        """Return (num_envs,) array of current player indices."""
        return np.array([s.current_player_i for s in self.states], dtype=np.int32)

    def step_single(self, env_idx: int, action: int):
        """Apply action to a single environment."""
        if self.done[env_idx]:
            return
        s = self.states[env_idx]
        child = s.copy()
        child.apply_action(action)
        self.states[env_idx] = child
        if child.is_terminal:
            self.done[env_idx] = True

    def get_payouts(self, player: int = 0) -> np.ndarray:
        """Return (num_envs,) payouts for the given player."""
        return np.array(
            [s.payout[player] if s.is_terminal else 0 for s in self.states],
            dtype=np.float64,
        )


def fast_evaluate_vs_random(
    value_net: ValueNetwork,
    device: torch.device,
    n_games: int = 500,
    n_players: int = 2,
) -> float:
    """Evaluate trained agent (player 0) vs random opponents.

    Uses batched inference — all games that need player-0 decisions get a
    single forward pass together.

    Returns average payout (chips) per game for player 0.
    """
    env = VectorizedPokerEnv(n_games, n_players)
    env.reset()
    value_net.eval()

    max_steps = n_games * 50  # safety limit
    step_count = 0

    while not env.done.all() and step_count < max_steps:
        step_count += 1

        # Find undone games and handle inactive players first.
        for i in range(n_games):
            if env.done[i]:
                continue
            s = env.states[i]
            # Advance past any inactive players.
            while not s.is_terminal and not s.active[s.current_player_i]:
                child = s.copy()
                child.apply_action(None)
                env.states[i] = child
                s = child
                if s.is_terminal:
                    env.done[i] = True

        if env.done.all():
            break

        # Separate games by whose turn it is.
        agent_indices = []
        random_indices = []
        for i in range(n_games):
            if env.done[i]:
                continue
            if env.states[i].current_player_i == 0:
                agent_indices.append(i)
            else:
                random_indices.append(i)

        # Random opponents: pick uniformly from legal actions.
        for i in random_indices:
            mask = env.states[i].get_legal_mask()
            legal = np.where(mask > 0)[0]
            action = int(np.random.choice(legal))
            env.step_single(i, action)

        # Trained agent: batched inference.
        if agent_indices:
            features_list = [
                env.states[i].to_feature_vector() for i in agent_indices
            ]
            masks_list = [
                env.states[i].get_legal_mask() for i in agent_indices
            ]
            features_batch = torch.from_numpy(np.stack(features_list))
            with torch.no_grad():
                preds = value_net(features_batch.to(device)).cpu().numpy()

            for j, i in enumerate(agent_indices):
                strategy = regret_match(preds[j], masks_list[j])
                legal = np.where(masks_list[j] > 0)[0]
                probs = np.array([strategy[a] for a in legal], dtype=np.float64)
                probs /= probs.sum()
                action = int(np.random.choice(legal, p=probs))
                env.step_single(i, action)

    return float(env.get_payouts(0).mean())
