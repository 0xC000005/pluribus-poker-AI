"""Coroutine-based batched Deep CFR traversal with multi-process support.

Optimization #2 (batched inference) and #4 (multi-process traversal):
- traverse_coroutine: generator that yields InferenceRequests
- batched_traverse: scheduler that batches forward passes across N concurrent
  traversals, giving batch_size=N instead of 1
- worker_fn: multiprocessing entry point for distributed traversal
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch

from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.deep_cfr import regret_match
from poker_ai.deep_cfr.fast_state import (
    FastPokerState,
    N_ACTIONS,
    N_FEATURES,
    new_fast_game,
)
from poker_ai.deep_cfr.networks import ValueNetwork


@dataclass
class InferenceRequest:
    """Yielded by a traverse coroutine when it needs a network prediction."""
    features: np.ndarray   # (126,) float32
    legal_mask: np.ndarray  # (3,) float32


def traverse_coroutine(
    state: FastPokerState,
    traverser: int,
    buffer: ReservoirBuffer,
    iteration: int,
) -> Generator[InferenceRequest, np.ndarray, float]:
    """External-sampling MCCFR as a coroutine.

    Yields InferenceRequest when it needs a forward pass.
    Receives np.ndarray advantages back via .send().
    Returns (via StopIteration.value) the expected value for the traverser.
    """
    if state.is_terminal:
        return float(state.payout[traverser])

    pi = state.current_player_i

    # Skip inactive players.
    if not state.active[pi]:
        child = state.copy()
        child.apply_action(None)
        return (yield from traverse_coroutine(child, traverser, buffer, iteration))

    features = state.to_feature_vector()
    legal_mask = state.get_legal_mask()

    # Yield request, receive batched result.
    advantages: np.ndarray = yield InferenceRequest(features, legal_mask)
    strategy = regret_match(advantages, legal_mask)

    legal_actions = [a for a in state.legal_actions if a is not None]

    if pi == traverser:
        # Traverser node: explore ALL legal actions.
        action_values: dict[int, float] = {}
        for action in legal_actions:
            child = state.copy()
            child.apply_action(action)
            action_values[action] = yield from traverse_coroutine(
                child, traverser, buffer, iteration,
            )

        state_value = sum(
            strategy[a] * action_values[a] for a in legal_actions
        )
        regrets = np.zeros(N_ACTIONS, dtype=np.float32)
        for a in legal_actions:
            regrets[a] = action_values[a] - state_value

        # Normalize regrets to [-1, 1] range for stable NN training.
        regrets /= state.initial_chips
        buffer.add(features, iteration, regrets)
        return state_value

    else:
        # Opponent node: sample a single action.
        probs = np.array([strategy[a] for a in legal_actions], dtype=np.float64)
        probs /= probs.sum()
        action = int(np.random.choice(legal_actions, p=probs))
        child = state.copy()
        child.apply_action(action)
        return (yield from traverse_coroutine(
            child, traverser, buffer, iteration,
        ))


def batched_traverse(
    n_traversals: int,
    traverser: int,
    value_net: ValueNetwork,
    buffer: ReservoirBuffer,
    iteration: int,
    device: torch.device,
    n_players: int = 2,
    initial_chips: int = 10000,
) -> None:
    """Run N traversals concurrently, batching their inference requests.

    Instead of N×~6 single-sample forward passes, we get ~6 batched forward
    passes with batch_size up to N.
    """
    # Create coroutines.
    coroutines: list[Generator] = []
    for _ in range(n_traversals):
        state = new_fast_game(n_players, initial_chips=initial_chips)
        co = traverse_coroutine(state, traverser, buffer, iteration)
        coroutines.append(co)

    # Initialize: get first InferenceRequest from each.
    pending: dict[int, tuple[Generator, InferenceRequest]] = {}
    for i, co in enumerate(coroutines):
        try:
            req = next(co)
            pending[i] = (co, req)
        except StopIteration:
            pass  # Immediate terminal.

    # Main scheduling loop.
    while pending:
        ids = list(pending.keys())
        features_list = [pending[i][1].features for i in ids]
        features_batch = torch.from_numpy(np.stack(features_list))

        with torch.no_grad():
            predictions = value_net(features_batch.to(device)).cpu().numpy()

        # Resume each coroutine with its result.
        next_pending: dict[int, tuple[Generator, InferenceRequest]] = {}
        for j, i in enumerate(ids):
            co, _ = pending[i]
            try:
                next_req = co.send(predictions[j])
                next_pending[i] = (co, next_req)
            except StopIteration:
                pass  # Traversal complete.
        pending = next_pending


# ---------------------------------------------------------------------------
# Multi-process worker
# ---------------------------------------------------------------------------

def worker_fn(
    worker_id: int,
    n_traversals: int,
    traverser: int,
    value_net_state_dict: dict,
    iteration: int,
    n_players: int,
    buffer_capacity: int,
    hidden_dim: int,
    initial_chips: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Worker entry point for multiprocessing.

    Creates a local value network (CPU), runs batched_traverse, and
    returns buffer data as picklable numpy arrays.
    """
    value_net = ValueNetwork(N_FEATURES, hidden_dim, N_ACTIONS)
    value_net.load_state_dict(value_net_state_dict)
    value_net.eval()
    device = torch.device("cpu")

    buffer = ReservoirBuffer(buffer_capacity)
    batched_traverse(
        n_traversals=n_traversals,
        traverser=traverser,
        value_net=value_net,
        buffer=buffer,
        iteration=iteration,
        device=device,
        n_players=n_players,
        initial_chips=initial_chips,
    )

    return (
        buffer.features[: buffer.size].copy(),
        buffer.iterations[: buffer.size].copy(),
        buffer.advantages[: buffer.size].copy(),
        buffer.size,
    )
