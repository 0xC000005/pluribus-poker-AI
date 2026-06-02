"""Parity and throughput checks for native-style poker rollout substrates."""

from __future__ import annotations

import contextlib
import copy
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from poker_ai.deep_cfr.deep_cfr import get_legal_mask as get_full_deck_legal_mask
from poker_ai.deep_cfr.fast_state import FastPokerState, _get_orders, new_fast_game
from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, card_to_index, new_game
from poker_ai.poker.deck import Deck
from poker_ai.research.compiled_fast_rollout import (
    CompiledFastStateBatch,
    compiled_apply_actions,
    compiled_feature_vectors,
    compiled_legal_masks,
)


_STAGE_TO_FAST = {
    "pre_flop": FastPokerState.PREFLOP,
    "flop": FastPokerState.FLOP,
    "turn": FastPokerState.TURN,
    "river": FastPokerState.RIVER,
    "show_down": FastPokerState.SHOWDOWN,
    "terminal": FastPokerState.TERMINAL,
}

CENTRALIZED_FAST_Q_FEATURES = N_FEATURES + 52 + 8


@contextlib.contextmanager
def deterministic_full_deck_deals() -> Iterator[None]:
    """Make the reference deck deal from a deterministic stack during checks."""
    original_pick = Deck.pick

    def _pick_next(self: Deck, random: bool = True):  # noqa: A002 - matches API
        return original_pick(self, random=False)

    Deck.pick = _pick_next
    try:
        yield
    finally:
        Deck.pick = original_pick


def _deck_order_from_reference_state(
    state,
    *,
    future_deal_mode: str = "deterministic",
) -> tuple[np.ndarray, int]:
    deck = state._table.dealer.deck
    dealt = [card_to_index(card) for card in deck._dealt_cards]
    if future_deal_mode == "deterministic":
        future = [card_to_index(card) for card in reversed(deck._cards_in_deck)]
    elif future_deal_mode == "random":
        np_state = np.random.get_state()
        try:
            remaining = list(deck._cards_in_deck)
            future = []
            while remaining:
                index = int(np.random.randint(len(remaining), size=None))
                future.append(card_to_index(remaining.pop(index)))
        finally:
            np.random.set_state(np_state)
    else:
        raise ValueError("future_deal_mode must be one of: deterministic, random")
    order = np.array(dealt + future, dtype=np.int8)
    if order.shape != (52,):
        raise ValueError(f"expected full 52-card deck order, got {order.shape}")
    return order, len(dealt)


def fast_state_from_full_deck_state(
    state,
    *,
    future_deal_mode: str = "deterministic",
) -> FastPokerState:
    """Build a FastPokerState snapshot from the canonical full-deck state."""
    n_players = len(state.players)
    fast = FastPokerState.__new__(FastPokerState)
    fast.n_players = int(n_players)
    fast.small_blind = int(state.small_blind)
    fast.big_blind = int(state.big_blind)
    fast.initial_chips = int(state._initial_n_chips)
    fast.chips = np.array([int(p.n_chips) for p in state.players], dtype=np.int32)
    fast.bets = np.array([int(p.n_bet_chips) for p in state.players], dtype=np.int32)
    fast.active = np.array([bool(p.is_active) for p in state.players], dtype=np.bool_)
    fast.hole_cards = np.full((n_players, 2), -1, dtype=np.int8)
    for player_i, player in enumerate(state.players):
        for card_i, card in enumerate(player.cards[:2]):
            fast.hole_cards[player_i, card_i] = card_to_index(card)
    fast.community = np.full(5, -1, dtype=np.int8)
    for card_i, card in enumerate(state.community_cards[:5]):
        fast.community[card_i] = card_to_index(card)
    fast.deck_order, fast.deck_cursor = _deck_order_from_reference_state(
        state,
        future_deal_mode=future_deal_mode,
    )
    fast.stage = int(_STAGE_TO_FAST[str(state.betting_stage)])
    fast.n_raises = int(state._n_raises)
    fast._player_i_index = int(state._player_i_index)
    fast.n_actions = int(state._n_actions)
    fast.pot_total = int(state._table.pot.total)
    fast._skip_counter = int(state._skip_counter)
    fast._winners_computed = bool(state.is_terminal)
    fast.history = np.zeros((4, 3), dtype=np.int8)
    for round_i, stage in enumerate(("pre_flop", "flop", "turn", "river")):
        actions = state._history.get(stage, [])
        fast.history[round_i, 0] = sum(1 for action in actions if action == "call")
        fast.history[round_i, 1] = sum(1 for action in actions if action == "raise")
        fast.history[round_i, 2] = sum(1 for action in actions if action == "fold")
    fast._preflop_order, fast._postflop_order = _get_orders(n_players)
    fast.n_players_started_round = int(state._n_players_started_round)
    return fast


def _safe_full_deck_legal_mask(state) -> np.ndarray:
    if state.is_terminal:
        return np.zeros(N_ACTIONS, dtype=np.float32)
    return get_full_deck_legal_mask(state)


def _state_mismatches(reference, fast: FastPokerState, *, label: str) -> tuple[list[str], float]:
    mismatches: list[str] = []
    ref_mask = _safe_full_deck_legal_mask(reference)
    fast_mask = fast.get_legal_mask() if not fast.is_terminal else np.zeros(N_ACTIONS, dtype=np.float32)
    if not np.array_equal(ref_mask > 0, fast_mask > 0):
        mismatches.append(f"{label}: legal mask mismatch ref={ref_mask.tolist()} fast={fast_mask.tolist()}")
    if int(reference.player_i) != int(fast.current_player_i):
        mismatches.append(f"{label}: current player mismatch ref={reference.player_i} fast={fast.current_player_i}")
    if int(_STAGE_TO_FAST[str(reference.betting_stage)]) != int(fast.stage):
        mismatches.append(f"{label}: stage mismatch ref={reference.betting_stage} fast={fast.stage}")
    ref_chips = np.array([int(p.n_chips) for p in reference.players], dtype=np.int32)
    ref_bets = np.array([int(p.n_bet_chips) for p in reference.players], dtype=np.int32)
    ref_active = np.array([bool(p.is_active) for p in reference.players], dtype=np.bool_)
    if not np.array_equal(ref_chips, fast.chips):
        mismatches.append(f"{label}: chips mismatch ref={ref_chips.tolist()} fast={fast.chips.tolist()}")
    if not np.array_equal(ref_bets, fast.bets):
        mismatches.append(f"{label}: bets mismatch ref={ref_bets.tolist()} fast={fast.bets.tolist()}")
    if not np.array_equal(ref_active, fast.active):
        mismatches.append(f"{label}: active mismatch ref={ref_active.tolist()} fast={fast.active.tolist()}")
    if int(reference._table.pot.total) != int(fast.pot_total):
        mismatches.append(f"{label}: pot mismatch ref={reference._table.pot.total} fast={fast.pot_total}")
    if bool(reference.is_terminal) != bool(fast.is_terminal):
        mismatches.append(f"{label}: terminal mismatch ref={reference.is_terminal} fast={fast.is_terminal}")
    ref_features = reference.to_feature_vector().astype(np.float32, copy=False)
    fast_features = fast.to_feature_vector().astype(np.float32, copy=False)
    feature_max_abs_diff = float(np.max(np.abs(ref_features - fast_features)))
    if feature_max_abs_diff > 0.0:
        mismatches.append(f"{label}: feature mismatch max_abs_diff={feature_max_abs_diff}")
    if reference.is_terminal and fast.is_terminal:
        ref_payout = np.array([int(reference.payout[i]) for i in range(len(reference.players))])
        fast_payout = np.array([int(fast.payout[i]) for i in range(fast.n_players)])
        if not np.array_equal(ref_payout, fast_payout):
            mismatches.append(f"{label}: payout mismatch ref={ref_payout.tolist()} fast={fast_payout.tolist()}")
    return mismatches, feature_max_abs_diff


def _sample_legal_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask > 0)
    if legal.size == 0:
        raise ValueError("cannot sample action from empty legal mask")
    return int(rng.choice(legal))


def _default_policy_replay_action(
    features: np.ndarray,
    legal_mask: np.ndarray,
    *,
    player_i: int,
    game_i: int,
    step_i: int,
) -> int:
    legal = np.flatnonzero(legal_mask > 0)
    if legal.size == 0:
        raise ValueError("cannot select action from empty legal mask")
    feature_hash = int(np.dot(features, np.arange(1, features.size + 1, dtype=np.float32)) * 1000)
    offset = feature_hash + int(player_i) * 7 + int(game_i) * 17 + int(step_i) * 31
    return int(legal[offset % int(legal.size)])


def run_fast_state_parity_check(
    *,
    n_games: int = 16,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    seed: int = 20260750,
    max_mismatches: int = 10,
) -> dict[str, Any]:
    """Replay deterministic full-deck hands through FastPokerState."""
    rng = np.random.default_rng(seed)
    mismatches: list[str] = []
    checked_steps = 0
    max_feature_diff = 0.0
    with deterministic_full_deck_deals():
        for game_i in range(int(n_games)):
            random.seed(int(seed) + game_i)
            np.random.seed(int(seed) + game_i)
            reference = new_game(2, initial_chips=int(initial_chips))
            fast = fast_state_from_full_deck_state(reference)
            for step_i in range(int(max_steps_per_game)):
                current, feature_diff = _state_mismatches(
                    reference,
                    fast,
                    label=f"game={game_i} step={step_i} pre",
                )
                max_feature_diff = max(max_feature_diff, feature_diff)
                mismatches.extend(current)
                if mismatches or reference.is_terminal:
                    break
                action_idx = _sample_legal_action(_safe_full_deck_legal_mask(reference), rng)
                reference = reference.apply_action(INDEX_TO_ACTION[action_idx])
                fast.apply_action(action_idx)
                checked_steps += 1
                current, feature_diff = _state_mismatches(
                    reference,
                    fast,
                    label=f"game={game_i} step={step_i} post action={action_idx}",
                )
                max_feature_diff = max(max_feature_diff, feature_diff)
                mismatches.extend(current)
                if mismatches or reference.is_terminal:
                    break
            if len(mismatches) >= int(max_mismatches):
                break
    return {
        "backend": "fast-state",
        "canonical_backend": "python-full-deck",
        "n_games": int(n_games),
        "max_steps_per_game": int(max_steps_per_game),
        "checked_steps": int(checked_steps),
        "feature_max_abs_diff": float(max_feature_diff),
        "mismatches": mismatches[: int(max_mismatches)],
        "passed": not mismatches and checked_steps > 0,
    }


def run_fast_state_policy_replay_parity(
    *,
    n_games: int = 16,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    seed: int = 20260788,
    max_mismatches: int = 10,
    action_selector: Callable[..., int] | None = None,
    deterministic_deals: bool = True,
) -> dict[str, Any]:
    """Replay canonical hands through FastPokerState using feature-driven actions.

    This is stricter than random legal-action replay for H2H gates: each action
    is selected from the exact feature vector, legal mask, player id, game id,
    and step id on both backends before the transition is applied.
    """
    selector = action_selector or _default_policy_replay_action
    mismatches: list[str] = []
    checked_decisions = 0
    action_mismatches = 0
    max_feature_diff = 0.0
    deal_context = deterministic_full_deck_deals() if deterministic_deals else contextlib.nullcontext()
    future_deal_mode = "deterministic" if deterministic_deals else "random"
    with deal_context:
        for game_i in range(int(n_games)):
            random.seed(int(seed) + game_i)
            np.random.seed(int(seed) + game_i)
            reference = new_game(2, initial_chips=int(initial_chips))
            fast = fast_state_from_full_deck_state(
                reference,
                future_deal_mode=future_deal_mode,
            )
            for step_i in range(int(max_steps_per_game)):
                current, feature_diff = _state_mismatches(
                    reference,
                    fast,
                    label=f"game={game_i} step={step_i} pre",
                )
                max_feature_diff = max(max_feature_diff, feature_diff)
                mismatches.extend(current)
                if mismatches or reference.is_terminal:
                    break

                ref_features = reference.to_feature_vector().astype(np.float32, copy=False)
                fast_features = fast.to_feature_vector().astype(np.float32, copy=False)
                ref_mask = _safe_full_deck_legal_mask(reference)
                fast_mask = fast.get_legal_mask()
                ref_action = int(
                    selector(
                        ref_features,
                        ref_mask,
                        player_i=int(reference.player_i),
                        game_i=int(game_i),
                        step_i=int(step_i),
                    )
                )
                fast_action = int(
                    selector(
                        fast_features,
                        fast_mask,
                        player_i=int(fast.current_player_i),
                        game_i=int(game_i),
                        step_i=int(step_i),
                    )
                )
                checked_decisions += 1
                ref_legal = 0 <= ref_action < N_ACTIONS and ref_mask[ref_action] > 0
                fast_legal = 0 <= fast_action < N_ACTIONS and fast_mask[fast_action] > 0
                if not ref_legal or not fast_legal or ref_action != fast_action:
                    action_mismatches += 1
                    mismatches.append(
                        "game="
                        f"{game_i} step={step_i}: action mismatch "
                        f"ref={ref_action} legal={bool(ref_legal)} "
                        f"fast={fast_action} legal={bool(fast_legal)}"
                    )
                    break

                reference = reference.apply_action(INDEX_TO_ACTION[ref_action])
                fast.apply_action(fast_action)
                current, feature_diff = _state_mismatches(
                    reference,
                    fast,
                    label=f"game={game_i} step={step_i} post action={ref_action}",
                )
                max_feature_diff = max(max_feature_diff, feature_diff)
                mismatches.extend(current)
                if mismatches or reference.is_terminal:
                    break
            if len(mismatches) >= int(max_mismatches):
                break
    return {
        "backend": "fast-state",
        "canonical_backend": "python-full-deck",
        "mode": "policy_replay_parity",
        "n_games": int(n_games),
        "max_steps_per_game": int(max_steps_per_game),
        "checked_decisions": int(checked_decisions),
        "action_mismatches": int(action_mismatches),
        "deterministic_deals": bool(deterministic_deals),
        "feature_max_abs_diff": float(max_feature_diff),
        "mismatches": mismatches[: int(max_mismatches)],
        "passed": not mismatches and checked_decisions > 0,
    }


def run_rollout_throughput(
    *,
    backend: str,
    n_games: int = 128,
    max_steps_per_game: int = 128,
    initial_chips: int = 1000,
    seed: int = 20260751,
) -> dict[str, Any]:
    """Measure random legal-action environment-step throughput."""
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    steps = 0
    if backend == "python-full-deck":
        with deterministic_full_deck_deals():
            for game_i in range(int(n_games)):
                random.seed(int(seed) + game_i)
                np.random.seed(int(seed) + game_i)
                state = new_game(2, initial_chips=int(initial_chips))
                for _ in range(int(max_steps_per_game)):
                    if state.is_terminal:
                        break
                    action_idx = _sample_legal_action(_safe_full_deck_legal_mask(state), rng)
                    state = state.apply_action(INDEX_TO_ACTION[action_idx])
                    steps += 1
    elif backend == "fast-state":
        for game_i in range(int(n_games)):
            np.random.seed(int(seed) + game_i)
            state = new_fast_game(2, initial_chips=int(initial_chips))
            for _ in range(int(max_steps_per_game)):
                if state.is_terminal:
                    break
                action_idx = _sample_legal_action(state.get_legal_mask(), rng)
                state.apply_action(action_idx)
                steps += 1
    else:
        raise ValueError("backend must be one of: python-full-deck, fast-state")
    seconds = time.perf_counter() - started
    return {
        "backend": backend,
        "n_games": int(n_games),
        "max_steps_per_game": int(max_steps_per_game),
        "steps": int(steps),
        "seconds": float(seconds),
        "steps_per_second": float(steps / max(seconds, 1e-12)),
    }


def _resolve_policy_inference_device(requested_device: str):
    import torch

    requested = str(requested_device).strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    elif requested == "cuda":
        if not cuda_available:
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        resolved = "cuda"
    elif requested == "cpu":
        resolved = "cpu"
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(resolved), {
        "requested_device": requested,
        "resolved_device": resolved,
        "torch_cuda_available": cuda_available,
        "torch_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "torch_device_name": torch.cuda.get_device_name(0) if cuda_available else "",
    }


def _build_policy_inference_net(hidden_dim: int, device):
    import torch
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(N_FEATURES, int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), N_ACTIONS),
    ).to(device).eval()


def _policy_inference_actions(policy, features: np.ndarray, legal_masks: np.ndarray, device) -> np.ndarray:
    import torch

    x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
    masks = torch.as_tensor(np.asarray(legal_masks, dtype=np.float32), device=device)
    with torch.no_grad():
        logits = policy(x)
        logits = logits.masked_fill(masks <= 0, -1.0e30)
        actions = torch.argmax(logits, dim=1)
    return actions.detach().cpu().numpy().astype(np.int64)


def _new_seeded_fast_state(seed: int, game_i: int, initial_chips: int) -> FastPokerState:
    np.random.seed(int(seed) + int(game_i))
    return new_fast_game(2, initial_chips=int(initial_chips))


def run_fast_state_policy_inference_throughput(
    *,
    mode: str,
    n_games: int = 256,
    batch_size: int = 64,
    max_steps_per_game: int = 128,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    device: str = "auto",
    seed: int = 20260805,
) -> dict[str, Any]:
    """Measure fast-state rollout throughput with neural policy inference.

    This benchmark isolates the collector shape: sequential mode performs one
    policy forward per decision, while batched mode runs one forward over many
    live fast states before applying the chosen legal actions.
    """
    import torch

    mode = str(mode).strip().lower()
    if mode not in {"sequential", "batched"}:
        raise ValueError("mode must be one of: sequential, batched")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    resolved_device, device_info = _resolve_policy_inference_device(device)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    policy = _build_policy_inference_net(int(hidden_dim), resolved_device)

    started = time.perf_counter()
    steps = 0
    forward_calls = 0
    if mode == "sequential":
        for game_i in range(int(n_games)):
            state = _new_seeded_fast_state(seed, game_i, int(initial_chips))
            for _ in range(int(max_steps_per_game)):
                if state.is_terminal:
                    break
                features = state.to_feature_vector()[None, :].astype(np.float32, copy=False)
                legal_mask = state.get_legal_mask()[None, :].astype(np.float32, copy=False)
                action_idx = int(_policy_inference_actions(policy, features, legal_mask, resolved_device)[0])
                forward_calls += 1
                state.apply_action(action_idx)
                steps += 1
    else:
        states = [
            _new_seeded_fast_state(seed, game_i, int(initial_chips))
            for game_i in range(int(n_games))
        ]
        steps_per_game = np.zeros(int(n_games), dtype=np.int32)
        while True:
            live_indices = [
                i
                for i, state in enumerate(states)
                if not state.is_terminal and int(steps_per_game[i]) < int(max_steps_per_game)
            ]
            if not live_indices:
                break
            for start in range(0, len(live_indices), int(batch_size)):
                batch_indices = live_indices[start : start + int(batch_size)]
                features = np.stack(
                    [states[i].to_feature_vector().astype(np.float32, copy=False) for i in batch_indices]
                )
                legal_masks = np.stack(
                    [states[i].get_legal_mask().astype(np.float32, copy=False) for i in batch_indices]
                )
                actions = _policy_inference_actions(policy, features, legal_masks, resolved_device)
                forward_calls += 1
                for state_i, action_idx in zip(batch_indices, actions, strict=True):
                    states[state_i].apply_action(int(action_idx))
                    steps_per_game[state_i] += 1
                    steps += 1
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return {
        "mode": mode,
        "backend": "fast-state",
        "policy": "synthetic_masked_mlp_argmax",
        "n_games": int(n_games),
        "batch_size": int(batch_size),
        "max_steps_per_game": int(max_steps_per_game),
        "hidden_dim": int(hidden_dim),
        "initial_chips": int(initial_chips),
        "steps": int(steps),
        "seconds": float(seconds),
        "steps_per_second": float(steps / max(seconds, 1e-12)),
        "games_per_second": float(int(n_games) / max(seconds, 1e-12)),
        "policy_forward_calls": int(forward_calls),
        "mean_decisions_per_forward": float(steps / max(int(forward_calls), 1)),
        **device_info,
    }


def _empty_fast_self_play_batch() -> dict[str, np.ndarray]:
    return {
        "features": np.zeros((0, N_FEATURES), dtype=np.float32),
        "critic_features": np.zeros((0, CENTRALIZED_FAST_Q_FEATURES), dtype=np.float32),
        "legal_masks": np.zeros((0, N_ACTIONS), dtype=np.float32),
        "actions": np.zeros((0,), dtype=np.int64),
        "old_log_probs": np.zeros((0,), dtype=np.float32),
        "players": np.zeros((0,), dtype=np.int64),
        "game_indices": np.zeros((0,), dtype=np.int64),
        "step_indices": np.zeros((0,), dtype=np.int64),
        "next_decision_indices": np.zeros((0,), dtype=np.int64),
        "dones": np.zeros((0,), dtype=np.float32),
        "rewards": np.zeros((0,), dtype=np.float32),
        "payoffs": np.zeros((0, 2), dtype=np.float32),
    }


def _fast_state_centralized_q_features(
    state: FastPokerState,
    observation: np.ndarray,
) -> np.ndarray:
    features = np.zeros(CENTRALIZED_FAST_Q_FEATURES, dtype=np.float32)
    features[:N_FEATURES] = np.asarray(observation, dtype=np.float32)
    offset = N_FEATURES
    current_player = int(state.current_player_i)
    for player_i in range(int(state.n_players)):
        if int(player_i) == current_player:
            continue
        for card_idx in state.hole_cards[int(player_i), :2]:
            card_i = int(card_idx)
            if 0 <= card_i < 52:
                features[offset + card_i] = 1.0
    offset += 52
    scale = max(float(state.initial_chips), 1.0)
    for player_i in range(min(int(state.n_players), 2)):
        features[offset] = float(state.chips[player_i]) / scale
        features[offset + 1] = float(state.bets[player_i]) / scale
        features[offset + 2] = 1.0 if bool(state.active[player_i]) else 0.0
        features[offset + 3] = 1.0 if bool(state.active[player_i]) and int(state.chips[player_i]) <= 0 else 0.0
        offset += 4
    return features


def _fast_self_play_checksums(batch: dict[str, np.ndarray]) -> tuple[int, float]:
    if batch["actions"].size <= 0:
        return 0, 0.0
    weights = (
        (batch["game_indices"].astype(np.int64) + 1)
        * (batch["step_indices"].astype(np.int64) + 1)
        * (batch["players"].astype(np.int64) + 1)
    )
    action_checksum = int(np.sum((batch["actions"].astype(np.int64) + 1) * weights))
    payoff_weights = np.arange(1, batch["payoffs"].size + 1, dtype=np.float32).reshape(
        batch["payoffs"].shape
    )
    payoff_checksum = float(np.sum(batch["payoffs"].astype(np.float32) * payoff_weights))
    return action_checksum, payoff_checksum


def _final_fast_state_payoffs(state: FastPokerState, initial_chips: int) -> tuple[np.ndarray, bool]:
    if state.is_terminal:
        payout = state.payout
        return (
            np.array(
                [float(payout.get(i, 0)) / float(initial_chips) for i in range(state.n_players)],
                dtype=np.float32,
            ),
            False,
        )
    return np.zeros(state.n_players, dtype=np.float32), True


def collect_batched_fast_self_play(
    *,
    mode: str = "batched",
    n_games: int = 256,
    batch_size: int = 64,
    max_steps_per_game: int = 128,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    device: str = "auto",
    seed: int = 20260807,
) -> dict[str, Any]:
    """Collect native fast-state self-play transitions with batched actor inference.

    The policy is a synthetic masked MLP actor used only to exercise the
    substrate shape. It intentionally adds no strategic poker rule.
    """
    import torch

    mode = str(mode).strip().lower()
    if mode not in {"sequential", "batched"}:
        raise ValueError("mode must be one of: sequential, batched")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(n_games) <= 0:
        empty = _empty_fast_self_play_batch()
        return {
            "batch": empty,
            "metrics": {
                "mode": mode,
                "backend": "fast-state",
                "n_games": int(n_games),
                "steps": 0,
                "policy_forward_calls": 0,
                "mean_decisions_per_forward": 0.0,
                "uses_slumbot_training_data": False,
                "action_checksum": 0,
                "payoff_checksum": 0.0,
            },
        }

    resolved_device, device_info = _resolve_policy_inference_device(device)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    policy = _build_policy_inference_net(int(hidden_dim), resolved_device)
    states = [
        _new_seeded_fast_state(seed, game_i, int(initial_chips))
        for game_i in range(int(n_games))
    ]
    steps_per_game = np.zeros(int(n_games), dtype=np.int32)
    records: list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, int, float]] = []
    forward_calls = 0
    started = time.perf_counter()

    def _record_and_apply(game_i: int, action_idx: int, features: np.ndarray, legal_mask: np.ndarray) -> None:
        state = states[int(game_i)]
        player_i = int(state.current_player_i)
        records.append(
            (
                int(game_i),
                int(steps_per_game[int(game_i)]),
                player_i,
                np.asarray(features, dtype=np.float32),
                np.asarray(legal_mask, dtype=np.float32),
                int(action_idx),
            )
        )
        state.apply_action(int(action_idx))
        steps_per_game[int(game_i)] += 1

    if mode == "sequential":
        for game_i, state in enumerate(states):
            while not state.is_terminal and int(steps_per_game[game_i]) < int(max_steps_per_game):
                features = state.to_feature_vector().astype(np.float32, copy=False)
                legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
                action_idx = int(
                    _policy_inference_actions(
                        policy,
                        features[None, :],
                        legal_mask[None, :],
                        resolved_device,
                    )[0]
                )
                forward_calls += 1
                _record_and_apply(game_i, action_idx, features, legal_mask)
    else:
        while True:
            live_indices = [
                i
                for i, state in enumerate(states)
                if not state.is_terminal and int(steps_per_game[i]) < int(max_steps_per_game)
            ]
            if not live_indices:
                break
            for start in range(0, len(live_indices), int(batch_size)):
                batch_indices = live_indices[start : start + int(batch_size)]
                features = np.stack(
                    [states[i].to_feature_vector().astype(np.float32, copy=False) for i in batch_indices]
                )
                legal_masks = np.stack(
                    [states[i].get_legal_mask().astype(np.float32, copy=False) for i in batch_indices]
                )
                actions = _policy_inference_actions(policy, features, legal_masks, resolved_device)
                forward_calls += 1
                for row_i, game_i in enumerate(batch_indices):
                    _record_and_apply(game_i, int(actions[row_i]), features[row_i], legal_masks[row_i])

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    payoffs = np.zeros((int(n_games), 2), dtype=np.float32)
    truncated_games = 0
    for game_i, state in enumerate(states):
        payoff, truncated = _final_fast_state_payoffs(state, int(initial_chips))
        payoffs[game_i, : payoff.shape[0]] = payoff[:2]
        truncated_games += int(truncated)

    if records:
        game_indices = np.array([r[0] for r in records], dtype=np.int64)
        step_indices = np.array([r[1] for r in records], dtype=np.int64)
        players = np.array([r[2] for r in records], dtype=np.int64)
        features = np.stack([r[3] for r in records]).astype(np.float32, copy=False)
        legal_masks = np.stack([r[4] for r in records]).astype(np.float32, copy=False)
        actions = np.array([r[5] for r in records], dtype=np.int64)
        rewards = payoffs[game_indices, players].astype(np.float32, copy=False)
        batch = {
            "features": features,
            "legal_masks": legal_masks,
            "actions": actions,
            "players": players,
            "game_indices": game_indices,
            "step_indices": step_indices,
            "rewards": rewards,
            "payoffs": payoffs,
        }
    else:
        batch = _empty_fast_self_play_batch()
        batch["payoffs"] = payoffs

    action_checksum, payoff_checksum = _fast_self_play_checksums(batch)
    steps = int(len(records))
    metrics = {
        "mode": mode,
        "backend": "fast-state",
        "collector": "batched_live_state_self_play",
        "policy": "synthetic_masked_mlp_argmax",
        "n_games": int(n_games),
        "batch_size": int(batch_size),
        "max_steps_per_game": int(max_steps_per_game),
        "hidden_dim": int(hidden_dim),
        "initial_chips": int(initial_chips),
        "steps": steps,
        "seconds": float(seconds),
        "steps_per_second": float(steps / max(seconds, 1e-12)),
        "games_per_second": float(int(n_games) / max(seconds, 1e-12)),
        "policy_forward_calls": int(forward_calls),
        "mean_decisions_per_forward": float(steps / max(int(forward_calls), 1)),
        "truncated_games": int(truncated_games),
        "action_checksum": int(action_checksum),
        "payoff_checksum": float(payoff_checksum),
        "uses_slumbot_training_data": False,
        "promotion": False,
        **device_info,
    }
    return {"batch": batch, "metrics": metrics}


def collect_compiled_fast_self_play(
    *,
    n_games: int = 256,
    batch_size: int = 64,
    max_steps_per_game: int = 128,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    device: str = "auto",
    seed: int = 20260835,
) -> dict[str, Any]:
    """Collect self-play transitions through the compiled fast-state batch."""
    import torch

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(n_games) <= 0:
        empty = _empty_fast_self_play_batch()
        return {
            "batch": empty,
            "metrics": {
                "mode": "compiled",
                "backend": "compiled-fast-state",
                "n_games": int(n_games),
                "steps": 0,
                "policy_forward_calls": 0,
                "mean_decisions_per_forward": 0.0,
                "needs_python_showdown": 0,
                "uses_slumbot_training_data": False,
                "action_checksum": 0,
                "payoff_checksum": 0.0,
            },
        }

    resolved_device, device_info = _resolve_policy_inference_device(device)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    policy = _build_policy_inference_net(int(hidden_dim), resolved_device)
    states = [
        _new_seeded_fast_state(seed, game_i, int(initial_chips))
        for game_i in range(int(n_games))
    ]
    compiled = CompiledFastStateBatch.from_fast_states(states)
    steps_per_game = np.zeros(int(n_games), dtype=np.int32)
    records: list[tuple[int, int, int, np.ndarray, np.ndarray, int]] = []
    forward_calls = 0
    needs_python_showdown = 0
    started = time.perf_counter()

    while True:
        live_indices = [
            i
            for i in range(int(n_games))
            if int(compiled.stage[i]) < FastPokerState.SHOWDOWN
            and int(steps_per_game[i]) < int(max_steps_per_game)
        ]
        if not live_indices:
            break
        all_features = compiled_feature_vectors(compiled)
        all_masks = compiled_legal_masks(compiled)
        current_players = compiled.current_players()
        action_array = np.full(int(n_games), -1, dtype=np.int16)
        for start in range(0, len(live_indices), int(batch_size)):
            batch_indices = live_indices[start : start + int(batch_size)]
            features = all_features[batch_indices].astype(np.float32, copy=False)
            legal_masks = all_masks[batch_indices].astype(np.float32, copy=False)
            actions = _policy_inference_actions(policy, features, legal_masks, resolved_device)
            forward_calls += 1
            for row_i, game_i in enumerate(batch_indices):
                action_idx = int(actions[row_i])
                action_array[int(game_i)] = action_idx
                records.append(
                    (
                        int(game_i),
                        int(steps_per_game[int(game_i)]),
                        int(current_players[int(game_i)]),
                        np.asarray(features[row_i], dtype=np.float32),
                        np.asarray(legal_masks[row_i], dtype=np.float32),
                        action_idx,
                    )
                )
                steps_per_game[int(game_i)] += 1
        result = compiled_apply_actions(compiled, action_array)
        needs_python_showdown += int(result["needs_python_showdown"])

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    payoffs = (
        compiled.chips.astype(np.float32) - float(initial_chips)
    ) / float(initial_chips)
    terminal = compiled.stage >= FastPokerState.SHOWDOWN
    truncated_games = int(np.sum(~terminal))
    payoffs[~terminal, :] = 0.0

    if records:
        game_indices = np.array([r[0] for r in records], dtype=np.int64)
        step_indices = np.array([r[1] for r in records], dtype=np.int64)
        players = np.array([r[2] for r in records], dtype=np.int64)
        features = np.stack([r[3] for r in records]).astype(np.float32, copy=False)
        legal_masks = np.stack([r[4] for r in records]).astype(np.float32, copy=False)
        actions = np.array([r[5] for r in records], dtype=np.int64)
        rewards = payoffs[game_indices, players].astype(np.float32, copy=False)
        batch = {
            "features": features,
            "critic_features": np.zeros((features.shape[0], CENTRALIZED_FAST_Q_FEATURES), dtype=np.float32),
            "legal_masks": legal_masks,
            "actions": actions,
            "old_log_probs": np.zeros((features.shape[0],), dtype=np.float32),
            "players": players,
            "game_indices": game_indices,
            "step_indices": step_indices,
            "next_decision_indices": np.full((features.shape[0],), -1, dtype=np.int64),
            "dones": np.zeros((features.shape[0],), dtype=np.float32),
            "rewards": rewards,
            "payoffs": payoffs.astype(np.float32, copy=False),
        }
    else:
        batch = _empty_fast_self_play_batch()
        batch["payoffs"] = payoffs.astype(np.float32, copy=False)

    action_checksum, payoff_checksum = _fast_self_play_checksums(batch)
    steps = int(len(records))
    metrics = {
        "mode": "compiled",
        "backend": "compiled-fast-state",
        "collector": "compiled_batch_state_self_play",
        "policy": "synthetic_masked_mlp_argmax",
        "n_games": int(n_games),
        "batch_size": int(batch_size),
        "max_steps_per_game": int(max_steps_per_game),
        "hidden_dim": int(hidden_dim),
        "initial_chips": int(initial_chips),
        "steps": steps,
        "seconds": float(seconds),
        "steps_per_second": float(steps / max(seconds, 1e-12)),
        "games_per_second": float(int(n_games) / max(seconds, 1e-12)),
        "policy_forward_calls": int(forward_calls),
        "mean_decisions_per_forward": float(steps / max(int(forward_calls), 1)),
        "truncated_games": int(truncated_games),
        "needs_python_showdown": int(needs_python_showdown),
        "action_checksum": int(action_checksum),
        "payoff_checksum": float(payoff_checksum),
        "uses_slumbot_training_data": False,
        "promotion": False,
        **device_info,
    }
    return {"batch": batch, "metrics": metrics}


def _collect_compiled_fast_policy_gradient_rollout(
    policy,
    *,
    n_games: int,
    batch_size: int,
    max_steps_per_game: int,
    initial_chips: int,
    device,
    seed: int,
    opponent_policies: list[Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect learner-seat policy-gradient records from compiled batch state."""
    import torch

    opponents = list(opponent_policies or [])
    use_history_population = len(opponents) > 0
    learner_seats = np.arange(int(n_games), dtype=np.int64) % 2
    states = [
        _new_seeded_fast_state(seed, game_i, int(initial_chips))
        for game_i in range(int(n_games))
    ]
    compiled = CompiledFastStateBatch.from_fast_states(states)
    steps_per_game = np.zeros(int(n_games), dtype=np.int32)
    records: list[tuple[int, int, int, np.ndarray, np.ndarray, int, float]] = []
    forward_calls = 0
    learner_forward_calls = 0
    opponent_forward_calls = 0
    learner_controlled_steps = 0
    opponent_controlled_steps = 0
    needs_python_showdown = 0
    started = time.perf_counter()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1000)

    while True:
        live_indices = [
            i
            for i in range(int(n_games))
            if int(compiled.stage[i]) < FastPokerState.SHOWDOWN
            and int(steps_per_game[i]) < int(max_steps_per_game)
        ]
        if not live_indices:
            break
        all_features = compiled_feature_vectors(compiled)
        all_masks = compiled_legal_masks(compiled)
        current_players = compiled.current_players()
        action_array = np.full(int(n_games), -1, dtype=np.int16)
        for start in range(0, len(live_indices), int(batch_size)):
            batch_indices = live_indices[start : start + int(batch_size)]
            grouped_indices: dict[int, list[int]] = {}
            for game_i in batch_indices:
                player_i = int(current_players[int(game_i)])
                if use_history_population and player_i != int(learner_seats[int(game_i)]):
                    key = int(game_i) % len(opponents)
                else:
                    key = -1
                grouped_indices.setdefault(key, []).append(int(game_i))

            for key, group_indices in grouped_indices.items():
                acting_policy = policy if key == -1 else opponents[int(key)]
                features = all_features[group_indices].astype(np.float32, copy=False)
                legal_masks = all_masks[group_indices].astype(np.float32, copy=False)
                with torch.no_grad():
                    x = torch.as_tensor(features, dtype=torch.float32, device=device)
                    masks = torch.as_tensor(legal_masks, dtype=torch.float32, device=device)
                    logits = acting_policy(x).masked_fill(masks <= 0, -1.0e30)
                    probs_t = torch.softmax(logits, dim=1)
                    legal_mass = masks.sum(dim=1, keepdim=True)
                    fallback = masks / torch.clamp(legal_mass, min=1.0)
                    bad_rows = (
                        (legal_mass.squeeze(1) <= 0)
                        | (~torch.isfinite(probs_t).all(dim=1))
                        | (probs_t < 0).any(dim=1)
                        | (probs_t.sum(dim=1) <= 0)
                    )
                    probs_t = torch.where(
                        bad_rows.unsqueeze(1),
                        fallback,
                        probs_t,
                    )
                    probs = probs_t.detach().cpu()
                    actions = np.full(len(group_indices), -1, dtype=np.int64)
                    legal_rows = (legal_mass.squeeze(1).detach().cpu().numpy() > 0)
                    if np.any(legal_rows):
                        sampled = (
                            torch.multinomial(
                                probs[legal_rows],
                                num_samples=1,
                                generator=generator,
                            )
                            .squeeze(1)
                            .numpy()
                        )
                        actions[legal_rows] = sampled
                forward_calls += 1
                if key == -1:
                    learner_forward_calls += 1
                else:
                    opponent_forward_calls += 1
                for row_i, game_i in enumerate(group_indices):
                    player_i = int(current_players[int(game_i)])
                    action_idx = int(actions[row_i])
                    if action_idx < 0:
                        steps_per_game[int(game_i)] = int(max_steps_per_game)
                        continue
                    action_prob = float(probs[row_i, action_idx])
                    action_array[int(game_i)] = action_idx
                    if key == -1:
                        learner_controlled_steps += 1
                        records.append(
                            (
                                int(game_i),
                                int(steps_per_game[int(game_i)]),
                                player_i,
                                features[row_i].astype(np.float32, copy=False),
                                legal_masks[row_i].astype(np.float32, copy=False),
                                action_idx,
                                float(np.log(max(action_prob, 1e-45))),
                            )
                        )
                    else:
                        opponent_controlled_steps += 1
                    steps_per_game[int(game_i)] += 1
        result = compiled_apply_actions(compiled, action_array)
        needs_python_showdown += int(result["needs_python_showdown"])

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    payoffs = (
        compiled.chips.astype(np.float32) - float(initial_chips)
    ) / float(initial_chips)
    terminal = compiled.stage >= FastPokerState.SHOWDOWN
    truncated_games = int(np.sum(~terminal))
    payoffs[~terminal, :] = 0.0

    if not records:
        batch = _empty_fast_self_play_batch()
        batch["payoffs"] = payoffs.astype(np.float32, copy=False)
    else:
        game_indices = np.array([r[0] for r in records], dtype=np.int64)
        step_indices = np.array([r[1] for r in records], dtype=np.int64)
        players = np.array([r[2] for r in records], dtype=np.int64)
        features = np.stack([r[3] for r in records]).astype(np.float32, copy=False)
        legal_masks = np.stack([r[4] for r in records]).astype(np.float32, copy=False)
        actions = np.array([r[5] for r in records], dtype=np.int64)
        old_log_probs = np.array([r[6] for r in records], dtype=np.float32)
        next_decision_indices = np.full(actions.shape, -1, dtype=np.int64)
        grouped_record_indices: dict[tuple[int, int], list[int]] = {}
        for record_i, (game_i, player_i) in enumerate(zip(game_indices, players, strict=False)):
            grouped_record_indices.setdefault((int(game_i), int(player_i)), []).append(int(record_i))
        for indices in grouped_record_indices.values():
            ordered = sorted(indices, key=lambda i: int(step_indices[i]))
            for current_i, next_i in zip(ordered[:-1], ordered[1:], strict=False):
                next_decision_indices[int(current_i)] = int(next_i)
        rewards = payoffs[game_indices, players].astype(np.float32, copy=False)
        batch = {
            "features": features,
            "critic_features": np.zeros((features.shape[0], CENTRALIZED_FAST_Q_FEATURES), dtype=np.float32),
            "legal_masks": legal_masks,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "players": players,
            "game_indices": game_indices,
            "step_indices": step_indices,
            "next_decision_indices": next_decision_indices,
            "dones": (next_decision_indices < 0).astype(np.float32),
            "rewards": rewards,
            "payoffs": payoffs.astype(np.float32, copy=False),
        }
    steps = int(batch["actions"].shape[0])
    total_env_steps = int(learner_controlled_steps + opponent_controlled_steps)
    metrics = {
        "collector": "compiled_fast_policy_gradient_rollout",
        "backend": "compiled-fast-state",
        "n_games": int(n_games),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "total_env_steps": int(total_env_steps),
        "learner_controlled_steps": int(learner_controlled_steps),
        "opponent_controlled_steps": int(opponent_controlled_steps),
        "opponent_population_size": int(len(opponents)),
        "trajectory_links": int(np.sum(batch["next_decision_indices"] >= 0)),
        "seconds": float(seconds),
        "steps_per_second": float(steps / max(seconds, 1e-12)),
        "env_steps_per_second": float(total_env_steps / max(seconds, 1e-12)),
        "policy_forward_calls": int(forward_calls),
        "learner_policy_forward_calls": int(learner_forward_calls),
        "opponent_policy_forward_calls": int(opponent_forward_calls),
        "mean_decisions_per_forward": float(total_env_steps / max(int(forward_calls), 1)),
        "training_decisions_per_forward": float(steps / max(int(learner_forward_calls), 1)),
        "mean_payoff_p0": float(np.mean(payoffs[:, 0])) if payoffs.size else 0.0,
        "truncated_games": int(truncated_games),
        "needs_python_showdown": int(needs_python_showdown),
    }
    return batch, metrics


def benchmark_batched_fast_self_play_collection(
    *,
    n_games: int = 256,
    batch_size: int = 64,
    max_steps_per_game: int = 128,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    device: str = "auto",
    seed: int = 20260808,
) -> dict[str, Any]:
    """Compare sequential and batched collection with the same deterministic policy."""
    sequential = collect_batched_fast_self_play(
        mode="sequential",
        n_games=int(n_games),
        batch_size=int(batch_size),
        max_steps_per_game=int(max_steps_per_game),
        initial_chips=int(initial_chips),
        hidden_dim=int(hidden_dim),
        device=device,
        seed=int(seed),
    )["metrics"]
    batched = collect_batched_fast_self_play(
        mode="batched",
        n_games=int(n_games),
        batch_size=int(batch_size),
        max_steps_per_game=int(max_steps_per_game),
        initial_chips=int(initial_chips),
        hidden_dim=int(hidden_dim),
        device=device,
        seed=int(seed),
    )["metrics"]
    return {
        "mode": "batched_fast_state_self_play_collection_ab",
        "sequential": sequential,
        "batched": batched,
        "speedup": float(
            batched["steps_per_second"] / max(float(sequential["steps_per_second"]), 1e-12)
        ),
        "batched_forward_call_reduction": float(
            sequential["policy_forward_calls"] / max(float(batched["policy_forward_calls"]), 1.0)
        ),
        "same_decision_trace": bool(
            sequential["steps"] == batched["steps"]
            and sequential["action_checksum"] == batched["action_checksum"]
            and abs(float(sequential["payoff_checksum"]) - float(batched["payoff_checksum"])) < 1e-6
        ),
        "uses_slumbot_training_data": False,
        "promotion": False,
    }


def run_batched_fast_policy_fit_smoke(
    *,
    n_games: int = 256,
    collector_batch_size: int = 64,
    train_batch_size: int = 512,
    max_steps_per_game: int = 128,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    train_steps: int = 100,
    lr: float = 1e-3,
    device: str = "auto",
    seed: int = 20260810,
) -> dict[str, Any]:
    """Fit a masked policy head on one batched collector output.

    This is a training-interface smoke only. The labels are the collector
    policy's own actions, so the result is not poker-strength evidence.
    """
    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    collected = collect_batched_fast_self_play(
        mode="batched",
        n_games=int(n_games),
        batch_size=int(collector_batch_size),
        max_steps_per_game=int(max_steps_per_game),
        initial_chips=int(initial_chips),
        hidden_dim=int(hidden_dim),
        device=device,
        seed=int(seed),
    )
    batch = collected["batch"]
    collector_metrics = collected["metrics"]
    resolved_device, device_info = _resolve_policy_inference_device(device)
    torch.manual_seed(int(seed) + 1)
    learner = _build_policy_inference_net(int(hidden_dim), resolved_device)
    optimizer = optim.Adam(learner.parameters(), lr=float(lr))
    features = torch.as_tensor(batch["features"], dtype=torch.float32, device=resolved_device)
    legal_masks = torch.as_tensor(batch["legal_masks"], dtype=torch.float32, device=resolved_device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=resolved_device)
    n_samples = int(actions.numel())
    if n_samples <= 0:
        raise ValueError("collector produced no samples")

    def _loss_for(indices: torch.Tensor | None = None) -> torch.Tensor:
        if indices is None:
            logits = learner(features)
            masks = legal_masks
            target = actions
        else:
            logits = learner(features[indices])
            masks = legal_masks[indices]
            target = actions[indices]
        logits = logits.masked_fill(masks <= 0, -1.0e30)
        return F.cross_entropy(logits, target)

    with torch.no_grad():
        initial_loss = float(_loss_for().detach().cpu())
    rng = np.random.default_rng(int(seed) + 17)
    started = time.perf_counter()
    last_loss = initial_loss
    for _ in range(int(train_steps)):
        indices_np = rng.choice(
            n_samples,
            size=min(max(int(train_batch_size), 1), n_samples),
            replace=n_samples < int(train_batch_size),
        )
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=resolved_device)
        loss = _loss_for(indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    with torch.no_grad():
        final_loss = float(_loss_for().detach().cpu())
    return {
        "mode": "batched_fast_state_policy_fit_smoke",
        "role": "training_interface_smoke",
        "warning": (
            "Fits a policy to synthetic collector actions to validate the batched "
            "training interface; not poker-strength evidence."
        ),
        "backend": "fast-state",
        "collector_steps": int(n_samples),
        "collector_steps_per_second": float(collector_metrics["steps_per_second"]),
        "collector_policy_forward_calls": int(collector_metrics["policy_forward_calls"]),
        "n_games": int(n_games),
        "collector_batch_size": int(collector_batch_size),
        "train_batch_size": int(train_batch_size),
        "hidden_dim": int(hidden_dim),
        "train_steps": int(train_steps),
        "train_seconds": float(train_seconds),
        "samples_per_second": float(
            int(train_steps) * min(max(int(train_batch_size), 1), n_samples)
            / max(train_seconds, 1e-12)
        ),
        "initial_loss": float(initial_loss),
        "last_minibatch_loss": float(last_loss),
        "final_loss": float(final_loss),
        "loss_delta": float(initial_loss - final_loss),
        "loss_decreased": bool(final_loss < initial_loss),
        "uses_slumbot_training_data": False,
        "promotion": False,
        **device_info,
    }


def _collect_batched_fast_policy_gradient_rollout(
    policy,
    *,
    n_games: int,
    batch_size: int,
    max_steps_per_game: int,
    initial_chips: int,
    device,
    seed: int,
    opponent_policies: list[Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    opponents = list(opponent_policies or [])
    use_history_population = len(opponents) > 0
    learner_seats = np.arange(int(n_games), dtype=np.int64) % 2
    states = [
        _new_seeded_fast_state(seed, game_i, int(initial_chips))
        for game_i in range(int(n_games))
    ]
    steps_per_game = np.zeros(int(n_games), dtype=np.int32)
    records: list[tuple[int, int, int, np.ndarray, np.ndarray, int]] = []
    forward_calls = 0
    learner_forward_calls = 0
    opponent_forward_calls = 0
    learner_controlled_steps = 0
    opponent_controlled_steps = 0
    started = time.perf_counter()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1000)

    while True:
        live_indices = [
            i
            for i, state in enumerate(states)
            if not state.is_terminal and int(steps_per_game[i]) < int(max_steps_per_game)
        ]
        if not live_indices:
            break
        for start in range(0, len(live_indices), int(batch_size)):
            batch_indices = live_indices[start : start + int(batch_size)]
            grouped_indices: dict[int, list[int]] = {}
            for game_i in batch_indices:
                state = states[int(game_i)]
                player_i = int(state.current_player_i)
                if use_history_population and player_i != int(learner_seats[int(game_i)]):
                    key = int(game_i) % len(opponents)
                else:
                    key = -1
                grouped_indices.setdefault(key, []).append(int(game_i))

            for key, group_indices in grouped_indices.items():
                acting_policy = policy if key == -1 else opponents[int(key)]
                features = np.stack(
                    [
                        states[i].to_feature_vector().astype(np.float32, copy=False)
                        for i in group_indices
                    ]
                )
                legal_masks = np.stack(
                    [
                        states[i].get_legal_mask().astype(np.float32, copy=False)
                        for i in group_indices
                    ]
                )
                with torch.no_grad():
                    x = torch.as_tensor(features, dtype=torch.float32, device=device)
                    masks = torch.as_tensor(legal_masks, dtype=torch.float32, device=device)
                    logits = acting_policy(x).masked_fill(masks <= 0, -1.0e30)
                    probs_t = torch.softmax(logits, dim=1)
                    legal_mass = masks.sum(dim=1, keepdim=True)
                    fallback = masks / torch.clamp(legal_mass, min=1.0)
                    bad_rows = (
                        (legal_mass.squeeze(1) <= 0)
                        | (~torch.isfinite(probs_t).all(dim=1))
                        | (probs_t < 0).any(dim=1)
                        | (probs_t.sum(dim=1) <= 0)
                    )
                    probs_t = torch.where(
                        bad_rows.unsqueeze(1),
                        fallback,
                        probs_t,
                    )
                    probs = probs_t.detach().cpu()
                    actions = np.full(len(group_indices), -1, dtype=np.int64)
                    legal_rows = (legal_mass.squeeze(1).detach().cpu().numpy() > 0)
                    if np.any(legal_rows):
                        sampled = (
                            torch.multinomial(
                                probs[legal_rows],
                                num_samples=1,
                                generator=generator,
                            )
                            .squeeze(1)
                            .numpy()
                        )
                        actions[legal_rows] = sampled
                forward_calls += 1
                if key == -1:
                    learner_forward_calls += 1
                else:
                    opponent_forward_calls += 1
                for row_i, game_i in enumerate(group_indices):
                    state = states[int(game_i)]
                    player_i = int(state.current_player_i)
                    action_idx = int(actions[row_i])
                    if action_idx < 0:
                        steps_per_game[int(game_i)] = int(max_steps_per_game)
                        continue
                    action_prob = float(probs[row_i, action_idx])
                    if key == -1:
                        learner_controlled_steps += 1
                        records.append(
                            (
                                int(game_i),
                                int(steps_per_game[int(game_i)]),
                                player_i,
                                features[row_i].astype(np.float32, copy=False),
                                _fast_state_centralized_q_features(
                                    state,
                                    features[row_i],
                                ),
                                legal_masks[row_i].astype(np.float32, copy=False),
                                action_idx,
                                float(np.log(max(action_prob, 1e-45))),
                            )
                        )
                    else:
                        opponent_controlled_steps += 1
                    state.apply_action(action_idx)
                    steps_per_game[int(game_i)] += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    payoffs = np.zeros((int(n_games), 2), dtype=np.float32)
    truncated_games = 0
    for game_i, state in enumerate(states):
        payoff, truncated = _final_fast_state_payoffs(state, int(initial_chips))
        payoffs[game_i, : payoff.shape[0]] = payoff[:2]
        truncated_games += int(truncated)

    if not records:
        batch = _empty_fast_self_play_batch()
        batch["payoffs"] = payoffs
    else:
        game_indices = np.array([r[0] for r in records], dtype=np.int64)
        step_indices = np.array([r[1] for r in records], dtype=np.int64)
        players = np.array([r[2] for r in records], dtype=np.int64)
        features = np.stack([r[3] for r in records]).astype(np.float32, copy=False)
        critic_features = np.stack([r[4] for r in records]).astype(np.float32, copy=False)
        legal_masks = np.stack([r[5] for r in records]).astype(np.float32, copy=False)
        actions = np.array([r[6] for r in records], dtype=np.int64)
        old_log_probs = np.array([r[7] for r in records], dtype=np.float32)
        next_decision_indices = np.full(actions.shape, -1, dtype=np.int64)
        grouped_record_indices: dict[tuple[int, int], list[int]] = {}
        for record_i, (game_i, player_i) in enumerate(zip(game_indices, players, strict=False)):
            grouped_record_indices.setdefault((int(game_i), int(player_i)), []).append(int(record_i))
        for indices in grouped_record_indices.values():
            ordered = sorted(indices, key=lambda i: int(step_indices[i]))
            for current_i, next_i in zip(ordered[:-1], ordered[1:], strict=False):
                next_decision_indices[int(current_i)] = int(next_i)
        dones = (next_decision_indices < 0).astype(np.float32)
        rewards = payoffs[game_indices, players].astype(np.float32, copy=False)
        batch = {
            "features": features,
            "critic_features": critic_features,
            "legal_masks": legal_masks,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "players": players,
            "game_indices": game_indices,
            "step_indices": step_indices,
            "next_decision_indices": next_decision_indices,
            "dones": dones,
            "rewards": rewards,
            "payoffs": payoffs,
        }
    steps = int(batch["actions"].shape[0])
    total_env_steps = int(learner_controlled_steps + opponent_controlled_steps)
    metrics = {
        "collector": "batched_fast_policy_gradient_rollout",
        "backend": "fast-state",
        "n_games": int(n_games),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "total_env_steps": int(total_env_steps),
        "learner_controlled_steps": int(learner_controlled_steps),
        "opponent_controlled_steps": int(opponent_controlled_steps),
        "opponent_population_size": int(len(opponents)),
        "trajectory_links": int(np.sum(batch["next_decision_indices"] >= 0)),
        "seconds": float(seconds),
        "steps_per_second": float(steps / max(seconds, 1e-12)),
        "env_steps_per_second": float(total_env_steps / max(seconds, 1e-12)),
        "policy_forward_calls": int(forward_calls),
        "learner_policy_forward_calls": int(learner_forward_calls),
        "opponent_policy_forward_calls": int(opponent_forward_calls),
        "mean_decisions_per_forward": float(total_env_steps / max(int(forward_calls), 1)),
        "training_decisions_per_forward": float(steps / max(int(learner_forward_calls), 1)),
        "mean_payoff_p0": float(np.mean(payoffs[:, 0])) if payoffs.size else 0.0,
        "truncated_games": int(truncated_games),
    }
    return batch, metrics


def _clone_policy_for_history(policy, *, device):
    frozen = copy.deepcopy(policy).to(device)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def _q_boost_targets_and_advantages(
    *,
    policy,
    q_net,
    features,
    q_features,
    legal_masks,
    actions,
    rewards,
    next_decision_indices: np.ndarray,
    game_indices: np.ndarray,
    players: np.ndarray,
    step_indices: np.ndarray,
    gamma: float,
    trace_lambda: float,
) -> tuple[Any, Any]:
    """Compute a small Expected-SARSA(lambda)-style target for learner decisions."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = policy(features).masked_fill(legal_masks <= 0, -1.0e30)
        probs = F.softmax(logits, dim=1)
        q_all = q_net(q_features)
        q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        expected_q = (probs * q_all).sum(dim=1)
        traces = torch.zeros_like(rewards)
        grouped: dict[tuple[int, int], list[int]] = {}
        for record_i, (game_i, player_i) in enumerate(zip(game_indices, players, strict=False)):
            grouped.setdefault((int(game_i), int(player_i)), []).append(int(record_i))
        for indices in grouped.values():
            ordered = sorted(indices, key=lambda i: int(step_indices[i]), reverse=True)
            for record_i in ordered:
                next_i = int(next_decision_indices[record_i])
                if next_i >= 0:
                    delta = float(gamma) * expected_q[next_i] - q_selected[record_i]
                    traces[record_i] = delta + float(gamma) * float(trace_lambda) * traces[next_i]
                else:
                    traces[record_i] = rewards[record_i] - q_selected[record_i]
        q_targets = q_selected + traces
        advantages = q_selected - expected_q + traces
        std = advantages.std(unbiased=False)
        if torch.isfinite(std) and float(std.detach().cpu()) > 1e-6:
            advantages = (advantages - advantages.mean()) / std.clamp_min(1e-6)
        else:
            advantages = advantages - advantages.mean()
    return q_targets.detach(), advantages.detach()


def run_batched_fast_policy_gradient_pilot(
    *,
    train_iterations: int = 4,
    games_per_iteration: int = 256,
    collector_batch_size: int = 64,
    train_batch_size: int = 512,
    train_epochs: int = 2,
    max_steps_per_game: int = 128,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    entropy_weight: float = 0.01,
    value_loss_weight: float = 0.0,
    q_boost_lambda: float | None = None,
    q_loss_weight: float = 1.0,
    ppo_clip_epsilon: float = 0.0,
    gamma: float = 1.0,
    centralized_q_critic: bool = False,
    device: str = "auto",
    seed: int = 20260811,
    checkpoint_path: str | None = None,
    history_opponent_interval: int = 0,
    history_opponent_capacity: int = 8,
    rollout_backend: str = "fast-state",
) -> dict[str, Any]:
    """Train a small shared policy-gradient self-play pilot on batched fast states."""
    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    from poker_ai.research.native_ppo_policy import _PolicyMLP, _QMLP, _ValueMLP

    rollout_backend = str(rollout_backend).strip().lower()
    if rollout_backend not in {"fast-state", "compiled"}:
        raise ValueError("rollout_backend must be one of: fast-state, compiled")
    resolved_device, device_info = _resolve_policy_inference_device(device)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    policy = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    use_value_baseline = float(value_loss_weight) > 0.0
    use_q_boosting = q_boost_lambda is not None and float(q_boost_lambda) >= 0.0
    value_net = (
        _ValueMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
        if use_value_baseline and not use_q_boosting
        else None
    )
    q_net = (
        _QMLP(
            int(hidden_dim),
            input_dim=CENTRALIZED_FAST_Q_FEATURES if bool(centralized_q_critic) else N_FEATURES,
        ).to(resolved_device)
        if use_q_boosting
        else None
    )
    q_input_dim = CENTRALIZED_FAST_Q_FEATURES if bool(centralized_q_critic) else N_FEATURES
    optimizer_params = list(policy.parameters())
    if value_net is not None:
        optimizer_params.extend(value_net.parameters())
    if q_net is not None:
        optimizer_params.extend(q_net.parameters())
    optimizer = optim.Adam(optimizer_params, lr=float(lr))
    rng = np.random.default_rng(int(seed) + 2000)
    total_collector_steps = 0
    total_policy_updates = 0
    losses: list[float] = []
    entropies: list[float] = []
    value_losses: list[float] = []
    q_losses: list[float] = []
    collector_rates: list[float] = []
    env_collector_rates: list[float] = []
    total_env_steps = 0
    learner_controlled_steps = 0
    opponent_controlled_steps = 0
    history_opponents: list[Any] = []
    history_snapshots_added = 0
    compiled_needs_python_showdown = 0
    train_started = time.perf_counter()

    for iteration in range(int(train_iterations)):
        collector_fn = (
            _collect_compiled_fast_policy_gradient_rollout
            if rollout_backend == "compiled"
            else _collect_batched_fast_policy_gradient_rollout
        )
        batch, collector_metrics = collector_fn(
            policy,
            n_games=int(games_per_iteration),
            batch_size=int(collector_batch_size),
            max_steps_per_game=int(max_steps_per_game),
            initial_chips=int(initial_chips),
            device=resolved_device,
            seed=int(seed) + iteration * 10_000,
            opponent_policies=history_opponents if history_opponents else None,
        )
        compiled_needs_python_showdown += int(collector_metrics.get("needs_python_showdown", 0))
        n = int(batch["actions"].shape[0])
        total_env_steps += int(collector_metrics.get("total_env_steps", n))
        learner_controlled_steps += int(collector_metrics.get("learner_controlled_steps", n))
        opponent_controlled_steps += int(collector_metrics.get("opponent_controlled_steps", 0))
        if n <= 0:
            if (
                int(history_opponent_interval) > 0
                and (iteration + 1) % int(history_opponent_interval) == 0
            ):
                history_opponents.append(_clone_policy_for_history(policy, device=resolved_device))
                history_snapshots_added += 1
                history_opponents = history_opponents[-max(int(history_opponent_capacity), 1) :]
            continue
        total_collector_steps += n
        collector_rates.append(float(collector_metrics["steps_per_second"]))
        env_collector_rates.append(float(collector_metrics.get("env_steps_per_second", 0.0)))
        features = torch.as_tensor(batch["features"], dtype=torch.float32, device=resolved_device)
        legal_masks = torch.as_tensor(batch["legal_masks"], dtype=torch.float32, device=resolved_device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=resolved_device)
        returns = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=resolved_device)
        critic_features = torch.as_tensor(
            batch["critic_features"],
            dtype=torch.float32,
            device=resolved_device,
        )
        q_features = critic_features if bool(centralized_q_critic) else features
        old_log_probs = torch.as_tensor(
            batch["old_log_probs"],
            dtype=torch.float32,
            device=resolved_device,
        )
        if q_net is not None:
            q_targets, q_boost_advantages = _q_boost_targets_and_advantages(
                policy=policy,
                q_net=q_net,
                features=features,
                q_features=q_features,
                legal_masks=legal_masks,
                actions=actions,
                rewards=returns,
                next_decision_indices=batch["next_decision_indices"],
                game_indices=batch["game_indices"],
                players=batch["players"],
                step_indices=batch["step_indices"],
                gamma=float(gamma),
                trace_lambda=float(q_boost_lambda),
            )
        else:
            q_targets = None
            q_boost_advantages = None
        batch_size = min(max(int(train_batch_size), 1), n)
        for _epoch in range(int(train_epochs)):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx_np = order[start : start + batch_size]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=resolved_device)
                logits = policy(features[idx]).masked_fill(legal_masks[idx] <= 0, -1.0e30)
                log_probs = F.log_softmax(logits, dim=1)
                probs = torch.exp(log_probs)
                selected_log_probs = log_probs.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                entropy = -(probs * log_probs).sum(dim=1).mean()
                if q_net is not None:
                    q_values = q_net(q_features[idx])
                    q_selected = q_values.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                    q_loss = F.mse_loss(q_selected, q_targets[idx])
                    advantages = q_boost_advantages[idx]
                    if float(ppo_clip_epsilon) > 0.0:
                        ratio = torch.exp(selected_log_probs - old_log_probs[idx])
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - float(ppo_clip_epsilon),
                            1.0 + float(ppo_clip_epsilon),
                        )
                        policy_loss = -torch.minimum(
                            ratio * advantages,
                            clipped_ratio * advantages,
                        ).mean()
                    else:
                        policy_loss = -(selected_log_probs * advantages).mean()
                    loss = (
                        policy_loss
                        + float(q_loss_weight) * q_loss
                        - float(entropy_weight) * entropy
                    )
                elif value_net is not None:
                    values = value_net(features[idx])
                    value_loss = F.mse_loss(values, returns[idx])
                    advantages = returns[idx] - values.detach()
                    std = advantages.std(unbiased=False)
                    if torch.isfinite(std) and float(std.detach().cpu()) > 1e-6:
                        advantages = advantages / std.clamp_min(1e-6)
                    policy_loss = -(selected_log_probs * advantages).mean()
                    loss = (
                        policy_loss
                        + float(value_loss_weight) * value_loss
                        - float(entropy_weight) * entropy
                    )
                else:
                    value_loss = torch.zeros((), dtype=torch.float32, device=resolved_device)
                    advantages = returns[idx] - returns.mean()
                    std = advantages.std(unbiased=False)
                    if torch.isfinite(std) and float(std.detach().cpu()) > 1e-6:
                        advantages = advantages / std.clamp_min(1e-6)
                    policy_loss = -(selected_log_probs * advantages).mean()
                    loss = policy_loss - float(entropy_weight) * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_policy_updates += 1
                losses.append(float(loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                if value_net is not None:
                    value_losses.append(float(value_loss.detach().cpu()))
                if q_net is not None:
                    q_losses.append(float(q_loss.detach().cpu()))
        if (
            int(history_opponent_interval) > 0
            and (iteration + 1) % int(history_opponent_interval) == 0
        ):
            history_opponents.append(_clone_policy_for_history(policy, device=resolved_device))
            history_snapshots_added += 1
            history_opponents = history_opponents[-max(int(history_opponent_capacity), 1) :]

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    metrics = {
        "algorithm": "batched_fast_policy_gradient",
        "role": "batched_native_self_play_policy_gradient_pilot",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": (
            "Simple shared policy-gradient self-play pilot on the batched fast-state "
            "substrate. This is a falsifiable local candidate, not Slumbot/SOTA evidence."
        ),
        **device_info,
        "uses_slumbot_training_data": False,
        "promotion": False,
        "rollout_backend": rollout_backend,
        "collector_backend": "compiled-fast-state" if rollout_backend == "compiled" else "fast-state",
        "compiled_needs_python_showdown": int(compiled_needs_python_showdown),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "hidden_dim": int(hidden_dim),
        "train_iterations": int(train_iterations),
        "games_per_iteration": int(games_per_iteration),
        "collector_batch_size": int(collector_batch_size),
        "train_batch_size": int(train_batch_size),
        "train_epochs": int(train_epochs),
        "entropy_weight": float(entropy_weight),
        "value_loss_weight": float(value_loss_weight),
        "uses_value_baseline": bool(use_value_baseline),
        "value_updates": int(len(value_losses)),
        "uses_q_boosting": bool(use_q_boosting),
        "q_boost_lambda": float(q_boost_lambda) if q_boost_lambda is not None else None,
        "q_loss_weight": float(q_loss_weight),
        "ppo_clip_epsilon": float(ppo_clip_epsilon),
        "gamma": float(gamma),
        "uses_centralized_q_critic": bool(centralized_q_critic and use_q_boosting),
        "q_input_dim": int(q_input_dim),
        "q_updates": int(len(q_losses)),
        "max_steps_per_game": int(max_steps_per_game),
        "initial_chips": int(initial_chips),
        "total_collector_steps": int(total_collector_steps),
        "total_env_steps": int(total_env_steps),
        "learner_controlled_steps": int(learner_controlled_steps),
        "opponent_controlled_steps": int(opponent_controlled_steps),
        "history_opponent_interval": int(history_opponent_interval),
        "history_opponent_capacity": int(history_opponent_capacity),
        "history_snapshots_added": int(history_snapshots_added),
        "history_population_size": int(len(history_opponents)),
        "train_opponent_mode": (
            "history_population" if int(history_opponent_interval) > 0 else "self_play"
        ),
        "total_policy_updates": int(total_policy_updates),
        "train_seconds": float(train_seconds),
        "collector_steps_per_second": float(
            total_collector_steps / max(train_seconds, 1e-12)
        ),
        "env_steps_per_second": float(total_env_steps / max(train_seconds, 1e-12)),
        "mean_rollout_steps_per_second": float(np.mean(collector_rates)) if collector_rates else 0.0,
        "mean_env_rollout_steps_per_second": (
            float(np.mean(env_collector_rates)) if env_collector_rates else 0.0
        ),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "first_value_loss": value_losses[0] if value_losses else None,
        "last_value_loss": value_losses[-1] if value_losses else None,
        "first_q_loss": q_losses[0] if q_losses else None,
        "last_q_loss": q_losses[-1] if q_losses else None,
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    if checkpoint_path:
        import torch
        from pathlib import Path

        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "batched_fast_policy_gradient",
                "environment": "poker_ai:full_deck_hu_nlhe",
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "q_input_dim": int(q_input_dim),
                "hidden_dim": int(hidden_dim),
                "policy_net_state_dict": policy.state_dict(),
                "value_net_state_dict": (
                    value_net.state_dict() if value_net is not None else None
                ),
                "q_net_state_dict": q_net.state_dict() if q_net is not None else None,
                "config": {
                    "feature_mode": "flat",
                    "hidden_dim": int(hidden_dim),
                    "initial_chips": int(initial_chips),
                    "max_steps_per_hand": int(max_steps_per_game),
                    "fsp_average_policy": False,
                    "centralized_q_critic": bool(centralized_q_critic and use_q_boosting),
                    "rollout_backend": rollout_backend,
                },
                "metrics": metrics,
            },
            path,
        )
    return metrics


def summarize_native_rollout_gate(
    *,
    parity: dict[str, Any],
    policy_parity: dict[str, Any] | None = None,
    baseline_steps_per_second: float,
    candidate_steps_per_second: float,
    min_speedup: float = 5.0,
) -> dict[str, Any]:
    speedup = float(candidate_steps_per_second) / max(float(baseline_steps_per_second), 1e-12)
    parity_passed = bool(parity.get("passed", False))
    policy_parity_passed = True if policy_parity is None else bool(policy_parity.get("passed", False))
    speedup_passed = speedup >= float(min_speedup)
    return {
        "passed": bool(parity_passed and policy_parity_passed and speedup_passed),
        "parity_passed": parity_passed,
        "policy_parity_passed": bool(policy_parity_passed),
        "speedup_passed": bool(speedup_passed),
        "speedup": float(speedup),
        "min_speedup": float(min_speedup),
        "baseline_steps_per_second": float(baseline_steps_per_second),
        "candidate_steps_per_second": float(candidate_steps_per_second),
        "checked_steps": int(parity.get("checked_steps", 0)),
        "checked_policy_decisions": int((policy_parity or {}).get("checked_decisions", 0)),
        "mismatches": list(parity.get("mismatches", [])),
        "policy_mismatches": list((policy_parity or {}).get("mismatches", [])),
    }


def evaluate_native_rollout_substrate(
    *,
    n_parity_games: int = 16,
    parity_max_steps: int = 64,
    n_benchmark_games: int = 256,
    benchmark_max_steps: int = 128,
    initial_chips: int = 1000,
    seed: int = 20260752,
    min_speedup: float = 5.0,
) -> dict[str, Any]:
    parity = run_fast_state_parity_check(
        n_games=int(n_parity_games),
        max_steps_per_game=int(parity_max_steps),
        initial_chips=int(initial_chips),
        seed=int(seed),
    )
    policy_parity = run_fast_state_policy_replay_parity(
        n_games=int(n_parity_games),
        max_steps_per_game=int(parity_max_steps),
        initial_chips=int(initial_chips),
        seed=int(seed),
    )
    baseline = run_rollout_throughput(
        backend="python-full-deck",
        n_games=int(n_benchmark_games),
        max_steps_per_game=int(benchmark_max_steps),
        initial_chips=int(initial_chips),
        seed=int(seed) + 1,
    )
    candidate = run_rollout_throughput(
        backend="fast-state",
        n_games=int(n_benchmark_games),
        max_steps_per_game=int(benchmark_max_steps),
        initial_chips=int(initial_chips),
        seed=int(seed) + 1,
    )
    gate = summarize_native_rollout_gate(
        parity=parity,
        policy_parity=policy_parity,
        baseline_steps_per_second=float(baseline["steps_per_second"]),
        candidate_steps_per_second=float(candidate["steps_per_second"]),
        min_speedup=float(min_speedup),
    )
    return {
        "mode": "native_rollout_substrate_gate",
        "candidate_backend": "fast-state",
        "baseline_backend": "python-full-deck",
        "parity": parity,
        "policy_parity": policy_parity,
        "baseline": baseline,
        "candidate": candidate,
        "gate": gate,
        "passed": bool(gate["passed"]),
        "warning": (
            "Passing this gate proves a faithful faster rollout substrate only; "
            "it is not a poker-strength or Slumbot promotion claim."
        ),
    }
