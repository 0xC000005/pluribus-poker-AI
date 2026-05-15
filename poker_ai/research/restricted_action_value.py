"""Restricted early-action value diagnostics.

This module is deliberately not an exploitability estimator. It scores the
current legal actions under a simple showdown abstraction: the opponent calls
non-fold actions, then both players check to showdown. The purpose is to give
autoresearch a positive-control-first diagnostic before trusting expensive
checkpoint comparisons.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import torch

from poker_ai.games.full_deck.state import (
    ACTION_TO_INDEX,
    INDEX_TO_ACTION,
    RAISE_FRACTIONS,
    PokerState,
    new_game,
)
from poker_ai.poker.card import Card, get_all_suits
from poker_ai.poker.evaluation.evaluator import Evaluator
from poker_ai.research.evaluation import (
    _strategies_from_network,
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.native_nfsp import get_legal_mask, resolve_device


@dataclass(frozen=True)
class RestrictedActionValueConfig:
    n_roots: int = 64
    n_equity_samples: int = 512
    initial_chips: int = 1000
    br_player: int = 0
    seed: int = 20260515
    include_positive_controls: bool = True
    checkpoint: str | None = None
    strategy_source: str = "regret"
    device: str = "auto"


def _full_deck() -> list[Card]:
    return [
        Card(rank, suit)
        for suit in sorted(get_all_suits())
        for rank in range(2, 15)
    ]


def sample_seeded_hole_cards(*, seed: int, root_idx: int) -> tuple[Card, Card]:
    """Return deterministic hero hole cards for comparable root samples."""
    deck = _full_deck()
    seed_sequence = np.random.SeedSequence([int(seed), int(root_idx)])
    rng = np.random.default_rng(seed_sequence)
    indices = rng.choice(len(deck), size=2, replace=False)
    return deck[int(indices[0])], deck[int(indices[1])]


def _known_cards(state: PokerState, player_i: int) -> list[Card]:
    cards = list(state.players[int(player_i)].cards)
    cards.extend(state.community_cards)
    return cards


def estimate_showdown_equity(
    state: PokerState,
    *,
    player_i: int,
    n_samples: int,
    seed: int,
) -> float:
    """Estimate info-set equity versus a random opponent hand/range."""
    player_i = int(player_i)
    hero_cards = list(state.players[player_i].cards)
    board = list(state.community_cards)
    if len(hero_cards) != 2:
        raise ValueError("showdown equity requires exactly two hero cards")
    if len(board) > 5:
        raise ValueError("community board cannot contain more than five cards")

    known = set(_known_cards(state, player_i))
    unknown_deck = [card for card in _full_deck() if card not in known]
    n_board_needed = 5 - len(board)
    n_draw = 2 + n_board_needed
    if len(unknown_deck) < n_draw:
        raise ValueError("not enough unknown cards to sample opponent and board")

    rng = np.random.default_rng(int(seed))
    evaluator = Evaluator()
    wins = 0.0
    n_samples = max(int(n_samples), 1)
    hero_eval = [card.eval_card for card in hero_cards]
    board_eval_prefix = [card.eval_card for card in board]
    for _ in range(n_samples):
        indices = rng.choice(len(unknown_deck), size=n_draw, replace=False)
        opponent = [unknown_deck[int(indices[0])], unknown_deck[int(indices[1])]]
        board_tail = [unknown_deck[int(i)] for i in indices[2:]]
        board_eval = board_eval_prefix + [card.eval_card for card in board_tail]
        hero_rank = evaluator.evaluate(board_eval, hero_eval)
        opponent_rank = evaluator.evaluate(
            board_eval,
            [card.eval_card for card in opponent],
        )
        if hero_rank < opponent_rank:
            wins += 1.0
        elif hero_rank == opponent_rank:
            wins += 0.5
    return float(wins / n_samples)


def _raise_contribution(state: PokerState, action: str) -> int:
    player = state.current_player
    biggest_bet = max(p.n_bet_chips for p in state.players)
    to_call = max(0, biggest_bet - player.n_bet_chips)
    if action == "call":
        return min(to_call, player.n_chips)
    if action == "all_in":
        return player.n_chips
    if action.startswith("raise_"):
        frac = float(action.split("_")[1])
        contribution = int(frac * state._table.pot.total) + to_call
        if to_call <= 0:
            min_raise = state.big_blind
        else:
            min_raise = to_call + max(to_call, state.big_blind)
        return min(max(contribution, min_raise), player.n_chips)
    raise ValueError(f"unsupported non-showdown action: {action}")


def _payoff_for_called_action(state: PokerState, action: str, equity: float) -> float:
    player = state.current_player
    initial = float(state._initial_n_chips)
    if action == "fold":
        return float(player.n_chips - initial)

    contribution = float(_raise_contribution(state, action))
    player_bet_after = float(player.n_bet_chips) + contribution
    opponent_call = 0.0
    for opponent in state.players:
        if opponent is player or not opponent.is_active:
            continue
        call_gap = max(0.0, player_bet_after - opponent.n_bet_chips)
        opponent_call = max(
            opponent_call,
            min(call_gap, float(opponent.n_chips)),
        )
    final_pot = float(state._table.pot.total) + contribution + opponent_call
    final_stack_ev = float(player.n_chips) - contribution + float(equity) * final_pot
    return float(final_stack_ev - initial)


def score_legal_actions_by_showdown_equity(
    state: PokerState,
    *,
    player_i: int | None = None,
    n_equity_samples: int = 512,
    seed: int = 20260515,
) -> dict[str, Any]:
    """Score current legal actions by info-set equity and immediate pot odds."""
    acting_player = int(state.player_i)
    if player_i is not None and int(player_i) != acting_player:
        raise ValueError(
            f"state current player is {acting_player}, cannot score player {player_i}"
        )
    equity = estimate_showdown_equity(
        state,
        player_i=acting_player,
        n_samples=int(n_equity_samples),
        seed=int(seed),
    )
    action_values: dict[str, float] = {}
    for action in state.legal_actions:
        if action is None:
            continue
        action_values[str(action)] = _payoff_for_called_action(state, str(action), equity)
    if not action_values:
        raise ValueError("state has no indexed legal actions to score")
    best_action = max(action_values, key=action_values.__getitem__)
    return {
        "player_i": acting_player,
        "street": state.betting_stage,
        "equity": float(equity),
        "action_values": action_values,
        "best_action": best_action,
        "best_action_value": float(action_values[best_action]),
    }


def _set_private_cards_for_control(
    state: PokerState,
    player_i: int,
    cards: list[Card],
) -> None:
    state.players[int(player_i)].cards = list(cards)


def _positive_controls(
    n_equity_samples: int,
    initial_chips: int,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    premium = new_game(2, initial_chips=int(initial_chips))
    _set_private_cards_for_control(
        premium,
        0,
        [Card("ace", "spades"), Card("ace", "hearts")],
    )
    premium_values = score_legal_actions_by_showdown_equity(
        premium,
        player_i=0,
        n_equity_samples=n_equity_samples,
        seed=seed,
    )

    random.seed(seed + 1)
    np.random.seed(seed + 1)
    trash = new_game(2, initial_chips=int(initial_chips))
    _set_private_cards_for_control(
        trash,
        1,
        [Card("7", "clubs"), Card("2", "diamonds")],
    )
    trash = trash.apply_action("raise_2.0")
    trash_values = score_legal_actions_by_showdown_equity(
        trash,
        player_i=1,
        n_equity_samples=n_equity_samples,
        seed=seed + 1,
    )

    return {
        "premium_pressure": {
            "passed": bool(
                premium_values["best_action"] == "all_in"
                and premium_values["action_values"]["all_in"]
                > premium_values["action_values"]["call"]
                and premium_values["equity"] > 0.75
            ),
            **premium_values,
        },
        "trash_vs_large_raise": {
            "passed": bool(
                trash_values["best_action"] == "fold"
                and trash_values["action_values"]["fold"]
                > trash_values["action_values"]["call"]
                and trash_values["equity"] < 0.45
            ),
            **trash_values,
        },
    }


def _checkpoint_strategy(
    loaded: Any,
    state: PokerState,
    device: torch.device,
    *,
    strategy_source: str,
) -> np.ndarray:
    features = state.to_feature_vector().reshape(1, -1)
    legal_mask = get_legal_mask(state)
    return np.asarray(
        _strategies_from_network(
            loaded.value_net,
            features,
            [legal_mask],
            device,
            strategy_source=strategy_source,
            checkpoint_metadata=loaded.metadata,
        )[0],
        dtype=np.float64,
    )


def _checkpoint_advantages(
    loaded: Any,
    state: PokerState,
    device: torch.device,
) -> np.ndarray:
    features = state.to_feature_vector().reshape(1, -1).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(features).to(device)
        return loaded.value_net(tensor).detach().cpu().numpy().reshape(-1)


def _advantage_value_corr(
    advantages: np.ndarray,
    action_values: dict[str, float],
) -> float:
    legal_indices = [ACTION_TO_INDEX[action] for action in action_values]
    adv = np.asarray(advantages, dtype=np.float64)[legal_indices]
    vals = np.asarray([action_values[INDEX_TO_ACTION[idx]] for idx in legal_indices])
    if float(np.std(adv)) <= 1e-12 or float(np.std(vals)) <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(adv, vals)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def evaluate_restricted_action_values(cfg: RestrictedActionValueConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    loaded_checkpoint = None
    torch_device = None
    device_info: dict[str, Any] = {}
    if cfg.checkpoint:
        device_info = resolve_device(cfg.device)
        torch_device = torch.device(device_info["resolved_device"])
        loaded_checkpoint = load_value_network_checkpoint(cfg.checkpoint, torch_device)
        assert_strategy_source_supported(loaded_checkpoint, cfg.strategy_source)

    started = time.perf_counter()
    action_values_by_name: dict[str, list[float]] = defaultdict(list)
    best_action_counts: Counter[str] = Counter()
    oracle_values: list[float] = []
    call_values: list[float] = []
    checkpoint_action_counts: Counter[str] = Counter()
    checkpoint_selected_values: list[float] = []
    checkpoint_policy_values: list[float] = []
    checkpoint_oracle_gaps: list[float] = []
    checkpoint_matches: list[bool] = []
    checkpoint_advantage_value_corrs: list[float] = []

    for root_idx in range(max(int(cfg.n_roots), 1)):
        state = new_game(2, initial_chips=int(cfg.initial_chips))
        if int(cfg.br_player) != int(state.player_i):
            raise ValueError("restricted evaluator currently expects br_player to act at root")
        _set_private_cards_for_control(
            state,
            int(cfg.br_player),
            list(sample_seeded_hole_cards(seed=int(cfg.seed), root_idx=root_idx)),
        )
        values = score_legal_actions_by_showdown_equity(
            state,
            player_i=int(cfg.br_player),
            n_equity_samples=int(cfg.n_equity_samples),
            seed=int(cfg.seed) + 10_000 + root_idx,
        )
        for action, payoff in values["action_values"].items():
            action_values_by_name[action].append(float(payoff))
        best_action_counts[str(values["best_action"])] += 1
        oracle_values.append(float(values["best_action_value"]))
        if "call" in values["action_values"]:
            call_values.append(float(values["action_values"]["call"]))
        if loaded_checkpoint is not None and torch_device is not None:
            strategy = _checkpoint_strategy(
                loaded_checkpoint,
                state,
                torch_device,
                strategy_source=cfg.strategy_source,
            )
            advantages = _checkpoint_advantages(loaded_checkpoint, state, torch_device)
            selected_idx = int(np.argmax(strategy))
            selected_action = INDEX_TO_ACTION[selected_idx]
            selected_value = float(values["action_values"][selected_action])
            policy_value = 0.0
            for action, payoff in values["action_values"].items():
                policy_value += float(strategy[ACTION_TO_INDEX[action]]) * float(payoff)
            checkpoint_action_counts[selected_action] += 1
            checkpoint_selected_values.append(selected_value)
            checkpoint_policy_values.append(float(policy_value))
            checkpoint_oracle_gaps.append(float(values["best_action_value"] - selected_value))
            checkpoint_matches.append(selected_action == values["best_action"])
            checkpoint_advantage_value_corrs.append(
                _advantage_value_corr(advantages, values["action_values"])
            )

    elapsed = time.perf_counter() - started
    per_action_mean = {
        action: float(np.mean(payoffs)) if payoffs else 0.0
        for action, payoffs in sorted(action_values_by_name.items())
    }
    controls = (
        _positive_controls(int(cfg.n_equity_samples), int(cfg.initial_chips), int(cfg.seed))
        if cfg.include_positive_controls
        else {}
    )
    controls_passed = (
        all(item.get("passed") is True for item in controls.values())
        if controls
        else None
    )
    metrics = {
        "algorithm": "restricted_showdown_action_value",
        "role": "positive_control_first_early_action_value_diagnostic",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": (
            "Showdown-abstraction first-action diagnostic; not exploitability "
            "and not a Slumbot promotion gate by itself."
        ),
        "num_raise_fractions": len(RAISE_FRACTIONS),
        "n_roots": int(cfg.n_roots),
        "n_equity_samples": int(cfg.n_equity_samples),
        "initial_chips": int(cfg.initial_chips),
        "br_player": int(cfg.br_player),
        "seed": int(cfg.seed),
        "root_sampling": "deterministic_seeded_hero_holes",
        "seconds": float(elapsed),
        "roots_per_second": float(max(int(cfg.n_roots), 1) / max(elapsed, 1e-9)),
        "per_action_mean_payoff": per_action_mean,
        "best_action_counts": dict(best_action_counts),
        "oracle_best_mean_payoff": float(np.mean(oracle_values)) if oracle_values else 0.0,
        "call_mean_payoff": float(np.mean(call_values)) if call_values else 0.0,
        "positive_controls": controls,
        "positive_controls_passed": controls_passed,
        "promotion": False,
    }
    if loaded_checkpoint is not None:
        metrics.update(
            {
                **device_info,
                "checkpoint": str(cfg.checkpoint),
                "checkpoint_iteration": loaded_checkpoint.metadata.get("checkpoint_iteration"),
                "strategy_source": str(cfg.strategy_source),
                "checkpoint_action_counts": dict(checkpoint_action_counts),
                "checkpoint_mean_selected_payoff": (
                    float(np.mean(checkpoint_selected_values))
                    if checkpoint_selected_values
                    else 0.0
                ),
                "checkpoint_mean_policy_payoff": (
                    float(np.mean(checkpoint_policy_values))
                    if checkpoint_policy_values
                    else 0.0
                ),
                "checkpoint_mean_oracle_gap": (
                    float(np.mean(checkpoint_oracle_gaps))
                    if checkpoint_oracle_gaps
                    else 0.0
                ),
                "checkpoint_oracle_match_rate": (
                    float(np.mean(checkpoint_matches)) if checkpoint_matches else 0.0
                ),
                "checkpoint_mean_advantage_value_corr": (
                    float(np.mean(checkpoint_advantage_value_corrs))
                    if checkpoint_advantage_value_corrs
                    else 0.0
                ),
            }
        )
    return metrics
