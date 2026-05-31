"""Public-information early action rollout diagnostics.

This diagnostic samples hidden opponent cards and future deck order from the
public root, then scores each legal first action over the same sampled worlds.
It is not an exploitability estimator or a gameplay policy; it is a
positive-control-first teacher candidate for early-street action quality.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import (
    FastPokerState,
    N_ACTIONS,
)
from poker_ai.games.full_deck.state import INDEX_TO_ACTION
from poker_ai.research.evaluation import (
    _strategies_from_network,
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.native_nfsp import resolve_device


ContinuationPolicy = Callable[[FastPokerState, np.random.Generator], int]


@dataclass(frozen=True)
class PublicWorld:
    hero_cards: tuple[int, int]
    opponent_cards: tuple[int, int]
    deck_tail: tuple[int, ...]


@dataclass(frozen=True)
class FirstActionRolloutResult:
    legal_actions: tuple[int, ...]
    action_values: dict[int, float]
    action_standard_errors: dict[int, float]
    best_action: int
    best_action_value: float
    n_worlds: int
    n_truncated_rollouts: int


@dataclass(frozen=True)
class PublicActionRolloutConfig:
    n_roots: int = 16
    n_worlds: int = 64
    initial_chips: int = 1000
    small_blind: int = 50
    big_blind: int = 100
    max_steps_per_hand: int = 128
    seed: int = 20260521
    checkpoint: str | None = None
    strategy_source: str = "regret"
    continuation_checkpoint: str | None = None
    continuation_checkpoints: tuple[str, ...] = ()
    continuation_strategy_source: str | None = None
    greedy_continuation: bool = False
    device: str = "auto"
    include_positive_controls: bool = True


def _validate_cards(cards: tuple[int, ...] | list[int], *, label: str) -> tuple[int, ...]:
    normalized = tuple(int(card) for card in cards)
    for card in normalized:
        if card < 0 or card >= 52:
            raise ValueError(f"{label} contains invalid card index {card}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} contains duplicate cards")
    return normalized


def build_public_world_state(
    *,
    hero_cards: tuple[int, int],
    opponent_cards: tuple[int, int],
    deck_tail: tuple[int, ...] | list[int],
    initial_chips: int = 1000,
    small_blind: int = 50,
    big_blind: int = 100,
) -> FastPokerState:
    """Create a deterministic private world for one public root sample.

    The acting player's public root contains only their own private cards. The
    opponent cards and future deck are sampled per world and only become visible
    through the simulator when each player acts from their own perspective.
    """
    hero = _validate_cards(hero_cards, label="hero_cards")
    opponent = _validate_cards(opponent_cards, label="opponent_cards")
    tail_prefix = _validate_cards(tuple(deck_tail), label="deck_tail")
    known = hero + opponent + tail_prefix
    if len(set(known)) != len(known):
        raise ValueError("hero, opponent, and deck_tail cards must be disjoint")

    missing = tuple(card for card in range(52) if card not in set(known))
    deck_order = np.asarray(hero + opponent + tail_prefix + missing, dtype=np.int8)
    if deck_order.shape != (52,):
        raise ValueError("constructed deck_order must contain 52 cards")

    state = FastPokerState(
        n_players=2,
        small_blind=int(small_blind),
        big_blind=int(big_blind),
        initial_chips=int(initial_chips),
    )
    state.deck_order = deck_order
    state.deck_cursor = 4
    state.hole_cards[0, :] = np.asarray(hero, dtype=np.int8)
    state.hole_cards[1, :] = np.asarray(opponent, dtype=np.int8)
    state.community[:] = -1
    state.stage = FastPokerState.PREFLOP
    state.n_raises = 0
    state._player_i_index = 0
    state.n_actions = 0
    state.n_players_started_round = state._n_players_with_moves()
    state.history[:, :] = 0
    state._skip_counter = 0
    state._winners_computed = False
    return state


def sample_public_worlds(
    *,
    hero_cards: tuple[int, int],
    n_worlds: int,
    seed: int,
) -> list[PublicWorld]:
    """Sample opponent cards and future deck order conditional on hero cards."""
    hero = _validate_cards(hero_cards, label="hero_cards")
    if len(hero) != 2:
        raise ValueError("hero_cards must contain exactly two cards")
    rng = np.random.default_rng(int(seed))
    unknown = np.asarray([card for card in range(52) if card not in hero], dtype=np.int16)
    worlds: list[PublicWorld] = []
    for _ in range(int(n_worlds)):
        permuted = rng.permutation(unknown).astype(np.int16)
        opponent = tuple(int(card) for card in permuted[:2])
        tail = tuple(int(card) for card in permuted[2:])
        worlds.append(PublicWorld(hero_cards=hero, opponent_cards=opponent, deck_tail=tail))
    return worlds


def sample_seeded_hero_cards(*, seed: int, root_idx: int) -> tuple[int, int]:
    """Return deterministic root hole cards for comparable diagnostics."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(root_idx)]))
    cards = rng.choice(52, size=2, replace=False)
    return int(cards[0]), int(cards[1])


def masked_uniform_policy(state: FastPokerState, rng: np.random.Generator) -> int:
    mask = state.get_legal_mask()
    legal = np.flatnonzero(mask > 0)
    if legal.size == 0:
        raise ValueError("state has no legal actions")
    return int(rng.choice(legal))


def call_policy(state: FastPokerState, rng: np.random.Generator) -> int:
    mask = state.get_legal_mask()
    if mask[1] > 0:
        return 1
    legal = np.flatnonzero(mask > 0)
    if legal.size == 0:
        raise ValueError("state has no legal actions")
    return int(legal[0])


def _payoff_after_rollout(
    state: FastPokerState,
    *,
    action: int,
    acting_player: int,
    continuation_policy: ContinuationPolicy,
    rng: np.random.Generator,
    max_steps_per_hand: int,
) -> tuple[float, bool]:
    child = state.copy()
    child.apply_action(int(action))
    n_steps = 1
    while not child.is_terminal and n_steps < int(max_steps_per_hand):
        next_action = int(continuation_policy(child, rng))
        mask = child.get_legal_mask()
        if next_action < 0 or next_action >= N_ACTIONS or mask[next_action] <= 0:
            raise ValueError(f"continuation policy selected illegal action {next_action}")
        child.apply_action(next_action)
        n_steps += 1
    if not child.is_terminal:
        return float(child.chips[int(acting_player)] - child.initial_chips), True
    return float(child.payout[int(acting_player)]), False


def _payoff_after_forced_call_rollout(
    state: FastPokerState,
    *,
    action: int,
    acting_player: int,
    max_steps_per_hand: int,
) -> tuple[float, bool]:
    """Roll out the built-in call policy without repeated mask checks."""
    child = state.copy()
    child.apply_action(int(action))
    n_steps = 1
    while not child.is_terminal and n_steps < int(max_steps_per_hand):
        child.apply_action(1)
        n_steps += 1
    if not child.is_terminal:
        return float(child.chips[int(acting_player)] - child.initial_chips), True
    return float(child.payout[int(acting_player)]), False


def score_first_actions_across_worlds(
    *,
    worlds: list[PublicWorld],
    continuation_policy: ContinuationPolicy,
    seed: int,
    max_steps_per_hand: int = 128,
    initial_chips: int = 1000,
    small_blind: int = 50,
    big_blind: int = 100,
) -> FirstActionRolloutResult:
    """Score legal first actions using common sampled private worlds."""
    if not worlds:
        raise ValueError("at least one public world is required")
    root = build_public_world_state(
        hero_cards=worlds[0].hero_cards,
        opponent_cards=worlds[0].opponent_cards,
        deck_tail=worlds[0].deck_tail,
        initial_chips=initial_chips,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    acting_player = int(root.current_player_i)
    legal_actions = tuple(int(a) for a in np.flatnonzero(root.get_legal_mask() > 0))
    action_values: dict[int, float] = {}
    action_standard_errors: dict[int, float] = {}
    n_truncated = 0
    world_root_states = [
        build_public_world_state(
            hero_cards=world.hero_cards,
            opponent_cards=world.opponent_cards,
            deck_tail=world.deck_tail,
            initial_chips=initial_chips,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        for world in worlds
    ]
    use_forced_call_rollout = continuation_policy is call_policy
    for action in legal_actions:
        payoffs: list[float] = []
        for world_idx, state in enumerate(world_root_states):
            if use_forced_call_rollout:
                payoff, truncated = _payoff_after_forced_call_rollout(
                    state,
                    action=action,
                    acting_player=acting_player,
                    max_steps_per_hand=max_steps_per_hand,
                )
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), int(action), int(world_idx)])
                )
                payoff, truncated = _payoff_after_rollout(
                    state,
                    action=action,
                    acting_player=acting_player,
                    continuation_policy=continuation_policy,
                    rng=rng,
                    max_steps_per_hand=max_steps_per_hand,
                )
            payoffs.append(float(payoff))
            n_truncated += int(truncated)
        payoff_arr = np.asarray(payoffs, dtype=np.float64)
        action_values[int(action)] = float(payoff_arr.mean())
        stderr = 0.0
        if payoff_arr.size > 1:
            stderr = float(payoff_arr.std(ddof=1) / math.sqrt(payoff_arr.size))
        action_standard_errors[int(action)] = stderr
    best_action = max(action_values, key=action_values.__getitem__)
    return FirstActionRolloutResult(
        legal_actions=legal_actions,
        action_values=action_values,
        action_standard_errors=action_standard_errors,
        best_action=int(best_action),
        best_action_value=float(action_values[best_action]),
        n_worlds=int(len(worlds)),
        n_truncated_rollouts=int(n_truncated),
    )


def _strategy_for_state(
    loaded: Any,
    state: FastPokerState,
    device: torch.device,
    *,
    strategy_source: str,
) -> np.ndarray:
    _advantages, strategy = _network_outputs_for_state(
        loaded,
        state,
        device,
        strategy_source=strategy_source,
    )
    return strategy


def _network_outputs_for_state(
    loaded: Any,
    state: FastPokerState,
    device: torch.device,
    *,
    strategy_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = state.to_feature_vector().reshape(1, -1).astype(np.float32)
    mask = state.get_legal_mask()
    with torch.no_grad():
        advantages = loaded.value_net(torch.from_numpy(features).to(device)).cpu().numpy()[0]
    strategy = _strategies_from_network(
        loaded.value_net,
        features,
        [mask],
        device,
        strategy_source=strategy_source,
        checkpoint_metadata=loaded.metadata,
    )[0]
    return np.asarray(advantages, dtype=np.float64), np.asarray(strategy, dtype=np.float64)


def select_deployed_root_action(
    *,
    advantages: np.ndarray | None,
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    strategy_source: str,
    greedy: bool,
) -> int:
    """Select the deterministic root action using the live deployment contract."""
    mask = np.asarray(legal_mask, dtype=np.float32)
    legal = np.flatnonzero(mask > 0)
    if legal.size == 0:
        raise ValueError("legal_mask has no legal actions")
    if greedy and strategy_source == "regret" and advantages is not None:
        adv = np.asarray(advantages, dtype=np.float64)
        masked_adv = np.where(mask > 0, adv, -1e18)
        return int(np.argmax(masked_adv))
    probs = np.asarray(strategy, dtype=np.float64)
    return int(legal[int(np.argmax(probs[legal]))])


def checkpoint_continuation_policy(
    loaded: Any,
    device: torch.device,
    *,
    strategy_source: str,
    greedy: bool = False,
) -> ContinuationPolicy:
    def _policy(state: FastPokerState, rng: np.random.Generator) -> int:
        mask = state.get_legal_mask()
        legal = np.flatnonzero(mask > 0)
        if legal.size == 0:
            raise ValueError("state has no legal actions")
        strategy = _strategy_for_state(
            loaded,
            state,
            device,
            strategy_source=strategy_source,
        )
        probs = np.asarray(strategy[legal], dtype=np.float64)
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.ones_like(probs) / float(probs.size)
        else:
            probs = probs / total
        if greedy:
            return int(legal[int(np.argmax(probs))])
        return int(rng.choice(legal, p=probs))

    return _policy


def checkpoint_population_continuation_policy(
    loaded_population: list[Any],
    device: torch.device,
    *,
    strategy_source: str,
    greedy: bool = False,
) -> ContinuationPolicy:
    if not loaded_population:
        raise ValueError("loaded_population must not be empty")

    def _policy(state: FastPokerState, rng: np.random.Generator) -> int:
        mask = state.get_legal_mask()
        legal = np.flatnonzero(mask > 0)
        if legal.size == 0:
            raise ValueError("state has no legal actions")
        strategies = [
            _strategy_for_state(
                loaded,
                state,
                device,
                strategy_source=strategy_source,
            )
            for loaded in loaded_population
        ]
        mean_strategy = np.mean(np.stack(strategies, axis=0), axis=0)
        probs = np.asarray(mean_strategy[legal], dtype=np.float64)
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.ones_like(probs) / float(probs.size)
        else:
            probs = probs / total
        if greedy:
            return int(legal[int(np.argmax(probs))])
        return int(rng.choice(legal, p=probs))

    return _policy


def _policy_for_root(
    loaded: Any | None,
    state: FastPokerState,
    device: torch.device | None,
    *,
    strategy_source: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    mask = state.get_legal_mask()
    if loaded is None or device is None:
        total = float(mask.sum())
        return None, np.asarray(mask, dtype=np.float64) / total
    return _network_outputs_for_state(
        loaded,
        state,
        device,
        strategy_source=strategy_source,
    )


def _action_name(action: int) -> str:
    return str(INDEX_TO_ACTION.get(int(action), int(action)))


def _positive_controls(
    *,
    n_worlds: int,
    seed: int,
    initial_chips: int,
    small_blind: int,
    big_blind: int,
    max_steps_per_hand: int,
) -> dict[str, Any]:
    hero_aces = (50, 51)
    worlds = sample_public_worlds(
        hero_cards=hero_aces,
        n_worlds=max(int(n_worlds), 32),
        seed=int(seed) + 991,
    )
    premium = score_first_actions_across_worlds(
        worlds=worlds,
        continuation_policy=call_policy,
        seed=int(seed) + 992,
        max_steps_per_hand=max_steps_per_hand,
        initial_chips=initial_chips,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    return {
        "premium_aces_call_continuation": {
            "passed": bool(
                premium.n_truncated_rollouts == 0
                and premium.action_values.get(8, float("-inf"))
                > premium.action_values.get(0, float("inf"))
                and premium.best_action != 0
            ),
            "best_action": _action_name(premium.best_action),
            "best_action_value": premium.best_action_value,
            "fold_value": premium.action_values.get(0),
            "all_in_value": premium.action_values.get(8),
            "n_truncated_rollouts": premium.n_truncated_rollouts,
        }
    }


def evaluate_public_action_rollout_values(
    cfg: PublicActionRolloutConfig,
) -> dict[str, Any]:
    """Evaluate checkpoint first actions against public-world rollout values."""
    started = time.perf_counter()
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    loaded = None
    continuation_loaded = None
    continuation_population: list[Any] = []
    if cfg.checkpoint:
        loaded = load_value_network_checkpoint(cfg.checkpoint, device)
        assert_strategy_source_supported(loaded, cfg.strategy_source)
        checkpoint_metadata = dict(loaded.metadata)
    else:
        checkpoint_metadata = {}

    continuation_source = cfg.continuation_strategy_source or cfg.strategy_source
    continuation_paths = [str(path) for path in cfg.continuation_checkpoints]
    if not continuation_paths and cfg.continuation_checkpoint:
        continuation_paths = [str(cfg.continuation_checkpoint)]
    if continuation_paths:
        for path in continuation_paths:
            continuation_loaded_item = load_value_network_checkpoint(path, device)
            assert_strategy_source_supported(continuation_loaded_item, continuation_source)
            continuation_population.append(continuation_loaded_item)
        continuation_loaded = continuation_population[0]
    else:
        continuation_loaded = loaded

    if len(continuation_population) > 1:
        continuation = checkpoint_population_continuation_policy(
            continuation_population,
            device,
            strategy_source=continuation_source,
            greedy=cfg.greedy_continuation,
        )
        continuation_metadata = [dict(item.metadata) for item in continuation_population]
    elif continuation_loaded is not None:
        continuation = checkpoint_continuation_policy(
            continuation_loaded,
            device,
            strategy_source=continuation_source,
            greedy=cfg.greedy_continuation,
        )
        continuation_metadata = dict(continuation_loaded.metadata)
    else:
        continuation = masked_uniform_policy
        continuation_metadata = {}

    positive_controls: dict[str, Any] = {}
    if cfg.include_positive_controls:
        positive_controls = _positive_controls(
            n_worlds=cfg.n_worlds,
            seed=cfg.seed,
            initial_chips=cfg.initial_chips,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            max_steps_per_hand=cfg.max_steps_per_hand,
        )

    rows: list[dict[str, Any]] = []
    selected_payoffs: list[float] = []
    policy_evs: list[float] = []
    oracle_payoffs: list[float] = []
    n_truncated = 0
    for root_idx in range(int(cfg.n_roots)):
        hero_cards = sample_seeded_hero_cards(seed=cfg.seed, root_idx=root_idx)
        worlds = sample_public_worlds(
            hero_cards=hero_cards,
            n_worlds=cfg.n_worlds,
            seed=int(cfg.seed) + 1009 * (root_idx + 1),
        )
        root = build_public_world_state(
            hero_cards=worlds[0].hero_cards,
            opponent_cards=worlds[0].opponent_cards,
            deck_tail=worlds[0].deck_tail,
            initial_chips=cfg.initial_chips,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
        )
        root_advantages, root_strategy = _policy_for_root(
            loaded,
            root,
            device if loaded is not None else None,
            strategy_source=cfg.strategy_source,
        )
        legal = np.flatnonzero(root.get_legal_mask() > 0)
        legal_probs = np.asarray(root_strategy[legal], dtype=np.float64)
        legal_probs = legal_probs / max(float(legal_probs.sum()), 1e-12)
        selected_action = select_deployed_root_action(
            advantages=root_advantages,
            strategy=root_strategy,
            legal_mask=root.get_legal_mask(),
            strategy_source=cfg.strategy_source,
            greedy=cfg.greedy_continuation,
        )
        result = score_first_actions_across_worlds(
            worlds=worlds,
            continuation_policy=continuation,
            seed=int(cfg.seed) + 2003 * (root_idx + 1),
            max_steps_per_hand=cfg.max_steps_per_hand,
            initial_chips=cfg.initial_chips,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
        )
        policy_ev = float(
            sum(
                float(root_strategy[action]) * float(result.action_values[action])
                for action in result.legal_actions
            )
        )
        selected_value = float(result.action_values[selected_action])
        oracle_value = float(result.best_action_value)
        selected_payoffs.append(selected_value)
        policy_evs.append(policy_ev)
        oracle_payoffs.append(oracle_value)
        n_truncated += int(result.n_truncated_rollouts)
        rows.append(
            {
                "root_idx": int(root_idx),
                "hero_cards": [int(c) for c in hero_cards],
                "selected_action": _action_name(selected_action),
                "oracle_action": _action_name(result.best_action),
                "selected_action_value": selected_value,
                "oracle_action_value": oracle_value,
                "oracle_gap": float(oracle_value - selected_value),
                "policy_ev": policy_ev,
                "action_values": {
                    _action_name(action): value
                    for action, value in sorted(result.action_values.items())
                },
                "action_standard_errors": {
                    _action_name(action): value
                    for action, value in sorted(result.action_standard_errors.items())
                },
                "root_strategy": {
                    _action_name(action): float(root_strategy[action])
                    for action in result.legal_actions
                },
                "root_advantages": (
                    {
                        _action_name(action): float(root_advantages[action])
                        for action in result.legal_actions
                    }
                    if root_advantages is not None
                    else None
                ),
                "n_truncated_rollouts": int(result.n_truncated_rollouts),
            }
        )

    selected_arr = np.asarray(selected_payoffs, dtype=np.float64)
    oracle_arr = np.asarray(oracle_payoffs, dtype=np.float64)
    policy_ev_arr = np.asarray(policy_evs, dtype=np.float64)
    selected_counter = Counter(row["selected_action"] for row in rows)
    oracle_counter = Counter(row["oracle_action"] for row in rows)
    controls_passed = all(bool(item.get("passed")) for item in positive_controls.values())
    values_finite = bool(
        np.isfinite(selected_arr).all()
        and np.isfinite(oracle_arr).all()
        and np.isfinite(policy_ev_arr).all()
    )
    return {
        "mode": "public_information_early_action_rollout_value",
        "passed": bool(values_finite and controls_passed and n_truncated == 0),
        "promotable": False,
        "promotion_blockers": [
            "diagnostic_only_not_exploitability_estimator",
            "requires_training_or_resolver_gate_before_gameplay_use",
        ],
        "n_roots": int(cfg.n_roots),
        "n_worlds": int(cfg.n_worlds),
        "initial_chips": int(cfg.initial_chips),
        "small_blind": int(cfg.small_blind),
        "big_blind": int(cfg.big_blind),
        "seed": int(cfg.seed),
        "checkpoint": cfg.checkpoint,
        "strategy_source": cfg.strategy_source,
        "continuation_checkpoint": (
            continuation_paths[0]
            if len(continuation_paths) == 1
            else (cfg.checkpoint if not continuation_paths else None)
        ),
        "continuation_checkpoints": (
            continuation_paths
            if continuation_paths
            else ([cfg.checkpoint] if cfg.checkpoint else [])
        ),
        "continuation_strategy_source": continuation_source,
        "greedy_continuation": bool(cfg.greedy_continuation),
        "device": device_info,
        "checkpoint_metadata": checkpoint_metadata,
        "continuation_checkpoint_metadata": continuation_metadata,
        "positive_controls": positive_controls,
        "mean_selected_action_value": float(selected_arr.mean()) if selected_arr.size else 0.0,
        "mean_policy_ev": float(policy_ev_arr.mean()) if policy_ev_arr.size else 0.0,
        "mean_oracle_action_value": float(oracle_arr.mean()) if oracle_arr.size else 0.0,
        "mean_oracle_gap": float((oracle_arr - selected_arr).mean()) if selected_arr.size else 0.0,
        "oracle_match_rate": float(
            np.mean([row["selected_action"] == row["oracle_action"] for row in rows])
        ) if rows else 0.0,
        "selected_action_counts": dict(sorted(selected_counter.items())),
        "oracle_action_counts": dict(sorted(oracle_counter.items())),
        "n_truncated_rollouts": int(n_truncated),
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
    }
