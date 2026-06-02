#!/usr/bin/env python3
"""Build native all-action counterfactual rollout targets from local self-play.

This is a small falsification tool, not a promoted learner. For sampled native
full-deck decision states, it forces each legal root action and estimates the
acting player's continuation value by rolling out the local simulator. The goal
is to test whether action-wise local-simulator targets are coherent before using
them to train a policy/value checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import N_ACTIONS, FastPokerState, new_fast_game  # noqa: E402


def _sample_uniform_legal_action(state: FastPokerState, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(state.get_legal_mask() > 0)
    if legal.size <= 0:
        raise ValueError("cannot sample from an empty legal mask")
    return int(rng.choice(legal))


def _rollout_payoff(
    state: FastPokerState,
    *,
    player: int,
    rng: np.random.Generator,
    max_steps_per_rollout: int,
) -> tuple[float, bool]:
    rollout = state.copy()
    steps = 0
    while not rollout.is_terminal and steps < int(max_steps_per_rollout):
        rollout.apply_action(_sample_uniform_legal_action(rollout, rng))
        steps += 1
    if rollout.is_terminal:
        return float(rollout.payout.get(int(player), 0)) / float(rollout.initial_chips), False
    stack_delta = float(rollout.chips[int(player)] + rollout.bets[int(player)] - rollout.initial_chips)
    return stack_delta / float(rollout.initial_chips), True


def resample_hidden_world_for_observer(
    state: FastPokerState,
    *,
    observer: int,
    seed: int,
) -> FastPokerState:
    """Resample opponent private cards and future deck without changing observation."""
    if int(observer) < 0 or int(observer) >= int(state.n_players):
        raise ValueError("observer must be a valid player index")
    rng = np.random.default_rng(int(seed))
    resampled = state.copy()
    observer = int(observer)
    visible_cards: list[int] = []
    for card in resampled.hole_cards[observer, :2]:
        if int(card) >= 0:
            visible_cards.append(int(card))
    for card in resampled.community:
        if int(card) >= 0:
            visible_cards.append(int(card))
    remaining = [card for card in range(52) if card not in set(visible_cards)]
    rng.shuffle(remaining)
    cursor = 0
    for player_i in range(int(resampled.n_players)):
        if int(player_i) == observer:
            continue
        for card_i in range(2):
            resampled.hole_cards[player_i, card_i] = int(remaining[cursor])
            cursor += 1

    used_cards: list[int] = []
    for player_i in range(int(resampled.n_players)):
        for card in resampled.hole_cards[player_i, :2]:
            if int(card) >= 0:
                used_cards.append(int(card))
    for card in resampled.community:
        if int(card) >= 0:
            used_cards.append(int(card))
    used_set = set(used_cards)
    future = [card for card in remaining[cursor:] if int(card) not in used_set]
    if len(used_cards) + len(future) != 52:
        raise ValueError("resampled deck must contain exactly 52 unique cards")
    resampled.deck_order = np.asarray(used_cards + future, dtype=np.int8)
    resampled.deck_cursor = int(len(used_cards))
    return resampled


def estimate_all_action_rollout_values(
    state: FastPokerState,
    *,
    n_rollouts_per_action: int,
    max_steps_per_rollout: int,
    seed: int,
    paired_rollout_seeds: bool = True,
) -> dict[str, Any]:
    """Estimate legal action values by forcing each root action then rolling out."""
    if state.is_terminal:
        raise ValueError("state must be non-terminal")
    if int(n_rollouts_per_action) <= 0:
        raise ValueError("n_rollouts_per_action must be positive")
    if int(max_steps_per_rollout) < 0:
        raise ValueError("max_steps_per_rollout must be non-negative")

    player = int(state.current_player_i)
    legal_mask = state.get_legal_mask().astype(np.float32, copy=True)
    values = np.full(N_ACTIONS, np.nan, dtype=np.float32)
    std_errors = np.full(N_ACTIONS, np.nan, dtype=np.float32)
    truncations = np.zeros(N_ACTIONS, dtype=np.int64)

    for action in np.flatnonzero(legal_mask > 0):
        samples: list[float] = []
        for _rollout_i in range(int(n_rollouts_per_action)):
            if bool(paired_rollout_seeds):
                rng = np.random.default_rng(int(seed) + int(_rollout_i))
            else:
                rng = np.random.default_rng(int(seed) + int(action) * 100_000 + int(_rollout_i))
            child = state.copy()
            child.apply_action(int(action))
            payoff, truncated = _rollout_payoff(
                child,
                player=player,
                rng=rng,
                max_steps_per_rollout=int(max_steps_per_rollout),
            )
            samples.append(float(payoff))
            truncations[int(action)] += int(truncated)
        arr = np.asarray(samples, dtype=np.float32)
        values[int(action)] = float(arr.mean())
        std_errors[int(action)] = (
            float(arr.std(ddof=0) / np.sqrt(max(arr.size, 1))) if arr.size else np.nan
        )

    return {
        "player": player,
        "stage": int(state.stage),
        "legal_mask": legal_mask,
        "values": values,
        "std_errors": std_errors,
        "truncations": truncations,
        "n_rollouts_per_action": int(n_rollouts_per_action),
        "max_steps_per_rollout": int(max_steps_per_rollout),
        "paired_rollout_seeds": bool(paired_rollout_seeds),
    }


def estimate_world_averaged_all_action_values(
    state: FastPokerState,
    *,
    n_worlds: int,
    n_rollouts_per_action: int,
    max_steps_per_rollout: int,
    seed: int,
    paired_rollout_seeds: bool = True,
) -> dict[str, Any]:
    """Average all-action values over hidden worlds compatible with observation."""
    if int(n_worlds) <= 0:
        raise ValueError("n_worlds must be positive")
    player = int(state.current_player_i)
    legal_mask = state.get_legal_mask().astype(np.float32, copy=True)
    legal = legal_mask > 0
    world_values: list[np.ndarray] = []
    total_truncations = np.zeros(N_ACTIONS, dtype=np.int64)
    for world_i in range(int(n_worlds)):
        world = resample_hidden_world_for_observer(
            state,
            observer=player,
            seed=int(seed) + int(world_i) * 100_000,
        )
        estimate = estimate_all_action_rollout_values(
            world,
            n_rollouts_per_action=int(n_rollouts_per_action),
            max_steps_per_rollout=int(max_steps_per_rollout),
            seed=int(seed) + int(world_i) * 10_000,
            paired_rollout_seeds=bool(paired_rollout_seeds),
        )
        world_values.append(np.asarray(estimate["values"], dtype=np.float32))
        total_truncations += np.asarray(estimate["truncations"], dtype=np.int64)
    stacked = np.stack(world_values).astype(np.float32, copy=False)
    values = np.full(N_ACTIONS, np.nan, dtype=np.float32)
    std_errors = np.full(N_ACTIONS, np.nan, dtype=np.float32)
    values[legal] = np.nanmean(stacked[:, legal], axis=0)
    std_errors[legal] = np.nanstd(stacked[:, legal], axis=0) / np.sqrt(float(n_worlds))
    return {
        "player": player,
        "stage": int(state.stage),
        "legal_mask": legal_mask,
        "values": values,
        "std_errors": std_errors,
        "truncations": total_truncations,
        "n_worlds": int(n_worlds),
        "n_rollouts_per_action": int(n_rollouts_per_action),
        "max_steps_per_rollout": int(max_steps_per_rollout),
        "paired_rollout_seeds": bool(paired_rollout_seeds),
    }


def _top_legal_action(values: np.ndarray, legal_mask: np.ndarray) -> int:
    legal = np.flatnonzero(legal_mask > 0)
    if legal.size <= 0:
        raise ValueError("no legal actions")
    legal_values = values[legal]
    finite = np.isfinite(legal_values)
    if not np.any(finite):
        raise ValueError("no finite legal values")
    finite_legal = legal[finite]
    finite_values = legal_values[finite]
    return int(finite_legal[int(np.argmax(finite_values))])


def _top_legal_margin(values: np.ndarray, legal_mask: np.ndarray) -> float:
    legal = np.flatnonzero(legal_mask > 0)
    if legal.size <= 1:
        return 0.0
    legal_values = np.asarray(values, dtype=np.float32)[legal]
    finite_values = legal_values[np.isfinite(legal_values)]
    if finite_values.size <= 1:
        return 0.0
    sorted_values = np.sort(finite_values)
    return float(sorted_values[-1] - sorted_values[-2])


def _sample_decision_states(
    *,
    n_states: int,
    initial_chips: int,
    seed: int,
    max_steps_per_hand: int,
) -> list[FastPokerState]:
    rng = np.random.default_rng(int(seed))
    states: list[FastPokerState] = []
    game_i = 0
    while len(states) < int(n_states):
        np.random.seed(int(seed) + game_i)
        state = new_fast_game(2, initial_chips=int(initial_chips))
        for _step_i in range(int(max_steps_per_hand)):
            if state.is_terminal:
                break
            legal_mask = state.get_legal_mask()
            if int((legal_mask > 0).sum()) >= 2:
                states.append(state.copy())
                if len(states) >= int(n_states):
                    break
            state.apply_action(_sample_uniform_legal_action(state, rng))
        game_i += 1
    return states


def run_gate(
    *,
    n_states: int = 16,
    n_worlds: int = 1,
    low_rollouts_per_action: int = 2,
    high_rollouts_per_action: int = 8,
    max_steps_per_rollout: int = 64,
    initial_chips: int = 1000,
    seed: int = 20260711,
    paired_rollout_seeds: bool = True,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    if int(n_states) <= 0:
        raise ValueError("n_states must be positive")
    if int(n_worlds) <= 0:
        raise ValueError("n_worlds must be positive")
    if int(low_rollouts_per_action) <= 0 or int(high_rollouts_per_action) <= 0:
        raise ValueError("rollout counts must be positive")
    if int(high_rollouts_per_action) < int(low_rollouts_per_action):
        raise ValueError("high_rollouts_per_action must be >= low_rollouts_per_action")

    states = _sample_decision_states(
        n_states=int(n_states),
        initial_chips=int(initial_chips),
        seed=int(seed),
        max_steps_per_hand=max(int(max_steps_per_rollout), 1),
    )
    agreements = 0
    legal_counts: list[int] = []
    legal_l1s: list[float] = []
    low_top_actions: list[int] = []
    high_top_actions: list[int] = []
    high_top_margins: list[float] = []
    margin_002_agreements = 0
    margin_002_count = 0
    margin_005_agreements = 0
    margin_005_count = 0
    total_truncations = 0

    for state_i, state in enumerate(states):
        target_seed = int(seed) + 10_000 + state_i
        if int(n_worlds) > 1:
            low = estimate_world_averaged_all_action_values(
                state,
                n_worlds=int(n_worlds),
                n_rollouts_per_action=int(low_rollouts_per_action),
                max_steps_per_rollout=int(max_steps_per_rollout),
                seed=target_seed,
                paired_rollout_seeds=bool(paired_rollout_seeds),
            )
            high = estimate_world_averaged_all_action_values(
                state,
                n_worlds=int(n_worlds),
                n_rollouts_per_action=int(high_rollouts_per_action),
                max_steps_per_rollout=int(max_steps_per_rollout),
                seed=target_seed if bool(paired_rollout_seeds) else int(seed) + 20_000 + state_i,
                paired_rollout_seeds=bool(paired_rollout_seeds),
            )
        else:
            low = estimate_all_action_rollout_values(
                state,
                n_rollouts_per_action=int(low_rollouts_per_action),
                max_steps_per_rollout=int(max_steps_per_rollout),
                seed=target_seed,
                paired_rollout_seeds=bool(paired_rollout_seeds),
            )
            high = estimate_all_action_rollout_values(
                state,
                n_rollouts_per_action=int(high_rollouts_per_action),
                max_steps_per_rollout=int(max_steps_per_rollout),
                seed=target_seed if bool(paired_rollout_seeds) else int(seed) + 20_000 + state_i,
                paired_rollout_seeds=bool(paired_rollout_seeds),
            )
        legal_mask = low["legal_mask"]
        legal = legal_mask > 0
        low_values = low["values"]
        high_values = high["values"]
        low_top = _top_legal_action(low_values, legal_mask)
        high_top = _top_legal_action(high_values, legal_mask)
        high_margin = _top_legal_margin(high_values, legal_mask)
        agreements += int(low_top == high_top)
        if high_margin >= 0.02:
            margin_002_count += 1
            margin_002_agreements += int(low_top == high_top)
        if high_margin >= 0.05:
            margin_005_count += 1
            margin_005_agreements += int(low_top == high_top)
        legal_counts.append(int(legal.sum()))
        low_top_actions.append(int(low_top))
        high_top_actions.append(int(high_top))
        high_top_margins.append(float(high_margin))
        legal_l1s.append(float(np.mean(np.abs(low_values[legal] - high_values[legal]))))
        total_truncations += int(np.sum(low["truncations"]) + np.sum(high["truncations"]))

    metrics: dict[str, Any] = {
        "algorithm": "native_all_action_counterfactual_rollout_targets",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "gate": "native_all_action_counterfactual_target_consistency",
        "warning": "Local all-action rollout target builder; not checkpoint or Slumbot strength evidence.",
        "n_states": int(n_states),
        "n_worlds": int(n_worlds),
        "world_averaged_targets": bool(int(n_worlds) > 1),
        "low_rollouts_per_action": int(low_rollouts_per_action),
        "high_rollouts_per_action": int(high_rollouts_per_action),
        "max_steps_per_rollout": int(max_steps_per_rollout),
        "initial_chips": int(initial_chips),
        "seed": int(seed),
        "paired_rollout_seeds": bool(paired_rollout_seeds),
        "top_action_agreement": float(agreements / max(len(states), 1)),
        "mean_high_top_margin": float(np.mean(high_top_margins)) if high_top_margins else 0.0,
        "margin_0_02_state_count": int(margin_002_count),
        "top_action_agreement_margin_0_02": (
            float(margin_002_agreements / margin_002_count) if margin_002_count else None
        ),
        "margin_0_05_state_count": int(margin_005_count),
        "top_action_agreement_margin_0_05": (
            float(margin_005_agreements / margin_005_count) if margin_005_count else None
        ),
        "mean_legal_l1": float(np.mean(legal_l1s)) if legal_l1s else None,
        "mean_legal_action_count": float(np.mean(legal_counts)) if legal_counts else 0.0,
        "low_top_actions": low_top_actions,
        "high_top_actions": high_top_actions,
        "total_truncations": int(total_truncations),
        "uses_slumbot_training_data": False,
        "uses_solver_labels": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "passed": bool(states and np.isfinite(legal_l1s).all()),
    }
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=16)
    parser.add_argument("--n-worlds", type=int, default=1)
    parser.add_argument("--low-rollouts-per-action", type=int, default=2)
    parser.add_argument("--high-rollouts-per-action", type=int, default=8)
    parser.add_argument("--max-steps-per-rollout", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--independent-rollout-seeds",
        action="store_true",
        help="Disable paired/common-random continuation seeds for low-vs-high target comparison.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_gate(
        n_states=args.n_states,
        n_worlds=args.n_worlds,
        low_rollouts_per_action=args.low_rollouts_per_action,
        high_rollouts_per_action=args.high_rollouts_per_action,
        max_steps_per_rollout=args.max_steps_per_rollout,
        initial_chips=args.initial_chips,
        seed=args.seed,
        paired_rollout_seeds=not bool(args.independent_rollout_seeds),
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
