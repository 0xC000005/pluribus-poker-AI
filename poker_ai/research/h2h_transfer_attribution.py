"""Duplicate-swapped H2H attribution diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, RAISE_FRACTIONS
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.deep_cfr.vectorized_env import VectorizedPokerEnv
from poker_ai.research.evaluation import (
    _strategies_from_network,
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)


BIG_BLIND = 100
ACTION_NAMES = {
    0: "fold",
    1: "call",
    **{2 + i: f"raise_{frac}" for i, frac in enumerate(RAISE_FRACTIONS)},
    8: "all_in",
}
STREET_NAMES = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
    4: "showdown",
    5: "terminal",
}


def _new_raw_stats() -> dict[str, Any]:
    return {
        "payouts": [],
        "n_actions": 0,
        "street_action_counts": defaultdict(Counter),
        "first_action_outcomes": defaultdict(list),
    }


def _record_action(
    stats: dict[str, Any],
    first_actions: dict[str, dict[int, str]],
    *,
    label: str,
    env_idx: int,
    street: int,
    action_idx: int,
) -> None:
    action = ACTION_NAMES.get(int(action_idx), str(int(action_idx)))
    street_name = STREET_NAMES.get(int(street), str(int(street)))
    stats[label]["n_actions"] += 1
    stats[label]["street_action_counts"][street_name][action] += 1
    first_actions[label].setdefault(int(env_idx), action)


def _finalize_game(
    stats: dict[str, Any],
    first_actions: dict[str, dict[int, str]],
    *,
    state: Any,
    env_idx: int,
    label_to_seat: dict[str, int],
) -> None:
    payouts = state.payout
    for label, seat in label_to_seat.items():
        payoff = float(payouts[int(seat)])
        stats[label]["payouts"].append(payoff)
        action = first_actions[label].get(int(env_idx))
        if action is not None:
            stats[label]["first_action_outcomes"][action].append(payoff)


def _step_label_group(
    env: VectorizedPokerEnv,
    indices: list[int],
    value_net: ValueNetwork,
    device: torch.device,
    *,
    label: str,
    strategy_source: str,
    checkpoint_metadata: dict[str, Any],
    stats: dict[str, Any],
    first_actions: dict[str, dict[int, str]],
) -> None:
    if not indices:
        return
    features = np.stack([env.states[i].to_feature_vector() for i in indices])
    masks = [env.states[i].get_legal_mask() for i in indices]
    strategies = _strategies_from_network(
        value_net,
        features,
        masks,
        device,
        strategy_source=strategy_source,
        checkpoint_metadata=checkpoint_metadata,
    )

    for j, env_idx in enumerate(indices):
        legal = np.flatnonzero(masks[j] > 0)
        probs = np.asarray([strategies[j][action] for action in legal], dtype=np.float64)
        total = float(probs.sum())
        probs = probs / total if total > 1e-12 else np.ones_like(probs) / len(probs)
        action_idx = int(np.random.choice(legal, p=probs))
        _record_action(
            stats,
            first_actions,
            label=label,
            env_idx=env_idx,
            street=int(env.states[env_idx].stage),
            action_idx=action_idx,
        )
        env.step_single(env_idx, action_idx)


def _play_attributed(
    player0_net: ValueNetwork,
    player1_net: ValueNetwork,
    device: torch.device,
    *,
    n_games: int,
    initial_chips: int,
    player0_label: str,
    player1_label: str,
    player0_strategy_source: str,
    player1_strategy_source: str,
    player0_metadata: dict[str, Any],
    player1_metadata: dict[str, Any],
) -> dict[str, Any]:
    env = VectorizedPokerEnv(n_games, 2, initial_chips=initial_chips)
    env.reset()
    player0_net.eval()
    player1_net.eval()
    stats = {
        player0_label: _new_raw_stats(),
        player1_label: _new_raw_stats(),
    }
    first_actions = {
        player0_label: {},
        player1_label: {},
    }
    label_to_seat = {player0_label: 0, player1_label: 1}
    finalized = np.zeros(n_games, dtype=np.bool_)
    max_steps = n_games * 50

    for _step in range(max_steps):
        for i in range(n_games):
            if env.done[i]:
                continue
            state = env.states[i]
            while not state.is_terminal and not state.active[state.current_player_i]:
                child = state.copy()
                child.apply_action(None)
                env.states[i] = child
                state = child
                if state.is_terminal:
                    env.done[i] = True

        for i in range(n_games):
            if env.done[i] and not finalized[i]:
                _finalize_game(
                    stats,
                    first_actions,
                    state=env.states[i],
                    env_idx=i,
                    label_to_seat=label_to_seat,
                )
                finalized[i] = True

        if env.done.all():
            break

        player0_indices: list[int] = []
        player1_indices: list[int] = []
        for i in range(n_games):
            if env.done[i]:
                continue
            if env.states[i].current_player_i == 0:
                player0_indices.append(i)
            else:
                player1_indices.append(i)

        _step_label_group(
            env,
            player0_indices,
            player0_net,
            device,
            label=player0_label,
            strategy_source=player0_strategy_source,
            checkpoint_metadata=player0_metadata,
            stats=stats,
            first_actions=first_actions,
        )
        _step_label_group(
            env,
            player1_indices,
            player1_net,
            device,
            label=player1_label,
            strategy_source=player1_strategy_source,
            checkpoint_metadata=player1_metadata,
            stats=stats,
            first_actions=first_actions,
        )

    for i in range(n_games):
        if env.done[i] and not finalized[i]:
            _finalize_game(
                stats,
                first_actions,
                state=env.states[i],
                env_idx=i,
                label_to_seat=label_to_seat,
            )
            finalized[i] = True
    return stats


def _merge_raw_stats(raw_runs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    merged = _new_raw_stats()
    for raw in raw_runs:
        item = raw[label]
        merged["payouts"].extend(item["payouts"])
        merged["n_actions"] += int(item["n_actions"])
        for street, counts in item["street_action_counts"].items():
            merged["street_action_counts"][street].update(counts)
        for action, payoffs in item["first_action_outcomes"].items():
            merged["first_action_outcomes"][action].extend(payoffs)
    return merged


def _summarize_raw_stats(raw: dict[str, Any]) -> dict[str, Any]:
    payouts = np.asarray(raw["payouts"], dtype=np.float64)
    first_action_outcomes = {}
    for action, payoffs in sorted(raw["first_action_outcomes"].items()):
        arr = np.asarray(payoffs, dtype=np.float64)
        first_action_outcomes[action] = {
            "n": int(arr.size),
            "mean_payoff": float(arr.mean()) if arr.size else 0.0,
            "total_payoff": float(arr.sum()) if arr.size else 0.0,
        }
    return {
        "n_completed_games": int(payouts.size),
        "mean_payoff": float(payouts.mean()) if payouts.size else 0.0,
        "n_actions": int(raw["n_actions"]),
        "street_action_counts": {
            street: dict(counts)
            for street, counts in sorted(raw["street_action_counts"].items())
        },
        "first_action_outcomes": first_action_outcomes,
    }


def evaluate_h2h_transfer_attribution(
    *,
    candidate_checkpoint: str | Path,
    baseline_checkpoint: str | Path,
    n_games: int = 1000,
    initial_chips: int | None = None,
    seed: int = 0,
    device: str | torch.device = "auto",
    strategy_source: str = "regret",
    candidate_strategy_source: str | None = None,
    baseline_strategy_source: str | None = None,
) -> dict[str, Any]:
    """Run duplicate-swapped H2H and attribute action/outcome differences."""
    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    candidate_strategy_source = candidate_strategy_source or strategy_source
    baseline_strategy_source = baseline_strategy_source or strategy_source
    candidate = load_value_network_checkpoint(candidate_checkpoint, resolved_device)
    baseline = load_value_network_checkpoint(baseline_checkpoint, resolved_device)
    assert_strategy_source_supported(candidate, candidate_strategy_source)
    assert_strategy_source_supported(baseline, baseline_strategy_source)
    candidate_metadata = dict(candidate.metadata)
    baseline_metadata = dict(baseline.metadata)
    chips = int(initial_chips or candidate_metadata.get("initial_chips") or 10000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    candidate_seat0 = _play_attributed(
        candidate.value_net,
        baseline.value_net,
        resolved_device,
        n_games=int(n_games),
        initial_chips=chips,
        player0_label="candidate",
        player1_label="baseline",
        player0_strategy_source=candidate_strategy_source,
        player1_strategy_source=baseline_strategy_source,
        player0_metadata=candidate_metadata,
        player1_metadata=baseline_metadata,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    baseline_seat0 = _play_attributed(
        baseline.value_net,
        candidate.value_net,
        resolved_device,
        n_games=int(n_games),
        initial_chips=chips,
        player0_label="baseline",
        player1_label="candidate",
        player0_strategy_source=baseline_strategy_source,
        player1_strategy_source=candidate_strategy_source,
        player0_metadata=baseline_metadata,
        player1_metadata=candidate_metadata,
    )

    candidate_raw = _merge_raw_stats([candidate_seat0, baseline_seat0], "candidate")
    baseline_raw = _merge_raw_stats([candidate_seat0, baseline_seat0], "baseline")
    candidate_payouts = np.asarray(candidate_raw["payouts"], dtype=np.float64)
    avg = float(candidate_payouts.mean()) if candidate_payouts.size else 0.0
    std = float(candidate_payouts.std(ddof=1)) if candidate_payouts.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(candidate_payouts.size)) if candidate_payouts.size > 1 else 0.0
    return {
        "mode": "duplicate_swapped_transfer_attribution",
        "passed": bool(np.isfinite(candidate_payouts).all()),
        "promotion": False,
        "promotion_blockers": ["diagnostic_attribution_only"],
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_strategy_source": candidate_strategy_source,
        "baseline_strategy_source": baseline_strategy_source,
        "n_games": int(n_games * 2),
        "n_duplicate_pairs": int(n_games),
        "initial_chips": chips,
        "seed": int(seed),
        "device": str(resolved_device),
        "avg_chips_per_hand": avg,
        "ci95_chips_per_hand": ci95,
        "lower95_chips_per_hand": avg - ci95,
        "mbb_per_hand": avg / BIG_BLIND * 1000.0,
        "candidate": _summarize_raw_stats(candidate_raw),
        "baseline": _summarize_raw_stats(baseline_raw),
    }


def aggregate_attribution_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed-level attribution summaries without hiding per-seed detail."""
    if not runs:
        raise ValueError("cannot aggregate empty attribution runs")
    if len(runs) == 1:
        return runs[0]
    avgs = np.asarray([run["avg_chips_per_hand"] for run in runs], dtype=np.float64)
    ci95 = float(1.96 * avgs.std(ddof=1) / math.sqrt(avgs.size)) if avgs.size > 1 else 0.0
    merged = {
        key: value
        for key, value in runs[0].items()
        if key not in {"avg_chips_per_hand", "ci95_chips_per_hand", "lower95_chips_per_hand", "candidate", "baseline"}
    }
    merged.update(
        {
            "n_runs": int(len(runs)),
            "n_games": int(sum(run["n_games"] for run in runs)),
            "n_games_per_run": int(runs[0]["n_games"]),
            "avg_chips_per_hand": float(avgs.mean()),
            "ci95_chips_per_hand_across_seeds": ci95,
            "lower95_chips_per_hand_across_seeds": float(avgs.mean() - ci95),
            "runs": runs,
        }
    )
    return merged
