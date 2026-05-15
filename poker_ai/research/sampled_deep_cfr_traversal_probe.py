"""Research-only sampled traverser traversal probe.

This module deliberately stays outside the production Deep CFR trainer. It
tests whether inverse-probability sampled traverser actions can approximate
exhaustive root regrets on tiny full-deck states before any CUDA integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random
import time
from typing import Any

import numpy as np
import torch

from poker_ai.deep_cfr.deep_cfr import get_legal_mask, regret_match
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import (
    ACTION_TO_INDEX,
    INDEX_TO_ACTION,
    N_ACTIONS,
    N_FEATURES,
    PokerState,
    new_game,
)
from poker_ai.research.sampled_action_mccfr import (
    pps_without_replacement_inclusion_probs,
    priority_sample_without_replacement,
    sample_pps_without_replacement,
)


@dataclass(frozen=True)
class _TraversalResult:
    value: float
    root_regret: np.ndarray | None = None


def _strategy(
    value_net: ValueNetwork,
    state: PokerState,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = state.to_feature_vector()
    legal_mask = get_legal_mask(state)
    legal_actions = [str(action) for action in state.legal_actions if action is not None]
    with torch.no_grad():
        pred = (
            value_net(torch.from_numpy(features).to(device).unsqueeze(0))
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
    return regret_match(pred, legal_mask), legal_mask, legal_actions


def _initial_chips(state: PokerState) -> float:
    scale = float(getattr(state, "_initial_n_chips", 1) or 1)
    return scale if scale > 0 else 1.0


def _sampled_value_and_regret(
    *,
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    sampled_actions: np.ndarray,
    sampled_values: np.ndarray,
    sample_probs: np.ndarray,
    baseline_values: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    legal = legal_mask > 0.0
    if sampled_actions.size == 0:
        raise ValueError("sampled_actions must be non-empty")
    q = np.asarray(sample_probs, dtype=np.float64)
    q = np.where(legal, q, 0.0)
    q = q / float(q.sum())
    sigma = np.asarray(strategy, dtype=np.float64)
    sigma = np.where(legal, sigma, 0.0)
    sigma = sigma / float(sigma.sum())
    if np.any(q[legal] <= 0.0):
        raise ValueError("every legal action needs positive sampling probability")
    if baseline_values is None:
        baseline = np.zeros(N_ACTIONS, dtype=np.float64)
    else:
        baseline = np.asarray(baseline_values, dtype=np.float64)
        if baseline.shape != legal_mask.shape:
            raise ValueError("baseline_values must match legal_mask")
        baseline = np.where(legal, baseline, 0.0)

    estimated_values = baseline.copy()
    estimated_state_value = float(np.dot(sigma, baseline))
    denom = float(sampled_actions.size)
    for action, value in zip(sampled_actions, sampled_values):
        action = int(action)
        if action < 0 or action >= N_ACTIONS or not legal[action]:
            raise ValueError("sampled action must be legal")
        residual = float(value) - float(baseline[action])
        weight = 1.0 / (denom * float(q[action]))
        estimated_values[action] += residual * weight
        estimated_state_value += float(sigma[action]) * residual * weight

    regret = estimated_values - estimated_state_value
    regret[~legal] = 0.0
    return estimated_state_value, regret.astype(np.float32)


def _sampled_value_and_regret_without_replacement(
    *,
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    sampled_actions: np.ndarray,
    sampled_values: np.ndarray,
    inclusion_probs: np.ndarray,
    baseline_values: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    legal = legal_mask > 0.0
    sigma = np.asarray(strategy, dtype=np.float64)
    sigma = np.where(legal, sigma, 0.0)
    sigma = sigma / float(sigma.sum())
    inclusion = np.asarray(inclusion_probs, dtype=np.float64)
    if np.any(inclusion[legal] <= 0.0):
        raise ValueError("every legal action needs positive inclusion probability")
    if baseline_values is None:
        baseline = np.zeros(N_ACTIONS, dtype=np.float64)
    else:
        baseline = np.asarray(baseline_values, dtype=np.float64)
        if baseline.shape != legal_mask.shape:
            raise ValueError("baseline_values must match legal_mask")
        baseline = np.where(legal, baseline, 0.0)

    estimated_values = baseline.copy()
    estimated_state_value = float(np.dot(sigma, baseline))
    seen: set[int] = set()
    for action, value in zip(sampled_actions, sampled_values):
        action = int(action)
        if action in seen:
            raise ValueError("without-replacement sampled actions must be unique")
        seen.add(action)
        if action < 0 or action >= N_ACTIONS or not legal[action]:
            raise ValueError("sampled action must be legal")
        residual = float(value) - float(baseline[action])
        weight = 1.0 / float(inclusion[action])
        estimated_values[action] += residual * weight
        estimated_state_value += float(sigma[action]) * residual * weight

    regret = estimated_values - estimated_state_value
    regret[~legal] = 0.0
    return estimated_state_value, regret.astype(np.float32)


def _sample_action_indices(
    rng: np.random.Generator,
    legal_indices: np.ndarray,
    sample_probs: np.ndarray,
    *,
    sample_count: int,
) -> tuple[np.ndarray, bool]:
    legal_indices = np.asarray(legal_indices, dtype=np.int64)
    if legal_indices.size == 0:
        raise ValueError("legal_indices must be non-empty")
    if int(sample_count) >= int(legal_indices.size):
        return legal_indices.copy(), True
    probs = np.asarray(sample_probs, dtype=np.float64)[legal_indices]
    probs = probs / float(probs.sum())
    return (
        rng.choice(
            legal_indices,
            size=max(1, int(sample_count)),
            replace=True,
            p=probs,
        ).astype(np.int64),
        False,
    )


def _exhaustive_traverse(
    state: PokerState,
    *,
    traverser: int,
    value_net: ValueNetwork,
    device: torch.device,
    opponent_rng: np.random.Generator,
    depth: int,
) -> _TraversalResult:
    if state.is_terminal:
        return _TraversalResult(value=float(state.payout[traverser]))
    if not state.current_player.is_active:
        return _exhaustive_traverse(
            state.apply_action(None),
            traverser=traverser,
            value_net=value_net,
            device=device,
            opponent_rng=opponent_rng,
            depth=depth + 1,
        )

    strategy, legal_mask, legal_actions = _strategy(value_net, state, device)
    if int(state.player_i) == int(traverser):
        action_values = np.zeros(N_ACTIONS, dtype=np.float32)
        for action in legal_actions:
            child = _exhaustive_traverse(
                state.apply_action(action),
                traverser=traverser,
                value_net=value_net,
                device=device,
                opponent_rng=opponent_rng,
                depth=depth + 1,
            )
            action_values[ACTION_TO_INDEX[action]] = float(child.value)
        state_value = float(np.dot(strategy, action_values))
        regret = (action_values - state_value) * legal_mask / _initial_chips(state)
        return _TraversalResult(
            value=state_value,
            root_regret=regret.astype(np.float32) if depth == 0 else None,
        )

    probs = np.array([strategy[ACTION_TO_INDEX[action]] for action in legal_actions])
    probs = probs / float(probs.sum())
    action = str(opponent_rng.choice(legal_actions, p=probs))
    return _exhaustive_traverse(
        state.apply_action(action),
        traverser=traverser,
        value_net=value_net,
        device=device,
        opponent_rng=opponent_rng,
        depth=depth + 1,
    )


def _sampled_traverse(
    state: PokerState,
    *,
    traverser: int,
    value_net: ValueNetwork,
    device: torch.device,
    opponent_rng: np.random.Generator,
    traverser_rng: np.random.Generator,
    depth: int,
    sample_count: int,
    uniform_mix: float,
    sampling_mode: str,
    priority_forced_count: int,
) -> _TraversalResult:
    if state.is_terminal:
        return _TraversalResult(value=float(state.payout[traverser]))
    if not state.current_player.is_active:
        return _sampled_traverse(
            state.apply_action(None),
            traverser=traverser,
            value_net=value_net,
            device=device,
            opponent_rng=opponent_rng,
            traverser_rng=traverser_rng,
            depth=depth + 1,
            sample_count=sample_count,
            uniform_mix=uniform_mix,
            sampling_mode=sampling_mode,
            priority_forced_count=priority_forced_count,
        )

    strategy, legal_mask, legal_actions = _strategy(value_net, state, device)
    legal_indices = np.array(
        [ACTION_TO_INDEX[action] for action in legal_actions],
        dtype=np.int64,
    )
    if int(state.player_i) == int(traverser):
        uniform = legal_mask / max(1.0, float(legal_mask.sum()))
        q = ((1.0 - uniform_mix) * strategy + uniform_mix * uniform).astype(np.float64)
        q = q / float(q.sum())
        if sampling_mode == "with-replacement":
            sampled, sampled_all_legal = _sample_action_indices(
                traverser_rng,
                legal_indices,
                q,
                sample_count=sample_count,
            )
            inclusion_probs = None
        elif sampling_mode == "without-replacement":
            sampled = sample_pps_without_replacement(
                traverser_rng,
                q,
                legal_mask,
                sample_count=sample_count,
            )
            sampled_all_legal = int(sampled.size) >= int(legal_indices.size)
            inclusion_probs = pps_without_replacement_inclusion_probs(
                q,
                legal_mask,
                sample_count=sample_count,
            )
        elif sampling_mode == "priority-without-replacement":
            sampled, inclusion_probs = priority_sample_without_replacement(
                traverser_rng,
                q,
                legal_mask,
                sample_count=sample_count,
                forced_count=priority_forced_count,
                priority_scores=strategy,
            )
            sampled_all_legal = int(sampled.size) >= int(legal_indices.size)
        else:
            raise ValueError(f"unknown sampling mode: {sampling_mode}")
        sampled_values = np.zeros(sampled.shape[0], dtype=np.float32)
        for idx, action_idx in enumerate(sampled):
            action = INDEX_TO_ACTION[int(action_idx)]
            child = _sampled_traverse(
                state.apply_action(action),
                traverser=traverser,
                value_net=value_net,
                device=device,
                opponent_rng=opponent_rng,
                traverser_rng=traverser_rng,
                depth=depth + 1,
                sample_count=sample_count,
                uniform_mix=uniform_mix,
                sampling_mode=sampling_mode,
                priority_forced_count=priority_forced_count,
            )
            sampled_values[idx] = float(child.value)
        if sampled_all_legal:
            action_values = np.zeros(N_ACTIONS, dtype=np.float32)
            for action_idx, value in zip(sampled, sampled_values):
                action_values[int(action_idx)] = float(value)
            state_value = float(np.dot(strategy, action_values))
            regret = (action_values - state_value) * legal_mask
        else:
            if sampling_mode == "with-replacement":
                state_value, regret = _sampled_value_and_regret(
                    strategy=strategy,
                    legal_mask=legal_mask,
                    sampled_actions=sampled,
                    sampled_values=sampled_values,
                    sample_probs=q,
                )
            else:
                state_value, regret = _sampled_value_and_regret_without_replacement(
                    strategy=strategy,
                    legal_mask=legal_mask,
                    sampled_actions=sampled,
                    sampled_values=sampled_values,
                    inclusion_probs=inclusion_probs,
                )
        regret = regret / _initial_chips(state)
        return _TraversalResult(
            value=state_value,
            root_regret=regret.astype(np.float32) if depth == 0 else None,
        )

    probs = np.array([strategy[ACTION_TO_INDEX[action]] for action in legal_actions])
    probs = probs / float(probs.sum())
    action = str(opponent_rng.choice(legal_actions, p=probs))
    return _sampled_traverse(
        state.apply_action(action),
        traverser=traverser,
        value_net=value_net,
        device=device,
        opponent_rng=opponent_rng,
        traverser_rng=traverser_rng,
        depth=depth + 1,
        sample_count=sample_count,
        uniform_mix=uniform_mix,
        sampling_mode=sampling_mode,
        priority_forced_count=priority_forced_count,
    )


def _mean_root_regret(
    state: PokerState,
    *,
    traverser: int,
    value_net: ValueNetwork,
    device: torch.device,
    n_repeats: int,
    seed: int,
    sampled: bool,
    sample_count: int,
    uniform_mix: float,
    sampling_mode: str,
    priority_forced_count: int,
) -> tuple[np.ndarray, float]:
    regrets = []
    started = time.perf_counter()
    for repeat in range(max(1, int(n_repeats))):
        opponent_rng = np.random.default_rng([int(seed), repeat, 0])
        traverser_rng = np.random.default_rng([int(seed), repeat, 1])
        state_copy = copy.deepcopy(state)
        if sampled:
            result = _sampled_traverse(
                state_copy,
                traverser=traverser,
                value_net=value_net,
                device=device,
                opponent_rng=opponent_rng,
                traverser_rng=traverser_rng,
                depth=0,
                sample_count=sample_count,
                uniform_mix=uniform_mix,
                sampling_mode=sampling_mode,
                priority_forced_count=priority_forced_count,
            )
        else:
            result = _exhaustive_traverse(
                state_copy,
                traverser=traverser,
                value_net=value_net,
                device=device,
                opponent_rng=opponent_rng,
                depth=0,
            )
        if result.root_regret is None:
            raise RuntimeError("probe root did not produce traverser regrets")
        regrets.append(result.root_regret.astype(np.float64))
    return np.mean(np.stack(regrets, axis=0), axis=0), time.perf_counter() - started


def _top_margin(values: np.ndarray, legal_mask: np.ndarray) -> float:
    legal_values = np.asarray(values, dtype=np.float64)[legal_mask]
    if legal_values.size < 2:
        return 0.0
    ordered = np.sort(legal_values)[::-1]
    return float(ordered[0] - ordered[1])


def run_probe(
    *,
    n_repeats: int = 64,
    n_reference_repeats: int | None = None,
    initial_chips: int = 300,
    sample_count: int = 4,
    hidden_dim: int = 64,
    n_layers: int = 1,
    uniform_mix: float = 0.25,
    sampling_mode: str = "with-replacement",
    priority_forced_count: int = 0,
    seed: int = 20260525,
    device: str = "cpu",
) -> dict[str, Any]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    state = new_game(2, initial_chips=int(initial_chips))
    value_net = ValueNetwork(
        N_FEATURES,
        hidden_dim=int(hidden_dim),
        output_dim=N_ACTIONS,
        n_layers=int(n_layers),
    ).to(resolved_device)
    value_net.eval()

    reference_repeats = int(n_reference_repeats or n_repeats)
    exhaustive_mean, exhaustive_seconds = _mean_root_regret(
        state,
        traverser=0,
        value_net=value_net,
        device=resolved_device,
        n_repeats=reference_repeats,
        seed=int(seed) + 10_000,
        sampled=False,
        sample_count=sample_count,
        uniform_mix=uniform_mix,
        sampling_mode=sampling_mode,
        priority_forced_count=priority_forced_count,
    )
    sampled_mean, sampled_seconds = _mean_root_regret(
        state,
        traverser=0,
        value_net=value_net,
        device=resolved_device,
        n_repeats=int(n_repeats),
        seed=int(seed) + 10_000,
        sampled=True,
        sample_count=sample_count,
        uniform_mix=uniform_mix,
        sampling_mode=sampling_mode,
        priority_forced_count=priority_forced_count,
    )
    legal_mask = get_legal_mask(state) > 0.0
    bias = sampled_mean - exhaustive_mean
    legal_bias = bias[legal_mask]
    exhaustive_top = int(np.argmax(np.where(legal_mask, exhaustive_mean, -1e9)))
    sampled_top = int(np.argmax(np.where(legal_mask, sampled_mean, -1e9)))
    exhaustive_per_run = exhaustive_seconds / max(1, reference_repeats)
    sampled_per_run = sampled_seconds / max(1, int(n_repeats))
    return {
        "mode": "sampled_deep_cfr_traversal_probe",
        "warning": "Tiny stochastic CPU probe; not a trainer or promotion gate.",
        "initial_chips": int(initial_chips),
        "n_repeats": int(n_repeats),
        "n_reference_repeats": int(reference_repeats),
        "sample_count": int(sample_count),
        "sampling_mode": sampling_mode,
        "priority_forced_count": int(priority_forced_count),
        "uniform_mix": float(uniform_mix),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "device": str(resolved_device),
        "mean_abs_bias": round(float(np.mean(np.abs(legal_bias))), 6),
        "mean_l2_bias": round(float(np.sqrt(np.mean(legal_bias ** 2))), 6),
        "exhaustive_top_action": exhaustive_top,
        "sampled_top_action": sampled_top,
        "exhaustive_top_margin": round(_top_margin(exhaustive_mean, legal_mask), 6),
        "sampled_top_margin": round(_top_margin(sampled_mean, legal_mask), 6),
        "top_action_match": bool(exhaustive_top == sampled_top),
        "exhaustive_seconds": round(float(exhaustive_seconds), 6),
        "sampled_seconds": round(float(sampled_seconds), 6),
        "exhaustive_seconds_per_run": round(float(exhaustive_per_run), 6),
        "sampled_seconds_per_run": round(float(sampled_per_run), 6),
        "per_run_speedup": round(
            float(exhaustive_per_run / sampled_per_run) if sampled_per_run > 0 else 0.0,
            6,
        ),
        "promotion": False,
    }


def run_probe_grid(
    *,
    seeds: list[int],
    initial_chips_values: list[int],
    n_repeats: int = 64,
    n_reference_repeats: int | None = None,
    sample_count: int = 4,
    hidden_dim: int = 64,
    n_layers: int = 1,
    uniform_mix: float = 0.25,
    sampling_mode: str = "with-replacement",
    priority_forced_count: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for initial_chips in initial_chips_values:
        for seed in seeds:
            cases.append(
                run_probe(
                    n_repeats=n_repeats,
                    n_reference_repeats=n_reference_repeats,
                    initial_chips=int(initial_chips),
                    sample_count=sample_count,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    uniform_mix=uniform_mix,
                    sampling_mode=sampling_mode,
                    priority_forced_count=priority_forced_count,
                    seed=int(seed),
                    device=device,
                )
            )
    if not cases:
        raise ValueError("probe grid requires at least one case")
    top_matches = [bool(case["top_action_match"]) for case in cases]
    return {
        "mode": "sampled_deep_cfr_traversal_probe_grid",
        "warning": "Aggregate of tiny stochastic CPU probes; not a trainer or promotion gate.",
        "n_cases": len(cases),
        "seeds": [int(seed) for seed in seeds],
        "initial_chips_values": [int(value) for value in initial_chips_values],
        "sample_count": int(sample_count),
        "sampling_mode": sampling_mode,
        "priority_forced_count": int(priority_forced_count),
        "n_repeats": int(n_repeats),
        "n_reference_repeats": int(n_reference_repeats or n_repeats),
        "top_action_match_rate": round(float(np.mean(top_matches)), 6),
        "mean_abs_bias": round(
            float(np.mean([float(case["mean_abs_bias"]) for case in cases])),
            6,
        ),
        "mean_l2_bias": round(
            float(np.mean([float(case["mean_l2_bias"]) for case in cases])),
            6,
        ),
        "mean_speedup": round(
            float(np.mean([float(case["per_run_speedup"]) for case in cases])),
            6,
        ),
        "min_speedup": round(
            float(np.min([float(case["per_run_speedup"]) for case in cases])),
            6,
        ),
        "cases": cases,
        "promotion": False,
    }
