"""SD-CFR-style checkpoint mixture diagnostics."""

from __future__ import annotations

import math
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from poker_ai.deep_cfr.vectorized_env import VectorizedPokerEnv
from poker_ai.research.evaluation import (
    BIG_BLIND,
    LoadedValueNetwork,
    aggregate_seed_runs,
    assert_strategy_source_supported,
    load_value_network_checkpoint,
    _strategies_from_network,
)


@dataclass(frozen=True)
class CheckpointPolicySet:
    checkpoints: tuple[LoadedValueNetwork, ...]
    weights: np.ndarray

    @property
    def size(self) -> int:
        return len(self.checkpoints)

    @property
    def metadata(self) -> list[dict[str, Any]]:
        return [dict(checkpoint.metadata) for checkpoint in self.checkpoints]


def _checkpoint_iteration(loaded: LoadedValueNetwork, fallback: int) -> int:
    iteration = loaded.metadata.get("checkpoint_iteration")
    try:
        value = int(iteration)
    except (TypeError, ValueError):
        value = fallback
    return max(value, 1)


def load_checkpoint_policy_set(
    checkpoint_paths: Iterable[str | Path],
    device: torch.device,
    *,
    strategy_source: str = "regret",
) -> CheckpointPolicySet:
    """Load saved iteration networks and weight them linearly by iteration."""
    paths = tuple(Path(path) for path in checkpoint_paths)
    if not paths:
        raise ValueError("checkpoint mixture requires at least one checkpoint")
    loaded = tuple(load_value_network_checkpoint(path, device) for path in paths)
    for checkpoint in loaded:
        assert_strategy_source_supported(checkpoint, strategy_source)
        checkpoint.value_net.eval()
    order = sorted(
        range(len(loaded)),
        key=lambda index: (
            _checkpoint_iteration(loaded[index], index + 1),
            loaded[index].metadata.get("checkpoint", ""),
        ),
    )
    ordered = tuple(loaded[index] for index in order)
    raw_weights = np.asarray(
        [_checkpoint_iteration(checkpoint, i + 1) for i, checkpoint in enumerate(ordered)],
        dtype=np.float64,
    )
    weights = raw_weights / raw_weights.sum()
    return CheckpointPolicySet(checkpoints=ordered, weights=weights)


def discover_checkpoint_paths(patterns: Iterable[str]) -> list[Path]:
    """Resolve one or more glob patterns into a stable path list."""
    paths: list[Path] = []
    for pattern in patterns:
        expanded = (
            [Path(path) for path in sorted(glob.glob(pattern))]
            if any(ch in pattern for ch in "*?[")
            else [Path(pattern)]
        )
        paths.extend(path for path in expanded if path.is_file())
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise ValueError("no checkpoint paths matched the requested mixture")
    return unique


def _advance_inactive(env: VectorizedPokerEnv) -> None:
    for i in range(env.num_envs):
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


def _step_single_net_group(
    env: VectorizedPokerEnv,
    indices: list[int],
    value_net: torch.nn.Module,
    device: torch.device,
    *,
    strategy_source: str,
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
    )
    for j, env_index in enumerate(indices):
        legal = np.flatnonzero(masks[j] > 0)
        probs = np.asarray([strategies[j][action] for action in legal], dtype=np.float64)
        probs /= probs.sum()
        env.step_single(env_index, int(np.random.choice(legal, p=probs)))


def _step_mixture_group(
    env: VectorizedPokerEnv,
    indices: list[int],
    policy_set: CheckpointPolicySet,
    model_indices: np.ndarray,
    device: torch.device,
    *,
    strategy_source: str,
) -> None:
    if not indices:
        return
    for model_index in np.unique(model_indices[indices]):
        grouped = [index for index in indices if model_indices[index] == model_index]
        _step_single_net_group(
            env,
            grouped,
            policy_set.checkpoints[int(model_index)].value_net,
            device,
            strategy_source=strategy_source,
        )


def _evaluate_mixture_vs_single_payouts(
    mixture: CheckpointPolicySet,
    baseline: LoadedValueNetwork,
    device: torch.device,
    *,
    n_games: int,
    initial_chips: int,
    seed: int,
    candidate_player: int,
    model_indices: np.ndarray,
    strategy_source: str,
) -> np.ndarray:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = VectorizedPokerEnv(n_games, 2, initial_chips=initial_chips)
    env.reset()
    max_steps = n_games * 50

    for _ in range(max_steps):
        _advance_inactive(env)
        if env.done.all():
            break

        mixture_indices: list[int] = []
        baseline_indices: list[int] = []
        for i in range(n_games):
            if env.done[i]:
                continue
            if env.states[i].current_player_i == candidate_player:
                mixture_indices.append(i)
            else:
                baseline_indices.append(i)

        _step_mixture_group(
            env,
            mixture_indices,
            mixture,
            model_indices,
            device,
            strategy_source=strategy_source,
        )
        _step_single_net_group(
            env,
            baseline_indices,
            baseline.value_net,
            device,
            strategy_source=strategy_source,
        )
    return env.get_payouts(0)


def evaluate_checkpoint_mixture_head_to_head(
    candidate_paths: Iterable[str | Path],
    baseline_checkpoint: str | Path,
    device: torch.device,
    *,
    n_games: int,
    seed: int,
    initial_chips: int | None = None,
    strategy_source: str = "regret",
) -> dict[str, Any]:
    """Duplicate-swapped local head-to-head for a sampled checkpoint mixture."""
    mixture = load_checkpoint_policy_set(candidate_paths, device, strategy_source=strategy_source)
    baseline = load_value_network_checkpoint(baseline_checkpoint, device)
    assert_strategy_source_supported(baseline, strategy_source)
    resolved_initial_chips = int(
        initial_chips or baseline.metadata.get("initial_chips") or 20_000
    )
    rng = np.random.default_rng(seed)
    model_indices = rng.choice(mixture.size, size=n_games, p=mixture.weights)

    candidate_seat0 = _evaluate_mixture_vs_single_payouts(
        mixture,
        baseline,
        device,
        n_games=n_games,
        initial_chips=resolved_initial_chips,
        seed=seed,
        candidate_player=0,
        model_indices=model_indices,
        strategy_source=strategy_source,
    )
    baseline_seat0 = _evaluate_mixture_vs_single_payouts(
        mixture,
        baseline,
        device,
        n_games=n_games,
        initial_chips=resolved_initial_chips,
        seed=seed,
        candidate_player=1,
        model_indices=model_indices,
        strategy_source=strategy_source,
    )
    paired = (candidate_seat0 - baseline_seat0) / 2.0
    avg = float(paired.mean())
    std = float(paired.std(ddof=1)) if len(paired) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(paired))) if len(paired) > 1 else 0.0
    return {
        "passed": bool(np.isfinite(paired).all()),
        "mode": "sd_cfr_checkpoint_mixture_head_to_head",
        "n_games": int(n_games * 2),
        "n_duplicate_pairs": int(n_games),
        "n_players": 2,
        "initial_chips": resolved_initial_chips,
        "seed": int(seed),
        "strategy_source": strategy_source,
        "avg_chips_per_hand": avg,
        "ci95_chips_per_hand": ci95,
        "paired_delta_lower95_chips_per_hand": avg - ci95,
        "mbb_per_hand": avg / BIG_BLIND * 1000.0,
        "ci95_mbb_per_hand": ci95 / BIG_BLIND * 1000.0,
        "candidate_checkpoints": [meta["checkpoint"] for meta in mixture.metadata],
        "candidate_iterations": [
            meta.get("checkpoint_iteration") for meta in mixture.metadata
        ],
        "candidate_weights": [round(float(weight), 8) for weight in mixture.weights],
        "baseline_checkpoint": baseline.metadata.get("checkpoint"),
        "baseline_iteration": baseline.metadata.get("checkpoint_iteration"),
        "promotable": False,
        "promotion_blockers": [
            "diagnostic_checkpoint_mixture_requires_standard_falsification_ladder",
            "local_head_to_head_requires_slumbot_confirmation",
        ],
    }


def evaluate_checkpoint_mixture_across_seeds(
    candidate_paths: Iterable[str | Path],
    baseline_checkpoint: str | Path,
    device: torch.device,
    *,
    n_games: int,
    seeds: Iterable[int],
    initial_chips: int | None = None,
    strategy_source: str = "regret",
) -> dict[str, Any]:
    paths = tuple(candidate_paths)
    runs = [
        evaluate_checkpoint_mixture_head_to_head(
            paths,
            baseline_checkpoint,
            device,
            n_games=n_games,
            seed=int(seed),
            initial_chips=initial_chips,
            strategy_source=strategy_source,
        )
        for seed in seeds
    ]
    metrics = aggregate_seed_runs(runs)
    lower95 = float(metrics.get("lower95_chips_per_hand_across_seeds", 0.0))
    if lower95 <= 0.0:
        blockers = list(metrics.get("promotion_blockers") or [])
        blocker = "local_across_seed_lower95_not_positive"
        if blocker not in blockers:
            blockers.append(blocker)
        metrics["promotion_blockers"] = blockers
        metrics["passed"] = False
    return metrics
