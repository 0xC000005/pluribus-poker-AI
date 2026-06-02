"""Online compiled Rainbow response learning for native heads-up NLHE.

The collector records semi-MDP learner transitions: one replay row starts at
the learner's decision, applies that action, advances opponent actions locally,
and ends only at the next learner decision or terminal/truncation.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import FastPokerState
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.compiled_fast_rollout import (
    CompiledFastStateBatch,
    compiled_apply_actions,
    compiled_feature_vectors,
    compiled_legal_masks,
)
from poker_ai.research.compiled_joint_experience import (
    CompiledJointPolicy,
    load_compiled_joint_policy,
)
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.native_rollout_substrate import _new_seeded_fast_state


def _terminal_next_observation() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros(N_FEATURES, dtype=np.float32),
        np.ones(N_ACTIONS, dtype=np.bool_),
    )


def _q_values_from_distribution_model(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    num_atoms: int,
    device: torch.device,
) -> torch.Tensor:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        distribution = output[0] if isinstance(output, tuple) else output
        support = torch.linspace(-1.0, 1.0, int(num_atoms), dtype=torch.float32, device=device)
        return torch.sum(distribution * support.view(1, 1, -1), dim=-1)


def _select_masked_actions(
    scores: torch.Tensor,
    masks: np.ndarray,
    *,
    rng: np.random.Generator,
    epsilon: float,
) -> np.ndarray:
    masks_bool = np.asarray(masks, dtype=np.bool_)
    score_array = scores.detach().cpu().numpy().astype(np.float32, copy=False)
    actions = np.zeros(score_array.shape[0], dtype=np.int16)
    for row_i in range(score_array.shape[0]):
        legal = np.flatnonzero(masks_bool[row_i])
        if legal.size <= 0:
            actions[row_i] = 0
            continue
        if float(epsilon) > 0.0 and float(rng.random()) < float(epsilon):
            actions[row_i] = int(rng.choice(legal))
            continue
        masked = np.full(N_ACTIONS, -1.0e30, dtype=np.float32)
        masked[legal] = score_array[row_i, legal]
        actions[row_i] = int(np.argmax(masked))
    return actions


def _policy_scores(
    policy: CompiledJointPolicy,
    features: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        output = policy.module(x)
        scores = output[0] if isinstance(output, tuple) else output
        if scores.ndim != 2 or scores.shape[1] != N_ACTIONS:
            raise ValueError("compiled opponent policy must return shape (batch, N_ACTIONS)")
        return scores


def _select_learner_actions(
    model: torch.nn.Module,
    features: np.ndarray,
    masks: np.ndarray,
    *,
    num_atoms: int,
    device: torch.device,
    rng: np.random.Generator,
    epsilon: float,
) -> np.ndarray:
    scores = _q_values_from_distribution_model(
        model,
        features,
        num_atoms=int(num_atoms),
        device=device,
    )
    return _select_masked_actions(scores, masks, rng=rng, epsilon=float(epsilon))


def _select_opponent_actions(
    *,
    model: torch.nn.Module,
    opponent_policies: Sequence[CompiledJointPolicy],
    opponent_indices: np.ndarray,
    hand_indices: Sequence[int],
    features: np.ndarray,
    masks: np.ndarray,
    num_atoms: int,
    device: torch.device,
    rng: np.random.Generator,
    epsilon: float,
) -> np.ndarray:
    if not opponent_policies:
        return _select_learner_actions(
            model,
            features,
            masks,
            num_atoms=int(num_atoms),
            device=device,
            rng=rng,
            epsilon=float(epsilon),
        )

    actions = np.full(len(hand_indices), -1, dtype=np.int16)
    grouped: dict[int, list[int]] = {}
    for row_i, hand_i in enumerate(hand_indices):
        policy_i = int(opponent_indices[int(hand_i)])
        grouped.setdefault(policy_i, []).append(row_i)
    for policy_i, local_rows in grouped.items():
        score = _policy_scores(
            opponent_policies[int(policy_i)],
            features[local_rows].astype(np.float32, copy=False),
            device=device,
        )
        selected = _select_masked_actions(
            score,
            masks[local_rows],
            rng=rng,
            epsilon=0.0,
        )
        for out_i, action in zip(local_rows, selected, strict=False):
            actions[out_i] = int(action)
    return actions


def _payoff_for_learner(
    compiled: CompiledFastStateBatch,
    hand_i: int,
    *,
    learner_seat: int,
    initial_chips: int,
) -> float:
    if int(compiled.stage[int(hand_i)]) < FastPokerState.SHOWDOWN:
        return 0.0
    return float(
        (float(compiled.chips[int(hand_i), int(learner_seat)]) - float(initial_chips))
        / float(initial_chips)
    )


def collect_compiled_rainbow_response_transitions(
    model: torch.nn.Module,
    *,
    num_atoms: int,
    opponent_policies: Sequence[CompiledJointPolicy] | None = None,
    opponent_meta_strategy: Sequence[float] | None = None,
    n_hands: int = 128,
    batch_size: int = 128,
    seed: int = 20260624,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    learner_seat: int = 0,
    device: str | torch.device = "auto",
    epsilon: float = 0.1,
) -> dict[str, Any]:
    """Collect replay-ready semi-MDP learner transitions from compiled states."""

    if int(learner_seat) not in {0, 1}:
        raise ValueError("learner_seat must be 0 or 1")
    n_hands = int(n_hands)
    batch_size = int(batch_size)
    max_steps = int(max_steps_per_hand)
    if n_hands <= 0:
        raise ValueError("n_hands must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps_per_hand must be positive")
    device_info = resolve_device(str(device)) if not isinstance(device, torch.device) else {
        "requested_device": str(device),
        "resolved_device": str(device),
        "device_policy": "explicit_torch_device",
    }
    resolved_device = torch.device(device_info["resolved_device"])
    rng = np.random.default_rng(int(seed))
    opponent_policies = list(opponent_policies or [])
    for policy in opponent_policies:
        policy.module.to(resolved_device)
        policy.module.eval()
        for parameter in policy.module.parameters():
            parameter.requires_grad_(False)

    if opponent_policies:
        if opponent_meta_strategy is None:
            opponent_probs = np.full(len(opponent_policies), 1.0 / len(opponent_policies), dtype=np.float64)
        else:
            opponent_probs = np.asarray(opponent_meta_strategy, dtype=np.float64)
            if opponent_probs.shape != (len(opponent_policies),):
                raise ValueError("opponent_meta_strategy must match opponent_policies")
            total = float(opponent_probs.sum())
            if total <= 0.0:
                raise ValueError("opponent_meta_strategy must have positive mass")
            opponent_probs = opponent_probs / total
        opponent_indices = rng.choice(len(opponent_policies), size=n_hands, p=opponent_probs)
    else:
        opponent_probs = np.zeros(0, dtype=np.float64)
        opponent_indices = np.zeros(n_hands, dtype=np.int64)

    states = [_new_seeded_fast_state(seed, hand_i, int(initial_chips)) for hand_i in range(n_hands)]
    compiled = CompiledFastStateBatch.from_fast_states(states)
    steps_per_hand = np.zeros(n_hands, dtype=np.int32)

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    next_legal_masks: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    hand_indices: list[int] = []
    step_indices: list[int] = []
    learner_forward_calls = 0
    opponent_forward_calls = 0
    needs_python_showdown = 0
    started = time.perf_counter()

    while True:
        live = [
            hand_i
            for hand_i in range(n_hands)
            if int(compiled.stage[hand_i]) < FastPokerState.SHOWDOWN
            and int(steps_per_hand[hand_i]) < max_steps
        ]
        if not live:
            break

        current_players = compiled.current_players()
        learner_ready = [hand_i for hand_i in live if int(current_players[hand_i]) == int(learner_seat)]
        if not learner_ready:
            opponent_ready = [hand_i for hand_i in live if int(current_players[hand_i]) != int(learner_seat)]
            for start in range(0, len(opponent_ready), batch_size):
                chunk = opponent_ready[start : start + batch_size]
                features = compiled_feature_vectors(compiled)[chunk].astype(np.float32, copy=False)
                masks = compiled_legal_masks(compiled).astype(bool, copy=False)[chunk]
                selected = _select_opponent_actions(
                    model=model,
                    opponent_policies=opponent_policies,
                    opponent_indices=opponent_indices,
                    hand_indices=chunk,
                    features=features,
                    masks=masks,
                    num_atoms=int(num_atoms),
                    device=resolved_device,
                    rng=rng,
                    epsilon=float(epsilon),
                )
                action_array = np.full(n_hands, -1, dtype=np.int16)
                for row_i, hand_i in enumerate(chunk):
                    action_array[int(hand_i)] = int(selected[row_i])
                    steps_per_hand[int(hand_i)] += 1
                result = compiled_apply_actions(compiled, action_array)
                needs_python_showdown += int(result["needs_python_showdown"])
                opponent_forward_calls += 1
            continue

        all_features = compiled_feature_vectors(compiled)
        all_masks = compiled_legal_masks(compiled).astype(bool, copy=False)
        for start in range(0, len(learner_ready), batch_size):
            chunk = learner_ready[start : start + batch_size]
            features = all_features[chunk].astype(np.float32, copy=False)
            masks = all_masks[chunk].astype(np.bool_, copy=False)
            selected = _select_learner_actions(
                model,
                features,
                masks,
                num_atoms=int(num_atoms),
                device=resolved_device,
                rng=rng,
                epsilon=float(epsilon),
            )
            learner_forward_calls += 1
            action_array = np.full(n_hands, -1, dtype=np.int16)
            pending: list[tuple[int, int, np.ndarray, np.ndarray, int]] = []
            for row_i, hand_i in enumerate(chunk):
                action_idx = int(selected[row_i])
                action_array[int(hand_i)] = action_idx
                pending.append(
                    (
                        int(hand_i),
                        int(steps_per_hand[int(hand_i)]),
                        features[row_i].astype(np.float32, copy=False),
                        masks[row_i].astype(np.bool_, copy=False),
                        action_idx,
                    )
                )
                steps_per_hand[int(hand_i)] += 1
            result = compiled_apply_actions(compiled, action_array)
            needs_python_showdown += int(result["needs_python_showdown"])

            pending_hands = [hand_i for hand_i, *_rest in pending]
            while True:
                live_pending = [
                    hand_i
                    for hand_i in pending_hands
                    if int(compiled.stage[hand_i]) < FastPokerState.SHOWDOWN
                    and int(steps_per_hand[hand_i]) < max_steps
                    and int(compiled.current_players()[hand_i]) != int(learner_seat)
                ]
                if not live_pending:
                    break
                for opp_start in range(0, len(live_pending), batch_size):
                    opp_chunk = live_pending[opp_start : opp_start + batch_size]
                    post_features = compiled_feature_vectors(compiled)[opp_chunk].astype(
                        np.float32,
                        copy=False,
                    )
                    post_masks = compiled_legal_masks(compiled).astype(bool, copy=False)[opp_chunk]
                    opp_selected = _select_opponent_actions(
                        model=model,
                        opponent_policies=opponent_policies,
                        opponent_indices=opponent_indices,
                        hand_indices=opp_chunk,
                        features=post_features,
                        masks=post_masks,
                        num_atoms=int(num_atoms),
                        device=resolved_device,
                        rng=rng,
                        epsilon=float(epsilon),
                    )
                    opp_action_array = np.full(n_hands, -1, dtype=np.int16)
                    for row_i, hand_i in enumerate(opp_chunk):
                        opp_action_array[int(hand_i)] = int(opp_selected[row_i])
                        steps_per_hand[int(hand_i)] += 1
                    opp_result = compiled_apply_actions(compiled, opp_action_array)
                    needs_python_showdown += int(opp_result["needs_python_showdown"])
                    opponent_forward_calls += 1

            post_all_features = compiled_feature_vectors(compiled)
            post_all_masks = compiled_legal_masks(compiled).astype(bool, copy=False)
            for hand_i, step_i, obs, mask, action_idx in pending:
                is_terminal = int(compiled.stage[hand_i]) >= FastPokerState.SHOWDOWN
                is_truncated = (not is_terminal) and int(steps_per_hand[hand_i]) >= max_steps
                if is_terminal or is_truncated:
                    next_obs, next_mask = _terminal_next_observation()
                else:
                    next_obs = post_all_features[hand_i].astype(np.float32, copy=False)
                    next_mask = post_all_masks[hand_i].astype(np.bool_, copy=False)
                observations.append(obs)
                legal_masks.append(mask)
                next_observations.append(next_obs)
                next_legal_masks.append(next_mask)
                actions.append(int(action_idx))
                rewards.append(
                    _payoff_for_learner(
                        compiled,
                        hand_i,
                        learner_seat=int(learner_seat),
                        initial_chips=int(initial_chips),
                    )
                )
                terminated.append(bool(is_terminal))
                truncated.append(bool(is_truncated))
                hand_indices.append(int(hand_i))
                step_indices.append(int(step_i))

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if observations:
        observation_array = np.stack(observations).astype(np.float32, copy=False)
        next_observation_array = np.stack(next_observations).astype(np.float32, copy=False)
        mask_array = np.stack(legal_masks).astype(np.bool_, copy=False)
        next_mask_array = np.stack(next_legal_masks).astype(np.bool_, copy=False)
    else:
        observation_array = np.zeros((0, N_FEATURES), dtype=np.float32)
        next_observation_array = np.zeros((0, N_FEATURES), dtype=np.float32)
        mask_array = np.zeros((0, N_ACTIONS), dtype=np.bool_)
        next_mask_array = np.zeros((0, N_ACTIONS), dtype=np.bool_)

    n_transitions = int(len(actions))
    return {
        "algorithm": "compiled_rainbow_response_transition_batch",
        "role": "online_response_oracle_replay",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "collector_backend": "compiled-fast-state",
        "semi_mdp_transitions": True,
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "learner_seat": int(learner_seat),
        "n_hands": int(n_hands),
        "n_transitions": n_transitions,
        "batch_size": int(batch_size),
        "initial_chips": int(initial_chips),
        "max_steps_per_hand": int(max_steps),
        "seconds": float(seconds),
        "transitions_per_second": float(n_transitions / max(seconds, 1e-12)),
        "learner_forward_calls": int(learner_forward_calls),
        "opponent_forward_calls": int(opponent_forward_calls),
        "needs_python_showdown": int(needs_python_showdown),
        "truncated_hands": int(np.sum(compiled.stage < FastPokerState.SHOWDOWN)),
        "opponent_policy_kinds": [policy.kind for policy in opponent_policies],
        "opponent_policy_checkpoints": [policy.checkpoint_path for policy in opponent_policies],
        "opponent_policy_algorithms": [policy.algorithm for policy in opponent_policies],
        "opponent_meta_strategy": opponent_probs.tolist(),
        "observations": observation_array,
        "next_observations": next_observation_array,
        "legal_masks": mask_array,
        "next_legal_masks": next_mask_array,
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "hand_indices": np.asarray(hand_indices, dtype=np.int64),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
        "uses_slumbot_training_data": False,
        "native_action_projection": False,
        "promotion": False,
        **device_info,
    }


def _add_transitions_to_replay(replay: Any, transitions: dict[str, Any]) -> int:
    from tianshou.data import Batch

    observations = transitions["observations"]
    n_transitions = int(observations.shape[0])
    for row_i in range(n_transitions):
        replay.add(
            Batch(
                obs=Batch(
                    obs=transitions["observations"][row_i],
                    mask=transitions["legal_masks"][row_i],
                ),
                act=int(transitions["actions"][row_i]),
                rew=float(transitions["rewards"][row_i]),
                terminated=bool(transitions["terminated"][row_i]),
                truncated=bool(transitions["truncated"][row_i]),
                obs_next=Batch(
                    obs=transitions["next_observations"][row_i],
                    mask=transitions["next_legal_masks"][row_i],
                ),
                info={},
            )
        )
    return n_transitions


def _parse_policy_specs(
    specs: Sequence[str] | None,
    *,
    device: torch.device,
) -> list[CompiledJointPolicy]:
    policies: list[CompiledJointPolicy] = []
    for spec in specs or []:
        if ":" not in str(spec):
            raise ValueError("opponent policy specs must use KIND:PATH")
        kind, checkpoint = str(spec).split(":", 1)
        policies.append(load_compiled_joint_policy(checkpoint, kind=kind, device=device))
    return policies


def run_compiled_rainbow_response_oracle(
    *,
    collect_iterations: int = 4,
    games_per_iteration: int = 512,
    collector_batch_size: int = 128,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    learner_seat: int = 0,
    updates_per_collect: int = 4,
    batch_size: int = 256,
    replay_size: int = 100_000,
    hidden_dim: int = 256,
    num_atoms: int = 51,
    lr: float = 1e-3,
    gamma: float = 0.99,
    n_step: int = 1,
    target_update_freq: int = 50,
    epsilon: float = 0.1,
    seed: int = 20260624,
    device: str = "auto",
    opponent_policy_specs: Sequence[str] | None = None,
    opponent_meta_strategy: Sequence[float] | None = None,
    checkpoint_in: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Train or continue a Rainbow response oracle with compiled collection."""

    from tianshou.algorithm.algorithm_base import policy_within_training_step
    from tianshou.algorithm.modelfree.c51 import C51Policy
    from tianshou.algorithm.modelfree.rainbow import RainbowDQN
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import ReplayBuffer

    from scripts.run_tianshou_rainbow_native_control import (
        NativeRainbowPokerEnv,
        _RainbowDistributionNet,
        _rainbow_state_dict_from_payload,
    )

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device_info = resolve_device(str(device))
    resolved_device = torch.device(device_info["resolved_device"])
    checkpoint_payload = None
    if checkpoint_in is not None:
        checkpoint_payload = torch.load(checkpoint_in, map_location=resolved_device, weights_only=False)
        if int(checkpoint_payload.get("num_actions", -1)) != N_ACTIONS:
            raise ValueError("Rainbow checkpoint action count does not match native full-deck contract")
        if int(checkpoint_payload.get("num_features", -1)) != N_FEATURES:
            raise ValueError("Rainbow checkpoint feature count does not match native full-deck contract")
    opponent_policies = _parse_policy_specs(opponent_policy_specs, device=resolved_device)
    base_env = NativeRainbowPokerEnv(
        seed=int(seed),
        initial_chips=int(initial_chips),
        max_steps_per_hand=int(max_steps_per_game),
        state_backend="fast-state",
    )
    model = _RainbowDistributionNet(
        hidden_dim=int(hidden_dim),
        num_atoms=int(num_atoms),
        device=resolved_device,
    ).to(resolved_device)
    if checkpoint_payload is not None:
        model.load_state_dict(_rainbow_state_dict_from_payload(checkpoint_payload))
    policy = C51Policy(
        model=model,
        action_space=base_env.action_space,
        observation_space=base_env.observation_space,
        num_atoms=int(num_atoms),
        v_min=-1.0,
        v_max=1.0,
        eps_training=float(epsilon),
        eps_inference=0.0,
    )
    policy.to(resolved_device)
    algorithm = RainbowDQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=float(lr)),
        gamma=float(gamma),
        n_step_return_horizon=int(n_step),
        target_update_freq=int(target_update_freq),
    )
    replay = ReplayBuffer(size=max(int(replay_size), 1))

    losses: list[float] = []
    collector_seconds = 0.0
    collector_transitions = 0
    collector_hands = 0
    learner_updates = 0
    needs_python_showdown = 0
    started = time.perf_counter()
    for iteration_i in range(int(collect_iterations)):
        transitions = collect_compiled_rainbow_response_transitions(
            model,
            num_atoms=int(num_atoms),
            opponent_policies=opponent_policies,
            opponent_meta_strategy=opponent_meta_strategy,
            n_hands=int(games_per_iteration),
            batch_size=int(collector_batch_size),
            seed=int(seed) + iteration_i * 100_000,
            initial_chips=int(initial_chips),
            max_steps_per_hand=int(max_steps_per_game),
            learner_seat=int(learner_seat),
            device=resolved_device,
            epsilon=float(epsilon),
        )
        added = _add_transitions_to_replay(replay, transitions)
        collector_transitions += int(added)
        collector_hands += int(games_per_iteration)
        collector_seconds += float(transitions["seconds"])
        needs_python_showdown += int(transitions["needs_python_showdown"])
        if len(replay) <= 0:
            continue
        for _ in range(max(1, int(updates_per_collect))):
            with policy_within_training_step(algorithm.policy):
                stats = algorithm.update(replay, sample_size=min(int(batch_size), len(replay)))
            learner_updates += 1
            loss = getattr(stats, "loss", None)
            if loss is not None:
                losses.append(float(loss))
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started

    learner_mode = "continued" if checkpoint_payload is not None else "fresh"
    metrics: dict[str, Any] = {
        "algorithm": "compiled_tianshou_rainbow_response_oracle",
        "role": "online_compiled_response_oracle",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "collector_backend": "compiled-fast-state",
        "semi_mdp_transitions": True,
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_training_data": False,
        "warning": (
            f"{learner_mode.capitalize()} local Rainbow response learner using "
            "compiled semi-MDP poker transitions. This is not promotion evidence "
            "without H2H and league gates."
        ),
        **device_info,
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "learner_seat": int(learner_seat),
        "collect_iterations": int(collect_iterations),
        "games_per_iteration": int(games_per_iteration),
        "collector_batch_size": int(collector_batch_size),
        "collector_hands": int(collector_hands),
        "collector_transitions": int(collector_transitions),
        "collector_seconds": float(collector_seconds),
        "collector_transitions_per_second": float(
            collector_transitions / max(collector_seconds, 1e-12)
        ),
        "replay_transitions": int(len(replay)),
        "updates": int(learner_updates),
        "updates_per_collect": int(updates_per_collect),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "num_atoms": int(num_atoms),
        "lr": float(lr),
        "gamma": float(gamma),
        "n_step": int(n_step),
        "target_update_freq": int(target_update_freq),
        "epsilon": float(epsilon),
        "opponent_policy_specs": list(opponent_policy_specs or []),
        "opponent_meta_strategy": list(opponent_meta_strategy or []),
        "checkpoint_in": str(checkpoint_in) if checkpoint_in is not None else None,
        "checkpoint_in_algorithm": (
            str(checkpoint_payload.get("algorithm")) if checkpoint_payload is not None else None
        ),
        "needs_python_showdown": int(needs_python_showdown),
        "train_seconds": float(train_seconds),
        "updates_per_second": float(learner_updates / max(train_seconds, 1e-12)),
        "last_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "promotion": False,
        "league_eligible": False,
        "passed": bool(collector_transitions > 0 and learner_updates >= int(collect_iterations)),
    }
    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "environment": metrics["environment"],
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "num_atoms": int(num_atoms),
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "parent_checkpoint": str(checkpoint_in) if checkpoint_in is not None else None,
                "config": {
                    "initial_chips": int(initial_chips),
                    "max_steps_per_hand": int(max_steps_per_game),
                    "learner_seat": int(learner_seat),
                },
            },
            path,
        )
        metrics["checkpoint_path"] = str(path)
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
