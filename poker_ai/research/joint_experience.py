"""Joint trajectory collection for PSRO-style response-oracle data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import random

import numpy as np
import torch

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, new_game
from poker_ai.research.mixed_policy_h2h import PolicyAdapter
from poker_ai.research.native_nfsp import get_legal_mask, resolve_device


ARRAY_KEYS = (
    "observations",
    "next_observations",
    "legal_masks",
    "next_legal_masks",
    "actions",
    "dones",
    "player_indices",
    "behavior_policy_indices",
    "hand_indices",
    "step_indices",
    "returns",
    "hand_policy_indices",
    "hand_returns",
    "meta_strategy",
)


def _normalize_meta_strategy(meta_strategy: Sequence[float], n_policies: int) -> np.ndarray:
    probs = np.asarray(meta_strategy, dtype=np.float64)
    if probs.shape != (int(n_policies),):
        raise ValueError("meta_strategy length must match policies")
    if not np.isfinite(probs).all() or np.any(probs < 0.0):
        raise ValueError("meta_strategy must contain finite nonnegative probabilities")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("meta_strategy must have positive mass")
    return (probs / total).astype(np.float32)


def _state_observation_and_mask(state) -> tuple[np.ndarray, np.ndarray]:
    if state is None or state.is_terminal:
        return (
            np.zeros(N_FEATURES, dtype=np.float32),
            np.ones(N_ACTIONS, dtype=np.bool_),
        )
    return (
        state.to_feature_vector().astype(np.float32, copy=False),
        get_legal_mask(state).astype(np.bool_, copy=False),
    )


def collect_joint_experience(
    policies: Sequence[PolicyAdapter],
    *,
    meta_strategy: Sequence[float],
    n_hands: int = 128,
    seed: int = 20260527,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    device: str | torch.device = "auto",
) -> dict:
    """Collect full-deck trajectories under a policy-population meta-strategy.

    The dataset is intentionally behavior-only. It does not train or promote a
    response oracle; it creates a reusable local-data contract for later
    maintained-library or offline-response-oracle experiments.
    """
    if not policies:
        raise ValueError("policies must contain at least one PolicyAdapter")
    n_hands = int(n_hands)
    if n_hands <= 0:
        raise ValueError("n_hands must be positive")
    max_steps = int(max_steps_per_hand)
    if max_steps <= 0:
        raise ValueError("max_steps_per_hand must be positive")
    chips = int(initial_chips)
    if chips <= 0:
        raise ValueError("initial_chips must be positive")

    if isinstance(device, torch.device):
        resolved_device = device
        device_info = {
            "requested_device": str(device),
            "resolved_device": str(device),
            "device_policy": "explicit_torch_device",
        }
    else:
        device_info = resolve_device(str(device))
        resolved_device = torch.device(device_info["resolved_device"])

    meta = _normalize_meta_strategy(meta_strategy, len(policies))
    rng = np.random.default_rng(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    next_legal_masks: list[np.ndarray] = []
    actions: list[int] = []
    dones: list[bool] = []
    player_indices: list[int] = []
    behavior_policy_indices: list[int] = []
    hand_indices: list[int] = []
    step_indices: list[int] = []
    hand_policy_indices: list[list[int]] = []
    transition_refs_by_hand: list[list[tuple[int, int]]] = []
    hand_returns: list[list[float]] = []
    truncated_hands = 0

    for hand_i in range(n_hands):
        seat_policy_indices = [
            int(rng.choice(len(policies), p=meta)),
            int(rng.choice(len(policies), p=meta)),
        ]
        hand_policy_indices.append(seat_policy_indices)
        state = new_game(2, initial_chips=chips)
        refs: list[tuple[int, int]] = []
        step_i = 0
        while not state.is_terminal and step_i < max_steps:
            player_i = int(state.player_i)
            policy_i = int(seat_policy_indices[player_i])
            adapter = policies[policy_i]
            features = state.to_feature_vector().astype(np.float32, copy=False)
            legal_mask = get_legal_mask(state).astype(bool, copy=False)
            action_idx = adapter.select_action(
                state=state,
                features=features,
                legal_mask=legal_mask,
                device=resolved_device,
                rng=rng,
            )
            next_state = state.apply_action(INDEX_TO_ACTION[int(action_idx)])
            next_features, next_mask = _state_observation_and_mask(next_state)
            done = bool(next_state.is_terminal or (step_i + 1) >= max_steps)
            observations.append(np.asarray(features, dtype=np.float32))
            legal_masks.append(np.asarray(legal_mask, dtype=np.bool_))
            next_observations.append(np.asarray(next_features, dtype=np.float32))
            next_legal_masks.append(np.asarray(next_mask, dtype=np.bool_))
            actions.append(int(action_idx))
            dones.append(done)
            player_indices.append(player_i)
            behavior_policy_indices.append(policy_i)
            hand_indices.append(hand_i)
            step_indices.append(step_i)
            refs.append((len(actions) - 1, player_i))
            state = next_state
            step_i += 1
        if not state.is_terminal:
            truncated_hands += 1
        payouts = [
            float(state.payout.get(0, 0)) / float(chips),
            float(state.payout.get(1, 0)) / float(chips),
        ]
        hand_returns.append(payouts)
        transition_refs_by_hand.append(refs)

    returns = np.zeros(len(actions), dtype=np.float32)
    for hand_i, refs in enumerate(transition_refs_by_hand):
        payouts = hand_returns[hand_i]
        for transition_i, player_i in refs:
            returns[transition_i] = np.float32(payouts[player_i])

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

    return {
        "algorithm": "joint_experience_meta_policy_dataset",
        "role": "psro_joint_experience_response_oracle_data_contract",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "policy_kinds": [policy.kind for policy in policies],
        "policy_checkpoints": [policy.checkpoint_path for policy in policies],
        "meta_strategy": meta,
        "n_hands": n_hands,
        "n_transitions": int(len(actions)),
        "truncated_hands": int(truncated_hands),
        "initial_chips": chips,
        "max_steps_per_hand": max_steps,
        "observations": observation_array,
        "next_observations": next_observation_array,
        "legal_masks": mask_array,
        "next_legal_masks": next_mask_array,
        "actions": np.asarray(actions, dtype=np.int64),
        "dones": np.asarray(dones, dtype=np.bool_),
        "player_indices": np.asarray(player_indices, dtype=np.int64),
        "behavior_policy_indices": np.asarray(behavior_policy_indices, dtype=np.int64),
        "hand_indices": np.asarray(hand_indices, dtype=np.int64),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
        "returns": returns,
        "hand_policy_indices": np.asarray(hand_policy_indices, dtype=np.int64),
        "hand_returns": np.asarray(hand_returns, dtype=np.float32),
        "uses_slumbot_training_data": False,
        "promotion": False,
        **device_info,
    }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def joint_experience_manifest(dataset: dict, output_npz: str | Path | None = None) -> dict:
    """Return JSON metadata for a joint-experience dataset."""
    manifest = {
        key: _json_safe(value)
        for key, value in dataset.items()
        if key not in ARRAY_KEYS
    }
    manifest["array_shapes"] = {
        key: [int(dim) for dim in np.asarray(dataset[key]).shape]
        for key in ARRAY_KEYS
        if key in dataset
    }
    manifest["array_dtypes"] = {
        key: str(np.asarray(dataset[key]).dtype)
        for key in ARRAY_KEYS
        if key in dataset
    }
    if output_npz is not None:
        manifest["output_npz"] = str(output_npz)
    return manifest


def save_joint_experience_npz(
    dataset: dict,
    output_npz: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict:
    """Persist dataset arrays to NPZ and write a compact JSON manifest."""
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.asarray(dataset[key])
        for key in ARRAY_KEYS
        if key in dataset
    }
    np.savez_compressed(output_path, **arrays)
    manifest = joint_experience_manifest(dataset, output_path)
    if manifest_path is not None:
        manifest_out = Path(manifest_path)
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest["manifest_json"] = str(manifest_out)
        manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
