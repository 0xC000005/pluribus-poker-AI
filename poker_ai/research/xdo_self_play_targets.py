"""Collect all-street XDO-lite targets from local parent self-play."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import FastPokerState
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES
from poker_ai.research.decision_value_actor import score_first_actions_for_public_state
from poker_ai.research.mixed_policy_h2h import PolicyAdapter
from poker_ai.research.native_nfsp import resolve_device


def _action_name(action: int) -> str:
    return str(INDEX_TO_ACTION.get(int(action), int(action)))


def _as_device(device: str | torch.device) -> tuple[torch.device, dict]:
    if isinstance(device, torch.device):
        return device, {
            "requested_device": str(device),
            "resolved_device": str(device),
            "device_policy": "explicit_torch_device",
        }
    info = resolve_device(str(device))
    return torch.device(info["resolved_device"]), info


def _parse_streets(values: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(values, str):
        return tuple(int(item.strip()) for item in values.split(",") if item.strip())
    return tuple(int(value) for value in values)


def _adapter_action(
    adapter: PolicyAdapter,
    state: FastPokerState,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    features = state.to_feature_vector().astype(np.float32, copy=False)
    legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
    return int(
        adapter.select_action(
            state=state,
            features=features,
            legal_mask=legal_mask,
            device=device,
            rng=rng,
        )
    )


def _adapter_continuation(
    adapter: PolicyAdapter,
    device: torch.device,
):
    def _policy(state: FastPokerState, rng: np.random.Generator) -> int:
        return _adapter_action(adapter, state, device, rng)

    return _policy


def _coverage_gate(
    street_counts: Counter[str],
    required_streets: set[int],
    *,
    min_rows_per_required_street: int,
) -> dict:
    missing = [
        int(street)
        for street in sorted(required_streets)
        if int(street_counts.get(str(int(street)), 0)) < int(min_rows_per_required_street)
    ]
    return {
        "passed": not missing,
        "required_streets": [int(street) for street in sorted(required_streets)],
        "missing_streets": missing,
        "min_rows_per_required_street": int(min_rows_per_required_street),
    }


def collect_xdo_self_play_targets(
    adapter: PolicyAdapter,
    output_npz: str | Path,
    *,
    n_states: int = 64,
    n_worlds: int = 16,
    target_streets: Sequence[int] | str = (0, 1, 2, 3),
    required_streets: Sequence[int] | str = (0, 1, 2, 3),
    min_rows_per_required_street: int = 1,
    max_hands: int = 256,
    initial_chips: int = 1000,
    small_blind: int = 50,
    big_blind: int = 100,
    max_steps_per_hand: int = 128,
    collection_exploration_epsilon: float = 0.0,
    device: str | torch.device = "auto",
    seed: int = 20260527,
    output_json: str | Path | None = None,
) -> dict:
    """Generate policy targets from states reached by a local parent policy."""
    resolved_device, device_info = _as_device(device)
    target_street_set = set(_parse_streets(target_streets))
    required_street_set = set(_parse_streets(required_streets))
    if not required_street_set.issubset(target_street_set):
        raise ValueError("required_streets must be a subset of target_streets")
    rng = np.random.default_rng(int(seed))
    continuation = _adapter_continuation(adapter, resolved_device)

    features_out: list[np.ndarray] = []
    masks_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    root_indices: list[int] = []
    street_indices: list[int] = []
    oracle_actions: list[int] = []
    parent_actions: list[int] = []
    oracle_gaps: list[float] = []
    rows: list[dict] = []
    street_counts: Counter[str] = Counter()
    n_truncated = 0
    attempted_hands = 0
    exploratory_actions = 0
    epsilon = float(np.clip(collection_exploration_epsilon, 0.0, 1.0))

    def _needs_state(street: int) -> bool:
        if len(features_out) < int(n_states):
            return True
        return (
            street in required_street_set
            and street_counts.get(str(street), 0) < int(min_rows_per_required_street)
        )

    while attempted_hands < int(max_hands):
        if len(features_out) >= int(n_states):
            gate = _coverage_gate(
                street_counts,
                required_street_set,
                min_rows_per_required_street=int(min_rows_per_required_street),
            )
            if gate["passed"]:
                break
        state = FastPokerState(
            n_players=2,
            small_blind=int(small_blind),
            big_blind=int(big_blind),
            initial_chips=int(initial_chips),
        )
        attempted_hands += 1
        for step in range(int(max_steps_per_hand)):
            if state.is_terminal:
                break
            street = int(state.stage)
            parent_action = _adapter_action(adapter, state, resolved_device, rng)
            legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
            action = parent_action
            if epsilon > 0.0 and float(rng.random()) < epsilon:
                legal = np.flatnonzero(legal_mask > 0)
                if legal.size > 0:
                    action = int(rng.choice(legal))
                    exploratory_actions += 1
            if (
                street in target_street_set
                and _needs_state(street)
            ):
                result = score_first_actions_for_public_state(
                    state,
                    n_worlds=int(n_worlds),
                    continuation_policy=continuation,
                    seed=int(seed) + 4099 * (len(features_out) + 1),
                    max_steps_per_hand=int(max_steps_per_hand),
                )
                target = np.zeros(N_ACTIONS, dtype=np.float32)
                target[int(result.best_action)] = 1.0
                selected_value = float(result.action_values[int(parent_action)])
                oracle_value = float(result.best_action_value)
                features_out.append(state.to_feature_vector().astype(np.float32, copy=False))
                masks_out.append(legal_mask)
                targets_out.append(target)
                root_indices.append(len(features_out) - 1)
                street_indices.append(street)
                oracle_actions.append(int(result.best_action))
                parent_actions.append(int(parent_action))
                oracle_gaps.append(float(oracle_value - selected_value))
                street_counts[str(street)] += 1
                n_truncated += int(result.n_truncated_rollouts)
                rows.append(
                    {
                        "row_idx": int(len(rows)),
                        "hand_idx": int(attempted_hands - 1),
                        "step": int(step),
                        "street": street,
                        "acting_player": int(state.current_player_i),
                        "parent_action": _action_name(parent_action),
                        "behavior_action": _action_name(action),
                        "oracle_action": _action_name(result.best_action),
                        "parent_action_value": selected_value,
                        "oracle_action_value": oracle_value,
                        "oracle_gap": float(oracle_value - selected_value),
                        "n_truncated_rollouts": int(result.n_truncated_rollouts),
                        "action_values": {
                            _action_name(action_idx): float(value)
                            for action_idx, value in sorted(result.action_values.items())
                        },
                    }
                )
            state.apply_action(int(action))

    if features_out:
        features = np.stack(features_out).astype(np.float32, copy=False)
        legal_masks = np.stack(masks_out).astype(np.float32, copy=False)
        target_probs = np.stack(targets_out).astype(np.float32, copy=False)
    else:
        features = np.zeros((0, N_FEATURES), dtype=np.float32)
        legal_masks = np.zeros((0, N_ACTIONS), dtype=np.float32)
        target_probs = np.zeros((0, N_ACTIONS), dtype=np.float32)
    buffer = PolicyTargetBuffer(features, legal_masks, target_probs)
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=buffer.features,
        legal_masks=buffer.legal_masks,
        target_probs=buffer.target_probs,
        weights=buffer.weights,
        root_indices=np.asarray(root_indices, dtype=np.int32),
        street_indices=np.asarray(street_indices, dtype=np.int32),
        parent_actions=np.asarray(parent_actions, dtype=np.int32),
        oracle_actions=np.asarray(oracle_actions, dtype=np.int32),
        oracle_gaps=np.asarray(oracle_gaps, dtype=np.float32),
    )

    gate = _coverage_gate(
        street_counts,
        required_street_set,
        min_rows_per_required_street=int(min_rows_per_required_street),
    )
    blockers: list[str] = []
    if int(buffer.size) < int(n_states):
        blockers.append("insufficient_targets")
    if not gate["passed"]:
        blockers.append("target_street_coverage_failed")
    if n_truncated > 0:
        blockers.append("truncated_rollouts")
    metrics = {
        "mode": "xdo_lite_self_play_policy_target_dataset",
        "passed": bool(not blockers),
        "promotion": False,
        "promotable": False,
        "promotion_blockers": [
            "target_dataset_only",
            "requires_neural_consumer_parent_population_h2h",
            "slumbot_heldout_confirmation_required",
        ],
        "blockers": blockers,
        "policy_kind": adapter.kind,
        "policy_checkpoint": adapter.checkpoint_path,
        "policy_algorithm": adapter.algorithm,
        "output_npz": str(output_npz),
        "target_size": int(buffer.size),
        "n_states": int(n_states),
        "n_worlds": int(n_worlds),
        "target_streets": [int(street) for street in sorted(target_street_set)],
        "required_streets": [int(street) for street in sorted(required_street_set)],
        "target_street_counts": dict(sorted(street_counts.items())),
        "coverage_gate": gate,
        "attempted_hands": int(attempted_hands),
        "initial_chips": int(initial_chips),
        "small_blind": int(small_blind),
        "big_blind": int(big_blind),
        "max_steps_per_hand": int(max_steps_per_hand),
        "collection_exploration_epsilon": epsilon,
        "exploratory_actions": int(exploratory_actions),
        "n_truncated_rollouts": int(n_truncated),
        "mean_oracle_gap": float(np.mean(oracle_gaps)) if oracle_gaps else 0.0,
        "oracle_action_counts": dict(
            sorted(Counter(_action_name(action) for action in oracle_actions).items())
        ),
        "parent_action_counts": dict(
            sorted(Counter(_action_name(action) for action in parent_actions).items())
        ),
        "rows": rows,
        "uses_slumbot_training_data": False,
        **device_info,
    }
    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
