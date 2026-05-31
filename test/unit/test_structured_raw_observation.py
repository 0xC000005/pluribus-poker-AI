import copy

import numpy as np
import torch

from poker_ai.games.full_deck.state import ACTION_TO_INDEX, N_FEATURES, new_game
from poker_ai.research.native_ppo_policy import NativePPOConfig, run_native_ppo_policy_pilot
from poker_ai.research.structured_observation import (
    RAW_SEQUENCE_ACTION_AMOUNT_OFFSET,
    RAW_SEQUENCE_ACTION_OFFSET,
    RAW_SEQUENCE_ACTION_RECORD_WIDTH,
    RAW_SEQUENCE_ACTION_TOKEN_OFFSET,
    RAW_SEQUENCE_N_FEATURES,
    encode_raw_sequence_observation,
)


def test_raw_sequence_observation_records_ordered_actions_and_amounts():
    state = new_game(2, initial_chips=1000).apply_action("raise_1.0")

    records = getattr(state, "_action_records", [])
    assert records
    assert records[0]["action"] == "raise_1.0"
    assert records[0]["amount_added"] > 0

    features = encode_raw_sequence_observation(state)
    first_slot = features[
        RAW_SEQUENCE_ACTION_OFFSET : RAW_SEQUENCE_ACTION_OFFSET
        + RAW_SEQUENCE_ACTION_RECORD_WIDTH
    ]

    assert features.shape == (RAW_SEQUENCE_N_FEATURES,)
    assert RAW_SEQUENCE_N_FEATURES > N_FEATURES
    assert first_slot[RAW_SEQUENCE_ACTION_TOKEN_OFFSET + ACTION_TO_INDEX["raise_1.0"]] == 1.0
    assert first_slot[RAW_SEQUENCE_ACTION_AMOUNT_OFFSET] > 0.0


def test_raw_sequence_observation_is_order_sensitive_without_aggregate_counts():
    state = new_game(2, initial_chips=1000).apply_action("call").apply_action("raise_1.0")
    swapped_state = copy.deepcopy(state)
    swapped_state._action_records = list(reversed(swapped_state._action_records))

    ordered = encode_raw_sequence_observation(state)
    swapped = encode_raw_sequence_observation(swapped_state)

    assert np.allclose(
        ordered[:RAW_SEQUENCE_ACTION_OFFSET],
        swapped[:RAW_SEQUENCE_ACTION_OFFSET],
    )
    assert not np.allclose(ordered, swapped)


def test_native_ppo_policy_can_train_with_raw_sequence_features(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_raw_sequence.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            rollout_episodes_per_update=1,
            ppo_epochs=1,
            feature_mode="raw_sequence",
            device="cpu",
            seed=2040,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["feature_mode"] == "raw_sequence"
    assert metrics["num_features"] == RAW_SEQUENCE_N_FEATURES
    assert payload["num_features"] == RAW_SEQUENCE_N_FEATURES
    assert payload["config"]["feature_mode"] == "raw_sequence"
    assert "policy_net_state_dict" in payload
