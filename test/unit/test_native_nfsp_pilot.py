import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS, new_game
from poker_ai.research.native_nfsp import (
    NativeNFSPConfig,
    ReservoirPolicyBuffer,
    build_player_transitions,
    evaluate_native_nfsp_head_to_head,
    evaluate_native_nfsp_checkpoint,
    get_legal_mask,
    masked_uniform,
    run_native_nfsp_pilot,
    sample_episode_policy_modes,
    select_action,
)


def test_get_legal_mask_matches_full_deck_action_contract():
    state = new_game(2)
    mask = get_legal_mask(state)

    assert mask.shape == (N_ACTIONS,)
    assert mask.dtype == np.float32
    assert mask.sum() >= 2.0


def test_select_action_never_returns_illegal_index():
    legal_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = np.array([0.1, 0.2, 0.7, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    for seed in range(20):
        action = select_action(probs, legal_mask, rng=np.random.default_rng(seed))
        assert legal_mask[action] > 0


def test_masked_uniform_falls_back_to_legal_distribution():
    legal_mask = np.array([0, 1, 0, 1], dtype=np.float32)

    probs = masked_uniform(legal_mask)

    np.testing.assert_allclose(probs, np.array([0.0, 0.5, 0.0, 0.5]))


def test_sample_episode_policy_modes_once_per_player():
    always_br = sample_episode_policy_modes(
        n_players=2,
        anticipatory_param=1.0,
        rng=np.random.default_rng(1),
    )
    never_br = sample_episode_policy_modes(
        n_players=2,
        anticipatory_param=0.0,
        rng=np.random.default_rng(1),
    )

    assert always_br == (True, True)
    assert never_br == (False, False)


def test_build_player_transitions_links_next_decision_for_same_player():
    features0 = np.full(4, 0.1, dtype=np.float32)
    features1 = np.full(4, 0.2, dtype=np.float32)
    features2 = np.full(4, 0.3, dtype=np.float32)
    mask = np.array([1.0, 1.0], dtype=np.float32)
    records = [
        (0, features0, mask, 0, True),
        (1, features1, mask, 1, False),
        (0, features2, mask, 1, True),
    ]

    transitions = build_player_transitions(records, [0.75, -0.75])

    assert len(transitions) == 3
    np.testing.assert_allclose(transitions[0].next_features, features2)
    assert transitions[0].reward == 0.0
    assert transitions[0].done is False
    assert transitions[1].reward == -0.75
    assert transitions[1].done is True
    assert transitions[2].reward == 0.75
    assert transitions[2].done is True


def test_reservoir_policy_buffer_tracks_seen_items_without_fifo_bias():
    buffer = ReservoirPolicyBuffer(capacity=2)
    rng = np.random.default_rng(3)
    feature = np.array([0.0], dtype=np.float32)
    mask = np.array([1.0], dtype=np.float32)

    for action in range(10):
        buffer.add(feature + action, mask, action, rng=rng)

    stored_actions = [item[2] for item in buffer.items]
    assert len(buffer) == 2
    assert buffer.n_seen == 10
    assert stored_actions != [8, 9]


def test_run_native_nfsp_pilot_smoke_uses_nine_action_full_deck_contract():
    metrics = run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=2,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=123,
        )
    )

    assert metrics["algorithm"] == "native_nfsp_dqn"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["num_actions"] == 9
    assert metrics["resolved_device"] == "cpu"
    assert metrics["train_episodes"] == 2
    assert metrics["promotion"] is False
    assert metrics["train_seconds"] >= 0.0


def test_run_native_nfsp_pilot_writes_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "native_nfsp.pt"

    metrics = run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=321,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["checkpoint_path"] == str(checkpoint_path)
    assert payload["algorithm"] == "native_nfsp_dqn"
    assert payload["num_actions"] == 9
    assert "avg_net_state_dict" in payload
    assert "q_net_state_dict" in payload


def test_evaluate_native_nfsp_checkpoint_roundtrip(tmp_path):
    checkpoint_path = tmp_path / "native_nfsp.pt"
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=777,
            checkpoint_path=str(checkpoint_path),
        )
    )

    metrics = evaluate_native_nfsp_checkpoint(
        str(checkpoint_path),
        eval_games=2,
        device="cpu",
        seed=778,
    )

    assert metrics["algorithm"] == "native_nfsp_dqn"
    assert metrics["source_checkpoint"] == str(checkpoint_path)
    assert metrics["eval_games"] == 2
    assert metrics["resolved_device"] == "cpu"
    assert metrics["promotion"] is False


def test_evaluate_native_nfsp_head_to_head_roundtrip(tmp_path):
    checkpoint_path = tmp_path / "native_nfsp.pt"
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=779,
            checkpoint_path=str(checkpoint_path),
        )
    )

    metrics = evaluate_native_nfsp_head_to_head(
        str(checkpoint_path),
        str(checkpoint_path),
        n_games=2,
        device="cpu",
        seed=780,
    )

    assert metrics["algorithm"] == "native_nfsp_dqn_h2h"
    assert metrics["candidate_checkpoint"] == str(checkpoint_path)
    assert metrics["baseline_checkpoint"] == str(checkpoint_path)
    assert metrics["n_games"] == 2
    assert metrics["resolved_device"] == "cpu"
    assert abs(metrics["mean_candidate_payoff"]) < 1e-6
    assert metrics["promotion"] is False


def test_native_nfsp_cli_builds_config_from_args():
    from scripts.run_native_nfsp_pilot import build_config

    cfg = build_config(
        [
            "--train-episodes",
            "7",
            "--eval-games",
            "3",
            "--hidden-dim",
            "32",
            "--batch-size",
            "16",
            "--device",
            "cpu",
            "--checkpoint-out",
            "models/native.pt",
            "--seed",
            "42",
        ]
    )

    assert cfg.train_episodes == 7
    assert cfg.eval_games == 3
    assert cfg.hidden_dim == 32
    assert cfg.batch_size == 16
    assert cfg.device == "cpu"
    assert cfg.checkpoint_path == "models/native.pt"
    assert cfg.seed == 42
