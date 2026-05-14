import numpy as np

from poker_ai.games.full_deck.state import N_ACTIONS, new_game
from poker_ai.research.native_nfsp import (
    NativeNFSPConfig,
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

    assert metrics["algorithm"] == "native_nfsp_mc"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["num_actions"] == 9
    assert metrics["resolved_device"] == "cpu"
    assert metrics["train_episodes"] == 2
    assert metrics["promotion"] is False
    assert metrics["train_seconds"] >= 0.0


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
            "--seed",
            "42",
        ]
    )

    assert cfg.train_episodes == 7
    assert cfg.eval_games == 3
    assert cfg.hidden_dim == 32
    assert cfg.batch_size == 16
    assert cfg.device == "cpu"
    assert cfg.seed == 42
