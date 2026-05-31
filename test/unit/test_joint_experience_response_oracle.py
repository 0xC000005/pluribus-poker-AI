import numpy as np
import pytest

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES


pytest.importorskip("tianshou")


def _write_tiny_joint_dataset(path):
    n = 6
    observations = np.zeros((n, N_FEATURES), dtype=np.float32)
    next_observations = np.ones((n, N_FEATURES), dtype=np.float32) * 0.1
    legal_masks = np.ones((n, N_ACTIONS), dtype=np.bool_)
    next_legal_masks = np.ones((n, N_ACTIONS), dtype=np.bool_)
    actions = np.arange(n, dtype=np.int64) % N_ACTIONS
    returns = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    dones = np.array([False, False, True, False, False, True], dtype=np.bool_)
    np.savez_compressed(
        path,
        observations=observations,
        next_observations=next_observations,
        legal_masks=legal_masks,
        next_legal_masks=next_legal_masks,
        actions=actions,
        dones=dones,
        player_indices=np.arange(n, dtype=np.int64) % 2,
        behavior_policy_indices=np.zeros(n, dtype=np.int64),
        hand_indices=np.repeat(np.arange(2, dtype=np.int64), 3),
        step_indices=np.tile(np.arange(3, dtype=np.int64), 2),
        returns=returns,
        hand_policy_indices=np.zeros((2, 2), dtype=np.int64),
        hand_returns=np.array([[0.5, -0.5], [-0.25, 0.25]], dtype=np.float32),
        meta_strategy=np.array([1.0], dtype=np.float32),
    )


def test_load_joint_experience_replay_buffer_preserves_masks_and_rewards(tmp_path):
    from poker_ai.research.joint_experience_response_oracle import (
        load_joint_experience_replay_buffer,
    )

    dataset_path = tmp_path / "joint.npz"
    _write_tiny_joint_dataset(dataset_path)

    replay, summary = load_joint_experience_replay_buffer(dataset_path)
    batch, _indices = replay.sample(3)

    assert len(replay) == 6
    assert summary["n_transitions"] == 6
    assert summary["num_actions"] == N_ACTIONS
    assert summary["num_features"] == N_FEATURES
    assert batch.obs.obs.shape == (3, N_FEATURES)
    assert batch.obs.mask.shape == (3, N_ACTIONS)
    assert batch.obs_next.obs.shape == (3, N_FEATURES)
    assert batch.obs_next.mask.shape == (3, N_ACTIONS)
    assert np.all(np.isfinite(batch.rew))


def test_train_joint_experience_rainbow_response_oracle_writes_checkpoint(tmp_path):
    from poker_ai.research.joint_experience_response_oracle import (
        train_joint_experience_rainbow_response_oracle,
    )

    dataset_path = tmp_path / "joint.npz"
    checkpoint_path = tmp_path / "response.pt"
    _write_tiny_joint_dataset(dataset_path)

    metrics = train_joint_experience_rainbow_response_oracle(
        dataset_path,
        updates=2,
        batch_size=3,
        hidden_dim=16,
        num_atoms=11,
        device="cpu",
        checkpoint_out=checkpoint_path,
    )

    assert metrics["algorithm"] == "tianshou_rainbow_joint_experience_response_oracle"
    assert metrics["role"] == "psro_response_oracle_from_joint_experience"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["promotion"] is False
    assert metrics["updates"] == 2
    assert metrics["n_transitions"] == 6
    assert checkpoint_path.exists()


def test_train_joint_experience_response_oracle_cli_writes_metrics(monkeypatch, tmp_path):
    from scripts import train_joint_experience_response_oracle as cli

    output_json = tmp_path / "metrics.json"
    checkpoint_path = tmp_path / "response.pt"

    def _fake_train(dataset_npz, **kwargs):
        assert str(dataset_npz) == "joint.npz"
        assert kwargs["updates"] == 7
        assert kwargs["checkpoint_out"] == checkpoint_path
        return {
            "algorithm": "tianshou_rainbow_joint_experience_response_oracle",
            "updates": kwargs["updates"],
            "promotion": False,
        }

    monkeypatch.setattr(cli, "train_joint_experience_rainbow_response_oracle", _fake_train)

    exit_code = cli.main(
        [
            "--dataset-npz",
            "joint.npz",
            "--updates",
            "7",
            "--checkpoint-out",
            str(checkpoint_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    assert output_json.exists()
