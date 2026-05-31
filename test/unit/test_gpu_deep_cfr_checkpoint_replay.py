import numpy as np
import torch

from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from scripts.poker_autoresearch_train import summarize_resume_checkpoint_metadata


def _features(offset: float) -> np.ndarray:
    return (np.arange(N_FEATURES, dtype=np.float32) + offset) / 1000.0


def _advantages(offset: float) -> np.ndarray:
    return (np.arange(N_ACTIONS, dtype=np.float32) + offset) / 10.0


def test_gpu_checkpoint_can_restore_replay_buffers_for_true_continuation(tmp_path):
    trainer = GPUDeepCFRTrainer(
        n_players=2,
        buffer_capacity=4,
        hidden_dim=16,
        n_layers=1,
        batch_size=2,
        n_training_steps=1,
        n_traversals=1,
        device=torch.device("cpu"),
        average_strategy_memory_capacity=3,
    )
    trainer.iteration = 7
    trainer._adaptive_traversal_batch_size = 2
    trainer.buffers[0].add(_features(1), 5, _advantages(1))
    trainer.buffers[0].add(_features(2), 6, _advantages(2))
    trainer.buffers[1].add(_features(3), 7, _advantages(3))

    legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    legal_mask[:2] = 1.0
    target_probs = np.zeros(N_ACTIONS, dtype=np.float32)
    target_probs[0] = 0.25
    target_probs[1] = 0.75
    trainer.strategy_buffer.add(_features(4), legal_mask, target_probs, weight=0.5)
    trainer.strategy_buffer.add(_features(5), legal_mask, target_probs, weight=1.5)

    path = tmp_path / "full_state.pt"
    trainer.save(str(path), include_replay_buffers=True)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = summarize_resume_checkpoint_metadata(
        checkpoint,
        resume_path=str(path),
    )
    assert metadata["resume_restored_replay_buffers"] is True
    assert metadata["resume_semantics"] == "true_replay_buffer_continuation"

    loaded = GPUDeepCFRTrainer.load(str(path), device=torch.device("cpu"))

    assert loaded.iteration == 7
    assert loaded._adaptive_traversal_batch_size == 2
    assert [buffer.capacity for buffer in loaded.buffers] == [4, 4]
    assert [buffer.size for buffer in loaded.buffers] == [2, 1]
    assert [buffer._n_seen for buffer in loaded.buffers] == [2, 1]
    np.testing.assert_allclose(loaded.buffers[0].features[:2], trainer.buffers[0].features[:2])
    np.testing.assert_array_equal(loaded.buffers[0].iterations[:2], [5, 6])
    np.testing.assert_allclose(loaded.buffers[1].advantages[:1], trainer.buffers[1].advantages[:1])

    assert loaded.strategy_buffer.capacity == 3
    assert loaded.strategy_buffer.size == 2
    assert loaded.strategy_buffer._n_seen == 2
    np.testing.assert_allclose(
        loaded.strategy_buffer.target_probs[:2],
        trainer.strategy_buffer.target_probs[:2],
    )
    np.testing.assert_allclose(loaded.strategy_buffer.weights[:2], [0.5, 1.5])


def test_gpu_checkpoint_compact_save_omits_replay_buffers_by_default(tmp_path):
    trainer = GPUDeepCFRTrainer(
        n_players=2,
        buffer_capacity=4,
        hidden_dim=16,
        n_layers=1,
        batch_size=2,
        n_training_steps=1,
        n_traversals=1,
        device=torch.device("cpu"),
        average_strategy_memory_capacity=3,
    )
    trainer.buffers[0].add(_features(1), 1, _advantages(1))
    trainer.strategy_buffer.add(
        _features(2),
        np.ones(N_ACTIONS, dtype=np.float32),
        np.ones(N_ACTIONS, dtype=np.float32),
    )

    path = tmp_path / "compact.pt"
    trainer.save(str(path))

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert "replay_buffers" not in checkpoint
    assert "strategy_replay_buffer" not in checkpoint
    metadata = summarize_resume_checkpoint_metadata(
        checkpoint,
        resume_path=str(path),
    )
    assert metadata["resume_restored_replay_buffers"] is False
    assert metadata["resume_semantics"] == "model_warm_start_no_replay_buffers"
