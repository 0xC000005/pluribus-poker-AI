import torch

from poker_ai.deep_cfr.deep_cfr import DeepCFRTrainer
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES


def _tiny_policy_targets() -> PolicyTargetBuffer:
    features = torch.zeros((2, N_FEATURES), dtype=torch.float32).numpy()
    features[0, 104 + 2] = 1.0
    features[1, 104 + 3] = 1.0
    legal = torch.zeros((2, N_ACTIONS), dtype=torch.float32).numpy()
    legal[:, [1, 8]] = 1.0
    probs = torch.zeros((2, N_ACTIONS), dtype=torch.float32).numpy()
    probs[0, 1] = 1.0
    probs[1, 8] = 1.0
    return PolicyTargetBuffer(features, legal, probs)


def test_deep_cfr_uses_external_average_strategy_targets_in_checkpoint_metadata(tmp_path):
    targets = _tiny_policy_targets()
    trainer = DeepCFRTrainer(
        n_players=2,
        hidden_dim=16,
        n_traversals=1,
        n_training_steps=1,
        batch_size=4,
        device=torch.device("cpu"),
        average_strategy_target_buffer=targets,
        average_strategy_weight=0.25,
    )

    trainer.average_policy_net = trainer._train_average_policy_from_strategy_targets(
        n_epochs=1,
        batch_size=4,
    )
    trainer.has_average_policy_net = True
    checkpoint = tmp_path / "external_average_strategy.pt"
    trainer.save(str(checkpoint))

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["has_average_policy_net"] is True
    assert saved["average_strategy_target_size"] == 2
    assert saved["average_strategy_external_target_size"] == 2
    assert saved["average_strategy_collected_size"] == 0
    assert saved["average_strategy_weight"] == 0.25


def test_external_average_targets_can_mix_with_traversal_strategy_memory():
    targets = _tiny_policy_targets()
    trainer = DeepCFRTrainer(
        n_players=2,
        hidden_dim=16,
        n_traversals=1,
        n_training_steps=1,
        batch_size=4,
        device=torch.device("cpu"),
        average_strategy_target_buffer=targets,
        average_strategy_memory_capacity=8,
        average_strategy_weight=0.25,
    )

    feature = torch.zeros(N_FEATURES, dtype=torch.float32).numpy()
    feature[104] = 1.0
    legal = torch.zeros(N_ACTIONS, dtype=torch.float32).numpy()
    legal[[1, 8]] = 1.0
    probs = torch.zeros(N_ACTIONS, dtype=torch.float32).numpy()
    probs[1] = 1.0
    trainer.strategy_buffer.add(feature, legal, probs)

    training_buffer = trainer._average_strategy_training_buffer()

    assert trainer._traversal_strategy_buffer() is trainer.strategy_buffer
    assert trainer._average_strategy_external_target_size() == 2
    assert trainer.strategy_buffer.size == 1
    assert trainer._average_strategy_target_size() == 3
    batch = training_buffer.sample_batch(4, torch.device("cpu"))
    assert batch.features.shape == (3, N_FEATURES)


def test_zero_average_strategy_memory_keeps_external_targets_only():
    targets = _tiny_policy_targets()
    trainer = DeepCFRTrainer(
        n_players=2,
        hidden_dim=16,
        n_traversals=1,
        n_training_steps=1,
        batch_size=4,
        device=torch.device("cpu"),
        average_strategy_target_buffer=targets,
        average_strategy_memory_capacity=0,
        average_strategy_weight=0.25,
    )

    assert trainer.strategy_buffer.capacity == 0
    assert trainer._traversal_strategy_buffer() is None
    assert trainer._average_strategy_target_size() == 2


def test_gpu_trainer_can_seed_real_average_strategy_memory_from_targets():
    from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

    targets = _tiny_policy_targets()
    trainer = GPUDeepCFRTrainer(
        n_players=2,
        hidden_dim=16,
        n_traversals=1,
        n_training_steps=1,
        batch_size=4,
        device=torch.device("cpu"),
        average_strategy_memory_capacity=8,
        average_strategy_weight=0.25,
    )

    added = trainer.seed_average_strategy_memory_from_targets(targets, weight=3.0)

    assert added == 2
    assert trainer.strategy_buffer.size == 2
    assert trainer.average_strategy_seed_target_size == 2
    assert trainer._average_strategy_external_target_size() == 0
    batch = trainer.strategy_buffer.sample_batch(2, torch.device("cpu"))
    assert set(batch.weights.tolist()) == {3.0}
