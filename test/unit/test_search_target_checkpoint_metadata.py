import torch

from poker_ai.deep_cfr.deep_cfr import DeepCFRTrainer
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES


def _targets_for_turn_and_river() -> PolicyTargetBuffer:
    features = torch.zeros((3, N_FEATURES), dtype=torch.float32).numpy()
    features[0, 104 + 2] = 1.0
    features[1, 104 + 3] = 1.0
    features[2, 104 + 3] = 1.0
    legal = torch.zeros((3, N_ACTIONS), dtype=torch.float32).numpy()
    legal[:, [1, 8]] = 1.0
    probs = torch.zeros((3, N_ACTIONS), dtype=torch.float32).numpy()
    probs[:, 8] = 1.0
    return PolicyTargetBuffer(features, legal, probs)


def test_deep_cfr_checkpoint_records_search_target_policy_coverage(tmp_path):
    trainer = DeepCFRTrainer(
        n_players=2,
        hidden_dim=16,
        n_traversals=1,
        n_training_steps=1,
        batch_size=4,
        device=torch.device("cpu"),
        policy_target_buffer=_targets_for_turn_and_river(),
        policy_target_weight=0.05,
    )

    checkpoint = tmp_path / "with_search_targets.pt"
    trainer.save(str(checkpoint))

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["policy_calibration"]["target_streets"] == [2, 3]
    assert saved["policy_calibration"]["target_street_counts"] == {"2": 1, "3": 2}
    assert saved["policy_calibration"]["target_size"] == 3
    assert saved["policy_calibration"]["search_target_weight"] == 0.05


def test_deep_cfr_checkpoint_omits_policy_coverage_when_search_weight_zero(tmp_path):
    trainer = DeepCFRTrainer(
        n_players=2,
        hidden_dim=16,
        n_traversals=1,
        n_training_steps=1,
        batch_size=4,
        device=torch.device("cpu"),
        policy_target_buffer=_targets_for_turn_and_river(),
        policy_target_weight=0.0,
    )

    checkpoint = tmp_path / "without_search_targets.pt"
    trainer.save(str(checkpoint))

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "policy_calibration" not in saved
