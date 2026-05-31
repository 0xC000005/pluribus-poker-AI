from scripts.poker_autoresearch_train import (
    resolve_average_strategy_memory_capacity,
    resolve_average_strategy_target_buffers as resolve_autoresearch_target_buffers,
)
from scripts.run_gpu_deep_cfr import (
    resolve_average_strategy_memory_capacity as resolve_run_gpu_capacity,
    resolve_average_strategy_target_buffers,
)
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
import numpy as np


def test_zero_average_strategy_memory_capacity_stays_zero_for_external_only_training():
    assert resolve_average_strategy_memory_capacity(0) == 0
    assert resolve_run_gpu_capacity(0) == 0
    assert resolve_average_strategy_memory_capacity(128) == 128
    assert resolve_run_gpu_capacity(128) == 128


def test_run_gpu_can_route_average_strategy_targets_into_memory(tmp_path):
    features = np.zeros((1, N_FEATURES), dtype=np.float32)
    legal = np.zeros((1, N_ACTIONS), dtype=np.float32)
    probs = np.zeros((1, N_ACTIONS), dtype=np.float32)
    legal[0, [1, 8]] = 1.0
    probs[0, 1] = 1.0
    targets_path = tmp_path / "targets.npz"
    PolicyTargetBuffer(features, legal, probs).save_npz(targets_path)

    external, memory = resolve_average_strategy_target_buffers(
        str(targets_path),
        seed_memory=True,
    )

    assert external is None
    assert memory is not None
    assert memory.size == 1


def test_autoresearch_train_can_route_average_strategy_targets_into_memory(tmp_path):
    features = np.zeros((1, N_FEATURES), dtype=np.float32)
    legal = np.zeros((1, N_ACTIONS), dtype=np.float32)
    probs = np.zeros((1, N_ACTIONS), dtype=np.float32)
    legal[0, [1, 8]] = 1.0
    probs[0, 8] = 1.0
    targets_path = tmp_path / "targets.npz"
    PolicyTargetBuffer(features, legal, probs).save_npz(targets_path)

    external, memory = resolve_autoresearch_target_buffers(
        str(targets_path),
        seed_memory=True,
    )

    assert external is None
    assert memory is not None
    assert memory.size == 1
