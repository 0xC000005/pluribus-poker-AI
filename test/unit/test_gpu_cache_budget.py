import torch

from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.cuda.gpu_trainer import (
    GPUDeepCFRTrainer,
    _GPU_CACHE_COMPACT_SAMPLE_BYTES,
    _MultiBufferView,
    _gpu_cache_budget_allows,
    _gpu_cache_nbytes,
)
from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES


def test_gpu_cache_nbytes_matches_replay_tensor_shapes():
    assert _gpu_cache_nbytes(10) == 10 * (N_FEATURES + 1 + N_ACTIONS) * 4


def test_compact_gpu_cache_nbytes_uses_half_precision_samples():
    assert _GPU_CACHE_COMPACT_SAMPLE_BYTES == 2
    assert _gpu_cache_nbytes(
        10,
        sample_dtype_bytes=_GPU_CACHE_COMPACT_SAMPLE_BYTES,
    ) == 10 * ((N_FEATURES + N_ACTIONS) * 2 + 4)


def test_gpu_cache_budget_rejects_cache_that_would_consume_most_free_memory():
    estimated = _gpu_cache_nbytes(10_000_000)

    assert not _gpu_cache_budget_allows(
        n_samples=10_000_000,
        free_bytes=estimated,
        safety_fraction=0.60,
    )


def test_gpu_cache_budget_accepts_small_cache():
    estimated = _gpu_cache_nbytes(100_000)

    assert _gpu_cache_budget_allows(
        n_samples=100_000,
        free_bytes=estimated * 4,
        safety_fraction=0.60,
    )


def test_compact_gpu_cache_budget_accepts_20m_samples_on_8g_budget():
    eight_gib = 8 * 1024**3

    assert _gpu_cache_budget_allows(
        n_samples=20_000_000,
        free_bytes=eight_gib,
        safety_fraction=0.75,
        sample_dtype_bytes=_GPU_CACHE_COMPACT_SAMPLE_BYTES,
    )


def test_release_workspace_for_training_drops_traversal_workspace():
    trainer = GPUDeepCFRTrainer(device=torch.device("cpu"))
    sentinel = object()
    trainer._workspace = sentinel

    trainer._release_workspace_for_training()

    assert trainer._workspace is None


def test_gpu_cache_budget_rejection_warns_once_per_signature(monkeypatch):
    buffer = ReservoirBuffer(1)
    buffer.add(buffer.features[0], 1, buffer.advantages[0])
    view = _MultiBufferView([buffer])
    warnings = []

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))
    monkeypatch.setattr(
        "poker_ai.deep_cfr.cuda.gpu_trainer.logger.warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    assert not view._try_build_gpu_cache(torch.device("cuda"))
    assert not view._try_build_gpu_cache(torch.device("cuda"))

    assert len(warnings) == 1


def test_gpu_cache_release_drops_cached_tensors():
    view = _MultiBufferView([])
    sentinel = object()
    view._gpu_features = sentinel
    view._gpu_iterations = sentinel
    view._gpu_advantages = sentinel
    view._gpu_cache_sig = ((1, 1),)

    view.release_gpu_cache()

    assert view._gpu_features is None
    assert view._gpu_iterations is None
    assert view._gpu_advantages is None
    assert view._gpu_cache_sig is None


def test_gpu_cache_skips_when_training_budget_is_smaller_than_buffer(monkeypatch):
    buffer = ReservoirBuffer(2)
    buffer.add(buffer.features[0], 1, buffer.advantages[0])
    buffer.add(buffer.features[1], 1, buffer.advantages[1])
    view = _MultiBufferView([buffer])
    view.set_expected_sample_budget(1)
    mem_info_calls = []

    def fail_if_called(device):
        mem_info_calls.append(device)
        return 8 * 1024**3, 8 * 1024**3

    monkeypatch.setattr(torch.cuda, "mem_get_info", fail_if_called)

    assert not view._try_build_gpu_cache(torch.device("cuda"))
    assert mem_info_calls == []


def test_gpu_trainer_load_accepts_legacy_sequential_value_net(tmp_path):
    hidden_dim = 16
    checkpoint = {
        "value_net": {
            "net.0.weight": torch.randn(hidden_dim, N_FEATURES),
            "net.0.bias": torch.randn(hidden_dim),
            "net.2.weight": torch.randn(hidden_dim, hidden_dim),
            "net.2.bias": torch.randn(hidden_dim),
            "net.4.weight": torch.randn(N_ACTIONS, hidden_dim),
            "net.4.bias": torch.randn(N_ACTIONS),
        },
        "iteration": 7,
        "n_players": 2,
        "hidden_dim": hidden_dim,
        "n_layers": 2,
        "initial_chips": 20000,
    }
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)

    trainer = GPUDeepCFRTrainer.load(str(path), device=torch.device("cpu"))

    assert trainer.iteration == 7
    assert trainer.value_net.adv_head.weight.shape == (N_ACTIONS, hidden_dim)
