import torch

from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.cuda.gpu_trainer import (
    GPUDeepCFRTrainer,
    _GPU_CACHE_COMPACT_SAMPLE_BYTES,
    _MultiBufferView,
    _adapt_traversal_batch_size,
    _build_iteration_profile,
    _gpu_cache_budget_allows,
    _gpu_cache_nbytes,
    _nn_forward_chunk_size,
    _summarize_traversal_pool_stats,
    _traversal_batch_size,
    _traversal_pool_slots,
)
from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from scripts.benchmark_gpu_deep_cfr import evaluate_benchmark_gate_failures


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


def test_traversal_batch_size_keeps_fixed_pool_with_more_slots_per_traversal():
    assert _traversal_batch_size(
        n_traversals=2_000,
        pool_max_slots=1_000_000,
        slots_per_traversal=2_500,
    ) == 400


def test_traversal_batch_size_default_uses_conservative_fidelity_chunk():
    assert _traversal_batch_size(n_traversals=2_000) == 142


def test_traversal_batch_size_caps_to_requested_traversals():
    assert _traversal_batch_size(
        n_traversals=200,
        pool_max_slots=1_000_000,
        slots_per_traversal=2_500,
    ) == 200


def test_traversal_workspace_pool_uses_fixed_cap_for_small_chunks():
    assert _traversal_pool_slots(
        max_traversals=100,
        pool_max_slots=1_000_000,
        slots_per_traversal=2_000,
    ) == 1_000_000


def test_adaptive_traversal_batch_shrinks_on_observed_pool_demand():
    next_batch = _adapt_traversal_batch_size(
        current_batch=500,
        pool_max_slots=1_000_000,
        stats={
            "requested_slots": 3_313_926,
            "n_traversals": 500,
            "pool_exhausted_nodes": 100,
        },
    )

    assert 1 <= next_batch <= 137


def test_adaptive_traversal_batch_keeps_safe_chunk():
    next_batch = _adapt_traversal_batch_size(
        current_batch=500,
        pool_max_slots=1_000_000,
        stats={
            "requested_slots": 900_000,
            "n_traversals": 500,
            "pool_exhausted_nodes": 0,
        },
    )

    assert next_batch == 500


def test_nn_forward_chunk_size_reduces_when_cuda_memory_is_tight():
    net = torch.nn.Module()
    net.hidden_dim = 512

    chunk = _nn_forward_chunk_size(
        net,
        torch.device("cuda"),
        free_bytes=900 * 1024**2,
    )

    assert 8_192 <= chunk < 500_000


def test_nn_forward_chunk_size_keeps_fast_default_when_cuda_memory_is_plentiful():
    net = torch.nn.Module()
    net.hidden_dim = 512

    assert _nn_forward_chunk_size(
        net,
        torch.device("cuda"),
        free_bytes=3 * 1024**3,
    ) == 500_000


def test_summarize_traversal_pool_stats_reports_overflow_and_slot_pressure():
    summary = _summarize_traversal_pool_stats([
        {
            "requested_slots": 1_200,
            "pool_max_slots": 1_000,
            "n_traversals": 10,
            "regret_samples": 800,
            "pool_exhausted_nodes": 4,
            "max_nonterminal_slots": 200,
            "pool_exhausted_by_stage": [1, 2, 1, 0],
            "pool_exhausted_by_depth": [0, 1, 0, 3],
        },
        {
            "requested_slots": 500,
            "pool_max_slots": 1_000,
            "n_traversals": 5,
            "regret_samples": 300,
            "pool_exhausted_nodes": 1,
            "max_nonterminal_slots": 250,
            "pool_exhausted_by_stage": [0, 0, 0, 1],
            "pool_exhausted_by_depth": [0, 0, 1, 0],
        },
    ])

    assert summary["traversal_chunks"] == 2
    assert summary["traversal_overflow_chunks"] == 1
    assert summary["traversal_overflow_chunk_fraction"] == 0.5
    assert summary["traversal_mean_pool_demand_ratio"] == 0.85
    assert summary["traversal_max_pool_demand_ratio"] == 1.2
    assert summary["traversal_max_slots_per_traversal"] == 120.0
    assert summary["traversal_pool_exhausted_nodes"] == 5
    assert summary["traversal_pool_exhausted_per_traversal"] == 0.333333
    assert summary["traversal_pool_exhausted_stage_preflop"] == 1
    assert summary["traversal_pool_exhausted_stage_flop"] == 2
    assert summary["traversal_pool_exhausted_stage_turn"] == 1
    assert summary["traversal_pool_exhausted_stage_river"] == 1
    assert summary["traversal_pool_exhausted_first_depth"] == 1
    assert summary["traversal_pool_exhausted_peak_depth"] == 3
    assert summary["traversal_pool_exhausted_peak_depth_nodes"] == 3
    assert summary["traversal_pool_exhausted_last_depth"] == 3
    assert summary["traversal_max_nonterminal_slots"] == 250
    assert summary["traversal_mean_max_nonterminal_slots_per_traversal"] == 35.0
    assert summary["traversal_max_nonterminal_slots_per_traversal"] == 50.0
    assert summary["traversal_mean_allocated_to_live_ratio"] == 4.0
    assert summary["traversal_max_allocated_to_live_ratio"] == 6.0


def test_build_iteration_profile_reports_warmup_safe_throughput_metrics():
    profile = _build_iteration_profile(
        iteration=3,
        n_players=2,
        n_traversals=100,
        traversal_stats=[
            {
                "requested_slots": 900,
                "pool_max_slots": 1_000,
                "n_traversals": 100,
                "regret_samples": 600,
                "policy_samples": 40,
                "pool_exhausted_nodes": 2,
            },
            {
                "requested_slots": 800,
                "pool_max_slots": 1_000,
                "n_traversals": 100,
                "regret_samples": 500,
                "policy_samples": 30,
                "pool_exhausted_nodes": 3,
            },
        ],
        traverse_seconds=2.0,
        train_seconds=4.0,
        train_batch_size=128,
        train_steps=10,
    )

    assert profile["iteration"] == 3
    assert profile["requested_traversals"] == 200
    assert profile["regret_samples"] == 1100
    assert profile["policy_samples"] == 70
    assert profile["traversals_per_second"] == 100.0
    assert profile["regret_samples_per_second"] == 550.0
    assert profile["train_sample_budget"] == 1280
    assert profile["train_samples_per_second"] == 320.0
    assert profile["traversal_pool_exhausted_nodes"] == 5


def test_gpu_benchmark_gate_fails_on_pool_exhaustion():
    failures = evaluate_benchmark_gate_failures(
        {
            "mean_traversals_per_second": 2500.0,
            "max_pool_exhausted_per_traversal": 10.0,
            "max_overflow_chunk_fraction": 1.0,
        },
        min_mean_traversals_per_second=2000.0,
        max_pool_exhausted_per_traversal=0.0,
        max_overflow_chunk_fraction=0.0,
    )

    assert any("max_pool_exhausted_per_traversal" in failure for failure in failures)
    assert any("max_overflow_chunk_fraction" in failure for failure in failures)


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
    assert trainer.use_betting_history is False
    assert trainer.value_net.adv_head.weight.shape == (N_ACTIONS, hidden_dim)
