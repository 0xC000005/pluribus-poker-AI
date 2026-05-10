from poker_ai.deep_cfr.cuda.gpu_trainer import (
    _gpu_cache_budget_allows,
    _gpu_cache_nbytes,
)
from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES


def test_gpu_cache_nbytes_matches_replay_tensor_shapes():
    assert _gpu_cache_nbytes(10) == 10 * (N_FEATURES + 1 + N_ACTIONS) * 4


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
