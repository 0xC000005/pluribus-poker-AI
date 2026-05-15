import numpy as np

from poker_ai.research.sampled_deep_cfr_traversal_probe import (
    _sample_action_indices,
    run_probe_grid,
    run_probe,
)


def test_sample_action_indices_enumerates_when_budget_covers_legal_actions():
    rng = np.random.default_rng(7)
    legal_indices = np.array([1, 2, 4], dtype=np.int64)
    sample_probs = np.array([0.0, 0.2, 0.3, 0.0, 0.5], dtype=np.float64)

    sampled, is_exhaustive = _sample_action_indices(
        rng,
        legal_indices,
        sample_probs,
        sample_count=5,
    )

    assert is_exhaustive is True
    assert sampled.tolist() == [1, 2, 4]


def test_sampled_traversal_probe_emits_tiny_full_deck_metrics():
    metrics = run_probe(
        n_repeats=4,
        initial_chips=300,
        sample_count=4,
        hidden_dim=16,
        n_layers=1,
        seed=20260525,
    )

    assert metrics["mode"] == "sampled_deep_cfr_traversal_probe"
    assert metrics["sampling_mode"] == "with-replacement"
    assert metrics["sample_count"] == 4
    assert metrics["n_repeats"] == 4
    assert metrics["promotion"] is False
    assert metrics["mean_abs_bias"] >= 0.0
    assert metrics["exhaustive_top_margin"] >= 0.0
    assert metrics["sampled_top_margin"] >= 0.0
    assert metrics["sampled_seconds"] >= 0.0
    assert metrics["exhaustive_seconds"] >= 0.0


def test_sampled_traversal_probe_supports_without_replacement_mode():
    metrics = run_probe(
        n_repeats=4,
        initial_chips=300,
        sample_count=4,
        sampling_mode="without-replacement",
        hidden_dim=16,
        n_layers=1,
        seed=20260527,
    )

    assert metrics["sampling_mode"] == "without-replacement"
    assert metrics["sample_count"] == 4
    assert metrics["promotion"] is False


def test_sampled_traversal_probe_supports_priority_without_replacement_mode():
    metrics = run_probe(
        n_repeats=4,
        initial_chips=300,
        sample_count=4,
        sampling_mode="priority-without-replacement",
        priority_forced_count=2,
        hidden_dim=16,
        n_layers=1,
        seed=20260531,
    )

    assert metrics["sampling_mode"] == "priority-without-replacement"
    assert metrics["priority_forced_count"] == 2
    assert metrics["promotion"] is False


def test_sampled_traversal_probe_reproducible_for_same_seed():
    first = run_probe(
        n_repeats=2,
        n_reference_repeats=2,
        initial_chips=300,
        sample_count=4,
        hidden_dim=16,
        n_layers=1,
        seed=20260526,
    )
    second = run_probe(
        n_repeats=2,
        n_reference_repeats=2,
        initial_chips=300,
        sample_count=4,
        hidden_dim=16,
        n_layers=1,
        seed=20260526,
    )

    assert second["exhaustive_top_action"] == first["exhaustive_top_action"]
    assert second["sampled_top_action"] == first["sampled_top_action"]
    assert second["mean_abs_bias"] == first["mean_abs_bias"]


def test_sampled_traversal_probe_exact_budget_matches_exhaustive():
    metrics = run_probe(
        n_repeats=2,
        n_reference_repeats=2,
        initial_chips=300,
        sample_count=9,
        sampling_mode="without-replacement",
        hidden_dim=16,
        n_layers=1,
        seed=20260530,
    )

    assert metrics["mean_abs_bias"] == 0.0
    assert metrics["top_action_match"] is True


def test_sampled_traversal_probe_grid_aggregates_cases():
    metrics = run_probe_grid(
        seeds=[20260528, 20260529],
        initial_chips_values=[300],
        n_repeats=2,
        n_reference_repeats=2,
        sample_count=4,
        sampling_mode="without-replacement",
        priority_forced_count=0,
        hidden_dim=16,
        n_layers=1,
    )

    assert metrics["mode"] == "sampled_deep_cfr_traversal_probe_grid"
    assert metrics["n_cases"] == 2
    assert len(metrics["cases"]) == 2
    assert 0.0 <= metrics["top_action_match_rate"] <= 1.0
