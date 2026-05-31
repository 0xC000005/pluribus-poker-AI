import numpy as np
import torch

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.sampled_deep_cfr_traversal_probe import (
    _rng_for_path,
    _sample_action_indices,
    run_probe_grid,
    run_probe,
)
from scripts.eval_sampled_deep_cfr_traversal_grid import evaluate_gate_failures


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


def test_action_keyed_rng_is_path_stable_and_order_independent():
    first = _rng_for_path(123, (4, 2, 7)).choice(10, size=4).tolist()
    _ = _rng_for_path(123, (4, 9, 7)).choice(10, size=4).tolist()
    second = _rng_for_path(123, (4, 2, 7)).choice(10, size=4).tolist()

    assert second == first


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
    assert metrics["randomization_contract"] == "action-keyed"
    assert metrics["sample_count"] == 4
    assert metrics["n_repeats"] == 4
    assert metrics["seed"] == 20260525
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
    assert metrics["randomization_contract"] == "action-keyed"
    assert metrics["sample_count"] == 4
    assert metrics["promotion"] is False


def test_sampled_traversal_probe_keeps_legacy_randomization_available():
    metrics = run_probe(
        n_repeats=2,
        n_reference_repeats=2,
        initial_chips=300,
        sample_count=4,
        randomization_contract="legacy-sequential",
        hidden_dim=16,
        n_layers=1,
        seed=20260535,
    )

    assert metrics["randomization_contract"] == "legacy-sequential"
    assert metrics["promotion"] is False


def test_sampled_traversal_probe_supports_priority_without_replacement_mode():
    metrics = run_probe(
        n_repeats=4,
        initial_chips=300,
        sample_count=4,
        sampling_mode="priority-without-replacement",
        priority_forced_count=2,
        priority_source="advantage",
        hidden_dim=16,
        n_layers=1,
        seed=20260531,
    )

    assert metrics["sampling_mode"] == "priority-without-replacement"
    assert metrics["priority_forced_count"] == 2
    assert metrics["priority_source"] == "advantage"
    assert metrics["promotion"] is False


def test_sampled_traversal_probe_supports_priority_model_checkpoint(tmp_path):
    model = ValueNetwork(N_FEATURES, hidden_dim=16, output_dim=N_ACTIONS, n_layers=1)
    checkpoint = tmp_path / "priority.pt"
    torch.save(
        {
            "value_net": model.state_dict(),
            "hidden_dim": 16,
            "n_layers": 1,
            "initial_chips": 300,
            "uses_betting_history": True,
        },
        checkpoint,
    )

    metrics = run_probe(
        n_repeats=2,
        n_reference_repeats=2,
        initial_chips=300,
        sample_count=4,
        sampling_mode="priority-without-replacement",
        priority_forced_count=2,
        priority_source="priority-model",
        priority_checkpoint=str(checkpoint),
        hidden_dim=16,
        n_layers=1,
        seed=20260532,
    )

    assert metrics["priority_source"] == "priority-model"
    assert metrics["priority_checkpoint"] == str(checkpoint)
    assert metrics["promotion"] is False


def test_sampled_traversal_probe_supports_oracle_action_value_priority():
    metrics = run_probe(
        n_repeats=2,
        n_reference_repeats=2,
        initial_chips=300,
        sample_count=4,
        sampling_mode="priority-without-replacement",
        priority_forced_count=2,
        priority_source="oracle-action-value",
        hidden_dim=16,
        n_layers=1,
        seed=20260533,
    )

    assert metrics["priority_source"] == "oracle-action-value"
    assert metrics["promotion"] is False


def test_sampled_traversal_probe_can_use_oracle_priority_as_baseline():
    metrics = run_probe(
        n_repeats=2,
        n_reference_repeats=2,
        initial_chips=300,
        sample_count=4,
        sampling_mode="priority-without-replacement",
        priority_forced_count=2,
        priority_source="oracle-action-value",
        use_priority_baseline=True,
        hidden_dim=16,
        n_layers=1,
        seed=20260534,
    )

    assert metrics["priority_source"] == "oracle-action-value"
    assert metrics["use_priority_baseline"] is True
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
    assert metrics["randomization_contract"] == "action-keyed"
    assert len(metrics["cases"]) == 2
    assert 0.0 <= metrics["top_action_match_rate"] <= 1.0


def test_sampled_traversal_grid_gate_can_require_per_case_stability():
    metrics = {
        "top_action_match_rate": 0.75,
        "mean_speedup": 1.17,
        "mean_abs_bias": 0.04,
        "cases": [
            {
                "seed": 1,
                "initial_chips": 300,
                "top_action_match": True,
                "mean_abs_bias": 0.02,
            },
            {
                "seed": 2,
                "initial_chips": 300,
                "top_action_match": False,
                "mean_abs_bias": 0.095,
            },
        ],
    }

    failures = evaluate_gate_failures(
        metrics,
        min_top_match_rate=0.75,
        min_mean_speedup=1.05,
        max_mean_abs_bias=0.08,
        require_all_top_match=True,
        max_case_mean_abs_bias=0.08,
    )

    assert any("top_action_match failed" in failure for failure in failures)
    assert any("case mean_abs_bias exceeded" in failure for failure in failures)
