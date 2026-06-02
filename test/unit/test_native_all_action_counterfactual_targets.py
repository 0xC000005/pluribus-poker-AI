import numpy as np


def test_all_action_rollout_values_score_forced_fold_payoff():
    from poker_ai.deep_cfr.fast_state import new_fast_game
    from scripts.build_native_all_action_counterfactual_targets import (
        estimate_all_action_rollout_values,
    )

    state = new_fast_game(2, initial_chips=1000)
    result = estimate_all_action_rollout_values(
        state,
        n_rollouts_per_action=1,
        max_steps_per_rollout=8,
        seed=20260710,
        paired_rollout_seeds=True,
    )

    assert result["player"] == 0
    assert result["paired_rollout_seeds"] is True
    assert result["legal_mask"].shape == (9,)
    assert result["values"].shape == (9,)
    assert result["legal_mask"][0] == 1.0
    assert np.isclose(result["values"][0], -0.05)
    assert np.all(np.isfinite(result["values"][result["legal_mask"] > 0]))
    assert np.all(np.isnan(result["values"][result["legal_mask"] <= 0]))


def test_native_all_action_target_gate_compares_low_to_high_budget(tmp_path):
    from scripts.build_native_all_action_counterfactual_targets import run_gate

    output = tmp_path / "targets.json"
    metrics = run_gate(
        n_states=3,
        low_rollouts_per_action=1,
        high_rollouts_per_action=2,
        max_steps_per_rollout=16,
        initial_chips=1000,
        seed=20260711,
        paired_rollout_seeds=True,
        output_json=output,
    )

    assert output.exists()
    assert metrics["algorithm"] == "native_all_action_counterfactual_rollout_targets"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["n_states"] == 3
    assert metrics["low_rollouts_per_action"] == 1
    assert metrics["high_rollouts_per_action"] == 2
    assert metrics["paired_rollout_seeds"] is True
    assert 0.0 <= metrics["top_action_agreement"] <= 1.0
    assert metrics["mean_legal_action_count"] >= 2.0
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_solver_labels"] is False


def test_resample_hidden_world_preserves_actor_observation():
    from poker_ai.deep_cfr.fast_state import new_fast_game
    from scripts.build_native_all_action_counterfactual_targets import (
        resample_hidden_world_for_observer,
    )

    state = new_fast_game(2, initial_chips=1000)
    before_features = state.to_feature_vector().copy()
    observer = state.current_player_i

    resampled = resample_hidden_world_for_observer(
        state,
        observer=observer,
        seed=20260730,
    )

    assert resampled.current_player_i == observer
    assert np.array_equal(resampled.hole_cards[observer], state.hole_cards[observer])
    assert np.array_equal(resampled.community, state.community)
    assert np.array_equal(resampled.to_feature_vector(), before_features)
    assert len(set(int(card) for card in resampled.deck_order.tolist())) == 52


def test_world_averaged_all_action_values_are_observation_compatible():
    from poker_ai.deep_cfr.fast_state import new_fast_game
    from scripts.build_native_all_action_counterfactual_targets import (
        estimate_world_averaged_all_action_values,
    )

    state = new_fast_game(2, initial_chips=1000)
    result = estimate_world_averaged_all_action_values(
        state,
        n_worlds=3,
        n_rollouts_per_action=1,
        max_steps_per_rollout=8,
        seed=20260731,
    )

    assert result["n_worlds"] == 3
    assert result["player"] == state.current_player_i
    assert result["values"].shape == (9,)
    assert np.all(np.isfinite(result["values"][result["legal_mask"] > 0]))
    assert np.all(np.isnan(result["values"][result["legal_mask"] <= 0]))


def test_native_all_action_target_gate_supports_world_averaging(tmp_path):
    from scripts.build_native_all_action_counterfactual_targets import run_gate

    metrics = run_gate(
        n_states=2,
        low_rollouts_per_action=1,
        high_rollouts_per_action=2,
        max_steps_per_rollout=8,
        initial_chips=1000,
        seed=20260732,
        n_worlds=2,
        output_json=tmp_path / "world_targets.json",
    )

    assert metrics["n_worlds"] == 2
    assert metrics["world_averaged_targets"] is True
    assert "mean_high_top_margin" in metrics
    assert "top_action_agreement_margin_0_02" in metrics
    assert 0.0 <= metrics["top_action_agreement"] <= 1.0
