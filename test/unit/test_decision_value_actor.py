import numpy as np
import pytest
import torch

from poker_ai.deep_cfr.networks import PolicyNetwork, ValueNetwork
from poker_ai.deep_cfr.fast_state import FastPokerState
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.decision_value_actor import (
    DecisionValueTargetConfig,
    OnPolicyDecisionValueTargetConfig,
    build_on_policy_average_strategy_target_artifact,
    build_on_policy_behavior_average_strategy_target_artifact,
    build_on_policy_trajectory_return_average_strategy_target_artifact,
    build_decision_value_q_target_buffer,
    calibrate_average_policy_net_from_targets,
    calibrate_policy_head_from_decision_value_targets,
    calibrate_policy_head_with_kl_q_targets,
    collect_on_policy_behavior_average_strategy_targets,
    collect_on_policy_decision_value_policy_targets,
    collect_on_policy_trajectory_return_average_strategy_targets,
    decision_value_eval_gate,
    regularized_action_value_target,
    resample_hidden_world_state,
    select_decision_value_behavior_action,
    trajectory_return_weights,
)


def _write_average_policy_checkpoint(path, *, seed: int = 0):
    torch.manual_seed(seed)
    value_net = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=2)
    average_policy_net = PolicyNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=2)
    torch.save(
        {
            "value_net": value_net.state_dict(),
            "average_policy_net": average_policy_net.state_dict(),
            "hidden_dim": 32,
            "n_layers": 2,
            "n_players": 2,
            "initial_chips": 1000,
            "iteration": 3,
            "uses_betting_history": True,
        },
        path,
    )
    return value_net.state_dict(), average_policy_net.state_dict()


def _simple_targets() -> PolicyTargetBuffer:
    rng = np.random.default_rng(20260525)
    features = rng.normal(0.0, 0.25, size=(16, N_FEATURES)).astype(np.float32)
    features[:, 104:108] = 0.0
    features[:, 104] = 1.0
    legal_masks = np.zeros((16, N_ACTIONS), dtype=np.float32)
    legal_masks[:, [1, 2, 4]] = 1.0
    target_probs = np.zeros((16, N_ACTIONS), dtype=np.float32)
    target_probs[:8, 1] = 1.0
    target_probs[8:, 4] = 1.0
    return PolicyTargetBuffer(features, legal_masks, target_probs)


def test_regularized_action_value_target_moves_toward_high_value_legal_action():
    prior = np.array([0.65, 0.25, 0.10, 0.0], dtype=np.float32)
    legal_mask = np.array([1, 1, 1, 0], dtype=np.float32)
    action_values = np.array([0.0, 10.0, -5.0, 100.0], dtype=np.float32)

    target = regularized_action_value_target(
        prior,
        legal_mask,
        action_values,
        eta=1.0,
    )

    assert np.isclose(float(target.sum()), 1.0)
    assert target[3] == 0.0
    assert target[1] > prior[1]
    assert int(np.argmax(target)) == 1


def test_decision_value_eval_gate_rejects_policy_ev_regression():
    gate = decision_value_eval_gate(
        before_eval={
            "mean_selected_action_value": 10.0,
            "mean_policy_ev": 5.0,
            "mean_oracle_gap": 20.0,
        },
        after_eval={
            "mean_selected_action_value": 11.0,
            "mean_policy_ev": 4.0,
            "mean_oracle_gap": 19.0,
        },
        require_policy_ev_non_decrease=True,
    )

    assert gate["passed"] is False
    assert gate["checks"]["selected_action_value_non_decrease"] is True
    assert gate["checks"]["oracle_gap_non_increase"] is True
    assert gate["checks"]["policy_ev_non_decrease"] is False


def test_search_improved_behavior_action_uses_target_distribution():
    legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    legal_mask[[1, 4]] = 1.0
    prior = np.zeros(N_ACTIONS, dtype=np.float32)
    prior[1] = 0.95
    prior[4] = 0.05
    target = np.zeros(N_ACTIONS, dtype=np.float32)
    target[1] = 0.10
    target[4] = 0.90
    rng = np.random.default_rng(20260525)

    action = select_decision_value_behavior_action(
        prior,
        target,
        legal_mask,
        rng,
        behavior_policy="search-improved",
        greedy=True,
    )

    assert action == 4


def test_calibrate_average_policy_net_from_targets_preserves_value_net_and_records_metadata(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output = tmp_path / "decision_value_actor.pt"
    original_value, original_average = _write_average_policy_checkpoint(checkpoint, seed=7)
    targets = _simple_targets()

    metrics = calibrate_average_policy_net_from_targets(
        checkpoint,
        targets,
        output,
        n_steps=120,
        batch_size=16,
        lr=0.05,
        device="cpu",
        target_metadata={"n_roots": 16, "target_mode": "unit"},
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "decision_value_average_policy_calibration"
    assert metrics["after_loss"] < metrics["before_loss"]

    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert saved["decision_value_actor"]["target_metadata"]["target_mode"] == "unit"
    for key, value in original_value.items():
        assert torch.allclose(saved["value_net"][key], value)
    assert any(
        not torch.allclose(saved["average_policy_net"][key], original_average[key])
        for key in original_average
    )


def test_calibrate_policy_head_from_decision_value_targets_records_covered_metadata(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output = tmp_path / "decision_value_policy_head.pt"
    original_value, _original_average = _write_average_policy_checkpoint(checkpoint, seed=13)
    targets = _simple_targets()

    metrics = calibrate_policy_head_from_decision_value_targets(
        checkpoint,
        targets,
        output,
        n_steps=120,
        batch_size=16,
        lr=0.05,
        device="cpu",
        target_metadata={"target_mode": "policy-head-unit"},
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "decision_value_policy_head_calibration"
    assert metrics["after_loss"] < metrics["before_loss"]

    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert saved["decision_value_actor"]["actor_target"] == "policy-head"
    assert saved["policy_calibration"]["mode"] == "decision_value_policy_head_calibration"
    assert saved["policy_calibration"]["target_streets"] == [0]
    for key, value in original_value.items():
        if key.startswith("policy_head."):
            continue
        assert torch.allclose(saved["value_net"][key], value)


def test_calibrate_policy_head_with_kl_q_targets_improves_q_objective(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output = tmp_path / "decision_value_q_policy_head.pt"
    original_value, _original_average = _write_average_policy_checkpoint(checkpoint, seed=31)
    targets = _simple_targets()
    action_values = np.zeros((targets.size, N_ACTIONS), dtype=np.float32)
    action_values[:, 1] = 2.0
    action_values[:, 4] = -1.0
    base_probs = np.zeros((targets.size, N_ACTIONS), dtype=np.float32)
    base_probs[:, [1, 4]] = 0.5
    q_targets = build_decision_value_q_target_buffer(
        targets,
        action_values=action_values,
        base_probs=base_probs,
    )

    metrics = calibrate_policy_head_with_kl_q_targets(
        checkpoint,
        q_targets,
        output,
        n_steps=160,
        batch_size=16,
        lr=0.05,
        kl_beta=0.25,
        device="cpu",
        target_metadata={"target_mode": "q-unit"},
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "decision_value_policy_head_kl_q_calibration"
    assert metrics["after_expected_q"] > metrics["before_expected_q"]
    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert saved["decision_value_actor"]["mode"] == "decision_value_policy_head_kl_q_calibration"
    assert saved["decision_value_actor"]["target_metadata"]["target_mode"] == "q-unit"
    assert saved["policy_calibration"]["target_streets"] == [0]
    for key, value in original_value.items():
        if key.startswith("policy_head."):
            continue
        assert torch.allclose(saved["value_net"][key], value)


def test_resample_hidden_world_state_preserves_public_state_and_hero_cards():
    state = FastPokerState(n_players=2, initial_chips=1000)
    state.apply_action(1)
    state.apply_action(1)

    rng = np.random.default_rng(20260525)
    sampled = resample_hidden_world_state(state, rng)
    hero = int(state.current_player_i)

    assert sampled.stage == state.stage
    assert sampled.current_player_i == state.current_player_i
    assert np.array_equal(sampled.community, state.community)
    assert np.array_equal(sampled.chips, state.chips)
    assert np.array_equal(sampled.bets, state.bets)
    assert np.array_equal(sampled.get_legal_mask(), state.get_legal_mask())
    assert np.array_equal(sampled.hole_cards[hero], state.hole_cards[hero])
    visible = {
        int(card)
        for card in [*sampled.hole_cards[hero], *sampled.community]
        if int(card) >= 0
    }
    opponent = {
        int(card)
        for player in range(sampled.n_players)
        if player != hero
        for card in sampled.hole_cards[player]
    }
    assert not visible.intersection(opponent)
    assert len(set(int(card) for card in sampled.deck_order)) == 52


def test_collect_on_policy_decision_value_targets_returns_multi_street_metadata(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    _write_average_policy_checkpoint(checkpoint, seed=17)

    targets, metadata = collect_on_policy_decision_value_policy_targets(
        OnPolicyDecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_states=4,
            n_worlds=2,
            seed=20260525,
            device="cpu",
            target_streets=(0, 1),
            max_hands=20,
        )
    )

    assert targets.size == 4
    assert metadata["mode"] == "on_policy_decision_value_policy_targets"
    assert metadata["street_counts"]
    assert set(metadata["street_counts"]).issubset({"0", "1"})
    assert metadata["n_truncated_rollouts"] == 0


def test_on_policy_collection_continues_until_required_street_coverage(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"
    _write_average_policy_checkpoint(checkpoint, seed=19)

    def always_call_policy(_loaded, _device, *, strategy_source, greedy=False):
        def _policy(state, _rng):
            mask = state.get_legal_mask()
            if mask[1] > 0:
                return 1
            legal = np.flatnonzero(mask > 0)
            return int(legal[0])

        return _policy

    import poker_ai.research.decision_value_actor as module

    monkeypatch.setattr(module, "checkpoint_continuation_policy", always_call_policy)

    targets, metadata = collect_on_policy_decision_value_policy_targets(
        OnPolicyDecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_states=1,
            n_worlds=1,
            seed=20260525,
            device="cpu",
            target_streets=(0, 1),
            required_streets=(1,),
            max_hands=2,
        )
    )

    assert targets.size > 1
    assert metadata["street_counts"]["1"] >= 1
    assert metadata["coverage_gate"]["passed"] is True
    assert metadata["behavior_policy"] == "base"
    assert len(metadata["rows"][0]["action_values"]) == N_ACTIONS
    assert len(metadata["rows"][0]["prior_policy"]) == N_ACTIONS


def test_on_policy_collection_records_search_improved_behavior_actions(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"
    _write_average_policy_checkpoint(checkpoint, seed=29)

    def always_call_policy(_loaded, _device, *, strategy_source, greedy=False):
        return lambda _state, _rng: 1

    import poker_ai.research.decision_value_actor as module

    monkeypatch.setattr(module, "checkpoint_continuation_policy", always_call_policy)

    _targets, metadata = collect_on_policy_decision_value_policy_targets(
        OnPolicyDecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_states=2,
            n_worlds=1,
            seed=20260525,
            device="cpu",
            target_streets=(0,),
            behavior_policy="search-improved",
            max_hands=2,
        )
    )

    assert metadata["behavior_policy"] == "search-improved"
    assert all("behavior_action" in row for row in metadata["rows"])


def test_decision_value_targets_record_search_actor_selected_action(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"
    _write_average_policy_checkpoint(checkpoint, seed=31)

    def always_call_policy(_loaded, _device, *, strategy_source, greedy=False):
        return lambda _state, _rng: 1

    import poker_ai.research.decision_value_actor as module

    monkeypatch.setattr(module, "checkpoint_continuation_policy", always_call_policy)

    _targets, metadata = module.collect_decision_value_policy_targets(
        DecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_roots=2,
            n_worlds=1,
            seed=20260525,
            device="cpu",
        )
    )

    assert "selected_action_counts" in metadata
    assert "target_selected_action_counts" in metadata
    for row in metadata["rows"]:
        action = row["target_selected_action"]
        assert row["target_selected_action_value"] == row["action_values"][action]


def test_build_on_policy_average_strategy_target_artifact_saves_gpu_ready_targets(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"
    output_npz = tmp_path / "avg_strategy_targets.npz"
    output_json = tmp_path / "avg_strategy_targets.json"
    _write_average_policy_checkpoint(checkpoint, seed=37)

    def always_call_policy(_loaded, _device, *, strategy_source, greedy=False):
        return lambda _state, _rng: 1

    import poker_ai.research.decision_value_actor as module

    monkeypatch.setattr(module, "checkpoint_continuation_policy", always_call_policy)

    metrics = build_on_policy_average_strategy_target_artifact(
        OnPolicyDecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_states=2,
            n_worlds=1,
            seed=20260525,
            device="cpu",
            target_streets=(0,),
            behavior_policy="search-improved",
            max_hands=2,
        ),
        output_npz,
        output_json=output_json,
        recommended_average_strategy_weight=0.75,
    )

    reloaded = PolicyTargetBuffer.from_npz(output_npz)
    assert reloaded.size == 2
    assert metrics["passed"] is True
    assert metrics["mode"] == "on_policy_average_strategy_target_artifact"
    assert metrics["average_strategy_targets"] == str(output_npz)
    assert metrics["recommended_gpu_flags"]["average_strategy_targets"] == str(output_npz)
    assert metrics["recommended_gpu_flags"]["average_strategy_weight"] == 0.75
    assert metrics["target_metadata"]["behavior_policy"] == "search-improved"
    assert output_json.exists()


def test_on_policy_behavior_targets_train_on_actual_behavior_action(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"
    output_npz = tmp_path / "behavior_targets.npz"
    output_json = tmp_path / "behavior_targets.json"
    _write_average_policy_checkpoint(checkpoint, seed=41)

    def always_call_policy(_loaded, _device, *, strategy_source, greedy=False):
        return lambda _state, _rng: 1

    import poker_ai.research.decision_value_actor as module

    monkeypatch.setattr(module, "checkpoint_continuation_policy", always_call_policy)

    targets, metadata = collect_on_policy_behavior_average_strategy_targets(
        OnPolicyDecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_states=2,
            n_worlds=1,
            seed=20260525,
            device="cpu",
            target_streets=(0,),
            behavior_policy="search-improved",
            max_hands=2,
        )
    )

    for row_idx, row in enumerate(metadata["rows"]):
        action_idx = row["behavior_action_idx"]
        assert targets.target_probs[row_idx, action_idx] == 1.0
        assert targets.target_probs[row_idx].sum() == 1.0

    metrics = build_on_policy_behavior_average_strategy_target_artifact(
        OnPolicyDecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_states=1,
            n_worlds=1,
            seed=20260526,
            device="cpu",
            target_streets=(0,),
            behavior_policy="base",
            max_hands=2,
        ),
        output_npz,
        output_json=output_json,
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "on_policy_behavior_average_strategy_target_artifact"
    assert PolicyTargetBuffer.from_npz(output_npz).size == 1
    assert output_json.exists()


def test_trajectory_return_weights_boost_positive_realized_returns():
    returns = np.array([-100.0, 0.0, 100.0], dtype=np.float32)

    weights = trajectory_return_weights(
        returns,
        temperature=1.0,
        max_weight=10.0,
    )

    assert weights.shape == returns.shape
    assert np.all(weights >= 0.0)
    assert weights[2] > weights[1] > weights[0]
    assert np.isclose(float(weights.mean()), 1.0)


def test_trajectory_return_targets_weight_actual_behavior_actions(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"
    output_npz = tmp_path / "trajectory_targets.npz"
    output_json = tmp_path / "trajectory_targets.json"
    _write_average_policy_checkpoint(checkpoint, seed=43)

    def always_call_policy(_loaded, _device, *, strategy_source, greedy=False):
        return lambda _state, _rng: 1

    import poker_ai.research.decision_value_actor as module

    monkeypatch.setattr(module, "checkpoint_continuation_policy", always_call_policy)

    targets, metadata = collect_on_policy_trajectory_return_average_strategy_targets(
        OnPolicyDecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_states=2,
            n_worlds=1,
            seed=20260527,
            device="cpu",
            target_streets=(0,),
            behavior_policy="base",
            max_hands=4,
        )
    )

    assert targets.size == 2
    assert metadata["mode"] == "on_policy_trajectory_return_average_strategy_targets"
    assert metadata["target_policy"] == "one_hot_behavior_action"
    assert metadata["return_weight_mode"] == "centered_exp"
    assert metadata["n_terminal_hands"] >= 1
    assert np.all(targets.weights >= 0.0)
    for row_idx, row in enumerate(metadata["rows"]):
        action_idx = row["behavior_action_idx"]
        assert targets.target_probs[row_idx, action_idx] == 1.0
        assert row["trajectory_return_weight"] == pytest.approx(float(targets.weights[row_idx]))

    metrics = build_on_policy_trajectory_return_average_strategy_target_artifact(
        OnPolicyDecisionValueTargetConfig(
            checkpoint=str(checkpoint),
            strategy_source="average-policy",
            continuation_checkpoints=(str(checkpoint),),
            continuation_strategy_source="average-policy",
            n_states=1,
            n_worlds=1,
            seed=20260528,
            device="cpu",
            target_streets=(0,),
            behavior_policy="base",
            max_hands=3,
        ),
        output_npz,
        output_json=output_json,
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "on_policy_trajectory_return_average_strategy_target_artifact"
    assert PolicyTargetBuffer.from_npz(output_npz).size == 1
    assert output_json.exists()


def test_on_policy_collection_rejects_missing_required_street_coverage(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    _write_average_policy_checkpoint(checkpoint, seed=23)

    with pytest.raises(RuntimeError, match="missing required street coverage"):
        collect_on_policy_decision_value_policy_targets(
            OnPolicyDecisionValueTargetConfig(
                checkpoint=str(checkpoint),
                strategy_source="average-policy",
                continuation_checkpoints=(str(checkpoint),),
                continuation_strategy_source="average-policy",
                n_states=1,
                n_worlds=1,
                seed=20260525,
                device="cpu",
                target_streets=(0, 2),
                required_streets=(2,),
                max_steps_per_hand=1,
                max_hands=1,
            )
        )
