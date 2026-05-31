import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS
from poker_ai.research.neural_policy_iteration import (
    NeuralPolicyIterationCFRGateConfig,
    NeuralPolicyIterationPublicWorldGateConfig,
    LegalMixedPolicyImprovementTeacher,
    NeuralPolicyIterationConfig,
    PublicBeliefCFRPolicyImprovementTeacher,
    PolicyContinuationPublicStateRolloutTeacher,
    PolicyContinuationPublicWorldRolloutTeacher,
    PublicWorldRolloutPolicyImprovementTeacher,
    PublicWorldValuePolicyImprovementTeacher,
    SampledStateValuePolicyImprovementTeacher,
    SelfPlayPolicySample,
    _PolicyNet,
    _ValueNet,
    _batched_network_policies,
    _policy_target_sample_weights,
    _select_improvement_samples,
    _teacher_batching_summary,
    _target_stats,
    _train_on_policy_improvement_targets,
    collect_stochastic_self_play_samples,
    evaluate_neural_policy_iteration_head_to_head,
    evaluate_neural_policy_iteration_cfr_gate,
    evaluate_neural_policy_iteration_public_world_gate,
    run_neural_policy_iteration_pilot,
)


def test_policy_improvement_teacher_returns_legal_mixed_targets():
    teacher = LegalMixedPolicyImprovementTeacher(
        preferred_action=8,
        preferred_action_prob=0.7,
    )
    legal_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

    target = teacher.improve_one(legal_mask)

    assert target.shape == (N_ACTIONS,)
    assert np.isclose(float(target.sum()), 1.0)
    assert np.all(target[legal_mask <= 0] == 0.0)
    assert target[8] > target[0]
    assert target[0] > 0.0
    assert target[1] > 0.0


def test_stochastic_self_play_collector_uses_local_policy_actor_only():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=3,
        hidden_dim=16,
        max_steps_per_hand=32,
        seed=20260526,
        device="cpu",
    )

    samples = collect_stochastic_self_play_samples(cfg)

    assert len(samples) > 0
    assert {sample.source for sample in samples} == {"local_self_play"}
    assert all(sample.features.shape[0] == cfg.feature_dim for sample in samples)
    assert all(sample.legal_mask.shape == (N_ACTIONS,) for sample in samples)
    assert all(sample.behavior_policy.shape == (N_ACTIONS,) for sample in samples)
    assert all(np.isclose(float(sample.behavior_policy.sum()), 1.0) for sample in samples)
    assert all(sample.behavior_policy[int(sample.action)] > 0.0 for sample in samples)


def test_batched_network_policies_are_legal_normalized_distributions():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    policy_net = _PolicyNet(hidden_dim=16)
    state_a = new_game(2, initial_chips=1000)
    state_b = state_a.apply_action("call")
    features = np.stack([state_a.to_feature_vector(), state_b.to_feature_vector()])
    legal_masks = np.stack([get_legal_mask(state_a), get_legal_mask(state_b)])

    probs = _batched_network_policies(
        policy_net,
        features,
        legal_masks,
        torch.device("cpu"),
    )

    assert probs.shape == (2, N_ACTIONS)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs[legal_masks <= 0] == 0.0)


def test_inverse_target_top_sample_weights_counter_target_imbalance():
    targets = np.array(
        [
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        ],
        dtype=np.float32,
    )

    uniform = _policy_target_sample_weights(targets, mode="uniform")
    balanced = _policy_target_sample_weights(targets, mode="inverse_target_top")

    assert np.allclose(uniform, np.ones(4, dtype=np.float32))
    assert np.isclose(float(balanced.mean()), 1.0)
    assert balanced[3] > balanced[0]
    assert np.allclose(balanced[:3], balanced[0])


def test_target_stats_report_top_actions_by_street():
    class _State:
        def __init__(self, street: str) -> None:
            self.betting_stage = street

    mask = np.ones(N_ACTIONS, dtype=np.float32)
    samples = [
        SelfPlayPolicySample(
            features=np.zeros(126, dtype=np.float32),
            legal_mask=mask,
            behavior_policy=np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32),
            action=1,
            player=0,
            value_target=0.0,
            state=_State("turn"),
        ),
        SelfPlayPolicySample(
            features=np.zeros(126, dtype=np.float32),
            legal_mask=mask,
            behavior_policy=np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32),
            action=8,
            player=1,
            value_target=0.0,
            state=_State("river"),
        ),
        SelfPlayPolicySample(
            features=np.zeros(126, dtype=np.float32),
            legal_mask=mask,
            behavior_policy=np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32),
            action=8,
            player=0,
            value_target=0.0,
            state=_State("river"),
        ),
    ]
    targets = np.zeros((3, N_ACTIONS), dtype=np.float32)
    targets[0, 1] = 1.0
    targets[1, 8] = 1.0
    targets[2, 8] = 1.0

    stats = _target_stats(targets, samples)

    assert stats["target_top_action_counts_by_street"] == {
        "river": {"8": 2},
        "turn": {"1": 1},
    }
    assert stats["max_target_top_action_fraction_by_street"] == {
        "river": 1.0,
        "turn": 1.0,
    }


def test_policy_improvement_training_uses_all_self_play_value_samples():
    policy_net = _PolicyNet(hidden_dim=8)
    value_net = _ValueNet(hidden_dim=8)
    mask = np.ones(N_ACTIONS, dtype=np.float32)
    all_samples = [
        SelfPlayPolicySample(
            features=np.full(126, float(index), dtype=np.float32),
            legal_mask=mask,
            behavior_policy=np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32),
            action=1,
            player=index % 2,
            value_target=float(index % 2),
        )
        for index in range(4)
    ]
    policy_samples = all_samples[:2]
    targets = np.zeros((2, N_ACTIONS), dtype=np.float32)
    targets[:, 1] = 1.0

    losses = _train_on_policy_improvement_targets(
        policy_net,
        value_net,
        policy_samples,
        targets,
        NeuralPolicyIterationConfig(train_steps=0, hidden_dim=8, batch_size=4, device="cpu"),
        torch.device("cpu"),
        np.random.default_rng(20260584),
        value_samples=all_samples,
    )

    assert losses["policy_training_samples"] == 2
    assert losses["value_training_samples"] == 4


def test_parallel_self_play_collector_preserves_stochastic_contract():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=8,
        parallel_self_play_hands=4,
        hidden_dim=16,
        max_steps_per_hand=32,
        seed=20260533,
        device="cpu",
    )

    samples = collect_stochastic_self_play_samples(cfg)

    assert len(samples) > 0
    assert all(sample.source == "local_self_play" for sample in samples)
    assert all(np.isclose(float(sample.behavior_policy.sum()), 1.0) for sample in samples)


def test_public_cfr_collector_extends_until_min_searchable_states():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=1,
        max_self_play_hands=64,
        min_searchable_self_play_states=1,
        parallel_self_play_hands=8,
        hidden_dim=16,
        max_steps_per_hand=96,
        teacher_mode="public_belief_cfr",
        seed=20260535,
        device="cpu",
    )

    samples = collect_stochastic_self_play_samples(cfg)
    searchable = [
        sample
        for sample in samples
        if PublicBeliefCFRPolicyImprovementTeacher._is_street_root(sample.state)
    ]

    assert len(searchable) >= 1


def test_neural_policy_iteration_pilot_exports_alphazero_style_contract(tmp_path):
    checkpoint_path = tmp_path / "neural_policy_iteration.pt"

    metrics = run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=4,
            max_improvement_targets=16,
            train_steps=3,
            hidden_dim=16,
            batch_size=8,
            policy_sample_weighting="inverse_target_top",
            max_steps_per_hand=32,
            checkpoint_path=str(checkpoint_path),
            seed=20260527,
            device="cpu",
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["algorithm"] == "neural_self_play_policy_iteration"
    assert metrics["role"] == "alphazero_style_policy_iteration_contract_smoke"
    assert metrics["neural_policy_role"] == "main_stochastic_self_play_actor"
    assert metrics["cfr_role"] == "policy_improvement_teacher_interface"
    assert metrics["stochastic_policy_contract"] == "sample_mixed_strategy_not_argmax_by_default"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["promotion"] is False
    assert metrics["self_play_states"] > 0
    assert metrics["improvement_targets"] > 0
    assert metrics["policy_sample_weighting"] == "inverse_target_top"
    assert metrics["policy_sample_weight_mean"] > 0.0
    assert sum(metrics["target_top_action_counts"].values()) == metrics["improvement_targets"]
    assert 0.0 <= metrics["max_target_top_action_fraction"] <= 1.0
    assert metrics["self_play_collection_sec"] >= 0.0
    assert metrics["self_play_feature_mask_sec"] >= 0.0
    assert metrics["self_play_policy_inference_sec"] >= 0.0
    assert metrics["self_play_env_step_sec"] >= 0.0
    assert metrics["self_play_finalize_sec"] >= 0.0
    assert metrics["teacher_target_sec"] >= 0.0
    assert metrics["training_sec"] >= 0.0
    assert payload["algorithm"] == metrics["algorithm"]
    assert payload["config"]["self_play_hands"] == 4
    assert "policy_net_state_dict" in payload
    assert "value_net_state_dict" in payload


def test_neural_policy_iteration_public_cfr_mode_exports_teacher_coverage(tmp_path):
    checkpoint_path = tmp_path / "neural_policy_iteration_cfr.pt"

    metrics = run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=2,
            max_improvement_targets=4,
            train_steps=1,
            hidden_dim=16,
            batch_size=4,
            max_steps_per_hand=32,
            teacher_mode="public_belief_cfr",
            cfr_iterations=1,
            cfr_backend="cpu",
            checkpoint_path=str(checkpoint_path),
            seed=20260531,
            device="cpu",
        )
    )

    assert metrics["teacher_mode"] == "public_belief_cfr"
    assert metrics["cfr_role"] == "public_belief_cfr_policy_improvement_teacher"
    assert metrics["cfr_iterations"] == 1
    assert metrics["cfr_backend"] == "cpu"
    assert metrics["uses_resolver_teacher"] is True
    assert "searchable_self_play_states" in metrics
    assert "resolver_targets" in metrics
    assert "fallback_targets" in metrics
    assert metrics["resolver_targets"] + metrics["fallback_targets"] == metrics["improvement_targets"]


def test_neural_policy_iteration_public_cfr_mode_can_target_searchable_coverage(tmp_path):
    checkpoint_path = tmp_path / "neural_policy_iteration_cfr_coverage.pt"

    metrics = run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=1,
            max_self_play_hands=64,
            min_searchable_self_play_states=1,
            max_improvement_targets=4,
            train_steps=1,
            hidden_dim=16,
            batch_size=4,
            parallel_self_play_hands=8,
            max_steps_per_hand=96,
            teacher_mode="public_belief_cfr",
            cfr_iterations=1,
            cfr_backend="cpu",
            min_resolver_targets=1,
            checkpoint_path=str(checkpoint_path),
            seed=20260535,
            device="cpu",
        )
    )

    assert metrics["self_play_hands"] == 1
    assert metrics["max_self_play_hands"] == 64
    assert metrics["min_searchable_self_play_states"] == 1
    assert metrics["searchable_self_play_states"] >= 1
    assert metrics["resolver_targets"] >= 1
    assert metrics["passed"] is True


def test_neural_policy_iteration_public_cfr_mode_fails_min_target_gate(tmp_path):
    checkpoint_path = tmp_path / "neural_policy_iteration_cfr.pt"

    metrics = run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=0,
            max_improvement_targets=4,
            train_steps=1,
            hidden_dim=16,
            batch_size=4,
            teacher_mode="public_belief_cfr",
            cfr_iterations=1,
            cfr_backend="cpu",
            min_resolver_targets=1,
            checkpoint_path=str(checkpoint_path),
            seed=20260532,
            device="cpu",
        )
    )

    assert metrics["teacher_mode"] == "public_belief_cfr"
    assert metrics["resolver_targets"] == 0
    assert metrics["min_resolver_targets"] == 1
    assert metrics["passed"] is False
    assert "resolver_targets 0 < min_resolver_targets 1" in metrics["gate_failures"]


def test_public_belief_cfr_teacher_returns_legal_turn_target():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    state = new_game(2, initial_chips=1000)
    for _ in range(12):
        if state.betting_stage == "turn":
            break
        state = state.apply_action("call")
    assert state.betting_stage == "turn"
    legal_mask = get_legal_mask(state)
    sample = SelfPlayPolicySample(
        features=state.to_feature_vector(),
        legal_mask=legal_mask,
        behavior_policy=legal_mask / float(legal_mask.sum()),
        action=1,
        player=int(state.player_i),
        value_target=0.0,
        state=state,
    )
    teacher = PublicBeliefCFRPolicyImprovementTeacher(
        n_iterations=1,
        backend="cpu",
    )

    target = teacher.improve_one(sample)

    assert target.shape == (N_ACTIONS,)
    assert np.isclose(float(target.sum()), 1.0)
    assert np.all(target[legal_mask <= 0] == 0.0)
    assert teacher.last_stats["resolver_targets"] == 1
    assert teacher.last_stats["fallback_targets"] == 0


def test_public_belief_cfr_teacher_batches_same_topology_turn_roots():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    state = new_game(2, initial_chips=1000)
    for _ in range(12):
        if state.betting_stage == "turn":
            break
        state = state.apply_action("call")
    legal_mask = get_legal_mask(state)
    samples = [
        SelfPlayPolicySample(
            features=state.to_feature_vector(),
            legal_mask=legal_mask,
            behavior_policy=legal_mask / float(legal_mask.sum()),
            action=1,
            player=int(state.player_i),
            value_target=0.0,
            state=state,
        )
        for _ in range(2)
    ]
    teacher = PublicBeliefCFRPolicyImprovementTeacher(
        n_iterations=1,
        backend="torch-levelsync-cpu",
        device="cpu",
        batch_roots=True,
        batch_min_roots=2,
    )

    targets = teacher.improve(samples)

    assert targets.shape == (2, N_ACTIONS)
    np.testing.assert_allclose(targets.sum(axis=1), np.ones(2), atol=1e-6)
    assert np.all(targets[:, legal_mask <= 0] == 0.0)
    assert teacher.last_stats["resolver_targets"] == 2
    assert teacher.last_stats["fallback_targets"] == 0
    assert teacher.last_stats["batched_solver_roots"] == 2
    assert teacher.last_stats["batched_solver_groups"] == 1
    assert teacher.last_stats["serial_solver_roots"] == 0
    assert teacher.last_stats["solver_context_roots"] == 2
    assert teacher.last_stats["solver_context_sec"] >= 0.0


def test_public_belief_cfr_batching_does_not_mix_turn_and_river_hand_counts():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    state = new_game(2, initial_chips=1000)
    turn_state = None
    river_state = None
    for _ in range(20):
        if state.betting_stage == "turn" and turn_state is None:
            turn_state = state
        if state.betting_stage == "river":
            river_state = state
            break
        state = state.apply_action("call")
    assert turn_state is not None
    assert river_state is not None
    samples = []
    for root_state in (turn_state, river_state):
        legal_mask = get_legal_mask(root_state)
        samples.append(
            SelfPlayPolicySample(
                features=root_state.to_feature_vector(),
                legal_mask=legal_mask,
                behavior_policy=legal_mask / float(legal_mask.sum()),
                action=1,
                player=int(root_state.player_i),
                value_target=0.0,
                state=root_state,
            )
        )
    teacher = PublicBeliefCFRPolicyImprovementTeacher(
        n_iterations=1,
        backend="torch-levelsync-cpu",
        device="cpu",
        batch_roots=True,
        batch_min_roots=2,
    )

    targets = teacher.improve(samples)

    assert targets.shape == (2, N_ACTIONS)
    assert teacher.last_stats["solver_context_roots"] == 2
    assert teacher.last_stats["batchable_solver_groups"] == 0
    assert teacher.last_stats["batched_solver_failures"] == 0
    assert teacher.last_stats["serial_solver_roots"] == 2


def test_public_world_rollout_teacher_returns_legal_preflop_root_target():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    state = new_game(2, initial_chips=1000)
    legal_mask = get_legal_mask(state)
    sample = SelfPlayPolicySample(
        features=state.to_feature_vector(),
        legal_mask=legal_mask,
        behavior_policy=legal_mask / float(legal_mask.sum()),
        action=1,
        player=int(state.player_i),
        value_target=0.0,
        state=state,
    )
    teacher = PublicWorldRolloutPolicyImprovementTeacher(
        n_worlds=2,
        temperature=100.0,
        seed=20260539,
    )

    target = teacher.improve_one(sample)

    assert target.shape == (N_ACTIONS,)
    assert np.isclose(float(target.sum()), 1.0)
    assert np.all(target[legal_mask <= 0] == 0.0)
    assert teacher.last_stats["rollout_targets"] == 1
    assert teacher.last_stats["fallback_targets"] == 0


def test_policy_continuation_public_world_rollout_teacher_uses_actor_policy():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    class CountingPolicyNet(_PolicyNet):
        def __init__(self) -> None:
            super().__init__(hidden_dim=16)
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return super().forward(x)

    state = new_game(2, initial_chips=1000)
    legal_mask = get_legal_mask(state)
    sample = SelfPlayPolicySample(
        features=state.to_feature_vector(),
        legal_mask=legal_mask,
        behavior_policy=legal_mask / float(legal_mask.sum()),
        action=1,
        player=int(state.player_i),
        value_target=0.0,
        state=state,
    )
    policy_net = CountingPolicyNet()
    teacher = PolicyContinuationPublicWorldRolloutTeacher(
        policy_net=policy_net,
        device=torch.device("cpu"),
        n_worlds=2,
        temperature=100.0,
        seed=20260541,
    )

    target = teacher.improve_one(sample)

    assert policy_net.calls > 0
    assert target.shape == (N_ACTIONS,)
    assert np.isclose(float(target.sum()), 1.0)
    assert np.all(target[legal_mask <= 0] == 0.0)
    assert teacher.last_stats["rollout_targets"] == 1
    assert teacher.last_stats["policy_continuation_targets"] == 1
    assert teacher.last_stats["fallback_targets"] == 0


def test_policy_continuation_public_state_rollout_teacher_targets_later_street():
    from poker_ai.deep_cfr.fast_state import FastPokerState

    class CountingPolicyNet(_PolicyNet):
        def __init__(self) -> None:
            super().__init__(hidden_dim=16)
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return super().forward(x)

    state = FastPokerState(n_players=2, initial_chips=1000)
    for _ in range(12):
        if int(state.stage) >= int(FastPokerState.FLOP) or state.is_terminal:
            break
        state.apply_action(1)
    assert not state.is_terminal
    assert int(state.stage) >= int(FastPokerState.FLOP)
    legal_mask = state.get_legal_mask().astype(np.float32)
    sample = SelfPlayPolicySample(
        features=state.to_feature_vector(),
        legal_mask=legal_mask,
        behavior_policy=legal_mask / float(legal_mask.sum()),
        action=1,
        player=int(state.current_player_i),
        value_target=0.0,
        state=state,
    )
    policy_net = CountingPolicyNet()
    teacher = PolicyContinuationPublicStateRolloutTeacher(
        policy_net=policy_net,
        device=torch.device("cpu"),
        n_worlds=2,
        temperature=100.0,
        seed=20260543,
    )

    target = teacher.improve_one(sample)

    assert policy_net.calls > 0
    assert target.shape == (N_ACTIONS,)
    assert np.isclose(float(target.sum()), 1.0)
    assert np.all(target[legal_mask <= 0] == 0.0)
    assert teacher.last_stats["public_state_rollout_targets"] == 1
    assert teacher.last_stats["policy_continuation_targets"] == 1
    assert teacher.last_stats["fallback_targets"] == 0


def test_public_world_value_teacher_consumes_value_network():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    class CountingValueNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.zeros(x.shape[0], device=x.device)

    state = new_game(2, initial_chips=1000)
    legal_mask = get_legal_mask(state)
    sample = SelfPlayPolicySample(
        features=state.to_feature_vector(),
        legal_mask=legal_mask,
        behavior_policy=legal_mask / float(legal_mask.sum()),
        action=1,
        player=int(state.player_i),
        value_target=0.0,
        state=state,
    )
    value_net = CountingValueNet()
    teacher = PublicWorldValuePolicyImprovementTeacher(
        value_net=value_net,
        device=torch.device("cpu"),
        n_worlds=2,
        temperature=1.0,
        seed=20260585,
    )

    target = teacher.improve_one(sample)

    assert value_net.calls > 0
    assert target.shape == (N_ACTIONS,)
    assert np.isclose(float(target.sum()), 1.0)
    assert np.all(target[legal_mask <= 0] == 0.0)
    assert teacher.last_stats["value_search_targets"] == 1
    assert teacher.last_stats["fallback_targets"] == 0


def test_public_world_value_teacher_can_regularize_with_actor_prior():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    class ZeroValueNet(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], device=x.device)

    state = new_game(2, initial_chips=1000)
    legal_mask = get_legal_mask(state)
    legal_mask[0] = 0.0
    legal_mask[8] = 0.0
    prior = legal_mask.astype(np.float32)
    legal = np.flatnonzero(legal_mask > 0)
    prior[legal] = np.linspace(1.0, 2.0, num=legal.size, dtype=np.float32)
    prior = prior / float(prior.sum())
    sample = SelfPlayPolicySample(
        features=state.to_feature_vector(),
        legal_mask=legal_mask,
        behavior_policy=prior,
        action=int(legal[-1]),
        player=int(state.player_i),
        value_target=0.0,
        state=state,
    )
    teacher = PublicWorldValuePolicyImprovementTeacher(
        value_net=ZeroValueNet(),
        device=torch.device("cpu"),
        n_worlds=2,
        temperature=1.0,
        seed=20260588,
        use_policy_prior=True,
    )

    target = teacher.improve_one(sample)

    assert np.allclose(target, prior, atol=1e-6)


def test_sampled_state_value_prior_teacher_targets_turn_state_without_fallback():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    class ZeroValueNet(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], device=x.device)

    state = new_game(2, initial_chips=1000)
    for _ in range(12):
        if state.betting_stage == "turn":
            break
        state = state.apply_action("call")
    assert state.betting_stage == "turn"
    legal_mask = get_legal_mask(state)
    legal_mask[0] = 0.0
    prior = legal_mask.astype(np.float32) / float(legal_mask.sum())
    sample = SelfPlayPolicySample(
        features=state.to_feature_vector(),
        legal_mask=legal_mask,
        behavior_policy=prior,
        action=1,
        player=int(state.player_i),
        value_target=0.0,
        state=state,
    )
    teacher = SampledStateValuePolicyImprovementTeacher(
        value_net=ZeroValueNet(),
        device=torch.device("cpu"),
        temperature=1.0,
        use_policy_prior=True,
    )

    target = teacher.improve_one(sample)

    assert target.shape == (N_ACTIONS,)
    assert np.isclose(float(target.sum()), 1.0)
    assert np.all(target[legal_mask <= 0] == 0.0)
    assert np.allclose(target, prior, atol=1e-6)
    assert teacher.last_stats["state_value_targets"] == 1
    assert teacher.last_stats["fallback_targets"] == 0


def test_public_world_rollout_pilot_reports_dense_teacher_coverage():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=3,
        max_improvement_targets=3,
        train_steps=1,
        hidden_dim=16,
        batch_size=4,
        teacher_mode="public_world_rollout",
        rollout_worlds=2,
        rollout_temperature=100.0,
        max_steps_per_hand=32,
        seed=20260540,
        device="cpu",
    )

    metrics = run_neural_policy_iteration_pilot(cfg)

    assert metrics["teacher_mode"] == "public_world_rollout"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_resolver_teacher"] is False
    assert metrics["improvement_targets"] == 3
    assert metrics["rollout_targets"] == metrics["improvement_targets"]
    assert metrics["fallback_targets"] == 0
    assert metrics["rollout_eligible_self_play_states"] >= metrics["improvement_targets"]
    assert metrics["dropped_ineligible_self_play_states"] > 0


def test_policy_public_world_rollout_pilot_reports_policy_continuation_teacher():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=3,
        max_improvement_targets=3,
        train_steps=1,
        hidden_dim=16,
        batch_size=4,
        teacher_mode="policy_public_world_rollout",
        rollout_worlds=2,
        rollout_temperature=100.0,
        max_steps_per_hand=32,
        seed=20260542,
        device="cpu",
    )

    metrics = run_neural_policy_iteration_pilot(cfg)

    assert metrics["teacher_mode"] == "policy_public_world_rollout"
    assert metrics["cfr_role"] == "current_neural_policy_public_world_policy_improvement_teacher"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_resolver_teacher"] is False
    assert metrics["improvement_targets"] == 3
    assert metrics["rollout_targets"] == metrics["improvement_targets"]
    assert metrics["policy_continuation_targets"] == metrics["improvement_targets"]
    assert metrics["fallback_targets"] == 0


def test_policy_public_state_rollout_pilot_reports_all_street_teacher():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=8,
        max_improvement_targets=8,
        train_steps=1,
        hidden_dim=16,
        batch_size=4,
        teacher_mode="policy_public_state_rollout",
        self_play_state_backend="fast",
        rollout_worlds=2,
        rollout_temperature=100.0,
        max_steps_per_hand=64,
        min_target_streets=2,
        seed=20260544,
        device="cpu",
    )

    metrics = run_neural_policy_iteration_pilot(cfg)

    assert metrics["teacher_mode"] == "policy_public_state_rollout"
    assert metrics["cfr_role"] == "all_street_current_neural_policy_public_state_improvement_teacher"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_resolver_teacher"] is False
    assert metrics["improvement_targets"] == 8
    assert metrics["public_state_rollout_targets"] == metrics["improvement_targets"]
    assert metrics["policy_continuation_targets"] == metrics["improvement_targets"]
    assert len(metrics["target_count_by_street"]) >= 2
    assert metrics["gate_failures"] == []
    assert metrics["fallback_targets"] == 0


def test_public_world_value_pilot_reports_value_search_teacher():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=3,
        max_improvement_targets=3,
        train_steps=1,
        hidden_dim=16,
        batch_size=4,
        teacher_mode="public_world_value",
        self_play_state_backend="fast",
        rollout_worlds=2,
        rollout_temperature=1.0,
        max_steps_per_hand=32,
        seed=20260586,
        device="cpu",
    )

    metrics = run_neural_policy_iteration_pilot(cfg)

    assert metrics["teacher_mode"] == "public_world_value"
    assert metrics["cfr_role"] == "learned_value_public_world_policy_improvement_teacher"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_resolver_teacher"] is False
    assert metrics["value_search_targets"] == metrics["improvement_targets"]
    assert metrics["fallback_targets"] == 0
    assert metrics["value_training_samples"] >= metrics["policy_training_samples"]


def test_public_world_value_prior_pilot_reports_prior_guided_teacher():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=3,
        max_improvement_targets=3,
        train_steps=1,
        hidden_dim=16,
        batch_size=4,
        teacher_mode="public_world_value_prior",
        self_play_state_backend="fast",
        rollout_worlds=2,
        rollout_temperature=1.0,
        max_steps_per_hand=32,
        seed=20260589,
        device="cpu",
    )

    metrics = run_neural_policy_iteration_pilot(cfg)

    assert metrics["teacher_mode"] == "public_world_value_prior"
    assert metrics["cfr_role"] == "prior_guided_learned_value_public_world_policy_improvement_teacher"
    assert metrics["value_search_targets"] == metrics["improvement_targets"]
    assert metrics["fallback_targets"] == 0


def test_sampled_state_value_prior_pilot_reports_multi_street_teacher():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=4,
        max_improvement_targets=12,
        train_steps=1,
        hidden_dim=16,
        batch_size=4,
        teacher_mode="sampled_state_value_prior",
        self_play_state_backend="fast",
        rollout_temperature=1.0,
        max_steps_per_hand=64,
        seed=20260594,
        device="cpu",
    )

    metrics = run_neural_policy_iteration_pilot(cfg)

    assert metrics["teacher_mode"] == "sampled_state_value_prior"
    assert metrics["cfr_role"] == "prior_guided_sampled_state_value_policy_improvement_teacher"
    assert metrics["state_value_targets"] == metrics["improvement_targets"]
    assert metrics["fallback_targets"] == 0
    assert metrics["target_count_by_street"]
    assert any(street != "pre_flop" for street in metrics["target_count_by_street"])


def test_public_world_rollout_pilot_can_use_fast_self_play_backend():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=4,
        parallel_self_play_hands=4,
        max_improvement_targets=4,
        train_steps=1,
        hidden_dim=16,
        batch_size=4,
        teacher_mode="public_world_rollout",
        self_play_state_backend="fast",
        rollout_worlds=2,
        rollout_temperature=100.0,
        max_steps_per_hand=32,
        seed=20260542,
        device="cpu",
    )

    metrics = run_neural_policy_iteration_pilot(cfg)

    assert metrics["self_play_state_backend"] == "fast"
    assert metrics["teacher_mode"] == "public_world_rollout"
    assert metrics["rollout_targets"] == metrics["improvement_targets"]
    assert metrics["fallback_targets"] == 0
    assert metrics["uses_slumbot_training_data"] is False


def test_public_belief_cfr_teacher_rejects_fast_self_play_backend():
    cfg = NeuralPolicyIterationConfig(
        self_play_hands=1,
        teacher_mode="public_belief_cfr",
        self_play_state_backend="fast",
        train_steps=1,
        hidden_dim=16,
        seed=20260543,
        device="cpu",
    )

    try:
        run_neural_policy_iteration_pilot(cfg)
    except ValueError as exc:
        assert "public_belief_cfr requires self_play_state_backend='full_deck'" in str(exc)
    else:
        raise AssertionError("public_belief_cfr should reject fast self-play states")


def test_neural_policy_iteration_public_world_gate_evaluates_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=4,
            parallel_self_play_hands=4,
            self_play_state_backend="fast",
            max_improvement_targets=4,
            train_steps=1,
            hidden_dim=16,
            batch_size=4,
            teacher_mode="public_world_rollout",
            rollout_worlds=2,
            checkpoint_path=str(checkpoint_path),
            seed=20260544,
            device="cpu",
        )
    )

    metrics = evaluate_neural_policy_iteration_public_world_gate(
        NeuralPolicyIterationPublicWorldGateConfig(
            checkpoint_path=str(checkpoint_path),
            n_roots=3,
            n_worlds=2,
            seed=20260545,
            device="cpu",
        )
    )

    assert metrics["algorithm"] == "neural_policy_iteration_public_world_gate"
    assert metrics["checkpoint_algorithm"] == "neural_self_play_policy_iteration"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["promotion"] is False
    assert metrics["n_roots"] == 3
    assert metrics["n_worlds"] == 2
    assert metrics["mean_policy_ev"] <= metrics["mean_oracle_ev"] + 1e-6
    assert metrics["mean_oracle_gap"] >= -1e-6
    assert "mean_policy_minus_selected_ev" in metrics
    assert "policy_minus_selected_ev" in metrics["rows"][0]
    assert len(metrics["rows"]) == 3


def test_neural_policy_iteration_public_world_gate_cli_writes_metrics(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    output_json = tmp_path / "gate.json"
    run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=2,
            parallel_self_play_hands=2,
            self_play_state_backend="fast",
            max_improvement_targets=2,
            train_steps=1,
            hidden_dim=16,
            batch_size=2,
            teacher_mode="public_world_rollout",
            rollout_worlds=2,
            checkpoint_path=str(checkpoint_path),
            seed=20260546,
            device="cpu",
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_neural_policy_iteration_public_world_gate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--n-roots",
            "2",
            "--n-worlds",
            "2",
            "--seed",
            "20260547",
            "--device",
            "cpu",
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    saved = json.loads(output_json.read_text())
    assert metrics["algorithm"] == "neural_policy_iteration_public_world_gate"
    assert saved["algorithm"] == metrics["algorithm"]
    assert saved["n_roots"] == 2


def test_neural_policy_iteration_cfr_gate_evaluates_resolver_targets(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=2,
            max_self_play_hands=64,
            min_searchable_self_play_states=1,
            parallel_self_play_hands=8,
            max_improvement_targets=1,
            train_steps=1,
            hidden_dim=16,
            batch_size=4,
            teacher_mode="public_belief_cfr",
            cfr_iterations=1,
            cfr_backend="cpu",
            min_resolver_targets=1,
            max_steps_per_hand=96,
            checkpoint_path=str(checkpoint_path),
            seed=20260549,
            device="cpu",
        )
    )

    metrics = evaluate_neural_policy_iteration_cfr_gate(
        NeuralPolicyIterationCFRGateConfig(
            checkpoint_path=str(checkpoint_path),
            n_roots=1,
            max_self_play_hands=64,
            cfr_iterations=1,
            cfr_backend="cpu",
            seed=20260550,
            device="cpu",
        )
    )

    assert metrics["algorithm"] == "neural_policy_iteration_exact_cfr_gate"
    assert metrics["checkpoint_algorithm"] == "neural_self_play_policy_iteration"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["promotion"] is False
    assert metrics["n_roots"] == 1
    assert metrics["resolver_targets"] == 1
    assert metrics["fallback_targets"] == 0
    assert metrics["evaluated_roots"] == 1
    assert sum(metrics["policy_top_action_counts"].values()) == metrics["evaluated_roots"]
    assert sum(metrics["cfr_top_action_counts"].values()) == metrics["evaluated_roots"]
    assert 0.0 <= metrics["max_policy_top_action_fraction"] <= 1.0
    assert metrics["mean_l1_to_cfr"] >= 0.0
    assert 0.0 <= metrics["mean_policy_mass_on_cfr_top_action"] <= 1.0
    assert metrics["mean_policy_entropy"] >= 0.0
    assert metrics["mean_cfr_target_entropy"] >= 0.0
    assert metrics["max_illegal_mass"] == 0.0
    assert "policy_mass_on_cfr_top_action" in metrics["rows"][0]
    assert "policy_entropy" in metrics["rows"][0]
    assert "cfr_target_entropy" in metrics["rows"][0]
    assert len(metrics["rows"]) == 1


def test_neural_policy_iteration_cfr_gate_cli_writes_metrics(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    output_json = tmp_path / "cfr_gate.json"
    run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=1,
            max_self_play_hands=64,
            min_searchable_self_play_states=1,
            parallel_self_play_hands=8,
            max_improvement_targets=1,
            train_steps=1,
            hidden_dim=16,
            batch_size=4,
            teacher_mode="public_belief_cfr",
            cfr_iterations=1,
            cfr_backend="cpu",
            min_resolver_targets=1,
            max_steps_per_hand=96,
            checkpoint_path=str(checkpoint_path),
            seed=20260551,
            device="cpu",
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_neural_policy_iteration_cfr_gate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--n-roots",
            "1",
            "--max-self-play-hands",
            "64",
            "--max-steps-per-hand",
            "96",
            "--cfr-iterations",
            "1",
            "--cfr-backend",
            "cpu",
            "--seed",
            "20260552",
            "--device",
            "cpu",
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    saved = json.loads(output_json.read_text())
    assert metrics["algorithm"] == "neural_policy_iteration_exact_cfr_gate"
    assert saved["algorithm"] == metrics["algorithm"]
    assert saved["resolver_targets"] == 1


def test_neural_policy_iteration_cfr_gate_accepts_fixed_state_source(tmp_path):
    candidate_path = tmp_path / "candidate.pt"
    state_source_path = tmp_path / "state_source.pt"
    for path, seed in [(candidate_path, 20260556), (state_source_path, 20260557)]:
        run_neural_policy_iteration_pilot(
            NeuralPolicyIterationConfig(
                self_play_hands=1,
                max_improvement_targets=0,
                train_steps=0,
                hidden_dim=16,
                teacher_mode="legal_mixed",
                checkpoint_path=str(path),
                seed=seed,
                device="cpu",
            )
        )

    metrics = evaluate_neural_policy_iteration_cfr_gate(
        NeuralPolicyIterationCFRGateConfig(
            checkpoint_path=str(candidate_path),
            state_source_checkpoint_path=str(state_source_path),
            n_roots=1,
            max_self_play_hands=64,
            max_steps_per_hand=96,
            cfr_iterations=1,
            cfr_backend="cpu",
            seed=20260558,
            device="cpu",
        )
    )

    assert metrics["checkpoint"] == str(candidate_path)
    assert metrics["state_source_checkpoint"] == str(state_source_path)
    assert metrics["resolver_targets"] == 1


def test_neural_policy_iteration_head_to_head_runs_duplicate_swapped_pairs(tmp_path):
    candidate_path = tmp_path / "candidate.pt"
    baseline_path = tmp_path / "baseline.pt"
    for path, seed in [(candidate_path, 20260561), (baseline_path, 20260562)]:
        run_neural_policy_iteration_pilot(
            NeuralPolicyIterationConfig(
                self_play_hands=1,
                max_improvement_targets=1,
                train_steps=1,
                hidden_dim=16,
                max_steps_per_hand=32,
                checkpoint_path=str(path),
                seed=seed,
                device="cpu",
            )
        )

    metrics = evaluate_neural_policy_iteration_head_to_head(
        str(candidate_path),
        str(baseline_path),
        n_games=2,
        device="cpu",
        seed=20260563,
    )

    assert metrics["algorithm"] == "neural_policy_iteration_h2h"
    assert metrics["role"] == "duplicate_swapped_neural_policy_iteration_league_smoke"
    assert metrics["candidate_checkpoint"] == str(candidate_path)
    assert metrics["baseline_checkpoint"] == str(baseline_path)
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_resolver"] is False
    assert metrics["n_games"] == 2
    assert metrics["n_pairs"] == 1
    assert np.isfinite(metrics["mean_candidate_payoff"])


def test_public_belief_cfr_selection_prioritizes_searchable_states():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    preflop = new_game(2, initial_chips=1000)
    turn = preflop
    for _ in range(12):
        if turn.betting_stage == "turn":
            break
        turn = turn.apply_action("call")
    preflop_mask = get_legal_mask(preflop)
    turn_mask = get_legal_mask(turn)
    preflop_sample = SelfPlayPolicySample(
        features=preflop.to_feature_vector(),
        legal_mask=preflop_mask,
        behavior_policy=preflop_mask / float(preflop_mask.sum()),
        action=1,
        player=int(preflop.player_i),
        value_target=0.0,
        state=preflop,
    )
    turn_sample = SelfPlayPolicySample(
        features=turn.to_feature_vector(),
        legal_mask=turn_mask,
        behavior_policy=turn_mask / float(turn_mask.sum()),
        action=1,
        player=int(turn.player_i),
        value_target=0.0,
        state=turn,
    )

    selected = _select_improvement_samples(
        [preflop_sample, turn_sample],
        max_targets=1,
        teacher_mode="public_belief_cfr",
    )

    assert len(selected) == 1
    assert selected[0] is turn_sample


def test_public_belief_cfr_selection_skips_ineligible_states_instead_of_fallback():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    preflop = new_game(2, initial_chips=1000)
    turn = preflop
    for _ in range(12):
        if turn.betting_stage == "turn":
            break
        turn = turn.apply_action("call")
    preflop_mask = get_legal_mask(preflop)
    turn_mask = get_legal_mask(turn)
    preflop_sample = SelfPlayPolicySample(
        features=preflop.to_feature_vector(),
        legal_mask=preflop_mask,
        behavior_policy=preflop_mask / float(preflop_mask.sum()),
        action=1,
        player=int(preflop.player_i),
        value_target=0.0,
        state=preflop,
    )
    turn_sample = SelfPlayPolicySample(
        features=turn.to_feature_vector(),
        legal_mask=turn_mask,
        behavior_policy=turn_mask / float(turn_mask.sum()),
        action=1,
        player=int(turn.player_i),
        value_target=0.0,
        state=turn,
    )

    selected = _select_improvement_samples(
        [preflop_sample, turn_sample],
        max_targets=2,
        teacher_mode="public_belief_cfr",
    )

    assert len(selected) == 1
    assert selected[0] is turn_sample


def test_teacher_batching_summary_groups_eligible_state_shapes():
    from poker_ai.games.full_deck.state import new_game
    from poker_ai.research.native_nfsp import get_legal_mask

    state = new_game(2, initial_chips=1000)
    for _ in range(12):
        if state.betting_stage == "turn":
            break
        state = state.apply_action("call")
    legal_mask = get_legal_mask(state)
    sample = SelfPlayPolicySample(
        features=state.to_feature_vector(),
        legal_mask=legal_mask,
        behavior_policy=legal_mask / float(legal_mask.sum()),
        action=1,
        player=int(state.player_i),
        value_target=0.0,
        state=state,
    )

    summary = _teacher_batching_summary([sample, sample])

    assert summary["eligible_teacher_states"] == 2
    assert summary["eligible_street_counts"]["turn"] == 2
    assert summary["legal_mask_pattern_groups"] == 1
    assert summary["largest_legal_mask_pattern_group"] == 2


def test_run_neural_policy_iteration_pilot_cli_writes_metrics(tmp_path):
    output_json = tmp_path / "metrics.json"
    checkpoint_path = tmp_path / "checkpoint.pt"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_neural_policy_iteration_pilot.py",
            "--self-play-hands",
            "2",
            "--max-improvement-targets",
            "8",
            "--train-steps",
            "2",
            "--hidden-dim",
            "16",
            "--batch-size",
            "4",
            "--policy-sample-weighting",
            "inverse_target_top",
            "--max-steps-per-hand",
            "32",
            "--device",
            "cpu",
            "--seed",
            "20260528",
            "--checkpoint-out",
            str(checkpoint_path),
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    saved = json.loads(output_json.read_text())
    assert metrics["algorithm"] == "neural_self_play_policy_iteration"
    assert saved["algorithm"] == metrics["algorithm"]
    assert saved["policy_sample_weighting"] == "inverse_target_top"
    assert metrics["uses_slumbot_training_data"] is False
    assert checkpoint_path.exists()


def test_neural_policy_iteration_pilot_initializes_from_parent_checkpoint(tmp_path):
    parent_path = tmp_path / "parent.pt"
    child_path = tmp_path / "child.pt"
    parent_metrics = run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=1,
            max_improvement_targets=0,
            train_steps=0,
            hidden_dim=16,
            max_steps_per_hand=8,
            checkpoint_path=str(parent_path),
            seed=20260573,
            device="cpu",
        )
    )

    child_metrics = run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=1,
            max_improvement_targets=0,
            train_steps=0,
            hidden_dim=16,
            max_steps_per_hand=8,
            initial_checkpoint_path=str(parent_path),
            checkpoint_path=str(child_path),
            seed=20260574,
            device="cpu",
        )
    )
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    child_payload = torch.load(child_path, map_location="cpu", weights_only=False)

    assert parent_metrics["initialized_from_checkpoint"] is False
    assert child_metrics["initialized_from_checkpoint"] is True
    assert child_metrics["initial_checkpoint_path"] == str(parent_path)
    assert child_payload["parent_checkpoint_path"] == str(parent_path)
    for name, tensor in parent_payload["policy_net_state_dict"].items():
        torch.testing.assert_close(tensor, child_payload["policy_net_state_dict"][name])
    for name, tensor in parent_payload["value_net_state_dict"].items():
        torch.testing.assert_close(tensor, child_payload["value_net_state_dict"][name])


def test_neural_policy_iteration_loop_cli_writes_generations_and_league(tmp_path):
    output_json = tmp_path / "loop_summary.json"
    output_dir = tmp_path / "loop"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_neural_policy_iteration_loop.py",
            "--n-generations",
            "2",
            "--output-dir",
            str(output_dir),
            "--prefix",
            "az_loop",
            "--self-play-hands",
            "1",
            "--max-improvement-targets",
            "1",
            "--train-steps",
            "1",
            "--hidden-dim",
            "16",
            "--batch-size",
            "1",
            "--max-steps-per-hand",
            "8",
            "--h2h-games",
            "2",
            "--device",
            "cpu",
            "--seed",
            "20260575",
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    saved = json.loads(output_json.read_text())
    assert metrics["algorithm"] == "neural_policy_iteration_generation_loop"
    assert metrics["workflow"] == "alphazero_style_repeatable_policy_iteration"
    assert metrics["uses_slumbot_training_data"] is False
    assert saved["algorithm"] == metrics["algorithm"]
    assert len(metrics["generations"]) == 2
    assert len(metrics["league_results"]) == 1
    assert metrics["generations"][0]["initialized_from_checkpoint"] is False
    assert metrics["generations"][1]["initialized_from_checkpoint"] is True
    assert metrics["generations"][1]["initial_checkpoint_path"] == metrics["generations"][0][
        "checkpoint_path"
    ]
    assert metrics["league_results"][0]["candidate_checkpoint"] == metrics["generations"][1][
        "checkpoint_path"
    ]
    assert metrics["league_results"][0]["baseline_checkpoint"] == metrics["generations"][0][
        "checkpoint_path"
    ]


def test_neural_policy_iteration_loop_defaults_to_positive_league_gate(
    tmp_path,
    monkeypatch,
):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_neural_policy_iteration_loop.py"
    spec = importlib.util.spec_from_file_location("run_neural_policy_iteration_loop", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_pilot(cfg):
        return {
            "algorithm": "neural_self_play_policy_iteration",
            "checkpoint_path": cfg.checkpoint_path,
            "initial_checkpoint_path": cfg.initial_checkpoint_path,
            "initialized_from_checkpoint": cfg.initial_checkpoint_path is not None,
            "passed": True,
        }

    def fake_h2h(*_args, **kwargs):
        assert kwargs["min_lower95_candidate_payoff"] == 0.0
        return {
            "algorithm": "neural_policy_iteration_h2h",
            "lower95_candidate_payoff": -0.01,
            "passed": False,
        }

    monkeypatch.setattr(module, "run_neural_policy_iteration_pilot", fake_pilot)
    monkeypatch.setattr(module, "evaluate_neural_policy_iteration_head_to_head", fake_h2h)
    args = module._parser().parse_args(
        [
            "--n-generations",
            "2",
            "--output-dir",
            str(tmp_path / "loop"),
            "--h2h-games",
            "2",
        ]
    )

    metrics = module.run_loop(args)

    assert metrics["passed"] is False
    assert len(metrics["league_results"]) == 1


def test_neural_policy_iteration_loop_compares_each_generation_to_controls(
    tmp_path,
    monkeypatch,
):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_neural_policy_iteration_loop.py"
    spec = importlib.util.spec_from_file_location("run_neural_policy_iteration_loop", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    control = tmp_path / "control.pt"
    control.write_bytes(b"control")
    calls = []

    def fake_pilot(cfg):
        return {
            "algorithm": "neural_self_play_policy_iteration",
            "checkpoint_path": cfg.checkpoint_path,
            "initial_checkpoint_path": cfg.initial_checkpoint_path,
            "initialized_from_checkpoint": cfg.initial_checkpoint_path is not None,
            "passed": True,
        }

    def fake_h2h(candidate, baseline, **kwargs):
        calls.append((candidate, baseline, kwargs))
        return {
            "algorithm": "neural_policy_iteration_h2h",
            "candidate_checkpoint": candidate,
            "baseline_checkpoint": baseline,
            "lower95_candidate_payoff": 0.01,
            "passed": True,
        }

    monkeypatch.setattr(module, "run_neural_policy_iteration_pilot", fake_pilot)
    monkeypatch.setattr(module, "evaluate_neural_policy_iteration_head_to_head", fake_h2h)
    args = module._parser().parse_args(
        [
            "--n-generations",
            "2",
            "--output-dir",
            str(tmp_path / "loop"),
            "--h2h-games",
            "2",
            "--control-checkpoint",
            str(control),
        ]
    )

    metrics = module.run_loop(args)

    assert metrics["passed"] is True
    assert metrics["control_checkpoints"] == [str(control)]
    assert len(metrics["league_results"]) == 1
    assert len(metrics["control_results"]) == 2
    assert calls[0][1] == str(control)
    assert calls[1][1] == str(control)
    assert calls[2][1] == metrics["generations"][0]["checkpoint_path"]


def test_neural_policy_iteration_loop_compares_each_generation_to_mixed_controls(
    tmp_path,
    monkeypatch,
):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_neural_policy_iteration_loop.py"
    spec = importlib.util.spec_from_file_location("run_neural_policy_iteration_loop", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    control = tmp_path / "rainbow.pt"
    control.write_bytes(b"rainbow")
    calls = []

    def fake_pilot(cfg):
        return {
            "algorithm": "neural_self_play_policy_iteration",
            "checkpoint_path": cfg.checkpoint_path,
            "initial_checkpoint_path": cfg.initial_checkpoint_path,
            "initialized_from_checkpoint": cfg.initial_checkpoint_path is not None,
            "passed": True,
        }

    def fake_mixed_h2h(**kwargs):
        calls.append(kwargs)
        return {
            "algorithm": "mixed_native_policy_h2h",
            "candidate_checkpoint": kwargs["candidate_checkpoint"],
            "candidate_kind": kwargs["candidate_kind"],
            "baseline_checkpoint": kwargs["baseline_checkpoint"],
            "baseline_kind": kwargs["baseline_kind"],
            "lower95_candidate_payoff": 0.01,
            "passed": True,
        }

    monkeypatch.setattr(module, "run_neural_policy_iteration_pilot", fake_pilot)
    monkeypatch.setattr(
        module,
        "evaluate_neural_policy_iteration_head_to_head",
        lambda candidate, baseline, **_kwargs: {
            "algorithm": "neural_policy_iteration_h2h",
            "candidate_checkpoint": candidate,
            "baseline_checkpoint": baseline,
            "lower95_candidate_payoff": 0.01,
            "passed": True,
        },
    )
    monkeypatch.setattr(module, "evaluate_mixed_policy_head_to_head", fake_mixed_h2h, raising=False)
    args = module._parser().parse_args(
        [
            "--n-generations",
            "2",
            "--output-dir",
            str(tmp_path / "loop"),
            "--h2h-games",
            "2",
            "--mixed-control-checkpoint",
            f"tianshou-rainbow:{control}",
        ]
    )

    metrics = module.run_loop(args)

    assert metrics["passed"] is True
    assert metrics["mixed_control_checkpoints"] == [
        {"kind": "tianshou-rainbow", "checkpoint": str(control)}
    ]
    assert len(metrics["control_results"]) == 2
    assert calls[0]["candidate_kind"] == "npi"
    assert calls[0]["baseline_kind"] == "tianshou-rainbow"
    assert calls[0]["baseline_checkpoint"] == str(control)


def test_neural_policy_iteration_loop_can_require_control_results(
    tmp_path,
    monkeypatch,
):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_neural_policy_iteration_loop.py"
    spec = importlib.util.spec_from_file_location("run_neural_policy_iteration_loop", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_pilot(cfg):
        return {
            "algorithm": "neural_self_play_policy_iteration",
            "checkpoint_path": cfg.checkpoint_path,
            "initial_checkpoint_path": cfg.initial_checkpoint_path,
            "initialized_from_checkpoint": cfg.initial_checkpoint_path is not None,
            "passed": True,
        }

    monkeypatch.setattr(module, "run_neural_policy_iteration_pilot", fake_pilot)
    args = module._parser().parse_args(
        [
            "--n-generations",
            "2",
            "--output-dir",
            str(tmp_path / "loop"),
            "--h2h-games",
            "0",
            "--require-control-gate",
        ]
    )

    metrics = module.run_loop(args)

    assert metrics["passed"] is False
    assert metrics["control_gate_required"] is True
    assert metrics["required_control_results"] == 2
    assert metrics["actual_control_results"] == 0
    assert "missing_required_control_results" in metrics["gate_failures"]


def test_neural_policy_iteration_pilot_fails_on_target_top_action_collapse():
    metrics = run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=4,
            max_improvement_targets=4,
            train_steps=1,
            hidden_dim=16,
            batch_size=2,
            max_steps_per_hand=8,
            preferred_action=1,
            preferred_action_prob=0.9,
            max_target_top_action_fraction=0.5,
            seed=20260580,
            device="cpu",
        )
    )

    assert metrics["max_target_top_action_fraction"] > 0.5
    assert metrics["passed"] is False
    assert "max_target_top_action_fraction" in " ".join(metrics["gate_failures"])
