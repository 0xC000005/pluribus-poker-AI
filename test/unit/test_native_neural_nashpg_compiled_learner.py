import json

import numpy as np
import torch


def test_sampled_counterfactual_decision_weights_remove_prior_own_reach():
    from scripts.run_native_neural_nashpg_compiled_learner import (
        _sampled_counterfactual_decision_weights,
    )

    batch = {
        "game_indices": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "players": np.asarray([0, 1, 0, 0], dtype=np.int64),
        "step_indices": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "old_log_probs": np.log(np.asarray([0.5, 0.25, 0.25, 0.5], dtype=np.float32)),
    }

    weights = _sampled_counterfactual_decision_weights(
        batch,
        mode="inverse-own-reach",
        max_decision_weight=10.0,
    )

    assert torch.allclose(
        weights,
        torch.tensor([1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 8.0 / 3.0]),
        atol=1e-6,
    )


def test_native_neural_nashpg_compiled_learner_writes_native_checkpoint(tmp_path):
    from poker_ai.research.mixed_policy_h2h import load_policy_adapter
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    checkpoint = tmp_path / "native_nashpg.pt"
    output = tmp_path / "native_nashpg.json"
    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        reference_update_every=1,
        seed=20260661,
        device="cpu",
        checkpoint_out=checkpoint,
        output_json=output,
    )

    assert checkpoint.exists()
    assert output.exists()
    saved_metrics = json.loads(output.read_text(encoding="utf-8"))
    assert saved_metrics["checkpoint_path"] == str(checkpoint)
    assert metrics["algorithm"] == "native_neural_nashpg_compiled"
    assert metrics["gate"] == "native_neural_nashpg_compiled_smoke"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["fresh_initial_policy"] is True
    assert metrics["moving_reference"] is True
    assert metrics["reference_updates"] == 1
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_solver_labels"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert metrics["rlcard_candidate"] is False
    assert metrics["promotion"] is False
    assert metrics["passed"] is True
    assert metrics["n_samples"] > 0
    assert metrics["compiled_needs_python_showdown"] == 0

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "native_neural_nashpg_compiled"
    assert payload["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert payload["num_actions"] == 9
    assert payload["num_features"] == metrics["num_features"]
    assert payload["trained_environment_native"] is True
    assert payload["native_action_projection"] is False
    assert payload["rlcard_candidate"] is False
    assert payload["uses_solver_labels"] is False

    adapter = load_policy_adapter(str(checkpoint), kind="native-ppo", device=torch.device("cpu"))
    features = np.zeros((payload["num_features"],), dtype=np.float32)
    legal_mask = np.zeros((payload["num_actions"],), dtype=np.float32)
    legal_mask[[0, 1, 3]] = 1.0
    probs = adapter.probs(features, legal_mask, torch.device("cpu"))
    assert probs.shape == (payload["num_actions"],)
    assert np.isclose(float(probs.sum()), 1.0)
    assert np.all(probs[legal_mask == 0] == 0.0)


def test_compiled_rollout_opponent_loader_supports_policy_router(tmp_path):
    from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
    from poker_ai.research.native_nfsp import _MLP
    from poker_ai.research.policy_router import PolicyRouterNet
    from scripts.run_local_vtrace_compiled_native_learner import (
        _load_compiled_rollout_opponents,
    )

    member = tmp_path / "member.pt"
    q_net = _MLP(hidden_dim=16)
    avg_net = _MLP(hidden_dim=16)
    torch.save(
        {
            "algorithm": "native_nfsp_dqn",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_actions": N_ACTIONS,
            "num_features": N_FEATURES,
            "hidden_dim": 16,
            "q_net_state_dict": q_net.state_dict(),
            "avg_net_state_dict": avg_net.state_dict(),
            "config": {"hidden_dim": 16},
        },
        member,
    )
    router = PolicyRouterNet(hidden_dim=8, n_policies=1)
    checkpoint = tmp_path / "router.pt"
    torch.save(
        {
            "algorithm": "policy_population_router",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_features": N_FEATURES,
            "num_policies": 1,
            "hidden_dim": 8,
            "member_policy_kinds": ["native-nfsp"],
            "member_checkpoints": [str(member)],
            "router_state_dict": router.state_dict(),
        },
        checkpoint,
    )

    opponents, kinds = _load_compiled_rollout_opponents([checkpoint], torch.device("cpu"))
    scores = opponents[0](torch.zeros(2, N_FEATURES, dtype=torch.float32))

    assert kinds == ["policy-router"]
    assert len(opponents) == 1
    assert scores.shape == (2, N_ACTIONS)


def test_native_neural_nashpg_compiled_learner_supports_parent_reference(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        reference_update_every=1,
        seed=20260662,
        device="cpu",
        checkpoint_out=parent,
    )

    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        reference_policy_checkpoint=parent,
        reference_kl_weight=0.05,
        seed=20260663,
        device="cpu",
        checkpoint_out=child,
    )

    assert child.exists()
    assert metrics["fresh_initial_policy"] is True
    assert metrics["moving_reference"] is False
    assert metrics["reference_policy_checkpoint"] == str(parent)
    assert metrics["reference_policy_kind"] == "native-ppo"
    assert metrics["reference_kl_weight"] == 0.05
    assert metrics["passed"] is True
    payload = torch.load(child, map_location="cpu", weights_only=False)
    assert payload["metrics"]["reference_policy_checkpoint"] == str(parent)
    assert payload["config"]["reference_policy_kind"] == "native-ppo"


def test_native_neural_nashpg_compiled_learner_supports_checkpoint_continuation(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        seed=20260664,
        device="cpu",
        checkpoint_out=parent,
    )

    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        checkpoint_in=parent,
        reference_update_every=1,
        seed=20260665,
        device="cpu",
        checkpoint_out=child,
    )

    assert child.exists()
    assert metrics["fresh_initial_policy"] is False
    assert metrics["checkpoint_in"] == str(parent)
    assert metrics["checkpoint_in_algorithm"] == "native_neural_nashpg_compiled"
    assert metrics["passed"] is True
    payload = torch.load(child, map_location="cpu", weights_only=False)
    assert payload["metrics"]["checkpoint_in"] == str(parent)


def test_native_neural_nashpg_compiled_learner_supports_gae_targets(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    checkpoint = tmp_path / "gae.pt"
    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        advantage_target="gae",
        gamma=1.0,
        gae_lambda=0.75,
        seed=20260679,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    assert checkpoint.exists()
    assert metrics["advantage_target"] == "gae"
    assert metrics["gamma"] == 1.0
    assert metrics["gae_lambda"] == 0.75
    assert metrics["passed"] is True
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["advantage_target"] == "gae"
    assert payload["metrics"]["advantage_target"] == "gae"


def test_player_perspective_gae_flips_values_across_alternating_turns():
    from scripts.run_native_neural_nashpg_compiled_learner import (
        _player_perspective_bootstrap_targets_and_advantages,
    )

    values = torch.tensor([0.2, 0.3, -0.1], dtype=torch.float32)
    terminal_returns = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)
    batch = {
        "game_indices": np.asarray([0, 0, 0], dtype=np.int64),
        "players": np.asarray([0, 1, 0], dtype=np.int64),
        "step_indices": np.asarray([0, 1, 2], dtype=np.int64),
    }

    targets, advantages = _player_perspective_bootstrap_targets_and_advantages(
        values=values,
        terminal_returns=terminal_returns,
        batch=batch,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(targets, terminal_returns, atol=1e-6)
    assert torch.allclose(advantages, torch.tensor([0.8, -1.3, 1.1]), atol=1e-6)


def test_native_neural_nashpg_compiled_learner_supports_player_gae_targets(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    checkpoint = tmp_path / "player_gae.pt"
    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        inner_update="ppo",
        ppo_epochs=1,
        ppo_minibatches=1,
        advantage_target="player-gae",
        gamma=1.0,
        gae_lambda=0.95,
        seed=20260767,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    assert checkpoint.exists()
    assert metrics["advantage_target"] == "player-gae"
    assert metrics["passed"] is True
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["advantage_target"] == "player-gae"


def test_native_neural_nashpg_compiled_learner_supports_ppo_inner_update(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    checkpoint = tmp_path / "ppo_inner.pt"
    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=8,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        lr=0.00025,
        entropy_weight=0.05,
        inner_update="ppo",
        ppo_epochs=2,
        ppo_minibatches=2,
        clip_coef=0.1,
        advantage_target="gae",
        gamma=0.99,
        gae_lambda=0.95,
        seed=20260680,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    assert checkpoint.exists()
    assert metrics["inner_update"] == "ppo"
    assert metrics["ppo_epochs"] == 2
    assert metrics["ppo_minibatches"] == 2
    assert metrics["clip_coef"] == 0.1
    assert metrics["advantage_target"] == "gae"
    assert metrics["passed"] is True
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["inner_update"] == "ppo"
    assert payload["metrics"]["inner_update"] == "ppo"


def test_native_neural_nashpg_compiled_learner_supports_inverse_own_reach_weights(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    checkpoint = tmp_path / "inverse_reach.pt"
    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=8,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        decision_weight_mode="inverse-own-reach",
        max_decision_weight=10.0,
        seed=20260695,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    assert checkpoint.exists()
    assert metrics["decision_weight_mode"] == "inverse-own-reach"
    assert metrics["max_decision_weight"] == 10.0
    assert metrics["mean_decision_weight"] > 0.0
    assert metrics["max_observed_decision_weight"] >= metrics["mean_decision_weight"]
    assert metrics["passed"] is True
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["decision_weight_mode"] == "inverse-own-reach"


def test_compiled_q_expected_lambda_targets_use_same_player_next_decision():
    from scripts.run_native_neural_nashpg_compiled_learner import (
        _q_expected_lambda_targets,
    )

    batch = {
        "features": np.zeros((3, 2), dtype=np.float32),
        "legal_masks": np.ones((3, 2), dtype=np.float32),
        "rewards": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        "next_decision_indices": np.asarray([1, -1, -1], dtype=np.int64),
    }
    policy_logits = torch.zeros((3, 2), dtype=torch.float32)
    q_values = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=torch.float32,
    )

    targets = _q_expected_lambda_targets(
        batch=batch,
        policy_logits=policy_logits,
        q_values=q_values,
        gamma=1.0,
        trace_lambda=0.25,
    )

    assert torch.allclose(targets, torch.tensor([2.5, 1.0, 2.0]))


def test_native_neural_nashpg_compiled_learner_supports_q_expected_neurd(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    checkpoint = tmp_path / "q_expected_neurd.pt"
    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=8,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        inner_update="ppo",
        ppo_epochs=1,
        ppo_minibatches=1,
        advantage_target="q_expected_lambda",
        actor_update_mode="neurd",
        gamma=1.0,
        gae_lambda=0.5,
        seed=20260761,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    assert checkpoint.exists()
    assert metrics["advantage_target"] == "q_expected_lambda"
    assert metrics["actor_update_mode"] == "neurd"
    assert metrics["uses_q_critic"] is True
    assert metrics["passed"] is True
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["advantage_target"] == "q_expected_lambda"
    assert payload["config"]["actor_update_mode"] == "neurd"
    assert "q_net_state_dict" in payload


def test_native_neural_nashpg_compiled_learner_resumes_q_expected_critic(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    parent = tmp_path / "q_parent.pt"
    child = tmp_path / "q_child.pt"
    run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        inner_update="ppo",
        ppo_epochs=1,
        ppo_minibatches=1,
        advantage_target="q_expected_mc",
        actor_update_mode="neurd",
        seed=20260765,
        device="cpu",
        checkpoint_out=parent,
    )

    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        inner_update="ppo",
        ppo_epochs=1,
        ppo_minibatches=1,
        advantage_target="q_expected_mc",
        actor_update_mode="neurd",
        checkpoint_in=parent,
        seed=20260766,
        device="cpu",
        checkpoint_out=child,
    )

    assert metrics["checkpoint_in_q_loaded"] is True
    assert metrics["uses_q_critic"] is True
    assert child.exists()
    payload = torch.load(child, map_location="cpu", weights_only=False)
    assert payload["metrics"]["checkpoint_in_q_loaded"] is True


def test_compiled_rollout_opponent_recognizes_online_rainbow_response_payload():
    from scripts.run_local_vtrace_compiled_native_learner import _is_rainbow_payload

    assert _is_rainbow_payload({"algorithm": "compiled_tianshou_rainbow_response_oracle"})


def test_compiled_rollout_opponent_loads_online_rainbow_response_payload(tmp_path):
    from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
    from scripts.run_local_vtrace_compiled_native_learner import (
        _RainbowDistributionNet,
        _load_compiled_rollout_opponents,
    )

    checkpoint = tmp_path / "compiled_rainbow_response.pt"
    model = _RainbowDistributionNet(
        hidden_dim=8,
        num_atoms=3,
        device=torch.device("cpu"),
    )
    torch.save(
        {
            "algorithm": "compiled_tianshou_rainbow_response_oracle",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_actions": N_ACTIONS,
            "num_features": N_FEATURES,
            "hidden_dim": 8,
            "num_atoms": 3,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    opponents, kinds = _load_compiled_rollout_opponents(
        [checkpoint],
        torch.device("cpu"),
    )

    assert kinds == ["tianshou-rainbow"]
    q_values = opponents[0](torch.zeros((1, N_FEATURES), dtype=torch.float32))
    assert q_values.shape == (1, N_ACTIONS)


def test_compiled_rollout_opponent_reports_native_nfsp_payload_kind(tmp_path):
    from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
    from poker_ai.research.native_ppo_policy import _PolicyMLP
    from scripts.run_local_vtrace_compiled_native_learner import (
        _load_compiled_rollout_opponents,
    )

    checkpoint = tmp_path / "native_nfsp.pt"
    avg_net = _PolicyMLP(8, input_dim=N_FEATURES)
    torch.save(
        {
            "algorithm": "native_nfsp_dqn",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_actions": N_ACTIONS,
            "num_features": N_FEATURES,
            "hidden_dim": 8,
            "config": {"feature_mode": "flat", "fsp_average_policy": True},
            "avg_net_state_dict": avg_net.state_dict(),
        },
        checkpoint,
    )

    opponents, kinds = _load_compiled_rollout_opponents(
        [checkpoint],
        torch.device("cpu"),
    )

    assert kinds == ["native-nfsp"]
    logits = opponents[0](torch.zeros((1, N_FEATURES), dtype=torch.float32))
    assert logits.shape == (1, N_ACTIONS)
