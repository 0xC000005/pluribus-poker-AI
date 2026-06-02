import json

import numpy as np
import torch


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
