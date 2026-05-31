import json

import numpy as np
import torch


def test_local_vtrace_compiled_native_learner_writes_native_ppo_checkpoint(tmp_path):
    from poker_ai.research.mixed_policy_h2h import load_policy_adapter
    from scripts.run_local_vtrace_compiled_native_learner import run_learner

    checkpoint = tmp_path / "local_vtrace_native.pt"
    output = tmp_path / "local_vtrace_native.json"
    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        seed=20260559,
        device="cpu",
        checkpoint_out=checkpoint,
        output_json=output,
    )

    assert checkpoint.exists()
    assert output.exists()
    saved_metrics = json.loads(output.read_text(encoding="utf-8"))
    assert saved_metrics["checkpoint_path"] == str(checkpoint)
    assert metrics["algorithm"] == "local_vtrace_compiled_native"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["collector_backend"] == "compiled-fast-state"
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["uses_slumbot_data"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert metrics["rlcard_candidate"] is False
    assert metrics["promotion"] is False
    assert metrics["passed"] is True
    assert metrics["n_samples"] > 0
    assert metrics["compiled_needs_python_showdown"] == 0

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "local_vtrace_compiled_native"
    assert payload["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert payload["num_actions"] == 9
    assert payload["trained_environment_native"] is True
    assert payload["native_action_projection"] is False
    assert payload["rlcard_candidate"] is False

    adapter = load_policy_adapter(str(checkpoint), kind="native-ppo", device=torch.device("cpu"))
    features = np.zeros((payload["num_features"],), dtype=np.float32)
    legal_mask = np.zeros((payload["num_actions"],), dtype=np.float32)
    legal_mask[[0, 1, 3]] = 1.0
    probs = adapter.probs(features, legal_mask, torch.device("cpu"))
    assert probs.shape == (payload["num_actions"],)
    assert np.isclose(float(probs.sum()), 1.0)
    assert np.all(probs[legal_mask == 0] == 0.0)


def test_local_vtrace_compiled_native_learner_checkpoint_h2h_self_smoke(tmp_path):
    from poker_ai.research.mixed_policy_h2h import evaluate_mixed_policy_head_to_head
    from scripts.run_local_vtrace_compiled_native_learner import run_learner

    checkpoint = tmp_path / "local_vtrace_native.pt"
    run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        seed=20260560,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    metrics = evaluate_mixed_policy_head_to_head(
        candidate_checkpoint=str(checkpoint),
        candidate_kind="native-ppo",
        baseline_checkpoint=str(checkpoint),
        baseline_kind="native-ppo",
        n_games=4,
        device="cpu",
        seed=20260560,
        eval_state_backend="fast-state-canonical-deal",
    )

    assert metrics["candidate_kind"] == "native-ppo"
    assert metrics["baseline_kind"] == "native-ppo"
    assert metrics["candidate_algorithm"] == "local_vtrace_compiled_native"
    assert metrics["baseline_algorithm"] == "local_vtrace_compiled_native"
    assert metrics["native_action_projection"] is False
    assert metrics["n_games"] == 4


def test_local_vtrace_compiled_native_learner_supports_frozen_parent_population(tmp_path):
    from scripts.run_local_vtrace_compiled_native_learner import run_learner

    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        seed=20260561,
        device="cpu",
        checkpoint_out=parent,
    )

    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        seed=20260562,
        device="cpu",
        opponent_checkpoints=[parent],
        checkpoint_out=child,
    )

    assert child.exists()
    assert metrics["population_training"] is True
    assert metrics["opponent_population_size"] == 1
    assert metrics["opponent_checkpoints"] == [str(parent)]
    assert metrics["opponent_kind"] == "native-ppo"
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["passed"] is True
    payload = torch.load(child, map_location="cpu", weights_only=False)
    assert payload["metrics"]["population_training"] is True
    assert payload["metrics"]["opponent_population_size"] == 1
