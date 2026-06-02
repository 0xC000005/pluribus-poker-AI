import json

import numpy as np
import torch


def test_rnad_compiled_native_learner_writes_native_policy_checkpoint(tmp_path):
    from poker_ai.research.mixed_policy_h2h import load_policy_adapter
    from scripts.run_rnad_compiled_native_learner import run_learner

    parent = tmp_path / "rnad_parent.pt"
    checkpoint = tmp_path / "rnad_native.pt"
    output = tmp_path / "rnad_native.json"
    metrics = run_learner(
        train_iterations=1,
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=32,
        hidden_dim=16,
        seed=20260602,
        device="cpu",
        checkpoint_out=checkpoint,
        parent_checkpoint_out=parent,
        output_json=output,
    )

    assert checkpoint.exists()
    assert parent.exists()
    assert output.exists()
    saved_metrics = json.loads(output.read_text(encoding="utf-8"))
    assert saved_metrics["checkpoint_path"] == str(checkpoint)
    assert saved_metrics["parent_checkpoint_path"] == str(parent)
    assert metrics["algorithm"] == "rnad_compiled_native"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["collector_backend"] == "compiled-fast-state"
    assert metrics["num_actions"] == 9
    assert metrics["num_features"] == 126
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
    assert payload["algorithm"] == "rnad_compiled_native"
    assert payload["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert payload["num_actions"] == 9
    assert payload["native_policy_kind"] == "native-ppo"
    assert payload["rnad_policy_export"] == "target"
    assert payload["trained_environment_native"] is True
    assert payload["native_action_projection"] is False
    assert payload["rlcard_candidate"] is False
    assert "rnad_net_state_dict" in payload
    assert "policy_net_state_dict" in payload

    adapter = load_policy_adapter(str(checkpoint), kind="native-ppo", device=torch.device("cpu"))
    features = np.zeros((payload["num_features"],), dtype=np.float32)
    legal_mask = np.zeros((payload["num_actions"],), dtype=np.float32)
    legal_mask[[0, 1, 3]] = 1.0
    probs = adapter.probs(features, legal_mask, torch.device("cpu"))
    assert probs.shape == (payload["num_actions"],)
    assert np.isclose(float(probs.sum()), 1.0)
    assert np.all(probs[legal_mask == 0] == 0.0)


def test_rnad_compiled_native_learner_can_export_learner_policy(tmp_path):
    from scripts.run_rnad_compiled_native_learner import run_learner

    target_checkpoint = tmp_path / "rnad_target.pt"
    learner_checkpoint = tmp_path / "rnad_learner.pt"
    metrics = run_learner(
        train_iterations=1,
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=32,
        hidden_dim=16,
        seed=20260606,
        device="cpu",
        checkpoint_out=target_checkpoint,
        learner_checkpoint_out=learner_checkpoint,
    )

    assert target_checkpoint.exists()
    assert learner_checkpoint.exists()
    assert metrics["checkpoint_path"] == str(target_checkpoint)
    assert metrics["learner_checkpoint_path"] == str(learner_checkpoint)
    assert metrics["rnad_policy_export"] == "target"
    assert metrics["learner_rnad_policy_export"] == "learner"

    target_payload = torch.load(target_checkpoint, map_location="cpu", weights_only=False)
    learner_payload = torch.load(learner_checkpoint, map_location="cpu", weights_only=False)
    assert target_payload["rnad_policy_export"] == "target"
    assert learner_payload["rnad_policy_export"] == "learner"
    assert learner_payload["metrics"]["checkpoint_role"] == "learner_export"


def test_rnad_compiled_native_learner_can_continue_from_prior_checkpoint(tmp_path):
    from scripts.run_rnad_compiled_native_learner import run_learner

    source = tmp_path / "rnad_source.pt"
    continued_parent = tmp_path / "rnad_continued_parent.pt"
    continued_child = tmp_path / "rnad_continued_child.pt"

    run_learner(
        train_iterations=1,
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=32,
        hidden_dim=16,
        seed=20260604,
        device="cpu",
        checkpoint_out=source,
    )
    metrics = run_learner(
        train_iterations=1,
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=32,
        hidden_dim=16,
        seed=20260605,
        device="cpu",
        checkpoint_in=source,
        parent_checkpoint_out=continued_parent,
        checkpoint_out=continued_child,
    )

    assert metrics["checkpoint_in"] == str(source)
    assert metrics["continued_from_checkpoint"] is True
    assert continued_parent.exists()
    assert continued_child.exists()
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    parent_payload = torch.load(continued_parent, map_location="cpu", weights_only=False)
    for key, tensor in source_payload["rnad_net_state_dict"].items():
        assert torch.equal(tensor, parent_payload["rnad_net_state_dict"][key])


def test_rnad_compiled_native_learner_accepts_population_opponent(tmp_path):
    from scripts.run_rnad_compiled_native_learner import run_learner

    opponent = tmp_path / "rnad_opponent.pt"
    child = tmp_path / "rnad_population_child.pt"

    run_learner(
        train_iterations=1,
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=32,
        hidden_dim=16,
        seed=20260607,
        device="cpu",
        checkpoint_out=opponent,
    )
    metrics = run_learner(
        train_iterations=1,
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=32,
        hidden_dim=16,
        seed=20260608,
        device="cpu",
        checkpoint_out=child,
        opponent_checkpoints=[opponent],
        opponent_kinds=["native-ppo"],
    )

    assert child.exists()
    assert metrics["passed"] is True
    assert metrics["train_opponent_mode"] == "population"
    assert metrics["population_opponent_size"] == 1
    assert metrics["population_opponent_kinds"] == ["native-ppo"]
    assert metrics["opponent_controlled_steps"] > 0
    assert metrics["learner_controlled_steps"] > 0
    assert metrics["n_samples"] == metrics["learner_controlled_steps"] + metrics["opponent_controlled_steps"]
    assert metrics["illegal_records"] == 0


def test_rnad_compiled_native_checkpoint_h2h_self_smoke(tmp_path):
    from poker_ai.research.mixed_policy_h2h import evaluate_mixed_policy_head_to_head
    from scripts.run_rnad_compiled_native_learner import run_learner

    checkpoint = tmp_path / "rnad_native.pt"
    run_learner(
        train_iterations=1,
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=32,
        hidden_dim=16,
        seed=20260603,
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
        seed=20260603,
        eval_state_backend="fast-state-canonical-deal",
    )

    assert metrics["candidate_kind"] == "native-ppo"
    assert metrics["baseline_kind"] == "native-ppo"
    assert metrics["candidate_algorithm"] == "rnad_compiled_native"
    assert metrics["baseline_algorithm"] == "rnad_compiled_native"
    assert metrics["native_action_projection"] is False
    assert metrics["n_games"] == 4
