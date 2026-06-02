import numpy as np
import torch


def test_target_policy_from_values_masks_illegal_and_prefers_best_action():
    from scripts.train_native_all_action_target_policy import target_policy_from_values

    values = np.array([0.0, 0.2, np.nan, -0.1, np.nan, np.nan, np.nan, np.nan, 0.5], dtype=np.float32)
    legal_mask = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1], dtype=np.float32)

    target = target_policy_from_values(values, legal_mask, temperature=0.1)

    assert target.shape == (9,)
    assert np.isclose(float(target.sum()), 1.0)
    assert np.all(target[legal_mask <= 0] == 0.0)
    assert int(np.argmax(target)) == 8


def test_native_all_action_target_policy_training_writes_checkpoint(tmp_path):
    from poker_ai.research.mixed_policy_h2h import load_policy_adapter
    from scripts.train_native_all_action_target_policy import run_training_gate

    checkpoint = tmp_path / "target_policy.pt"
    output = tmp_path / "target_policy.json"
    metrics = run_training_gate(
        n_train_states=4,
        n_eval_states=3,
        rollouts_per_action=2,
        max_steps_per_rollout=16,
        hidden_dim=16,
        n_steps=20,
        batch_size=4,
        seed=20260720,
        device="cpu",
        checkpoint_out=checkpoint,
        output_json=output,
    )

    assert checkpoint.exists()
    assert output.exists()
    assert metrics["algorithm"] == "native_all_action_target_policy"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_solver_labels"] is False
    assert np.isfinite(metrics["eval_cross_entropy"])
    assert np.isfinite(metrics["uniform_eval_cross_entropy"])

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "native_all_action_target_policy"
    assert payload["num_actions"] == 9
    adapter = load_policy_adapter(str(checkpoint), kind="native-ppo", device=torch.device("cpu"))
    probs = adapter.probs(
        np.zeros((payload["num_features"],), dtype=np.float32),
        np.array([1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        torch.device("cpu"),
    )
    assert np.isclose(float(probs.sum()), 1.0)
    assert probs[2] == 0.0


def test_native_all_action_target_policy_supports_world_averaged_margin_gate(tmp_path):
    from scripts.train_native_all_action_target_policy import run_training_gate

    checkpoint = tmp_path / "world_margin.pt"
    metrics = run_training_gate(
        n_train_states=4,
        n_eval_states=3,
        n_worlds=2,
        rollouts_per_action=1,
        max_steps_per_rollout=12,
        hidden_dim=16,
        n_steps=10,
        batch_size=4,
        target_temperature=0.5,
        min_target_margin=0.0,
        margin_weight_power=1.0,
        seed=20260740,
        device="cpu",
        checkpoint_out=checkpoint,
    )

    assert checkpoint.exists()
    assert metrics["n_worlds"] == 2
    assert metrics["world_averaged_targets"] is True
    assert metrics["min_target_margin"] == 0.0
    assert metrics["margin_weight_power"] == 1.0
    assert metrics["train_retained_states"] == 4
    assert metrics["eval_retained_states"] == 3
    assert metrics["mean_eval_target_margin"] >= 0.0

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["n_worlds"] == 2
    assert payload["config"]["min_target_margin"] == 0.0
