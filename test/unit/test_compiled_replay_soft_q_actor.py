import json

import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES


def _fake_dataset(n: int = 32) -> dict:
    masks = np.zeros((n, N_ACTIONS), dtype=np.bool_)
    masks[:, [0, 1, 3]] = True
    actions = np.arange(n, dtype=np.int64) % 3
    action_map = np.array([0, 1, 3], dtype=np.int64)
    actions = action_map[actions]
    returns = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    return {
        "algorithm": "compiled_joint_experience_meta_policy_dataset",
        "backend": "compiled-fast-state",
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "n_hands": 8,
        "n_transitions": n,
        "initial_chips": 20000,
        "max_steps_per_hand": 256,
        "exploration_epsilon": 0.1,
        "needs_python_showdown": 0,
        "policy_kinds": ["tianshou-rainbow"],
        "policy_checkpoints": ["support.pt"],
        "meta_strategy": np.array([1.0], dtype=np.float32),
        "observations": np.zeros((n, N_FEATURES), dtype=np.float32),
        "legal_masks": masks,
        "actions": actions,
        "returns": returns,
    }


def test_train_soft_q_actor_from_compiled_replay_writes_native_checkpoint(tmp_path):
    from poker_ai.research.compiled_replay_soft_q_actor import (
        train_soft_q_actor_from_compiled_replay,
    )
    from poker_ai.research.mixed_policy_h2h import load_policy_adapter

    checkpoint = tmp_path / "soft_q_actor.pt"
    output = tmp_path / "soft_q_actor.json"
    metrics = train_soft_q_actor_from_compiled_replay(
        _fake_dataset(),
        hidden_dim=16,
        q_train_steps=2,
        policy_train_steps=2,
        batch_size=8,
        seed=20260844,
        device="cpu",
        checkpoint_out=checkpoint,
        output_json=output,
    )

    assert checkpoint.exists()
    assert output.exists()
    saved_metrics = json.loads(output.read_text(encoding="utf-8"))
    assert saved_metrics["checkpoint_path"] == str(checkpoint)
    assert metrics["algorithm"] == "compiled_replay_soft_q_actor"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["dataset_backend"] == "compiled-fast-state"
    assert metrics["dataset_initial_chips"] == 20000
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_solver_labels"] is False
    assert metrics["promotion"] is False
    assert metrics["passed"] is True

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "compiled_replay_soft_q_actor"
    assert payload["num_actions"] == N_ACTIONS
    assert payload["num_features"] == N_FEATURES
    assert payload["config"]["feature_mode"] == "flat"
    assert payload["config"]["initial_chips"] == 20000
    assert payload["trained_environment_native"] is True
    assert payload["uses_slumbot_training_data"] is False

    adapter = load_policy_adapter(str(checkpoint), kind="native-ppo", device=torch.device("cpu"))
    legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    legal_mask[[0, 1, 3]] = 1.0
    probs = adapter.probs(np.zeros(N_FEATURES, dtype=np.float32), legal_mask, torch.device("cpu"))
    assert probs.shape == (N_ACTIONS,)
    assert np.isclose(float(probs.sum()), 1.0)
    assert np.all(probs[legal_mask == 0] == 0.0)


def test_train_soft_q_actor_rejects_illegal_dataset_action():
    from poker_ai.research.compiled_replay_soft_q_actor import (
        train_soft_q_actor_from_compiled_replay,
    )

    dataset = _fake_dataset()
    dataset["legal_masks"][0, dataset["actions"][0]] = False

    try:
        train_soft_q_actor_from_compiled_replay(
            dataset,
            hidden_dim=16,
            q_train_steps=1,
            policy_train_steps=1,
            batch_size=4,
            device="cpu",
        )
    except ValueError as exc:
        assert "illegal recorded action" in str(exc)
    else:
        raise AssertionError("expected illegal action validation failure")


def test_compiled_replay_soft_q_actor_cli_forwards_options(monkeypatch, tmp_path):
    from scripts import run_compiled_replay_soft_q_actor as cli

    checkpoint = tmp_path / "candidate.pt"
    output = tmp_path / "candidate.json"
    calls = []

    def _fake_run(**kwargs):
        calls.append(kwargs)
        checkpoint.write_bytes(b"checkpoint")
        output.write_text(json.dumps({"passed": True}) + "\n", encoding="utf-8")
        return {"algorithm": "compiled_replay_soft_q_actor", "passed": True}

    monkeypatch.setattr(cli, "run_compiled_replay_soft_q_actor", _fake_run)
    metrics = cli.main(
        [
            "--policy",
            "tianshou-rainbow:support.pt",
            "--meta-strategy",
            "1.0",
            "--n-hands",
            "16",
            "--q-train-steps",
            "3",
            "--policy-train-steps",
            "2",
            "--checkpoint-out",
            str(checkpoint),
            "--output-json",
            str(output),
        ]
    )

    assert metrics["passed"] is True
    assert calls[0]["policy_specs"] == [("tianshou-rainbow", "support.pt")]
    assert calls[0]["meta_strategy"] == [1.0]
    assert calls[0]["n_hands"] == 16
    assert calls[0]["q_train_steps"] == 3
    assert calls[0]["policy_train_steps"] == 2
    assert calls[0]["checkpoint_out"] == checkpoint
    assert calls[0]["output_json"] == output
