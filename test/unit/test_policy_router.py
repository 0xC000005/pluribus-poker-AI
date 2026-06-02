import json

import numpy as np
import torch

from poker_ai.games.full_deck.state import N_FEATURES
from poker_ai.research.policy_router import (
    load_policy_router_checkpoint,
    router_policy_scores,
    train_policy_router_from_joint_experience,
)


def test_train_policy_router_learns_contextual_policy_values(tmp_path):
    observations = []
    behavior_policy_indices = []
    returns = []
    for _ in range(32):
        context_a = np.zeros(N_FEATURES, dtype=np.float32)
        context_a[0] = 1.0
        context_b = np.zeros(N_FEATURES, dtype=np.float32)
        context_b[1] = 1.0
        observations.extend([context_a, context_a, context_b, context_b])
        behavior_policy_indices.extend([0, 1, 0, 1])
        returns.extend([1.0, -1.0, -1.0, 1.0])

    dataset = tmp_path / "router_dataset.npz"
    np.savez(
        dataset,
        observations=np.stack(observations).astype(np.float32),
        behavior_policy_indices=np.asarray(behavior_policy_indices, dtype=np.int64),
        returns=np.asarray(returns, dtype=np.float32),
    )
    checkpoint = tmp_path / "router.pt"
    output_json = tmp_path / "router.json"

    metrics = train_policy_router_from_joint_experience(
        dataset,
        member_specs=[("native-ppo", "a.pt"), ("native-ppo", "b.pt")],
        hidden_dim=32,
        train_steps=300,
        batch_size=64,
        lr=0.01,
        seed=123,
        device="cpu",
        checkpoint_out=checkpoint,
        output_json=output_json,
    )

    assert metrics["algorithm"] == "policy_population_router"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["uses_solver_labels"] is False
    assert metrics["last_loss"] < metrics["first_loss"]
    assert checkpoint.exists()
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed"] is True

    payload, router = load_policy_router_checkpoint(checkpoint, device=torch.device("cpu"))
    assert payload["member_policy_kinds"] == ["native-ppo", "native-ppo"]

    context_a = np.zeros(N_FEATURES, dtype=np.float32)
    context_a[0] = 1.0
    context_b = np.zeros(N_FEATURES, dtype=np.float32)
    context_b[1] = 1.0

    assert int(np.argmax(router_policy_scores(router, context_a, torch.device("cpu")))) == 0
    assert int(np.argmax(router_policy_scores(router, context_b, torch.device("cpu")))) == 1


def test_train_policy_router_rejects_member_count_mismatch(tmp_path):
    dataset = tmp_path / "router_dataset.npz"
    np.savez(
        dataset,
        observations=np.zeros((2, N_FEATURES), dtype=np.float32),
        behavior_policy_indices=np.asarray([0, 1], dtype=np.int64),
        returns=np.asarray([0.0, 0.0], dtype=np.float32),
    )

    try:
        train_policy_router_from_joint_experience(
            dataset,
            member_specs=[("native-ppo", "only.pt")],
            train_steps=1,
            batch_size=1,
            device="cpu",
        )
    except ValueError as exc:
        assert "behavior_policy_indices exceed supplied member policies" in str(exc)
    else:
        raise AssertionError("expected member count mismatch to fail")
