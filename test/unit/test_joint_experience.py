from pathlib import Path

import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.joint_experience import collect_joint_experience, save_joint_experience_npz
from poker_ai.research.mixed_policy_h2h import PolicyAdapter


def _first_legal(_features, legal_mask, _device):
    probs = np.zeros(N_ACTIONS, dtype=np.float32)
    probs[int(np.flatnonzero(legal_mask)[0])] = 1.0
    return probs


def _last_legal(_features, legal_mask, _device):
    probs = np.zeros(N_ACTIONS, dtype=np.float32)
    probs[int(np.flatnonzero(legal_mask)[-1])] = 1.0
    return probs


def test_collect_joint_experience_records_meta_policy_trajectory_contract():
    adapters = [
        PolicyAdapter(
            kind="fake-first",
            checkpoint_path="first.pt",
            algorithm="fake",
            action_probs_fn=_first_legal,
        ),
        PolicyAdapter(
            kind="fake-last",
            checkpoint_path="last.pt",
            algorithm="fake",
            action_probs_fn=_last_legal,
        ),
    ]

    dataset = collect_joint_experience(
        adapters,
        meta_strategy=[0.25, 0.75],
        n_hands=8,
        seed=20260527,
        initial_chips=100,
        max_steps_per_hand=16,
        device=torch.device("cpu"),
    )

    assert dataset["algorithm"] == "joint_experience_meta_policy_dataset"
    assert dataset["uses_slumbot_training_data"] is False
    assert dataset["promotion"] is False
    assert dataset["n_hands"] == 8
    assert dataset["n_transitions"] == len(dataset["actions"])
    assert dataset["observations"].shape == (dataset["n_transitions"], N_FEATURES)
    assert dataset["next_observations"].shape == (dataset["n_transitions"], N_FEATURES)
    assert dataset["legal_masks"].shape == (dataset["n_transitions"], N_ACTIONS)
    assert dataset["next_legal_masks"].shape == (dataset["n_transitions"], N_ACTIONS)
    assert dataset["dones"].shape == (dataset["n_transitions"],)
    assert dataset["returns"].shape == (dataset["n_transitions"],)
    assert set(np.unique(dataset["player_indices"])) <= {0, 1}
    assert set(np.unique(dataset["behavior_policy_indices"])) <= {0, 1}
    assert np.all(dataset["legal_masks"][np.arange(dataset["n_transitions"]), dataset["actions"]])
    assert np.all(np.isfinite(dataset["returns"]))
    assert np.isclose(dataset["meta_strategy"].sum(), 1.0)


def test_collect_compiled_joint_experience_records_next_state_contract():
    from poker_ai.research.compiled_joint_experience import (
        CompiledJointPolicy,
        collect_compiled_joint_experience,
    )

    class FixedScorePolicy(torch.nn.Module):
        def __init__(self, descending: bool = False):
            super().__init__()
            scores = torch.arange(N_ACTIONS, dtype=torch.float32)
            if descending:
                scores = torch.flip(scores, dims=[0])
            self.register_buffer("scores", scores)

        def forward(self, features):
            return self.scores.expand(features.shape[0], -1)

    policies = [
        CompiledJointPolicy(
            kind="fixed-low",
            checkpoint_path="low.pt",
            algorithm="fixed",
            module=FixedScorePolicy(descending=True),
        ),
        CompiledJointPolicy(
            kind="fixed-high",
            checkpoint_path="high.pt",
            algorithm="fixed",
            module=FixedScorePolicy(descending=False),
        ),
    ]

    dataset = collect_compiled_joint_experience(
        policies,
        meta_strategy=[0.5, 0.5],
        n_hands=16,
        batch_size=8,
        seed=20260618,
        initial_chips=100,
        max_steps_per_hand=16,
        device=torch.device("cpu"),
    )

    assert dataset["algorithm"] == "compiled_joint_experience_meta_policy_dataset"
    assert dataset["backend"] == "compiled-fast-state"
    assert dataset["uses_slumbot_training_data"] is False
    assert dataset["promotion"] is False
    assert dataset["n_hands"] == 16
    assert dataset["n_transitions"] == len(dataset["actions"])
    assert dataset["observations"].shape == (dataset["n_transitions"], N_FEATURES)
    assert dataset["next_observations"].shape == (dataset["n_transitions"], N_FEATURES)
    assert dataset["legal_masks"].shape == (dataset["n_transitions"], N_ACTIONS)
    assert dataset["next_legal_masks"].shape == (dataset["n_transitions"], N_ACTIONS)
    assert np.all(dataset["legal_masks"][np.arange(dataset["n_transitions"]), dataset["actions"]])
    assert np.all(np.isfinite(dataset["returns"]))
    assert np.isclose(dataset["meta_strategy"].sum(), 1.0)
    assert set(np.unique(dataset["behavior_policy_indices"])) <= {0, 1}
    assert dataset["exploration_epsilon"] == 0.0
    assert dataset["exploratory_actions"] == 0
    assert dataset["needs_python_showdown"] == 0


def test_collect_compiled_joint_experience_supports_legal_exploration():
    from poker_ai.research.compiled_joint_experience import (
        CompiledJointPolicy,
        collect_compiled_joint_experience,
    )

    class FixedScorePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scores", torch.arange(N_ACTIONS, dtype=torch.float32))

        def forward(self, features):
            return self.scores.expand(features.shape[0], -1)

    dataset = collect_compiled_joint_experience(
        [
            CompiledJointPolicy(
                kind="fixed",
                checkpoint_path="fixed.pt",
                algorithm="fixed",
                module=FixedScorePolicy(),
            )
        ],
        meta_strategy=[1.0],
        n_hands=16,
        batch_size=8,
        seed=20260686,
        initial_chips=100,
        max_steps_per_hand=16,
        exploration_epsilon=1.0,
        device=torch.device("cpu"),
    )

    assert dataset["exploration_epsilon"] == 1.0
    assert dataset["exploratory_actions"] == dataset["n_transitions"]
    assert dataset["exploratory_action_fraction"] == 1.0
    assert np.all(dataset["legal_masks"][np.arange(dataset["n_transitions"]), dataset["actions"]])


def test_load_compiled_joint_policy_supports_native_nfsp_checkpoint(tmp_path):
    from poker_ai.research.compiled_joint_experience import load_compiled_joint_policy
    from poker_ai.research.native_nfsp import _MLP

    checkpoint = tmp_path / "native_nfsp.pt"
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
        checkpoint,
    )

    policy = load_compiled_joint_policy(checkpoint, kind="native-nfsp", device="cpu")
    scores = policy.module(torch.zeros(3, N_FEATURES, dtype=torch.float32))

    assert policy.kind == "native-nfsp"
    assert policy.algorithm == "native_nfsp_dqn"
    assert scores.shape == (3, N_ACTIONS)


def test_load_compiled_joint_policy_supports_policy_router_checkpoint(tmp_path):
    from poker_ai.research.compiled_joint_experience import (
        collect_compiled_joint_experience,
        load_compiled_joint_policy,
    )
    from poker_ai.research.native_nfsp import _MLP
    from poker_ai.research.policy_router import PolicyRouterNet

    def write_member(name: str, best_action: int) -> Path:
        checkpoint = tmp_path / f"{name}.pt"
        q_net = _MLP(hidden_dim=16)
        avg_net = _MLP(hidden_dim=16)
        for parameter in q_net.parameters():
            parameter.data.zero_()
        for parameter in avg_net.parameters():
            parameter.data.zero_()
        avg_net.net[-1].bias.data[int(best_action)] = 10.0
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
            checkpoint,
        )
        return checkpoint

    first = write_member("first", 0)
    second = write_member("second", N_ACTIONS - 1)
    router = PolicyRouterNet(hidden_dim=8, n_policies=2)
    for parameter in router.parameters():
        parameter.data.zero_()
    router.net[-1].bias.data[:] = torch.tensor([0.0, 1.0])
    checkpoint = tmp_path / "router.pt"
    torch.save(
        {
            "algorithm": "policy_population_router",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_features": N_FEATURES,
            "num_policies": 2,
            "hidden_dim": 8,
            "member_policy_kinds": ["native-nfsp", "native-nfsp"],
            "member_checkpoints": [str(first), str(second)],
            "router_state_dict": router.state_dict(),
        },
        checkpoint,
    )

    policy = load_compiled_joint_policy(checkpoint, kind="policy-router", device="cpu")
    scores = policy.module(torch.zeros(3, N_FEATURES, dtype=torch.float32))
    dataset = collect_compiled_joint_experience(
        [policy],
        meta_strategy=[1.0],
        n_hands=4,
        batch_size=4,
        seed=20261020,
        initial_chips=100,
        max_steps_per_hand=8,
        device=torch.device("cpu"),
    )

    assert policy.kind == "policy-router"
    assert policy.algorithm == "policy_population_router"
    assert scores.shape == (3, N_ACTIONS)
    assert torch.all(scores[:, N_ACTIONS - 1] > scores[:, 0])
    assert dataset["n_transitions"] > 0
    assert dataset["policy_kinds"] == ["policy-router"]


def test_load_compiled_joint_policy_rejects_recursive_policy_router_member(tmp_path):
    from poker_ai.research.compiled_joint_experience import load_compiled_joint_policy
    from poker_ai.research.policy_router import PolicyRouterNet

    router = PolicyRouterNet(hidden_dim=8, n_policies=1)
    checkpoint = tmp_path / "router.pt"
    torch.save(
        {
            "algorithm": "policy_population_router",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_features": N_FEATURES,
            "num_policies": 1,
            "hidden_dim": 8,
            "member_policy_kinds": ["policy-router"],
            "member_checkpoints": [str(checkpoint)],
            "router_state_dict": router.state_dict(),
        },
        checkpoint,
    )

    import pytest

    with pytest.raises(ValueError, match="may not themselves be policy-router"):
        load_compiled_joint_policy(checkpoint, kind="policy-router", device="cpu")


def test_save_joint_experience_npz_writes_arrays_and_manifest(tmp_path):
    dataset = {
        "algorithm": "joint_experience_meta_policy_dataset",
        "role": "psro_joint_experience_response_oracle_data_contract",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "policy_kinds": ["fake"],
        "policy_checkpoints": ["fake.pt"],
        "meta_strategy": np.array([1.0], dtype=np.float32),
        "n_hands": 1,
        "n_transitions": 1,
        "truncated_hands": 0,
        "initial_chips": 100,
        "max_steps_per_hand": 16,
        "observations": np.zeros((1, N_FEATURES), dtype=np.float32),
        "next_observations": np.zeros((1, N_FEATURES), dtype=np.float32),
        "legal_masks": np.ones((1, N_ACTIONS), dtype=np.bool_),
        "next_legal_masks": np.ones((1, N_ACTIONS), dtype=np.bool_),
        "actions": np.array([0], dtype=np.int64),
        "dones": np.array([True], dtype=np.bool_),
        "player_indices": np.array([0], dtype=np.int64),
        "behavior_policy_indices": np.array([0], dtype=np.int64),
        "hand_indices": np.array([0], dtype=np.int64),
        "step_indices": np.array([0], dtype=np.int64),
        "returns": np.array([0.5], dtype=np.float32),
        "hand_policy_indices": np.array([[0, 0]], dtype=np.int64),
        "hand_returns": np.array([[0.5, -0.5]], dtype=np.float32),
        "uses_slumbot_training_data": False,
        "promotion": False,
    }
    output = tmp_path / "joint.npz"
    manifest = tmp_path / "joint.json"

    summary = save_joint_experience_npz(dataset, output, manifest_path=manifest)

    loaded = np.load(output)
    assert loaded["observations"].shape == (1, N_FEATURES)
    assert loaded["next_observations"].shape == (1, N_FEATURES)
    assert loaded["legal_masks"].shape == (1, N_ACTIONS)
    assert loaded["next_legal_masks"].shape == (1, N_ACTIONS)
    assert loaded["dones"].shape == (1,)
    assert summary["output_npz"] == str(output)
    assert summary["manifest_json"] == str(manifest)
    assert summary["promotion"] is False
    assert manifest.exists()


def test_build_joint_experience_dataset_cli_writes_npz_and_manifest(monkeypatch, tmp_path):
    from scripts import build_joint_experience_dataset as cli

    output = tmp_path / "joint_cli.npz"
    manifest = tmp_path / "joint_cli.json"
    fake_dataset = {
        "algorithm": "joint_experience_meta_policy_dataset",
        "n_hands": 2,
        "n_transitions": 1,
        "observations": np.zeros((1, N_FEATURES), dtype=np.float32),
        "next_observations": np.zeros((1, N_FEATURES), dtype=np.float32),
        "legal_masks": np.ones((1, N_ACTIONS), dtype=np.bool_),
        "next_legal_masks": np.ones((1, N_ACTIONS), dtype=np.bool_),
        "actions": np.array([0], dtype=np.int64),
        "dones": np.array([True], dtype=np.bool_),
        "player_indices": np.array([0], dtype=np.int64),
        "behavior_policy_indices": np.array([0], dtype=np.int64),
        "hand_indices": np.array([0], dtype=np.int64),
        "step_indices": np.array([0], dtype=np.int64),
        "returns": np.array([0.0], dtype=np.float32),
        "hand_policy_indices": np.array([[0, 0]], dtype=np.int64),
        "hand_returns": np.array([[0.0, 0.0]], dtype=np.float32),
        "meta_strategy": np.array([1.0], dtype=np.float32),
        "promotion": False,
        "uses_slumbot_training_data": False,
    }
    monkeypatch.setattr(cli, "load_policy_adapter", lambda path, *, kind, device: object())
    monkeypatch.setattr(cli, "collect_joint_experience", lambda *_args, **_kwargs: fake_dataset)

    exit_code = cli.main(
        [
            "--policy",
            "tianshou-rainbow:rainbow.pt",
            "--meta-strategy",
            "1.0",
            "--n-hands",
            "2",
            "--output-npz",
            str(output),
            "--manifest-json",
            str(manifest),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    assert manifest.exists()


def test_build_compiled_joint_experience_dataset_cli_writes_npz_and_manifest(monkeypatch, tmp_path):
    from scripts import build_compiled_joint_experience_dataset as cli

    output = tmp_path / "compiled_joint_cli.npz"
    manifest = tmp_path / "compiled_joint_cli.json"
    fake_dataset = {
        "algorithm": "compiled_joint_experience_meta_policy_dataset",
        "backend": "compiled-fast-state",
        "n_hands": 2,
        "n_transitions": 1,
        "observations": np.zeros((1, N_FEATURES), dtype=np.float32),
        "next_observations": np.zeros((1, N_FEATURES), dtype=np.float32),
        "legal_masks": np.ones((1, N_ACTIONS), dtype=np.bool_),
        "next_legal_masks": np.ones((1, N_ACTIONS), dtype=np.bool_),
        "actions": np.array([0], dtype=np.int64),
        "dones": np.array([True], dtype=np.bool_),
        "player_indices": np.array([0], dtype=np.int64),
        "behavior_policy_indices": np.array([0], dtype=np.int64),
        "hand_indices": np.array([0], dtype=np.int64),
        "step_indices": np.array([0], dtype=np.int64),
        "returns": np.array([0.0], dtype=np.float32),
        "hand_policy_indices": np.array([[0, 0]], dtype=np.int64),
        "hand_returns": np.array([[0.0, 0.0]], dtype=np.float32),
        "meta_strategy": np.array([1.0], dtype=np.float32),
        "promotion": False,
        "uses_slumbot_training_data": False,
    }
    monkeypatch.setattr(cli, "load_compiled_joint_policy", lambda path, *, kind, device: object())
    monkeypatch.setattr(cli, "collect_compiled_joint_experience", lambda *_args, **_kwargs: fake_dataset)

    exit_code = cli.main(
        [
            "--policy",
            "tianshou-rainbow:rainbow.pt",
            "--meta-strategy",
            "1.0",
            "--n-hands",
            "2",
            "--batch-size",
            "2",
            "--exploration-epsilon",
            "0.25",
            "--output-npz",
            str(output),
            "--manifest-json",
            str(manifest),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    assert manifest.exists()
