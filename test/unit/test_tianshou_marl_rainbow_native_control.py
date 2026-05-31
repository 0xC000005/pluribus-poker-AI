import json
from pathlib import Path

import pytest
import torch


pytest.importorskip("pettingzoo")
pytest.importorskip("tianshou")


from scripts.run_tianshou_marl_rainbow_native_control import run_marl_control  # noqa: E402
from scripts.run_tianshou_rainbow_native_control import _load_rainbow_checkpoint_policy  # noqa: E402


def test_tianshou_marl_rainbow_native_control_smoke_writes_metrics(tmp_path):
    output = tmp_path / "marl_rainbow.json"
    checkpoint = tmp_path / "marl_rainbow.pt"

    metrics = run_marl_control(
        train_steps=8,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=8,
        num_envs=1,
        device="cpu",
        seed=20260637,
        output_json=str(output),
        checkpoint_out=str(checkpoint),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "tianshou_marl_rainbow_dqn"
    assert payload["environment"] == "poker_ai:pettingzoo_full_deck_hu_nlhe"
    assert payload["role"] == "plug_in_multi_agent_rl_control"
    assert payload["num_agents"] == 2
    assert payload["num_actions"] == 9
    assert payload["train_steps"] >= 8
    assert payload["updates"] == 1
    assert payload["promotion"] is False
    assert Path(metrics["checkpoint_path"]).exists()


def test_tianshou_marl_rainbow_checkpoint_loads_as_player0_rainbow_policy(tmp_path):
    checkpoint = tmp_path / "marl_rainbow.pt"
    run_marl_control(
        train_steps=8,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=8,
        num_envs=1,
        device="cpu",
        seed=20260638,
        checkpoint_out=str(checkpoint),
    )

    payload, policy = _load_rainbow_checkpoint_policy(str(checkpoint), resolved_device="cpu")

    assert payload["algorithm"] == "tianshou_marl_rainbow_dqn"
    assert payload["loaded_agent_id"] == "player_0"
    assert policy is not None


def test_tianshou_marl_rainbow_supports_shared_policy_checkpoint(tmp_path):
    output = tmp_path / "shared_marl_rainbow.json"
    checkpoint = tmp_path / "shared_marl_rainbow.pt"

    metrics = run_marl_control(
        train_steps=8,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=8,
        num_envs=1,
        device="cpu",
        seed=20260645,
        shared_policy=True,
        output_json=str(output),
        checkpoint_out=str(checkpoint),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    checkpoint_payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["shared_policy"] is True
    assert metrics["shared_policy"] is True
    assert checkpoint_payload["shared_policy"] is True
    assert "shared_model_state_dict" in checkpoint_payload
    assert sorted(checkpoint_payload["agent_model_state_dicts"]) == ["player_0", "player_1"]


def test_tianshou_marl_rainbow_records_vector_backend(tmp_path):
    output = tmp_path / "vector_marl_rainbow.json"

    metrics = run_marl_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        num_envs=2,
        vector_env_backend="dummy",
        device="cpu",
        seed=20260651,
        output_json=str(output),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["num_envs"] == 2
    assert payload["vector_env_backend"] == "dummy"
    assert metrics["vector_env_backend"] == "dummy"


def test_tianshou_marl_rainbow_records_fast_state_backend(tmp_path):
    output = tmp_path / "fast_state_marl_rainbow.json"

    metrics = run_marl_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        num_envs=1,
        state_backend="fast-state",
        device="cpu",
        seed=20260760,
        output_json=str(output),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["state_backend"] == "fast-state"
    assert metrics["state_backend"] == "fast-state"


def test_tianshou_marl_rainbow_can_continue_from_shared_checkpoint(tmp_path):
    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    output = tmp_path / "child.json"
    run_marl_control(
        train_steps=8,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=8,
        num_envs=1,
        shared_policy=True,
        device="cpu",
        seed=20260659,
        checkpoint_out=str(parent),
    )

    metrics = run_marl_control(
        train_steps=0,
        updates=0,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=0,
        num_envs=1,
        shared_policy=True,
        checkpoint_in=str(parent),
        device="cpu",
        seed=20260660,
        output_json=str(output),
        checkpoint_out=str(child),
    )

    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    child_payload = torch.load(child, map_location="cpu", weights_only=False)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["checkpoint_in"] == str(parent)
    assert metrics["checkpoint_in"] == str(parent)
    assert child_payload["parent_checkpoint"] == str(parent)
    for key, tensor in parent_payload["shared_model_state_dict"].items():
        assert torch.equal(tensor, child_payload["shared_model_state_dict"][key])
