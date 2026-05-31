import json
from pathlib import Path

import pytest


pytest.importorskip("pettingzoo")
pytest.importorskip("tianshou")


from scripts.run_tianshou_marl_ppo_native_control import run_marl_ppo_control  # noqa: E402
from scripts.run_tianshou_ppo_native_control import _load_ppo_checkpoint_policy  # noqa: E402


def test_tianshou_marl_ppo_smoke_writes_shared_policy_metrics(tmp_path):
    output = tmp_path / "marl_ppo.json"
    checkpoint = tmp_path / "marl_ppo.pt"

    metrics = run_marl_ppo_control(
        rollout_steps=8,
        updates=1,
        repeat=1,
        batch_size=4,
        hidden_dim=16,
        num_envs=1,
        shared_policy=True,
        device="cpu",
        seed=20260666,
        output_json=str(output),
        checkpoint_out=str(checkpoint),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "tianshou_marl_ppo"
    assert payload["environment"] == "poker_ai:pettingzoo_full_deck_hu_nlhe"
    assert payload["role"] == "plug_in_multi_agent_policy_value_rl_control"
    assert payload["shared_policy"] is True
    assert payload["num_agents"] == 2
    assert payload["num_actions"] == 9
    assert payload["train_steps"] >= 8
    assert payload["updates"] == 1
    assert payload["promotion"] is False
    assert Path(metrics["checkpoint_path"]).exists()


def test_tianshou_marl_ppo_checkpoint_loads_as_ppo_policy(tmp_path):
    checkpoint = tmp_path / "marl_ppo.pt"
    run_marl_ppo_control(
        rollout_steps=8,
        updates=1,
        repeat=1,
        batch_size=4,
        hidden_dim=16,
        num_envs=1,
        shared_policy=True,
        device="cpu",
        seed=20260667,
        checkpoint_out=str(checkpoint),
    )

    payload, policy = _load_ppo_checkpoint_policy(str(checkpoint), resolved_device="cpu")

    assert payload["algorithm"] == "tianshou_marl_ppo"
    assert policy is not None
