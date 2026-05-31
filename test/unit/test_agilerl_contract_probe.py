import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gymnasium import spaces

from poker_ai.research.agilerl_contract import (
    make_agilerl_native_parallel_env,
    probe_agilerl_offpolicy_algorithms,
    probe_native_agilerl_contract,
)


def test_probe_native_agilerl_contract_reports_parallel_mask_contract():
    pytest.importorskip("pettingzoo")

    metrics = probe_native_agilerl_contract(seed=123, max_steps_per_hand=32)

    assert metrics["algorithm"] == "agilerl_pettingzoo_contract_probe"
    assert metrics["parallel_conversion"] == "turn_based_aec_to_parallel"
    assert metrics["num_actions"] == 9
    assert metrics["num_features"] == 126
    assert metrics["num_agents"] == 2
    assert metrics["active_mask_agents"] == 1
    assert metrics["inactive_zero_mask_agents"] == 1
    assert metrics["recommendation"] == "candidate_agilerl_parallel_spike"
    assert metrics["promotion"] is False


def test_make_agilerl_native_parallel_env_moves_masks_to_infos():
    pytest.importorskip("pettingzoo")

    env = make_agilerl_native_parallel_env(seed=123, max_steps_per_hand=32)
    observations, infos = env.reset(seed=123)

    assert isinstance(env.observation_space("player_0"), spaces.Box)
    assert observations["player_0"].shape == (126,)
    assert "action_mask" not in observations["player_0"]
    assert len(infos["player_0"]["action_mask"]) == 9
    assert int(np.asarray(infos["player_0"]["action_mask"]).sum()) > 0
    assert int(np.asarray(infos["player_1"]["action_mask"]).sum()) == 0
    assert infos["player_0"]["env_defined_actions"] is None
    assert np.asarray(infos["player_1"]["env_defined_actions"]).shape == (1,)
    assert int(np.asarray(infos["player_1"]["env_defined_actions"])[0]) == 0

    actions = {
        agent: env.action_space(agent).sample(
            np.asarray(infos[agent]["action_mask"], dtype=np.int8)
        )
        if np.asarray(infos[agent]["action_mask"]).sum() > 0
        else 0
        for agent in env.agents
    }
    next_observations, rewards, terminations, truncations, next_infos = env.step(actions)

    assert set(next_observations) == {"player_0", "player_1"}
    assert set(rewards) == {"player_0", "player_1"}
    assert set(terminations) == {"player_0", "player_1"}
    assert set(truncations) == {"player_0", "player_1"}
    assert all("action_mask" in info for info in next_infos.values())


def test_probe_agilerl_contract_cli_writes_json(tmp_path):
    pytest.importorskip("pettingzoo")

    output = tmp_path / "agilerl_contract.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/probe_agilerl_pettingzoo_contract.py",
            "--seed",
            "123",
            "--max-steps-per-hand",
            "32",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "agilerl_pettingzoo_contract_probe"
    assert payload["promotion"] is False


def test_probe_agilerl_offpolicy_contract_cli_writes_json(tmp_path):
    pytest.importorskip("agilerl")
    pytest.importorskip("pettingzoo")

    output = tmp_path / "agilerl_offpolicy_contract.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/probe_agilerl_offpolicy_contract.py",
            "--algorithms",
            "MADDPG",
            "--seed",
            "123",
            "--max-steps",
            "8",
            "--evo-steps",
            "4",
            "--batch-size",
            "4",
            "--hidden-dim",
            "16",
            "--max-steps-per-hand",
            "32",
            "--device",
            "cpu",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "agilerl_offpolicy_contract_probe"
    assert payload["promotion"] is False
    assert set(payload["results"]) == {"MADDPG"}


def test_probe_agilerl_offpolicy_algorithms_reports_train_smoke_status():
    pytest.importorskip("agilerl")
    pytest.importorskip("pettingzoo")

    metrics = probe_agilerl_offpolicy_algorithms(
        algorithms=("MADDPG", "MATD3"),
        seed=123,
        max_steps=8,
        evo_steps=4,
        batch_size=4,
        hidden_dim=16,
        max_steps_per_hand=32,
        device="cpu",
    )

    assert metrics["algorithm"] == "agilerl_offpolicy_contract_probe"
    assert metrics["environment"] == "poker_ai:pettingzoo_full_deck_hu_nlhe"
    assert metrics["promotion"] is False
    assert set(metrics["results"]) == {"MADDPG", "MATD3"}
    for result in metrics["results"].values():
        assert result["constructs"] is True
        assert result["train_smoke_ok"] in {True, False}
        assert result["recommendation"] in {
            "candidate_bounded_training",
            "adapter_or_library_contract_fix_required",
        }


def test_probe_agilerl_offpolicy_algorithms_records_shape_diagnostics():
    pytest.importorskip("agilerl")
    pytest.importorskip("pettingzoo")

    metrics = probe_agilerl_offpolicy_algorithms(
        algorithms=("MADDPG",),
        seed=123,
        max_steps=8,
        evo_steps=4,
        batch_size=4,
        hidden_dim=16,
        max_steps_per_hand=32,
        device="cpu",
    )

    diagnostics = metrics["results"]["MADDPG"]["contract_diagnostics"]

    assert diagnostics["reset_observation_shapes"] == {
        "player_0": [126],
        "player_1": [126],
    }
    assert diagnostics["reset_action_mask_shapes"] == {
        "player_0": [9],
        "player_1": [9],
    }
    assert set(diagnostics["sample_action_values"]) == {"player_0", "player_1"}
    assert diagnostics["step_observation_shapes"] == {
        "player_0": [126],
        "player_1": [126],
    }
