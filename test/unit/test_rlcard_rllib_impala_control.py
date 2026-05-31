from pathlib import Path

import pytest

pytest.importorskip("ray")
pytest.importorskip("rlcard")


def test_rlcard_env_can_emit_rllib_action_mask_observation():
    from poker_ai.research.rlcard_tianshou_ppo import RLCardNoLimitHoldemSingleAgentEnv

    env = RLCardNoLimitHoldemSingleAgentEnv(seed=20, observation_format="rllib")
    obs, info = env.reset(seed=20)

    assert set(obs) == {"observations", "action_mask"}
    assert obs["observations"].shape == (54,)
    assert obs["action_mask"].shape == (5,)
    assert obs["action_mask"].sum() >= 1
    assert "observations" in env.observation_space.spaces
    assert "action_mask" in env.observation_space.spaces
    assert info["trained_environment_native"] is True
    assert info["native_action_projection"] is False


def test_rllib_impala_dry_run_reports_same_environment_contract(tmp_path):
    from scripts.run_rllib_impala_rlcard_reference_control import run_control

    metrics = run_control(
        train_iterations=0,
        seed=21,
        hidden_dim=32,
        device="cpu",
        dry_run=True,
        output_json=tmp_path / "dry_run.json",
    )

    assert metrics["algorithm"] == "rllib_impala_rlcard"
    assert metrics["environment"] == "rlcard:no-limit-holdem"
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["uses_slumbot_data"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert Path(metrics["output_json"]).exists()


def test_rllib_impala_cuda_actual_run_fails_closed(monkeypatch):
    from scripts import run_rllib_impala_rlcard_reference_control as runner

    with pytest.raises(RuntimeError, match="CPU-only"):
        monkeypatch.setattr(
            runner,
            "resolve_device",
            lambda _device: {
                "requested_device": "cuda",
                "resolved_device": "cuda",
                "device_policy": "explicit_cuda",
                "torch_cuda_available": True,
                "torch_device_count": 1,
                "torch_device_name": "test-gpu",
            },
        )
        runner.run_control(train_iterations=1, seed=22, device="cuda")
