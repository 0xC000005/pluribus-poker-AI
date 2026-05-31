import pytest


pytest.importorskip("ray.rllib")

from scripts.run_rllib_ppo_native_control import run_control
from scripts.run_rllib_ppo_native_control import _resolve_checkpoint_list


def test_rllib_response_oracle_resolves_checkpoint_paths_for_ray_workers(tmp_path, monkeypatch):
    checkpoint = tmp_path / "opponent.pt"
    checkpoint.write_text("placeholder", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _resolve_checkpoint_list(["opponent.pt"]) == [str(checkpoint.resolve())]


def test_rllib_ppo_native_control_runs_action_masked_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_json = "metrics.json"
    checkpoint_out = "checkpoint"

    metrics = run_control(
        train_iterations=1,
        train_batch_size=64,
        rollout_fragment_length=32,
        num_env_runners=0,
        num_cpus=2,
        num_gpus=0,
        max_steps_per_hand=16,
        seed=20260718,
        output_json=str(output_json),
        checkpoint_out=str(checkpoint_out),
    )

    assert metrics["algorithm"] == "rllib_ppo_action_masked"
    assert metrics["num_actions"] == 9
    assert metrics["train_iterations"] == 1
    assert metrics["disable_env_checking"] is True
    assert metrics["env_steps_sampled"] >= 64
    assert metrics["steps_per_second"] > 0
    assert "checkpoint_path" in metrics
    assert (tmp_path / output_json).exists()
    assert (tmp_path / checkpoint_out).exists()


def test_rllib_ppo_native_control_runs_response_oracle_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_json = "response_metrics.json"

    metrics = run_control(
        train_iterations=1,
        train_batch_size=64,
        rollout_fragment_length=32,
        num_env_runners=0,
        num_cpus=2,
        num_gpus=0,
        max_steps_per_hand=16,
        seed=20260725,
        opponent_kind="random",
        output_json=str(output_json),
    )

    assert metrics["algorithm"] == "rllib_ppo_action_masked_response_oracle"
    assert metrics["role"] == "maintained_library_population_response_oracle"
    assert metrics["opponent_kind"] == "random"
    assert metrics["train_opponent_mode"] == "random_policy"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["env_steps_sampled"] >= 64
    assert (tmp_path / output_json).exists()
