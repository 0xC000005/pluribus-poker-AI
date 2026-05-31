from pathlib import Path

import pytest

pytest.importorskip("ray")
pytest.importorskip("rlcard")


def test_rllib_appo_dry_run_reports_same_environment_contract(tmp_path):
    from scripts.run_rllib_appo_rlcard_reference_control import run_control

    metrics = run_control(
        train_iterations=0,
        seed=23,
        hidden_dim=32,
        device="cpu",
        dry_run=True,
        output_json=tmp_path / "dry_run.json",
    )

    assert metrics["algorithm"] == "rllib_appo_rlcard"
    assert metrics["environment"] == "rlcard:no-limit-holdem"
    assert metrics["learner_family"] == "appo_vtrace"
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["uses_slumbot_data"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert Path(metrics["output_json"]).exists()


def test_rllib_appo_timeout_fails_closed(monkeypatch):
    from scripts import run_rllib_appo_rlcard_reference_control as runner

    def slow_train(_metrics, **_kwargs):
        raise runner.APPOTimeoutError("simulated timeout")

    monkeypatch.setattr(runner, "_train_appo_in_process", slow_train)

    metrics = runner.run_control(
        train_iterations=1,
        seed=24,
        device="cpu",
        timeout_seconds=1.0,
    )

    assert metrics["status"] == "timeout"
    assert metrics["passed"] is False
    assert metrics["promotion"] is False
    assert "simulated timeout" in metrics["failure_reason"]


def test_rllib_appo_supervisor_timeout_writes_metrics(monkeypatch, tmp_path):
    from subprocess import TimeoutExpired

    from scripts import run_rllib_appo_rlcard_reference_control as runner

    output = tmp_path / "timeout.json"
    args = runner.parse_args(
        [
            "--train-iterations",
            "1",
            "--seed",
            "25",
            "--device",
            "cpu",
            "--timeout-seconds",
            "0.1",
            "--output-json",
            str(output),
        ]
    )

    def fake_run(*_args, **_kwargs):
        raise TimeoutExpired(cmd="worker", timeout=0.1)

    monkeypatch.setattr(runner, "_run_worker_subprocess", fake_run)

    exit_code = runner.run_supervised_worker(args)

    assert exit_code == 1
    assert output.exists()
    metrics = runner.json.loads(output.read_text(encoding="utf-8"))
    assert metrics["status"] == "timeout"
    assert metrics["passed"] is False
    assert metrics["promotion"] is False
