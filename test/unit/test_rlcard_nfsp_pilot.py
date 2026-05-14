import pytest

from scripts.run_rlcard_nfsp_pilot import resolve_device, run_pilot


def test_resolve_device_uses_cpu_when_requested():
    resolved = resolve_device("cpu")

    assert resolved["requested_device"] == "cpu"
    assert resolved["resolved_device"] == "cpu"


def test_resolve_device_auto_defaults_to_cpu_for_framework_control_even_with_cuda(monkeypatch):
    monkeypatch.setattr("scripts.run_rlcard_nfsp_pilot.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("scripts.run_rlcard_nfsp_pilot.torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("scripts.run_rlcard_nfsp_pilot.torch.cuda.get_device_name", lambda _: "test-gpu")

    resolved = resolve_device("auto")

    assert resolved["requested_device"] == "auto"
    assert resolved["resolved_device"] == "cpu"
    assert resolved["device_policy"] == "cpu_default_for_rlcard_framework_control"


def test_resolve_device_rejects_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr("scripts.run_rlcard_nfsp_pilot.torch.cuda.is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA requested"):
        resolve_device("cuda")


def test_run_pilot_returns_framework_control_metrics():
    metrics = run_pilot(
        train_episodes=2,
        eval_games=2,
        seed=11,
        hidden_dim=16,
        min_buffer_size_to_learn=100,
        device="cpu",
    )

    assert metrics["environment"] == "rlcard:no-limit-holdem"
    assert metrics["num_actions"] == 5
    assert metrics["requested_device"] == "cpu"
    assert metrics["resolved_device"] == "cpu"
    assert metrics["train_seconds"] >= 0.0
    assert metrics["eval_seconds"] >= 0.0
    assert metrics["total_seconds"] >= metrics["train_seconds"]
    assert metrics["role"] == "framework_control_pilot"
    assert metrics["algorithm"] == "nfsp"
    assert len(metrics["pre_payoffs"]) == 2
    assert len(metrics["post_payoffs"]) == 2
    assert metrics["train_episodes"] == 2
    assert metrics["agent_total_t"] >= 0
