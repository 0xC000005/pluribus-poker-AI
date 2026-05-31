from pathlib import Path

import pytest

pytest.importorskip("rlcard")


def test_local_vtrace_rlcard_smoke_writes_no_leakage_metrics(tmp_path):
    from scripts.run_local_vtrace_rlcard_smoke import run_smoke

    output = tmp_path / "smoke.json"
    metrics = run_smoke(
        n_envs=2,
        unroll_length=4,
        updates=1,
        hidden_dim=16,
        seed=26,
        device="cpu",
        output_json=output,
    )

    assert output.exists()
    assert metrics["algorithm"] == "local_vtrace_rlcard_smoke"
    assert metrics["environment"] == "rlcard:no-limit-holdem"
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["uses_slumbot_data"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert metrics["n_samples"] == 8
    assert metrics["illegal_action_probability"] == 0.0
    assert metrics["loss_is_finite"] is True
    assert Path(metrics["output_json"]).exists()
