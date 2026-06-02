import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


def test_small_nlhe_mmd_update_fidelity_bridge_tiny(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "bridge.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_small_nlhe_mmd_update_fidelity.py",
            "--start-mmd-steps",
            "0",
            "--target-mmd-delta",
            "1",
            "--fit-steps",
            "4",
            "--update-steps",
            "1",
            "--batch-size",
            "8",
            "--layers",
            "8",
            "--max-fit-current-kl",
            "10.0",
            "--min-target-kl-reduction",
            "-10.0",
            "--min-update-delta-cosine",
            "-1.0",
            "--output-json",
            str(out),
        ],
        check=False,
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())

    assert data["gate"] == "small_nlhe_sequence_form_mmd_update_fidelity_bridge"
    assert data["uses_slumbot_training_data"] is False
    assert data["promotion"] is False
    assert data["small_game_exact_only"] is True
    assert data["fingerprint"]["num_distinct_actions"] == 4
    assert math.isclose(data["fingerprint"]["uniform_nashconv"], 1.7, rel_tol=0, abs_tol=1e-3)
    for key in (
        "before_target_kl",
        "after_target_kl",
        "target_kl_reduction",
        "update_delta_cosine",
    ):
        assert math.isfinite(data["summary"][key])
    assert data["decision"]["slumbot_full_hunl_blocked"] is True


def test_small_nlhe_mmd_update_fidelity_bridge_supports_gae_targets(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "bridge_gae.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_small_nlhe_mmd_update_fidelity.py",
            "--start-mmd-steps",
            "0",
            "--target-mmd-delta",
            "1",
            "--fit-steps",
            "4",
            "--update-steps",
            "1",
            "--batch-size",
            "8",
            "--layers",
            "8",
            "--advantage-target",
            "gae",
            "--gamma",
            "1.0",
            "--gae-lambda",
            "0.5",
            "--max-fit-current-kl",
            "10.0",
            "--min-target-kl-reduction",
            "-10.0",
            "--min-update-delta-cosine",
            "-1.0",
            "--output-json",
            str(out),
        ],
        check=False,
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())

    assert data["config"]["advantage_target"] == "gae"
    assert data["config"]["gamma"] == 1.0
    assert data["config"]["gae_lambda"] == 0.5
    assert data["summary"]["last_step_logs"]["advantage_target"] == "gae"
