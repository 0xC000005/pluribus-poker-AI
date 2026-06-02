import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


def test_small_nlhe_mmd_policy_fit_probe_tiny(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "policy_fit.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_small_nlhe_mmd_policy_fit.py",
            "--mmd-steps",
            "2",
            "--fit-steps",
            "4",
            "--layers",
            "8",
            "--max-fit-nashconv",
            "10.0",
            "--max-mean-target-kl",
            "10.0",
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

    assert data["gate"] == "small_nlhe_mmd_policy_fit_probe"
    assert data["uses_slumbot_training_data"] is False
    assert data["promotion"] is False
    assert data["small_game_exact_only"] is True
    assert data["fingerprint"]["num_players"] == 2
    assert data["fingerprint"]["num_distinct_actions"] == 4
    assert math.isclose(data["fingerprint"]["uniform_nashconv"], 1.7, rel_tol=0, abs_tol=1e-3)
    assert data["dataset"]["n_information_states"] == 48
    assert math.isfinite(data["summary"]["target_mmd_nashconv"])
    assert math.isfinite(data["summary"]["fit_policy_nashconv"])
    assert data["decision"]["slumbot_full_hunl_blocked"] is True
