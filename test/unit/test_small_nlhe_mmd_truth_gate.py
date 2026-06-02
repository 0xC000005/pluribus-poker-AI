import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


def test_small_nlhe_mmd_truth_gate_tiny(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "mmd_gate.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_nlhe_mmd_truth_gate.py",
            "--steps",
            "2",
            "--eval-every",
            "1",
            "--alpha",
            "0.01",
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

    assert data["gate"] == "small_nlhe_mmd_truth_gate"
    assert data["algorithm"] == "open_spiel_mmd_dilated"
    assert data["uses_slumbot_training_data"] is False
    assert data["promotion"] is False
    assert data["fingerprint"]["num_players"] == 2
    assert data["fingerprint"]["num_distinct_actions"] == 4
    assert data["fingerprint"]["max_game_length"] == 7
    assert math.isclose(data["fingerprint"]["uniform_nashconv"], 1.7, rel_tol=0, abs_tol=1e-3)

    assert len(data["history"]) == 3
    for row in data["history"]:
        assert math.isfinite(row["current_nashconv"])
        assert math.isfinite(row["average_nashconv"])
        assert math.isfinite(row["regularized_gap"])
    assert data["decision"]["harness_passed"] is True
    assert data["decision"]["small_game_exact_only"] is True
