import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


def test_small_nlhe_neural_nashpg_gate_tiny(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "neural_nashpg_gate.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_nlhe_neural_nashpg_gate.py",
            "--steps",
            "2",
            "--eval-every",
            "1",
            "--batch-size",
            "8",
            "--seeds",
            "1",
            "--layers",
            "8",
            "--reference-update-every",
            "1",
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

    assert data["gate"] == "small_nlhe_neural_nashpg_truth_gate"
    assert data["algorithm"] == "neural_reference_regularized_policy_gradient"
    assert data["uses_slumbot_training_data"] is False
    assert data["promotion"] is False
    assert data["fingerprint"]["num_players"] == 2
    assert data["fingerprint"]["num_distinct_actions"] == 4
    assert data["fingerprint"]["max_game_length"] == 7
    assert math.isclose(data["fingerprint"]["uniform_nashconv"], 1.7, rel_tol=0, abs_tol=1e-3)
    assert data["decision"]["small_game_exact_only"] is True
    assert data["summary"]["mean_last_nashconv"] == data["arms"]["neural_nashpg"]["mean_last_nashconv"]
    assert len(data["arms"]["neural_nashpg"]["runs"]) == 1
    for _step, value in data["arms"]["neural_nashpg"]["runs"][0]["history"]:
        assert math.isfinite(value)
