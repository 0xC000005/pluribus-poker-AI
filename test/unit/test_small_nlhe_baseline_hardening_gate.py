import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


def test_small_nlhe_baseline_hardening_gate_tiny(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "gate.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_nlhe_baseline_hardening_gate.py",
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
            "--snapshot-every",
            "1",
            "--pool-size",
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

    assert data["fingerprint"]["num_players"] == 2
    assert data["fingerprint"]["num_distinct_actions"] == 4
    assert data["fingerprint"]["max_game_length"] == 7
    assert math.isclose(data["fingerprint"]["uniform_nashconv"], 1.7, rel_tol=0, abs_tol=1e-3)

    for arm in ("rnad", "ppo_fifo", "ppo_kbest"):
        assert data["arms"][arm]["seeds"] == [1]
        assert len(data["arms"][arm]["last_nashconv"]) == 1
        assert math.isfinite(data["arms"][arm]["mean_last_nashconv"])

    assert data["decision"]["lead"] in {"rnad", "ppo_fifo", "ppo_kbest"}


def test_small_nlhe_baseline_hardening_gate_supports_rnad_learner_policy_source(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "gate_learner.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_nlhe_baseline_hardening_gate.py",
            "--steps",
            "1",
            "--eval-every",
            "1",
            "--batch-size",
            "8",
            "--seeds",
            "1",
            "--layers",
            "8",
            "--snapshot-every",
            "1",
            "--pool-size",
            "1",
            "--rnad-policy-source",
            "learner",
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

    assert data["config"]["rnad_policy_source"] == "learner"
    assert data["arms"]["rnad"]["runs"][0]["policy_source"] == "learner"
