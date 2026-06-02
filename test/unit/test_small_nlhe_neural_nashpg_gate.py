import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch


class _TinyCollector:
    n_actions = 4
    obs_dim = 6
    n_players = 2


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


def test_small_nlhe_neural_nashpg_gate_supports_gae_targets(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "neural_nashpg_gate_gae.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_nlhe_neural_nashpg_gate.py",
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
            "--advantage-target",
            "gae",
            "--gamma",
            "1.0",
            "--gae-lambda",
            "0.5",
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
    assert data["arms"]["neural_nashpg"]["runs"][0]["loss_last"]["advantage_target"] == "gae"


def test_neural_reference_pg_solver_seed_controls_torch_initialization():
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))
    from scripts.run_small_nlhe_neural_nashpg_gate import NeuralReferencePGSolver

    first = NeuralReferencePGSolver(
        collector=_TinyCollector(),
        layers=(8,),
        lr=0.001,
        reference_kl_weight=0.05,
        entropy_weight=0.02,
        value_weight=0.5,
        reference_update_every=1,
        seed=123,
    )
    second = NeuralReferencePGSolver(
        collector=_TinyCollector(),
        layers=(8,),
        lr=0.001,
        reference_kl_weight=0.05,
        entropy_weight=0.02,
        value_weight=0.5,
        reference_update_every=1,
        seed=123,
    )

    for left, right in zip(first.net.parameters(), second.net.parameters(), strict=True):
        assert torch.equal(left, right)


def test_inverse_own_reach_decision_weights_use_prior_learner_reach():
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))
    from scripts.run_small_nlhe_neural_nashpg_gate import _decision_weights_from_own_reach

    valid = torch.tensor([[1.0], [1.0], [0.0]])
    action_oh = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 0.0]],
        ]
    )
    policy = torch.tensor(
        [
            [[0.25, 0.75]],
            [[0.50, 0.50]],
            [[0.50, 0.50]],
        ]
    )

    weights = _decision_weights_from_own_reach(
        action_oh=action_oh,
        behavior_policy=policy,
        valid=valid,
        max_decision_weight=10.0,
    )

    assert torch.isclose(weights[0, 0], torch.tensor(0.4))
    assert torch.isclose(weights[1, 0], torch.tensor(1.6))
    assert weights[2, 0] == 0.0
