import json
import subprocess
import sys
from pathlib import Path

import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.research.resolver_benchmark import (
    ResolverBenchmarkCase,
    run_resolver_benchmark,
)


def _small_checkpoint(path: Path) -> None:
    checkpoint = {
        "iteration": 9,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 20000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    torch.save(checkpoint, path)


def test_resolver_benchmark_reports_fixed_state_policy_and_solver_metrics():
    torch.manual_seed(0)
    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    case = ResolverBenchmarkCase(
        label="river-check",
        hole_cards=("Ac", "Kd"),
        board=("2c", "7d", "Jh", "4s", "9c"),
        action_str="ck/kk/kk/",
        client_pos=0,
    )

    metrics = run_resolver_benchmark(
        value_net,
        torch.device("cpu"),
        cases=[case],
        solver_iterations=1,
    )

    assert metrics["passed"] is True
    assert metrics["n_cases"] == 1
    assert metrics["n_solver_cases"] == 1
    result = metrics["cases"][0]
    assert result["label"] == "river-check"
    assert 0 <= result["blueprint_action"] < N_ACTIONS
    assert 0 <= result["solver_action"] < N_ACTIONS
    assert result["blueprint_action_legal"] is True
    assert result["solver_action_legal"] is True
    assert result["solver_latency_ms"] >= 0.0
    assert 0.0 <= result["action_l1_drift"] <= 2.0
    assert isinstance(result["advantage_delta_proxy"], float)


def test_resolver_benchmark_cli_emits_json_for_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    _small_checkpoint(checkpoint_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_resolver_benchmark.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            "cpu",
            "--max-cases",
            "1",
            "--solver-iterations",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["checkpoint_iteration"] == 9
    assert metrics["n_cases"] == 1
    assert metrics["promotion_blockers"] == [
        "fixed_state_resolver_benchmark_is_not_slumbot_confidence"
    ]
