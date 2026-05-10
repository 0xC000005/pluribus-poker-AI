import json
import subprocess
import sys
from pathlib import Path

import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.research.evaluation import (
    compare_checkpoint_metrics,
    evaluate_value_nets_head_to_head,
    evaluate_value_net_vs_random,
    load_value_network_checkpoint,
)


def test_load_value_network_checkpoint_accepts_legacy_sequential_keys(tmp_path):
    checkpoint = {
        "iteration": 7,
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
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)

    loaded = load_value_network_checkpoint(path, torch.device("cpu"))

    assert loaded.metadata["iteration"] == 7
    assert loaded.metadata["n_players"] == 2
    assert loaded.metadata["initial_chips"] == 20000
    assert loaded.value_net.hidden_dim == 16


def test_evaluate_value_net_vs_random_returns_ci_metrics():
    checkpoint = {
        "iteration": 0,
        "n_players": 2,
        "initial_chips": 1000,
    }
    from poker_ai.deep_cfr.networks import ValueNetwork

    value_net = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=1)
    metrics = evaluate_value_net_vs_random(
        value_net,
        torch.device("cpu"),
        n_games=8,
        n_players=2,
        initial_chips=1000,
        seed=123,
        checkpoint_metadata=checkpoint,
    )

    assert metrics["n_games"] == 8
    assert metrics["n_players"] == 2
    assert metrics["initial_chips"] == 1000
    assert isinstance(metrics["avg_chips_per_hand"], float)
    assert isinstance(metrics["ci95_chips_per_hand"], float)
    assert metrics["checkpoint_iteration"] == 0


def test_evaluate_value_nets_head_to_head_self_compare_is_zero_with_swapped_seats():
    from poker_ai.deep_cfr.networks import ValueNetwork

    value_net = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=1)
    metadata = {
        "checkpoint": "self.pt",
        "checkpoint_iteration": 0,
        "hidden_dim": 32,
        "n_layers": 1,
    }

    metrics = evaluate_value_nets_head_to_head(
        value_net,
        value_net,
        torch.device("cpu"),
        n_games=6,
        initial_chips=1000,
        seed=123,
        candidate_metadata=metadata,
        baseline_metadata=metadata,
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "duplicate_swapped_head_to_head"
    assert metrics["avg_chips_per_hand"] == 0.0
    assert metrics["paired_delta_lower95_chips_per_hand"] == 0.0
    assert metrics["promotable"] is False


def test_eval_cli_emits_json_for_legacy_checkpoint(tmp_path):
    checkpoint = {
        "iteration": 3,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, checkpoint_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--n-games",
            "6",
            "--device",
            "cpu",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["checkpoint_iteration"] == 3
    assert metrics["n_games"] == 6
    assert metrics["passed"] is True


def test_eval_cli_aggregates_multiple_seeds(tmp_path):
    checkpoint = {
        "iteration": 4,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, checkpoint_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--n-games",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "11,12",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["n_runs"] == 2
    assert [run["seed"] for run in metrics["runs"]] == [11, 12]
    assert isinstance(metrics["avg_chips_per_hand"], float)


def test_compare_checkpoint_metrics_keeps_local_result_non_promotable():
    candidate = {
        "passed": True,
        "avg_chips_per_hand": 15.0,
        "runs": [
            {"seed": 1, "avg_chips_per_hand": 10.0},
            {"seed": 2, "avg_chips_per_hand": 20.0},
        ],
        "checkpoint": "candidate.pt",
    }
    baseline = {
        "passed": True,
        "avg_chips_per_hand": 10.0,
        "runs": [
            {"seed": 1, "avg_chips_per_hand": 8.0},
            {"seed": 2, "avg_chips_per_hand": 12.0},
        ],
        "checkpoint": "baseline.pt",
    }

    metrics = compare_checkpoint_metrics(candidate, baseline)

    assert metrics["passed"] is True
    assert metrics["delta_avg_chips_per_hand"] == 5.0
    assert metrics["paired_delta_chips_per_hand"] == [2.0, 8.0]
    assert metrics["promotable"] is False
    assert "local_random_comparison_is_not_strategy_quality" in metrics["promotion_blockers"]


def test_eval_cli_compares_candidate_to_baseline_checkpoint(tmp_path):
    checkpoint = {
        "iteration": 5,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    candidate_path = tmp_path / "candidate.pt"
    baseline_path = tmp_path / "baseline.pt"
    torch.save(checkpoint, candidate_path)
    torch.save(checkpoint, baseline_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(candidate_path),
            "--baseline-checkpoint",
            str(baseline_path),
            "--n-games",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "21,22",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["candidate"]["checkpoint"] == str(candidate_path)
    assert metrics["baseline"]["checkpoint"] == str(baseline_path)
    assert metrics["delta_avg_chips_per_hand"] == 0.0
    assert metrics["promotable"] is False


def test_eval_cli_runs_head_to_head_self_compare(tmp_path):
    checkpoint = {
        "iteration": 6,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    candidate_path = tmp_path / "candidate.pt"
    baseline_path = tmp_path / "baseline.pt"
    torch.save(checkpoint, candidate_path)
    torch.save(checkpoint, baseline_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(candidate_path),
            "--baseline-checkpoint",
            str(baseline_path),
            "--head-to-head",
            "--n-games",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "31,32",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["mode"] == "duplicate_swapped_head_to_head"
    assert metrics["avg_chips_per_hand"] == 0.0
    assert metrics["promotable"] is False
