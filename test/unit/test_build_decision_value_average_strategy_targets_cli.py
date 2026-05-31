import json
import subprocess
import sys
from pathlib import Path

import torch

from poker_ai.deep_cfr.networks import PolicyNetwork, ValueNetwork
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES


def _write_average_policy_checkpoint(path: Path) -> None:
    torch.manual_seed(20260525)
    value_net = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=2)
    average_policy_net = PolicyNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=2)
    torch.save(
        {
            "value_net": value_net.state_dict(),
            "average_policy_net": average_policy_net.state_dict(),
            "hidden_dim": 32,
            "n_layers": 2,
            "n_players": 2,
            "initial_chips": 1000,
            "iteration": 1,
            "uses_betting_history": True,
        },
        path,
    )


def test_build_decision_value_average_strategy_targets_cli_writes_npz_and_json(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output_npz = tmp_path / "targets.npz"
    output_json = tmp_path / "targets.json"
    _write_average_policy_checkpoint(checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_decision_value_average_strategy_targets.py",
            "--checkpoint",
            str(checkpoint),
            "--continuation-checkpoint",
            str(checkpoint),
            "--strategy-source",
            "average-policy",
            "--continuation-strategy-source",
            "average-policy",
            "--behavior-policy",
            "base",
            "--n-states",
            "1",
            "--n-worlds",
            "1",
            "--target-streets",
            "0",
            "--max-hands",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output_npz),
            "--output-json",
            str(output_json),
            "--recommended-average-strategy-weight",
            "0.5",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["recommended_gpu_flags"]["average_strategy_weight"] == 0.5
    assert json.loads(output_json.read_text())["target_size"] == 1
    assert PolicyTargetBuffer.from_npz(output_npz).size == 1


def test_build_decision_value_average_strategy_targets_cli_behavior_mode(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output_npz = tmp_path / "behavior_targets.npz"
    output_json = tmp_path / "behavior_targets.json"
    _write_average_policy_checkpoint(checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_decision_value_average_strategy_targets.py",
            "--checkpoint",
            str(checkpoint),
            "--continuation-checkpoint",
            str(checkpoint),
            "--strategy-source",
            "average-policy",
            "--continuation-strategy-source",
            "average-policy",
            "--behavior-policy",
            "base",
            "--target-mode",
            "behavior",
            "--n-states",
            "1",
            "--n-worlds",
            "1",
            "--target-streets",
            "0",
            "--max-hands",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output_npz),
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    assert metrics["mode"] == "on_policy_behavior_average_strategy_target_artifact"
    assert metrics["recommended_gpu_flags"]["seed_average_strategy_memory_from_targets"] is True
    assert PolicyTargetBuffer.from_npz(output_npz).target_probs.sum() == 1.0


def test_build_decision_value_average_strategy_targets_cli_trajectory_return_mode(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output_npz = tmp_path / "trajectory_targets.npz"
    output_json = tmp_path / "trajectory_targets.json"
    _write_average_policy_checkpoint(checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_decision_value_average_strategy_targets.py",
            "--checkpoint",
            str(checkpoint),
            "--continuation-checkpoint",
            str(checkpoint),
            "--strategy-source",
            "average-policy",
            "--continuation-strategy-source",
            "average-policy",
            "--behavior-policy",
            "base",
            "--target-mode",
            "trajectory-return-behavior",
            "--n-states",
            "1",
            "--n-worlds",
            "1",
            "--target-streets",
            "0",
            "--max-hands",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output_npz),
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    assert metrics["mode"] == "on_policy_trajectory_return_average_strategy_target_artifact"
    assert metrics["target_metadata"]["return_weight_mode"] == "centered_exp"
    assert PolicyTargetBuffer.from_npz(output_npz).size == 1
