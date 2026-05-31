import json
import subprocess
import sys
from pathlib import Path

import torch

from poker_ai.deep_cfr.networks import PolicyNetwork, ValueNetwork
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


def test_eval_decision_value_search_actor_cli_writes_root_metrics(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output_json = tmp_path / "search_actor.json"
    _write_average_policy_checkpoint(checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_decision_value_search_actor.py",
            "--checkpoint",
            str(checkpoint),
            "--continuation-checkpoint",
            str(checkpoint),
            "--strategy-source",
            "average-policy",
            "--continuation-strategy-source",
            "average-policy",
            "--n-roots",
            "2",
            "--n-worlds",
            "1",
            "--device",
            "cpu",
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    saved = json.loads(output_json.read_text())
    assert metrics["mode"] == "decision_value_search_actor_root_gate"
    assert saved["target_metadata"]["n_roots"] == 2
    assert "target_selected_action_counts" in metrics["target_metadata"]
    assert metrics["selected_action_counts"] == metrics["search_actor_metrics"]["selected_action_counts"]
    assert "search_actor_metrics" in metrics
