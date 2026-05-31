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


def test_eval_decision_value_search_actor_h2h_cli_writes_latency_metrics(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output_json = tmp_path / "search_actor_h2h.json"
    _write_average_policy_checkpoint(checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_decision_value_search_actor_h2h.py",
            "--checkpoint",
            str(checkpoint),
            "--strategy-source",
            "average-policy",
            "--n-games",
            "2",
            "--n-worlds",
            "1",
            "--device",
            "cpu",
            "--record-decisions",
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
    assert metrics["mode"] == "decision_value_search_actor_h2h"
    assert saved["n_duplicate_pairs"] == 2
    assert "mean_search_decision_ms" in metrics
    assert "decision_summary" in metrics
    assert "decision_records" in metrics
    assert metrics["promotion_blockers"]


def test_eval_decision_value_search_actor_h2h_cli_supports_search_guided_scorer(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    output_json = tmp_path / "search_guided_h2h.json"
    _write_average_policy_checkpoint(checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_decision_value_search_actor_h2h.py",
            "--checkpoint",
            str(checkpoint),
            "--strategy-source",
            "average-policy",
            "--n-games",
            "1",
            "--n-worlds",
            "1",
            "--scorer-continuation",
            "search-guided",
            "--continuation-search-worlds",
            "1",
            "--max-steps-per-hand",
            "16",
            "--device",
            "cpu",
            "--record-decisions",
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
    assert metrics["scorer_continuation"] == "search-guided"
    assert saved["continuation_search_worlds"] == 1
    assert metrics["n_continuation_search_decisions"] >= 0
    assert "mean_continuation_search_decision_ms" in metrics
