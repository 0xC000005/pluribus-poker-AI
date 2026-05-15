import json
import subprocess
import sys
from pathlib import Path

import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.research.slumbot_trace_action_likelihood import (
    diagnose_trace_opponent_action_likelihood,
)


def _small_checkpoint(path: Path) -> None:
    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    torch.save(
        {
            "iteration": 3,
            "n_players": 2,
            "hidden_dim": 16,
            "n_layers": 1,
            "initial_chips": 20000,
            "value_net": value_net.state_dict(),
        },
        path,
    )


def _write_trace(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "decision",
                        "source": "policy",
                        "hand_index": 1,
                        "client_pos": 0,
                        "hole_cards": ["Ac", "Kd"],
                        "board": ["2c", "7d", "Jh"],
                        "action_str": "b200b650c/",
                        "street_index": 1,
                    }
                ),
                json.dumps(
                    {
                        "event": "hand_result",
                        "hand_index": 1,
                        "client_pos": 0,
                        "hole_cards": ["Ac", "Kd"],
                        "board": ["2c", "7d", "Jh"],
                        "bot_hole_cards": ["Qs", "Qd"],
                        "winnings": 650,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_diagnose_trace_opponent_action_likelihood_scores_revealed_bot_actions(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    trace = tmp_path / "trace.jsonl"
    _small_checkpoint(checkpoint)
    _write_trace(trace)

    metrics = diagnose_trace_opponent_action_likelihood(
        checkpoint,
        trace,
        strategy_source="regret",
        device="cpu",
    )

    assert metrics["passed"] is True
    assert metrics["n_scored_actions"] == 2
    assert metrics["n_scored_hands"] == 1
    assert metrics["by_street"]["preflop"]["n"] == 2
    record = metrics["records"][0]
    assert record["hand_index"] == 1
    assert record["bot_hole_cards"] == ["Qs", "Qd"]
    assert 0.0 <= record["action_prob"] <= 1.0
    assert record["uniform_prob"] > 0.0
    assert "log_lift_vs_uniform" in record


def test_diagnose_slumbot_trace_action_likelihood_cli_writes_metrics(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "metrics.json"
    _small_checkpoint(checkpoint)
    _write_trace(trace)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "diagnose_slumbot_trace_action_likelihood.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--trace",
            str(trace),
            "--strategy-source",
            "regret",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "slumbot_trace_opponent_action_likelihood"
    assert payload["n_scored_actions"] == 2
