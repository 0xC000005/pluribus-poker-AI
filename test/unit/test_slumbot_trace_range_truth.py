import json
import subprocess
import sys
from pathlib import Path

import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.research.slumbot_trace_range_truth import diagnose_trace_true_range


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


def test_diagnose_trace_true_range_scores_revealed_slumbot_hand(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _small_checkpoint(checkpoint)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "decision",
                        "source": "policy",
                        "hand_index": 1,
                        "client_pos": 0,
                        "hole_cards": ["Ac", "Kd"],
                        "board": ["2c", "7d", "Jh", "4s"],
                        "action_str": "ck/kk/",
                        "street_index": 2,
                    }
                ),
                json.dumps(
                    {
                        "event": "hand_result",
                        "hand_index": 1,
                        "client_pos": 0,
                        "hole_cards": ["Ac", "Kd"],
                        "board": ["2c", "7d", "Jh", "4s", "9c"],
                        "bot_hole_cards": ["Qs", "Qd"],
                        "winnings": -100,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = diagnose_trace_true_range(
        checkpoint,
        trace,
        strategy_source="regret",
        device="cpu",
    )

    assert metrics["passed"] is True
    assert metrics["n_scored"] == 1
    record = metrics["records"][0]
    assert record["hand_index"] == 1
    assert record["bot_hole_cards"] == ["Qs", "Qd"]
    assert record["true_hand_prob"] > 0.0
    assert record["uniform_prob"] > 0.0
    assert "log_lift_vs_uniform" in record
    assert "true_hand_percentile" in record
    assert metrics["by_street"]["turn"]["n"] == 1
    assert metrics["outcome_buckets"]["loss"]["n"] == 1


def test_diagnose_slumbot_trace_range_truth_cli_writes_metrics(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _small_checkpoint(checkpoint)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "decision",
                        "source": "policy",
                        "hand_index": 2,
                        "client_pos": 0,
                        "hole_cards": ["Ac", "Kd"],
                        "board": ["2c", "7d", "Jh", "4s"],
                        "action_str": "ck/kk/",
                        "street_index": 2,
                    }
                ),
                json.dumps(
                    {
                        "event": "hand_result",
                        "hand_index": 2,
                        "client_pos": 0,
                        "hole_cards": ["Ac", "Kd"],
                        "board": ["2c", "7d", "Jh", "4s", "9c"],
                        "bot_hole_cards": ["Qs", "Qd"],
                        "winnings": -100,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"
    script = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_slumbot_trace_range_truth.py"

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
    assert payload["mode"] == "slumbot_trace_true_range_likelihood"
    assert payload["n_scored"] == 1
    assert payload["records"][0]["hand_index"] == 2
    assert payload["by_street"]["turn"]["n"] == 1
