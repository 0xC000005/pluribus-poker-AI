import json
import subprocess
import sys
from pathlib import Path

from poker_ai.research.slumbot_opponent_response_baseline import (
    analyze_opponent_response_baseline,
)


def _write_action_likelihood(path: Path) -> None:
    payload = {
        "mode": "slumbot_trace_opponent_action_likelihood",
        "records": [
            {
                "hand_index": 1,
                "client_pos": 0,
                "street": "preflop",
                "action_str_before": "",
                "action_char": "b",
                "mapped_actions": [{"action_idx": 8, "weight": 1.0}],
                "action_prob": 0.80,
                "uniform_prob": 0.125,
                "log_lift_vs_uniform": 1.85,
            },
            {
                "hand_index": 1,
                "client_pos": 0,
                "street": "preflop",
                "action_str_before": "",
                "action_char": "b",
                "mapped_actions": [{"action_idx": 8, "weight": 1.0}],
                "action_prob": 0.70,
                "uniform_prob": 0.125,
                "log_lift_vs_uniform": 1.72,
            },
            {
                "hand_index": 2,
                "client_pos": 0,
                "street": "preflop",
                "action_str_before": "",
                "action_char": "c",
                "mapped_actions": [{"action_idx": 1, "weight": 1.0}],
                "action_prob": 0.05,
                "uniform_prob": 0.125,
                "log_lift_vs_uniform": -0.92,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_opponent_response_baseline_excludes_same_hand_examples(tmp_path):
    artifact = tmp_path / "action_likelihood.json"
    _write_action_likelihood(artifact)

    metrics = analyze_opponent_response_baseline(artifact, alpha=0.25)

    assert metrics["mode"] == "slumbot_opponent_response_baseline"
    assert metrics["n_records"] == 3
    assert metrics["n_hands"] == 2
    first = metrics["records"][0]
    assert first["global_loo_prob"] < first["uniform_prob"]
    assert metrics["aggregates"]["model"]["mean_log_lift_vs_uniform"] > 0.0
    assert "global_loo" in metrics["aggregates"]


def test_analyze_slumbot_opponent_response_baseline_cli_writes_metrics(tmp_path):
    artifact = tmp_path / "action_likelihood.json"
    output = tmp_path / "baseline.json"
    _write_action_likelihood(artifact)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "analyze_slumbot_opponent_response_baseline.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--action-likelihood",
            str(artifact),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "slumbot_opponent_response_baseline"
    assert payload["n_records"] == 3
