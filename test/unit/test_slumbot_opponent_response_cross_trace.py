import json
import subprocess
import sys
from pathlib import Path

from poker_ai.research.slumbot_opponent_response_cross_trace import (
    train_and_eval_cross_trace_response_probe,
)


def _write_action_likelihood(path: Path, offset: int = 0) -> None:
    records = []
    hands = [
        (1 + offset, ["Qs", "Qd"], 3),
        (2 + offset, ["Jh", "Td"], 3),
        (3 + offset, ["9s", "8s"], 3),
        (4 + offset, ["7h", "7d"], 1),
        (5 + offset, ["6s", "5s"], 1),
        (6 + offset, ["4h", "4d"], 1),
    ]
    for hand_index, bot_cards, action_idx in hands:
        records.append(
            {
                "hand_index": hand_index,
                "client_pos": 0,
                "hole_cards": ["Ac", "Kd"],
                "bot_hole_cards": bot_cards,
                "board": [],
                "visible_board": [],
                "action_str_before": "",
                "action_char": "b" if action_idx != 1 else "c",
                "bet_to": 200 if action_idx != 1 else 0,
                "mapped_actions": [{"action_idx": action_idx, "weight": 1.0}],
                "action_prob": 0.1,
                "uniform_prob": 0.125,
                "log_lift_vs_uniform": -0.2,
            }
        )
    path.write_text(
        json.dumps({"mode": "slumbot_trace_opponent_action_likelihood", "records": records}),
        encoding="utf-8",
    )


def test_train_and_eval_cross_trace_response_probe_reports_external_metrics(tmp_path):
    train_artifact = tmp_path / "train_action.json"
    eval_artifact = tmp_path / "eval_action.json"
    _write_action_likelihood(train_artifact, offset=0)
    _write_action_likelihood(eval_artifact, offset=100)

    metrics = train_and_eval_cross_trace_response_probe(
        train_artifact,
        eval_artifact,
        holdout_fraction=0.5,
        hidden_dim=16,
        n_layers=1,
        epochs=3,
        batch_size=4,
        device="cpu",
        seed=7,
    )

    assert metrics["mode"] == "slumbot_opponent_response_cross_trace"
    assert metrics["source_train"]["n"] > 0
    assert metrics["source_calibration"]["n"] > 0
    assert metrics["external_eval"]["n"] == 6
    assert "probe_mean_log_lift_vs_uniform" in metrics["external_eval"]
    assert "calibrated_external_eval" in metrics


def test_eval_slumbot_opponent_response_cross_trace_cli_writes_metrics(tmp_path):
    train_artifact = tmp_path / "train_action.json"
    eval_artifact = tmp_path / "eval_action.json"
    output = tmp_path / "cross_trace.json"
    _write_action_likelihood(train_artifact, offset=0)
    _write_action_likelihood(eval_artifact, offset=100)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval_slumbot_opponent_response_cross_trace.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--train-action-likelihood",
            str(train_artifact),
            "--eval-action-likelihood",
            str(eval_artifact),
            "--output",
            str(output),
            "--holdout-fraction",
            "0.5",
            "--epochs",
            "3",
            "--hidden-dim",
            "16",
            "--n-layers",
            "1",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "slumbot_opponent_response_cross_trace"
    assert payload["external_eval"]["n"] == 6
