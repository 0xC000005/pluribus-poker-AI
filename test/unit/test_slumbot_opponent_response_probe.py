import json
import subprocess
import sys
from pathlib import Path

from poker_ai.research.slumbot_opponent_response_probe import (
    load_opponent_response_dataset,
    train_opponent_response_probe,
)


def _write_action_likelihood(path: Path) -> None:
    records = []
    hands = [
        (1, ["Qs", "Qd"], "b200", 3),
        (2, ["Jh", "Td"], "b250", 1),
        (3, ["9s", "8s"], "b200", 3),
        (4, ["7h", "7d"], "b250", 1),
        (5, ["Ad", "Kh"], "b200", 3),
        (6, ["6s", "5s"], "b250", 1),
    ]
    for hand_index, bot_cards, before, action_idx in hands:
        records.append(
            {
                "hand_index": hand_index,
                "client_pos": 0,
                "hole_cards": ["Ac", "Kd"],
                "bot_hole_cards": bot_cards,
                "board": ["2c", "7d", "Jh"],
                "visible_board": [],
                "action_str_before": before,
                "action_char": "b" if action_idx == 3 else "c",
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


def test_load_opponent_response_dataset_builds_features_and_targets(tmp_path):
    artifact = tmp_path / "action_likelihood.json"
    _write_action_likelihood(artifact)

    dataset = load_opponent_response_dataset(artifact)

    assert dataset.features.shape[0] == 6
    assert dataset.features.shape[1] > 0
    assert dataset.legal_masks.shape == dataset.target_probs.shape
    assert dataset.target_probs[0, 3] == 1.0
    assert dataset.hand_indices.tolist() == [1, 2, 3, 4, 5, 6]


def test_train_opponent_response_probe_reports_heldout_metrics(tmp_path):
    artifact = tmp_path / "action_likelihood.json"
    _write_action_likelihood(artifact)

    metrics = train_opponent_response_probe(
        artifact,
        holdout_fraction=0.5,
        hidden_dim=16,
        n_layers=1,
        epochs=5,
        batch_size=4,
        device="cpu",
        seed=7,
    )

    assert metrics["mode"] == "slumbot_opponent_response_probe"
    assert metrics["train"]["n"] > 0
    assert metrics["holdout"]["n"] > 0
    assert metrics["temperature"] > 0.0
    assert "calibrated_holdout" in metrics
    assert "probe_mean_log_lift_vs_uniform" in metrics["holdout"]
    assert "probe_mean_log_lift_vs_uniform" in metrics["calibrated_holdout"]
    assert "model_mean_log_lift_vs_uniform" in metrics["holdout"]


def test_train_slumbot_opponent_response_probe_cli_writes_metrics(tmp_path):
    artifact = tmp_path / "action_likelihood.json"
    output = tmp_path / "probe.json"
    _write_action_likelihood(artifact)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "train_slumbot_opponent_response_probe.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--action-likelihood",
            str(artifact),
            "--output",
            str(output),
            "--epochs",
            "5",
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
    assert payload["mode"] == "slumbot_opponent_response_probe"
    assert payload["holdout"]["n"] > 0
