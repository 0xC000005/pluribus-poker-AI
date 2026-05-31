import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from poker_ai.research.trace_start_policy import (
    split_trace_records_by_source,
    split_trace_records_by_hand,
    value_soft_target,
)


def test_value_soft_target_masks_illegal_actions_and_prefers_high_value():
    action_values = np.asarray([0.0, 10.0, -5.0, 20.0], dtype=np.float32)

    target = value_soft_target(action_values, legal_actions=[1, 3], temperature=5.0)

    assert target.shape == action_values.shape
    assert target[0] == 0.0
    assert target[2] == 0.0
    assert target[3] > target[1]
    np.testing.assert_allclose(target.sum(), 1.0)


def test_split_trace_records_by_hand_keeps_same_hand_together():
    records = [
        {"label": "trace-hand1-decision1-street2"},
        {"label": "trace-hand1-decision2-street3"},
        {"label": "trace-hand2-decision1-street2"},
        {"label": "trace-hand3-decision1-street3"},
    ]

    train, holdout = split_trace_records_by_hand(records, holdout_fraction=0.5, seed=7)

    train_hands = {record["label"].split("-")[1] for record in train}
    holdout_hands = {record["label"].split("-")[1] for record in holdout}
    assert train
    assert holdout
    assert train_hands.isdisjoint(holdout_hands)


def test_split_trace_records_by_source_holds_out_named_trace():
    records = [
        {"label": "a", "_trace_source": "train.json"},
        {"label": "b", "_trace_source": "holdout.json"},
        {"label": "c", "_trace_source": "train.json"},
    ]

    train, holdout = split_trace_records_by_source(records, holdout_source="holdout.json")

    assert [record["label"] for record in train] == ["a", "c"]
    assert [record["label"] for record in holdout] == ["b"]


def test_train_trace_start_policy_cli_writes_metrics(tmp_path):
    ev_gate = tmp_path / "ev_gate.json"
    output_json = tmp_path / "metrics.json"
    records = []
    for hand_idx, action_value in [(1, 50.0), (2, 75.0), (3, 100.0)]:
        records.append(
            {
                "label": f"trace-hand{hand_idx}-decision1-street2",
                "hole_cards": ["Kc", "Js"],
                "board": ["Jc", "5s", "3d", "9d"],
                "action_str": "b200c/b100c/",
                "client_pos": 0,
                "source": "slumbot_trace",
                "legal_actions": [1, 2],
                "counterfactual_action_values": [0.0, action_value, -25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "baseline_action": 2,
            }
        )
    ev_gate.write_text(json.dumps({"records": records}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_trace_start_policy_head.py",
            "--ev-gate-json",
            str(ev_gate),
            "--output-json",
            str(output_json),
            "--n-steps",
            "2",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--seed",
            "13",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    assert metrics["mode"] == "trace_start_policy_head_probe"
    assert metrics["passed"] is True
    assert metrics["n_records"] == 3
    assert metrics["feature_context_dim"] == 127
    assert "holdout_mean_selected_ev_delta_vs_baseline" in metrics
    assert "mean_oracle_ev_delta_vs_baseline" in metrics["holdout_eval"]
    assert json.loads(output_json.read_text(encoding="utf-8")) == metrics
