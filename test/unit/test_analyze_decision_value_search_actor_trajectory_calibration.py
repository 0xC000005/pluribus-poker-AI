import json
import subprocess
import sys
from pathlib import Path


def _record(pair, local, *, street="flop", deployed="call", search="raise_1.0", pot=100.0):
    return {
        "duplicate_pair": pair,
        "deployed_action": deployed,
        "search_action": search,
        "local_search_minus_deployed_value": float(local),
        "street": street,
        "pot_total": float(pot),
        "to_call": 25.0,
    }


def test_trajectory_calibration_learns_holdout_pair_deltas(tmp_path):
    input_json = tmp_path / "h2h.json"
    output_json = tmp_path / "calibration.json"
    paired_deltas = [float(8 * idx - 20) for idx in range(10)]
    records = [
        _record(idx, delta / 8.0, street=("turn" if idx % 2 else "flop"), pot=100 + idx)
        for idx, delta in enumerate(paired_deltas)
    ]
    input_json.write_text(
        json.dumps(
            {
                "mode": "decision_value_search_actor_h2h",
                "paired_deltas": paired_deltas,
                "decision_records": records,
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_decision_value_search_actor_trajectory_calibration.py",
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--train-fraction",
            "0.6",
            "--ridge-alpha",
            "0.01",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    saved = json.loads(output_json.read_text())
    assert metrics["mode"] == "decision_value_search_actor_trajectory_calibration"
    assert saved["n_train_pairs"] == 6
    assert metrics["n_holdout_pairs"] == 4
    assert metrics["trajectory_calibration"]["holdout_pearson"] > 0.95
    assert metrics["trajectory_calibration"]["holdout_sign_agreement"] >= 0.75
    assert metrics["promotable"] is False


def test_trajectory_calibration_marks_tiny_inputs_not_calibrated(tmp_path):
    input_json = tmp_path / "tiny.json"
    input_json.write_text(
        json.dumps(
            {
                "mode": "decision_value_search_actor_h2h",
                "paired_deltas": [10.0, -3.0],
                "decision_records": [
                    _record(0, 1.0),
                    _record(1, -1.0),
                ],
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_decision_value_search_actor_trajectory_calibration.py",
            "--input-json",
            str(input_json),
            "--min-holdout-pairs",
            "3",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["trajectory_calibration_passed"] is False
    assert "insufficient_holdout_pairs" in metrics["calibration_blockers"]
