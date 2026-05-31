import json
import subprocess
import sys
from pathlib import Path

from poker_ai.research.slumbot_response_decision_gate import (
    evaluate_response_decision_gate,
)


def _record(idx: int, delta: float, drift: float) -> dict:
    response_strategy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, drift, 1.0 - drift]
    baseline_strategy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - drift, drift]
    return {
        "label": f"case-{idx}",
        "passed": True,
        "street": idx % 4,
        "client_pos": idx % 2,
        "legal_actions": [1, 7, 8],
        "n_opponent_updates": 2 + (idx % 3),
        "action_l1_drift": drift,
        "action_agreement": drift < 0.5,
        "baseline_action": 8,
        "response_action": 7 if drift > 0.5 else 8,
        "baseline_strategy": baseline_strategy,
        "response_strategy": response_strategy,
        "baseline_villain_range_normalized_entropy": 0.8 - 0.1 * drift,
        "baseline_villain_range_top10_mass": 0.2 + 0.1 * drift,
        "baseline_villain_range_top1_mass": 0.05 + 0.01 * drift,
        "response_villain_range_normalized_entropy": 0.9 - 0.2 * drift,
        "response_villain_range_top10_mass": 0.1 + 0.2 * drift,
        "response_villain_range_top1_mass": 0.02 + 0.02 * drift,
        "baseline_latency_ms": 100.0 + idx,
        "response_latency_ms": 120.0 + idx,
        "strategy_ev_delta": delta,
        "selected_action_ev_delta": delta * 0.5,
    }


def _write_ev_artifact(path: Path, *, invert: bool = False) -> None:
    records = []
    for idx, drift in enumerate([0.1, 0.2, 0.8, 0.9, 0.15, 0.85]):
        sign = 1.0 if drift > 0.5 else -1.0
        if invert:
            sign *= -1.0
        records.append(_record(idx, delta=10.0 * sign, drift=drift))
    path.write_text(json.dumps({"mode": "response_ev_gate", "records": records}), encoding="utf-8")


def test_evaluate_response_decision_gate_uses_observable_features(tmp_path):
    train_artifact = tmp_path / "train_ev.json"
    eval_artifact = tmp_path / "eval_ev.json"
    _write_ev_artifact(train_artifact)
    _write_ev_artifact(eval_artifact)

    metrics = evaluate_response_decision_gate(train_artifact, eval_artifact, ridge_l2=0.01)

    assert metrics["mode"] == "slumbot_response_decision_gate"
    assert metrics["train_n"] == 6
    assert metrics["eval_n"] == 6
    assert metrics["selected_count"] == 3
    assert metrics["selective_mean_strategy_ev_delta"] > metrics["direct_mean_strategy_ev_delta"]
    assert metrics["selective_mean_strategy_ev_delta"] > 0
    assert metrics["eval_selected_prediction_pearson"] > 0.9
    assert metrics["decision_gate_passed"] is True
    assert "response_true_hand_prob" not in metrics["feature_names"]
    assert "counterfactual_action_values" not in metrics["feature_names"]


def test_decision_gate_rejects_negative_selected_action_transfer(tmp_path):
    train_artifact = tmp_path / "train_ev.json"
    eval_artifact = tmp_path / "eval_ev.json"
    _write_ev_artifact(train_artifact)
    records = []
    for idx, drift in enumerate([0.1, 0.2, 0.8, 0.9]):
        strategy_delta = 20.0 if drift > 0.5 else -10.0
        record = _record(idx, delta=strategy_delta, drift=drift)
        record["selected_action_ev_delta"] = -20.0 if drift > 0.5 else 0.0
        records.append(record)
    eval_artifact.write_text(
        json.dumps({"mode": "response_ev_gate", "records": records}),
        encoding="utf-8",
    )

    metrics = evaluate_response_decision_gate(train_artifact, eval_artifact, ridge_l2=0.01)

    assert metrics["selective_mean_strategy_ev_delta"] > metrics["direct_mean_strategy_ev_delta"]
    assert metrics["selective_mean_selected_action_ev_delta"] < 0
    assert metrics["decision_gate_passed"] is False


def test_evaluate_response_decision_gate_cli_writes_metrics(tmp_path):
    train_artifact = tmp_path / "train_ev.json"
    eval_artifact = tmp_path / "eval_ev.json"
    output = tmp_path / "decision_gate.json"
    _write_ev_artifact(train_artifact)
    _write_ev_artifact(eval_artifact)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval_slumbot_response_decision_gate.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--train-ev-gate",
            str(train_artifact),
            "--eval-ev-gate",
            str(eval_artifact),
            "--output",
            str(output),
            "--ridge-l2",
            "0.01",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "slumbot_response_decision_gate"
    assert payload["selected_count"] == 3
