import json
import subprocess
import sys
from pathlib import Path

from poker_ai.research.slumbot_response_latent_conditioner import (
    evaluate_latent_response_conditioner,
)


def _write_ev_gate(path: Path, *, invert: bool = False) -> None:
    records = []
    for idx in range(40):
        high_drift = idx % 4 in (0, 1)
        action_changed = idx % 4 in (0, 2)
        positive = high_drift == action_changed
        if invert:
            positive = not positive
        strategy_delta = 4.0 if positive else -3.0
        selected_delta = 2.0 if positive else -2.0
        records.append(
            {
                "passed": True,
                "strategy_ev_delta": strategy_delta,
                "selected_action_ev_delta": selected_delta,
                "street": idx % 4,
                "client_pos": idx % 2,
                "legal_actions": [1, 2, 3, 4],
                "n_opponent_updates": idx % 3,
                "action_l1_drift": 0.9 if high_drift else 0.1,
                "action_agreement": not action_changed,
                "baseline_action": 1,
                "response_action": 3 if action_changed else 1,
                "baseline_strategy": [0.0, 0.7, 0.2, 0.1, 0, 0, 0, 0, 0],
                "response_strategy": [0.0, 0.1, 0.2, 0.7, 0, 0, 0, 0, 0]
                if action_changed
                else [0.0, 0.68, 0.22, 0.1, 0, 0, 0, 0, 0],
                "baseline_latency_ms": 100.0,
                "response_latency_ms": 120.0 if high_drift else 90.0,
                "baseline_villain_range_normalized_entropy": 0.4,
                "response_villain_range_normalized_entropy": 0.8 if high_drift else 0.3,
                "baseline_villain_range_top10_mass": 0.5,
                "response_villain_range_top10_mass": 0.2 if high_drift else 0.6,
                "baseline_villain_range_top1_mass": 0.2,
                "response_villain_range_top1_mass": 0.1 if high_drift else 0.3,
            }
        )
    path.write_text(
        json.dumps({"mode": "slumbot_response_range_counterfactual_ev_gate", "records": records}),
        encoding="utf-8",
    )


def test_latent_response_conditioner_reports_external_decision_metrics(tmp_path):
    train_ev = tmp_path / "train.json"
    eval_ev = tmp_path / "eval.json"
    _write_ev_gate(train_ev)
    _write_ev_gate(eval_ev)

    metrics = evaluate_latent_response_conditioner(
        train_ev,
        eval_ev,
        hidden_dim=16,
        epochs=120,
        batch_size=16,
        device="cpu",
        seed=7,
    )

    assert metrics["mode"] == "slumbot_response_latent_conditioner"
    assert metrics["eval_n"] == 40
    assert metrics["decision_gate_passed"]
    assert metrics["eval"]["selective_mean_selected_action_ev_delta"] > metrics["eval"][
        "direct_mean_selected_action_ev_delta"
    ]
    assert "counterfactual_action_values" not in metrics["feature_names"]
    assert "baseline_log_lift_vs_uniform" not in metrics["feature_names"]


def test_eval_slumbot_response_latent_conditioner_cli_writes_metrics(tmp_path):
    train_ev = tmp_path / "train.json"
    eval_ev = tmp_path / "eval.json"
    output = tmp_path / "metrics.json"
    _write_ev_gate(train_ev)
    _write_ev_gate(eval_ev)

    script = Path(__file__).resolve().parents[2] / "scripts" / "eval_slumbot_response_latent_conditioner.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--train-ev-gate",
            str(train_ev),
            "--eval-ev-gate",
            str(eval_ev),
            "--output",
            str(output),
            "--hidden-dim",
            "16",
            "--epochs",
            "120",
            "--batch-size",
            "16",
            "--device",
            "cpu",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "slumbot_response_latent_conditioner"
    assert payload["decision_gate_passed"]
