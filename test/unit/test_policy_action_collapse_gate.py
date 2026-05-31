import json
import subprocess
import sys
from pathlib import Path

from poker_ai.research.action_collapse_gate import evaluate_action_collapse


def test_evaluate_action_collapse_rejects_single_action_policy():
    metrics = evaluate_action_collapse(
        {"selected_action_counts": {"raise_1.0": 32}},
        max_top_action_fraction=0.75,
        min_distinct_actions=2,
    )

    assert metrics["passed"] is False
    assert metrics["top_action"] == "raise_1.0"
    assert metrics["top_action_fraction"] == 1.0
    assert any("top_action_fraction" in failure for failure in metrics["failures"])


def test_evaluate_action_collapse_accepts_diverse_policy():
    metrics = evaluate_action_collapse(
        {"selected_action_counts": {"call": 12, "raise_1.0": 10, "fold": 10}},
        max_top_action_fraction=0.75,
        min_distinct_actions=2,
    )

    assert metrics["passed"] is True
    assert metrics["distinct_actions"] == 3


def test_policy_action_collapse_gate_cli_writes_metrics(tmp_path):
    root_metrics = tmp_path / "root_rollout.json"
    output = tmp_path / "collapse_gate.json"
    root_metrics.write_text(
        json.dumps({"selected_action_counts": {"raise_1.0": 8}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_policy_action_collapse_gate.py",
            "--input-json",
            str(root_metrics),
            "--max-top-action-fraction",
            "0.75",
            "--min-distinct-actions",
            "2",
            "--output-json",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert json.loads(output.read_text())["passed"] is False
