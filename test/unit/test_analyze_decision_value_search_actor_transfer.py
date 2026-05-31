import json
import subprocess
import sys
from pathlib import Path


def test_analyze_decision_value_search_actor_transfer_reports_pair_correlation(tmp_path):
    input_json = tmp_path / "attribution.json"
    output_json = tmp_path / "analysis.json"
    input_json.write_text(
        json.dumps(
            {
                "mode": "decision_value_search_actor_h2h",
                "paired_deltas": [10.0, -5.0, 6.0],
                "decision_records": [
                    {
                        "duplicate_pair": 0,
                        "deployed_action": "call",
                        "search_action": "raise_1.0",
                        "local_search_minus_deployed_value": 7.0,
                    },
                    {
                        "duplicate_pair": 0,
                        "deployed_action": "call",
                        "search_action": "call",
                        "local_search_minus_deployed_value": 1.0,
                    },
                    {
                        "duplicate_pair": 1,
                        "deployed_action": "all_in",
                        "search_action": "fold",
                        "local_search_minus_deployed_value": -3.0,
                    },
                    {
                        "duplicate_pair": 2,
                        "deployed_action": "fold",
                        "search_action": "call",
                        "local_search_minus_deployed_value": 5.0,
                    },
                ],
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_decision_value_search_actor_transfer.py",
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    saved = json.loads(output_json.read_text())
    assert metrics["mode"] == "decision_value_search_actor_transfer_analysis"
    assert saved["n_pairs_with_decision_records"] == 3
    assert metrics["all_decisions"]["pair_local_delta_sum_to_h2h_delta_pearson"] > 0.9
    assert metrics["disagreement_decisions"]["n_decisions"] == 3
    assert metrics["promotable"] is False
