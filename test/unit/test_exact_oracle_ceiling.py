import json

from poker_ai.research.exact_oracle_ceiling import (
    summarize_exact_oracle_ceiling_records,
)
from scripts import eval_exact_oracle_ceiling as cli


def test_summarize_exact_oracle_ceiling_records_reports_positive_gap():
    records = [
        {
            "root_idx": 0,
            "parent_action": "call",
            "oracle_action": "raise_1x",
            "parent_action_value": 0.10,
            "oracle_action_value": 0.35,
            "n_truncated_rollouts": 0,
        },
        {
            "root_idx": 1,
            "parent_action": "call",
            "oracle_action": "call",
            "parent_action_value": 0.20,
            "oracle_action_value": 0.20,
            "n_truncated_rollouts": 0,
        },
    ]

    summary = summarize_exact_oracle_ceiling_records(
        records,
        min_roots=2,
        min_mean_oracle_gap=0.05,
    )

    assert summary["passed"] is True
    assert summary["ceiling_positive"] is True
    assert summary["promotable"] is False
    assert summary["n_roots"] == 2
    assert summary["oracle_match_rate"] == 0.5
    assert summary["mean_parent_action_value"] == 0.15
    assert summary["mean_oracle_action_value"] == 0.275
    assert summary["mean_oracle_gap"] == 0.125
    assert summary["parent_action_counts"] == {"call": 2}
    assert summary["oracle_action_counts"] == {"call": 1, "raise_1x": 1}


def test_summarize_exact_oracle_ceiling_records_blocks_truncated_or_small_samples():
    records = [
        {
            "root_idx": 0,
            "parent_action": "call",
            "oracle_action": "raise_1x",
            "parent_action_value": 0.10,
            "oracle_action_value": 0.35,
            "n_truncated_rollouts": 1,
        }
    ]

    summary = summarize_exact_oracle_ceiling_records(
        records,
        min_roots=2,
        min_mean_oracle_gap=0.05,
    )

    assert summary["passed"] is False
    assert summary["ceiling_positive"] is True
    assert "insufficient_roots" in summary["blockers"]
    assert "truncated_rollouts" in summary["blockers"]


def test_exact_oracle_ceiling_cli_summarizes_record_file(tmp_path, capsys):
    records_path = tmp_path / "records.json"
    output_path = tmp_path / "summary.json"
    records_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "root_idx": 0,
                        "parent_action": "call",
                        "oracle_action": "raise_1x",
                        "parent_action_value": 0.10,
                        "oracle_action_value": 0.35,
                        "n_truncated_rollouts": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--records-json",
            str(records_path),
            "--min-roots",
            "1",
            "--min-mean-oracle-gap",
            "0.05",
            "--output-json",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "native_exact_oracle_ceiling"
    assert payload["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True
