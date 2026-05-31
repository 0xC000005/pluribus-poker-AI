import json

import numpy as np

from poker_ai.research.exact_response_oracle import (
    mixture_policy_probs,
    summarize_exact_response_h2h,
)
from scripts import eval_exact_response_oracle_h2h as cli


class _StaticAdapter:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32)

    def probs(self, features, legal_mask, device):
        del features, device
        masked = self._probs * np.asarray(legal_mask, dtype=np.float32)
        total = float(masked.sum())
        return masked / total


def test_mixture_policy_probs_renormalizes_legal_average():
    features = np.zeros(4, dtype=np.float32)
    legal_mask = np.array([1, 1, 0, 1], dtype=np.float32)
    adapters = [
        _StaticAdapter([0.6, 0.4, 0.0, 0.0]),
        _StaticAdapter([0.0, 0.4, 0.0, 0.6]),
    ]

    probs = mixture_policy_probs(adapters, features, legal_mask, device=None)

    assert probs.shape == (4,)
    assert np.isclose(float(probs.sum()), 1.0)
    assert probs[2] == 0.0
    assert probs[0] > 0.0
    assert probs[1] > probs[0]
    assert probs[3] > 0.0


def test_summarize_exact_response_h2h_reports_confidence_gate():
    hand_payoffs = [0.20, 0.15, 0.10, 0.05]
    decision_records = [
        {
            "street": "preflop",
            "baseline_action": "all_in",
            "response_action": "call",
            "local_response_minus_baseline_value": 100.0,
            "n_truncated_rollouts": 0,
        },
        {
            "street": "flop",
            "baseline_action": "call",
            "response_action": "call",
            "local_response_minus_baseline_value": 0.0,
            "n_truncated_rollouts": 0,
        },
    ]

    summary = summarize_exact_response_h2h(
        hand_payoffs,
        decision_records,
        min_games=4,
        min_lower95_response_payoff=0.0,
    )

    assert summary["mode"] == "exact_restricted_response_h2h"
    assert summary["passed"] is True
    assert summary["promotion"] is False
    assert summary["n_games"] == 4
    assert summary["mean_response_payoff"] == 0.125
    assert summary["lower95_response_payoff"] > 0.0
    assert summary["decision_summary"]["n_recorded"] == 2
    assert summary["decision_summary"]["disagreement_count"] == 1
    assert summary["decision_summary"]["by_street"]["preflop"]["count"] == 1


def test_summarize_exact_response_h2h_blocks_small_or_truncated_samples():
    summary = summarize_exact_response_h2h(
        [0.1],
        [
            {
                "street": "preflop",
                "baseline_action": "all_in",
                "response_action": "fold",
                "local_response_minus_baseline_value": 50.0,
                "n_truncated_rollouts": 2,
            }
        ],
        min_games=2,
        min_lower95_response_payoff=0.0,
    )

    assert summary["passed"] is False
    assert "insufficient_games" in summary["blockers"]
    assert "truncated_rollouts" in summary["blockers"]


def test_exact_response_cli_summarizes_record_file(tmp_path, capsys):
    records_path = tmp_path / "records.json"
    output_path = tmp_path / "summary.json"
    records_path.write_text(
        json.dumps(
            {
                "response_payoffs": [0.2, 0.1, 0.15, 0.05],
                "decision_records": [],
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--records-json",
            str(records_path),
            "--min-games",
            "4",
            "--min-lower95-response-payoff",
            "0.0",
            "--output-json",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "exact_restricted_response_h2h"
    assert payload["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True
