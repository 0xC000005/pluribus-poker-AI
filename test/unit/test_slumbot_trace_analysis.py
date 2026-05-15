import json

from poker_ai.research.slumbot_trace_analysis import analyze_trace


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_analyze_trace_splits_risky_and_non_risky_hands(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "event": "decision",
                "source": "policy",
                "hand_index": 1,
                "action_idx": 8,
                "action_name": "all-in",
                "increment": "b20000",
                "last_bet_size": 0,
                "advantages": [0, 0.02, 0, 0, 0, 0, 0, 0.01, 0.021],
                "legal_mask": [0, 1, 0, 0, 0, 0, 0, 1, 1],
            },
            {
                "event": "hand_result",
                "hand_index": 1,
                "winnings": -20000,
                "hole_cards": ["Ah", "4s"],
            },
            {
                "event": "decision",
                "source": "policy",
                "hand_index": 2,
                "action_idx": 3,
                "action_name": "r0.5x",
                "increment": "b300",
                "last_bet_size": 0,
                "advantages": [0, 0.01, 0.02, 0.03, 0.01, 0, 0, 0, 0],
                "legal_mask": [0, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "event": "hand_result",
                "hand_index": 2,
                "winnings": 600,
                "hole_cards": ["Ks", "Kd"],
            },
            {
                "event": "decision",
                "source": "policy",
                "hand_index": 3,
                "action_idx": 1,
                "action_name": "call/chk",
                "increment": "c",
                "last_bet_size": 8000,
                "advantages": [0, 0.1, 0, 0, 0, 0, 0, 0, 0],
                "legal_mask": [1, 1, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "event": "hand_result",
                "hand_index": 3,
                "winnings": -9000,
                "hole_cards": ["Qh", "2c"],
            },
        ],
    )

    summary = analyze_trace(trace)

    assert summary["hands"] == 3
    assert summary["total_chips"] == -28400
    assert summary["policy_decisions"] == 3
    assert summary["risk_hands"]["n"] == 2
    assert summary["risk_hands"]["avg_chips"] == -14500.0
    assert summary["risk_hands"]["stack_losses"] == 1
    assert summary["no_risk_hands"]["n"] == 1
    assert summary["no_risk_hands"]["avg_chips"] == 600.0
    assert summary["low_margin_risk_hands"]["n"] == 1
    assert summary["allin_hands"]["n"] == 1
    assert summary["big_call_hands"]["n"] == 1
    assert summary["worst_hands"][0]["hand_index"] == 1
