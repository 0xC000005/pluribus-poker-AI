from poker_ai.research.slumbot_eval import add_runtime_metrics, parse_slumbot_summary


def test_parse_slumbot_summary_extracts_final_metrics():
    output = """
============================================================
FINAL: 5 hands | -1265 chips
  Avg: -253 +/- 216 chips/hand
  Rate: -2530 mbb/hand
  Win rate: 0.0%
============================================================
"""

    metrics = parse_slumbot_summary(output)

    assert metrics["passed"] is True
    assert metrics["hands"] == 5
    assert metrics["total_chips"] == -1265
    assert metrics["avg_chips_per_hand"] == -253
    assert metrics["ci95_chips_per_hand"] == 216
    assert metrics["mbb_per_hand"] == -2530
    assert metrics["win_rate"] == 0.0


def test_add_runtime_metrics_records_seconds_per_hand():
    metrics = {"hands": 10, "passed": True}

    enriched = add_runtime_metrics(metrics, elapsed_seconds=85.0)

    assert enriched["elapsed_seconds"] == 85.0
    assert enriched["seconds_per_hand"] == 8.5
