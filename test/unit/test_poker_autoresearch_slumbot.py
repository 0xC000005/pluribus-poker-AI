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


def test_parse_slumbot_summary_extracts_action_diagnostics():
    output = """
============================================================
FINAL: 2 hands | -100 chips
  Avg: -50 +/- 150 chips/hand
  Rate: -500 mbb/hand
  Win rate: 50.0%
  Decisions: total=4 policy=3 solver=1 fallback=0 parse_errors=0
  Action mix: fold=0 call/chk=2 r0.25x=1 r0.5x=0 r0.75x=0 r1.0x=0 r1.5x=0 r2.0x=0 all-in=0 solver=1
  Increments: f=0 k=1 c=1 b=2
  Mapping drift: n=3 mean=0.125 max=0.250
============================================================
"""

    metrics = parse_slumbot_summary(output)

    assert metrics["decision_total"] == 4
    assert metrics["decision_policy"] == 3
    assert metrics["decision_solver"] == 1
    assert metrics["decision_fallback"] == 0
    assert metrics["parse_errors"] == 0
    assert metrics["action_mix"]["call/chk"] == 2
    assert metrics["action_mix"]["r0.25x"] == 1
    assert metrics["increment_mix"] == {"f": 0, "k": 1, "c": 1, "b": 2}
    assert metrics["mapping_drift_n"] == 3
    assert metrics["mapping_drift_mean"] == 0.125
    assert metrics["mapping_drift_max"] == 0.25
