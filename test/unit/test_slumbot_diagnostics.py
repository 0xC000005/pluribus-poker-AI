import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import ActionDiagnostics, action_to_slumbot, parse_action


def test_action_diagnostics_records_policy_mapping_drift():
    diagnostics = ActionDiagnostics()
    action_str = ""
    parsed = parse_action(action_str)
    incr = action_to_slumbot(3, parsed, action_str, client_pos=1)

    diagnostics.record_policy_action(3, incr, action_str, client_pos=1, parsed=parsed)

    summary = diagnostics.as_summary()
    assert summary["decision_total"] == 1
    assert summary["decision_policy"] == 1
    assert summary["action_mix"]["r0.5x"] == 1
    assert summary["increment_mix"]["b"] == 1
    assert summary["mapping_drift_n"] == 1
    assert 0.0 <= summary["mapping_drift_mean"] <= 1.0


def test_action_diagnostics_records_fallback_and_parse_error():
    diagnostics = ActionDiagnostics()

    diagnostics.record_fallback("c")
    diagnostics.record_parse_error()

    summary = diagnostics.as_summary()
    assert summary["decision_total"] == 1
    assert summary["decision_fallback"] == 1
    assert summary["parse_errors"] == 1
    assert summary["increment_mix"]["c"] == 1
