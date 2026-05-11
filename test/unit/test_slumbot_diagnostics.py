import sys
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import ActionDiagnostics, _base_policy_action, action_to_slumbot, parse_action


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


class _PolicyHeadProbeNet(torch.nn.Module):
    def forward(self, features):
        advantages = torch.zeros((features.shape[0], 9), dtype=torch.float32)
        advantages[:, 8] = 10.0
        return advantages

    def forward_with_policy(self, features):
        advantages = self.forward(features)
        logits = torch.zeros_like(advantages)
        logits[:, 1] = 10.0
        return advantages, logits


def test_base_policy_action_can_use_policy_head_instead_of_regret_matching():
    parsed = parse_action("")
    net = _PolicyHeadProbeNet()

    regret_incr = _base_policy_action(
        ["Ac", "Kd"],
        [],
        "",
        1,
        parsed,
        net,
        torch.device("cpu"),
        greedy=True,
        no_allin=False,
        verbose=False,
        strategy_source="regret",
    )
    policy_incr = _base_policy_action(
        ["Ac", "Kd"],
        [],
        "",
        1,
        parsed,
        net,
        torch.device("cpu"),
        greedy=True,
        no_allin=False,
        verbose=False,
        strategy_source="policy-head",
    )

    assert regret_incr.startswith("b")
    assert policy_incr == "c"
