import json
import sys
import subprocess
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (
    ActionDiagnostics,
    _base_policy_action,
    _solver_iterations_for_profile,
    api_new_hand,
    action_to_slumbot,
    network_strategy,
    parse_action,
)


def test_action_diagnostics_records_policy_mapping_drift():
    diagnostics = ActionDiagnostics()
    action_str = ""
    parsed = parse_action(action_str)
    incr = action_to_slumbot(3, parsed, action_str, client_pos=1)

    diagnostics.record_policy_action(
        3, incr, action_str, client_pos=1, parsed=parsed, street=0
    )

    summary = diagnostics.as_summary()
    assert summary["decision_total"] == 1
    assert summary["decision_policy"] == 1
    assert summary["action_mix"]["r0.5x"] == 1
    assert summary["increment_mix"]["b"] == 1
    assert summary["mapping_drift_n"] == 1
    assert 0.0 <= summary["mapping_drift_mean"] <= 1.0
    assert summary["street_decisions"]["preflop"] == 1
    assert summary["street_all_in"]["preflop"] == 0


def test_action_diagnostics_records_street_all_in_and_solver_counts():
    diagnostics = ActionDiagnostics()

    diagnostics.record_policy_action(
        8, "b20000", "", client_pos=1, parsed=parse_action(""), street=0
    )
    diagnostics.record_solver_action("b300", street=2, latency_ms=125.5)

    summary = diagnostics.as_summary()

    assert summary["street_decisions"]["preflop"] == 1
    assert summary["street_decisions"]["turn"] == 1
    assert summary["street_all_in"]["preflop"] == 1
    assert summary["street_solver"]["turn"] == 1


def test_action_diagnostics_records_first_policy_action_outcome():
    diagnostics = ActionDiagnostics()

    diagnostics.begin_hand()
    diagnostics.record_policy_action(
        4, "b500", "", client_pos=1, parsed=parse_action(""), street=0
    )
    diagnostics.record_policy_action(
        1, "c", "b500c/kk", client_pos=1, parsed=parse_action("b500c/kk"), street=2
    )
    diagnostics.end_hand(-250)

    summary = diagnostics.as_summary()

    assert summary["first_policy_outcomes"]["r0.75x"]["n"] == 1
    assert summary["first_policy_outcomes"]["r0.75x"]["avg_chips"] == -250


def test_action_diagnostics_writes_jsonl_trace(tmp_path):
    trace_path = tmp_path / "slumbot_trace.jsonl"
    diagnostics = ActionDiagnostics(trace_path=trace_path)

    diagnostics.begin_hand(hand_index=7, client_pos=1, hole_cards=["Ac", "Kd"])
    diagnostics.record_policy_action(
        4,
        "b500",
        "",
        client_pos=1,
        parsed=parse_action(""),
        street=0,
        board=["Ah", "7d", "2c"],
        legal_mask=[0, 1, 1, 1, 1, 1, 0, 0, 1],
        advantages=[-1.0, 0.2, 0.3, 0.4, 1.5, 0.1, -0.4, -0.5, 0.0],
        strategy=[0.0, 0.1, 0.1, 0.1, 0.5, 0.1, 0.0, 0.0, 0.1],
        strategy_source="regret",
    )
    diagnostics.record_solver_action(
        "b300",
        street=2,
        latency_ms=125.5,
        n_hands=20,
        full_n_hands=100,
        board=["Ah", "7d", "2c", "Ts"],
        action_str="b500c/k",
        solver_action_idx=7,
        strategy={1: 0.25, 7: 0.75},
    )
    diagnostics.end_hand(-250, board=["Ah", "7d", "2c", "Ts", "9h"], bot_hole_cards=["Qs", "Qd"])

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "decision",
        "decision",
        "hand_result",
    ]
    assert records[0]["source"] == "policy"
    assert records[0]["action_name"] == "r0.75x"
    assert records[0]["hand_index"] == 7
    assert records[0]["board"] == ["Ah", "7d", "2c"]
    assert records[0]["legal_mask"] == [0, 1, 1, 1, 1, 1, 0, 0, 1]
    assert records[0]["strategy_source"] == "regret"
    assert records[0]["strategy_probs"][4] == 0.5
    assert records[0]["advantages"][4] == 1.5
    assert records[1]["source"] == "solver"
    assert records[1]["solver_latency_ms"] == 125.5
    assert records[1]["board"] == ["Ah", "7d", "2c", "Ts"]
    assert records[1]["action_str"] == "b500c/k"
    assert records[1]["solver_action_idx"] == 7
    assert records[1]["solver_strategy"][7] == 0.75
    assert records[2]["winnings"] == -250
    assert records[2]["first_policy_action"] == "r0.75x"
    assert records[2]["board"] == ["Ah", "7d", "2c", "Ts", "9h"]
    assert records[2]["bot_hole_cards"] == ["Qs", "Qd"]


def test_action_diagnostics_records_fallback_and_parse_error():
    diagnostics = ActionDiagnostics()

    diagnostics.record_fallback("c")
    diagnostics.record_parse_error()

    summary = diagnostics.as_summary()
    assert summary["decision_total"] == 1
    assert summary["decision_fallback"] == 1
    assert summary["parse_errors"] == 1
    assert summary["increment_mix"]["c"] == 1


def test_action_diagnostics_records_solver_performance_stats():
    diagnostics = ActionDiagnostics()

    diagnostics.record_solver_action("b300", latency_ms=125.5, n_hands=20, full_n_hands=100)
    diagnostics.record_solver_action("k", cached=True)

    summary = diagnostics.as_summary()
    assert summary["decision_solver"] == 2
    assert summary["solver_latency_n"] == 1
    assert summary["solver_latency_mean_ms"] == 125.5
    assert summary["solver_cache_hits"] == 1
    assert summary["solver_mean_hands"] == 20.0
    assert summary["solver_mean_full_hands"] == 100.0
    assert summary["solver_mean_prune_ratio"] == 0.2


def test_api_new_hand_uses_explicit_request_timeout(monkeypatch):
    calls = []

    class FakeResponse:
        def json(self):
            return {"token": "abc", "client_pos": 1, "hole_cards": ["Ac", "Kd"]}

    def fake_post(url, *, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("play_slumbot.requests.post", fake_post)

    result = api_new_hand(None, timeout_seconds=3.5)

    assert result["token"] == "abc"
    assert calls == [
        {
            "url": "https://slumbot.com/slumbot/api/new_hand",
            "json": {},
            "timeout": 3.5,
        }
    ]


def test_solver_iteration_profiles_keep_live_default_and_fast_live_candidate():
    assert _solver_iterations_for_profile(
        "live",
        to_call=0,
        pot=1000,
        hero_stack=5000,
        villain_stack=5000,
    ) == 150
    assert _solver_iterations_for_profile(
        "live",
        to_call=300,
        pot=1000,
        hero_stack=5000,
        villain_stack=5000,
    ) == 250
    assert _solver_iterations_for_profile(
        "live",
        to_call=600,
        pot=1000,
        hero_stack=5000,
        villain_stack=5000,
    ) == 350
    assert _solver_iterations_for_profile(
        "fast-live",
        to_call=0,
        pot=1000,
        hero_stack=5000,
        villain_stack=5000,
    ) == 100
    assert _solver_iterations_for_profile(
        "fast-live",
        to_call=300,
        pot=1000,
        hero_stack=12000,
        villain_stack=5000,
    ) == 150
    assert _solver_iterations_for_profile(
        "fast-live",
        to_call=600,
        pot=1000,
        hero_stack=5000,
        villain_stack=5000,
    ) == 250


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


class _AveragePolicyProbeNet(torch.nn.Module):
    def forward(self, features):
        logits = torch.zeros((features.shape[0], 9), dtype=torch.float32)
        logits[:, 1] = 10.0
        return logits


def test_base_policy_action_can_use_policy_head_instead_of_regret_matching():
    parsed = parse_action("")
    net = _PolicyHeadProbeNet()
    net.average_policy_net = _AveragePolicyProbeNet()

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
    average_policy_incr = _base_policy_action(
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
        strategy_source="average-policy",
    )

    assert regret_incr.startswith("b")
    assert policy_incr == "c"
    assert average_policy_incr == "c"


def test_network_strategy_policy_head_covered_routes_by_street():
    net = _PolicyHeadProbeNet()
    net.policy_calibration = {"target_streets": [2, 3]}
    legal_mask = torch.zeros((9,), dtype=torch.float32).numpy()
    legal_mask[[1, 8]] = 1.0
    preflop_features = torch.zeros((126,), dtype=torch.float32).numpy()
    preflop_features[104] = 1.0
    turn_features = torch.zeros((126,), dtype=torch.float32).numpy()
    turn_features[106] = 1.0

    _, preflop_strategy = network_strategy(
        net,
        preflop_features,
        legal_mask,
        torch.device("cpu"),
        strategy_source="policy-head-covered",
    )
    _, turn_strategy = network_strategy(
        net,
        turn_features,
        legal_mask,
        torch.device("cpu"),
        strategy_source="policy-head-covered",
    )

    assert preflop_strategy[8] > 0.99
    assert turn_strategy[1] > 0.99


def test_play_slumbot_script_help_imports_from_repo_root():
    script = SCRIPTS_DIR / "play_slumbot.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--strategy-source" in result.stdout
    assert "torch-levelsync-cuda" in result.stdout
    assert "--trace-jsonl" in result.stdout
    assert "--api-timeout-seconds" in result.stdout
