import json

from poker_ai.research.empirical_meta_strategy_gate import (
    evaluate_empirical_meta_strategy_h2h_gate,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _h2h(path, *, baseline, mean, lower95, upper95):
    _write_json(
        path,
        {
            "algorithm": "mixed_native_policy_h2h",
            "candidate_kind": "meta-strategy",
            "candidate_checkpoint": "empirical.json",
            "baseline_checkpoint": baseline,
            "mean_candidate_payoff": mean,
            "lower95_candidate_payoff": lower95,
            "upper95_candidate_payoff": upper95,
            "n_games": 20000,
        },
    )


def test_empirical_meta_strategy_gate_separates_support_ties_from_offsupport_wins(
    tmp_path,
):
    empirical = tmp_path / "empirical.json"
    _write_json(
        empirical,
        {
            "policies": ["support_a.pt", "off_support.pt", "support_b.pt"],
            "meta_strategy": {
                "solved": True,
                "row_strategy": [0.6, 0.0, 0.4],
            },
        },
    )
    support_a = tmp_path / "support_a_h2h.json"
    off_support = tmp_path / "off_support_h2h.json"
    support_b = tmp_path / "support_b_h2h.json"
    _h2h(support_a, baseline="support_a.pt", mean=0.001, lower95=-0.004, upper95=0.006)
    _h2h(off_support, baseline="off_support.pt", mean=0.03, lower95=0.02, upper95=0.04)
    _h2h(support_b, baseline="support_b.pt", mean=-0.002, lower95=-0.005, upper95=0.001)

    metrics = evaluate_empirical_meta_strategy_h2h_gate(
        empirical,
        [support_a, off_support, support_b],
        max_support_abs_mean=0.01,
        max_support_lower95_loss=0.01,
        min_off_support_lower95=0.0,
    )

    assert metrics["passed"] is True
    assert metrics["support_policy_count"] == 2
    assert metrics["off_support_policy_count"] == 1
    assert metrics["promotion_blockers"] == []


def test_empirical_meta_strategy_gate_blocks_offsupport_loss(tmp_path):
    empirical = tmp_path / "empirical.json"
    _write_json(
        empirical,
        {
            "policies": ["support.pt", "off_support.pt"],
            "meta_strategy": {
                "solved": True,
                "row_strategy": [1.0, 0.0],
            },
        },
    )
    support = tmp_path / "support_h2h.json"
    off_support = tmp_path / "off_support_h2h.json"
    _h2h(support, baseline="support.pt", mean=0.0, lower95=-0.001, upper95=0.001)
    _h2h(off_support, baseline="off_support.pt", mean=-0.01, lower95=-0.02, upper95=0.0)

    metrics = evaluate_empirical_meta_strategy_h2h_gate(
        empirical,
        [support, off_support],
    )

    assert metrics["passed"] is False
    assert "off_support_policy_not_beaten" in metrics["promotion_blockers"]


def test_eval_empirical_meta_strategy_gate_cli_writes_json(monkeypatch, tmp_path):
    from scripts import eval_empirical_meta_strategy_gate as cli

    output = tmp_path / "gate.json"
    calls = []

    def fake_gate(empirical_game_json, h2h_records, **kwargs):
        calls.append((empirical_game_json, h2h_records, kwargs))
        return {
            "algorithm": "empirical_meta_strategy_h2h_gate",
            "passed": True,
            "promotion_blockers": [],
        }

    monkeypatch.setattr(cli, "evaluate_empirical_meta_strategy_h2h_gate", fake_gate)

    exit_code = cli.main(
        [
            "--empirical-game-json",
            "empirical.json",
            "--h2h-record",
            "a.json",
            "--h2h-record",
            "b.json",
            "--max-support-abs-mean",
            "0.02",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls[0][0] == "empirical.json"
    assert calls[0][1] == ["a.json", "b.json"]
    assert calls[0][2]["max_support_abs_mean"] == 0.02
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
