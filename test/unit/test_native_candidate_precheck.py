import json


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _h2h_payload(*, candidate="candidate.pt", lower95=0.1):
    return {
        "algorithm": "mixed_native_policy_h2h",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "candidate_checkpoint": candidate,
        "baseline_checkpoint": "baseline.pt",
        "candidate_kind": "tianshou-rainbow",
        "baseline_kind": "native-ppo",
        "n_games": 20000,
        "mean_candidate_payoff": 0.2,
        "lower95_candidate_payoff": lower95,
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_training_data": False,
        "uses_alphanlholdem_training_data": False,
    }


def _empirical_payload(*, candidate="candidate.pt", support=1.0):
    return {
        "algorithm": "native_empirical_payoff_matrix",
        "policies": [candidate, "baseline.pt"],
        "off_diagonal_coverage": 1.0,
        "meta_strategy": {
            "solved": True,
            "row_strategy": [support, 1.0 - support],
            "column_strategy": [support, 1.0 - support],
            "value": 0.0,
        },
    }


def test_native_candidate_local_precheck_passes_local_but_blocks_external(tmp_path):
    from poker_ai.research.native_candidate_precheck import (
        evaluate_native_candidate_local_precheck,
    )

    h2h = tmp_path / "h2h.json"
    empirical = tmp_path / "empirical.json"
    _write_json(h2h, _h2h_payload())
    _write_json(empirical, _empirical_payload())

    metrics = evaluate_native_candidate_local_precheck(
        candidate_checkpoint="candidate.pt",
        native_h2h_jsons=[h2h],
        empirical_game_json=empirical,
    )

    assert metrics["algorithm"] == "native_candidate_local_precheck"
    assert metrics["passed"] is True
    assert metrics["local_precheck_passed"] is True
    assert metrics["promotion"] is False
    assert metrics["slumbot_confidence_eligible"] is False
    assert metrics["local_blockers"] == []
    assert "rlcard_reference_evidence_missing" in metrics["external_promotion_blockers"]
    assert "slumbot_adapter_missing_for_tianshou_rainbow" in metrics["external_promotion_blockers"]


def test_native_candidate_local_precheck_fails_bad_lower_bound(tmp_path):
    from poker_ai.research.native_candidate_precheck import (
        evaluate_native_candidate_local_precheck,
    )

    h2h = tmp_path / "h2h.json"
    empirical = tmp_path / "empirical.json"
    _write_json(h2h, _h2h_payload(lower95=-0.01))
    _write_json(empirical, _empirical_payload())

    metrics = evaluate_native_candidate_local_precheck(
        candidate_checkpoint="candidate.pt",
        native_h2h_jsons=[h2h],
        empirical_game_json=empirical,
    )

    assert metrics["passed"] is False
    assert "native_h2h_lower95_below_threshold" in metrics["local_blockers"]


def test_native_candidate_local_precheck_cli_writes_metrics(tmp_path):
    from scripts import eval_native_candidate_local_precheck as cli

    h2h = tmp_path / "h2h.json"
    empirical = tmp_path / "empirical.json"
    output = tmp_path / "precheck.json"
    _write_json(h2h, _h2h_payload())
    _write_json(empirical, _empirical_payload())

    exit_code = cli.main(
        [
            "--candidate-checkpoint",
            "candidate.pt",
            "--native-h2h-json",
            str(h2h),
            "--empirical-game-json",
            str(empirical),
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["local_precheck_passed"] is True
