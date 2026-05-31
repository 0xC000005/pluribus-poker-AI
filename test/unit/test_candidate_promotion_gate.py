import json


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rlcard_payload(
    *,
    lower95=0.1,
    baseline_kind="alphanlholdem",
    reference_integrity_passed=True,
):
    return {
        "algorithm": "alphanlholdem_rlcard_reference_h2h",
        "environment": "rlcard:no-limit-holdem",
        "candidate_kind": "rlcard-vtrace",
        "candidate_weights": "rlcard_candidate.pt",
        "baseline_kind": baseline_kind,
        "lower95_candidate_payoff": lower95,
        "trained_environment_native": True,
        "native_action_projection": False,
        "reference_integrity_passed": reference_integrity_passed,
        "reference_checkout_ignored": True,
        "tracked_external_reference_files": [],
        "native_slumbot_evidence": False,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
    }


def _native_payload(*, lower95=0.1):
    return {
        "algorithm": "mixed_native_policy_h2h",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "candidate_checkpoint": "native_candidate.pt",
        "baseline_checkpoint": "native_baseline.pt",
        "lower95_candidate_payoff": lower95,
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_training_data": False,
    }


def _empirical_payload(*, candidate_prob=1.0, coverage=1.0):
    return {
        "algorithm": "native_empirical_payoff_matrix",
        "policies": ["native_candidate.pt", "native_baseline.pt"],
        "off_diagonal_coverage": coverage,
        "recommendation": "solve_meta_strategy",
        "meta_strategy": {
            "solved": True,
            "row_strategy": [candidate_prob, 1.0 - candidate_prob],
            "column_strategy": [candidate_prob, 1.0 - candidate_prob],
            "value": 0.1,
        },
        "promotion": False,
    }


def test_candidate_promotion_gate_requires_all_surfaces(tmp_path):
    from poker_ai.research.candidate_promotion_gate import evaluate_candidate_promotion_evidence

    rlcard = tmp_path / "rlcard.json"
    native = tmp_path / "native.json"
    empirical = tmp_path / "empirical.json"
    _write_json(rlcard, _rlcard_payload(lower95=-0.01))
    _write_json(native, _native_payload(lower95=0.2))
    _write_json(empirical, _empirical_payload(candidate_prob=1.0))

    metrics = evaluate_candidate_promotion_evidence(
        rlcard_reference_json=rlcard,
        native_h2h_json=native,
        empirical_game_json=empirical,
        native_candidate_checkpoint="native_candidate.pt",
        min_lower95=0.0,
    )

    assert metrics["algorithm"] == "poker_candidate_promotion_gate"
    assert metrics["passed"] is False
    assert metrics["slumbot_confidence_eligible"] is False
    assert "rlcard_reference_lower95_below_threshold" in metrics["promotion_blockers"]
    assert metrics["checks"]["native_h2h"]["passed"] is True
    assert metrics["checks"]["empirical_game"]["passed"] is True


def test_candidate_promotion_gate_passes_complete_dual_surface_evidence(tmp_path):
    from poker_ai.research.candidate_promotion_gate import evaluate_candidate_promotion_evidence

    rlcard = tmp_path / "rlcard.json"
    native = tmp_path / "native.json"
    empirical = tmp_path / "empirical.json"
    _write_json(rlcard, _rlcard_payload(lower95=0.01))
    _write_json(native, _native_payload(lower95=0.02))
    _write_json(empirical, _empirical_payload(candidate_prob=0.75))

    metrics = evaluate_candidate_promotion_evidence(
        rlcard_reference_json=rlcard,
        native_h2h_json=native,
        empirical_game_json=empirical,
        native_candidate_checkpoint="native_candidate.pt",
        min_lower95=0.0,
    )

    assert metrics["passed"] is True
    assert metrics["slumbot_confidence_eligible"] is True
    assert metrics["promotion_blockers"] == []
    assert metrics["checks"]["rlcard_reference"]["passed"] is True
    assert metrics["checks"]["native_h2h"]["passed"] is True
    assert metrics["checks"]["empirical_game"]["candidate_support_probability"] == 0.75


def test_candidate_promotion_gate_requires_alphanlholdem_reference_baseline(tmp_path):
    from poker_ai.research.candidate_promotion_gate import evaluate_candidate_promotion_evidence

    rlcard = tmp_path / "rlcard.json"
    native = tmp_path / "native.json"
    empirical = tmp_path / "empirical.json"
    _write_json(rlcard, _rlcard_payload(lower95=0.1, baseline_kind="random"))
    _write_json(native, _native_payload(lower95=0.1))
    _write_json(empirical, _empirical_payload(candidate_prob=1.0))

    metrics = evaluate_candidate_promotion_evidence(
        rlcard_reference_json=rlcard,
        native_h2h_json=native,
        empirical_game_json=empirical,
        native_candidate_checkpoint="native_candidate.pt",
        min_lower95=0.0,
    )

    assert metrics["passed"] is False
    assert "rlcard_reference_not_alphanlholdem_baseline" in metrics["promotion_blockers"]


def test_candidate_promotion_gate_requires_reference_integrity(tmp_path):
    from poker_ai.research.candidate_promotion_gate import evaluate_candidate_promotion_evidence

    rlcard = tmp_path / "rlcard.json"
    native = tmp_path / "native.json"
    empirical = tmp_path / "empirical.json"
    _write_json(rlcard, _rlcard_payload(lower95=0.1, reference_integrity_passed=False))
    _write_json(native, _native_payload(lower95=0.1))
    _write_json(empirical, _empirical_payload(candidate_prob=1.0))

    metrics = evaluate_candidate_promotion_evidence(
        rlcard_reference_json=rlcard,
        native_h2h_json=native,
        empirical_game_json=empirical,
        native_candidate_checkpoint="native_candidate.pt",
        min_lower95=0.0,
    )

    assert metrics["passed"] is False
    assert "rlcard_reference_integrity_not_passed" in metrics["promotion_blockers"]


def test_candidate_promotion_gate_cli_writes_metrics_and_returns_nonzero_on_failure(tmp_path):
    from scripts import eval_candidate_promotion_gate as cli

    rlcard = tmp_path / "rlcard.json"
    native = tmp_path / "native.json"
    output = tmp_path / "promotion.json"
    _write_json(rlcard, _rlcard_payload(lower95=0.1))
    _write_json(native, _native_payload(lower95=-0.1))

    exit_code = cli.main(
        [
            "--rlcard-reference-json",
            str(rlcard),
            "--native-h2h-json",
            str(native),
            "--output-json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["passed"] is False
    assert "native_h2h_lower95_below_threshold" in payload["promotion_blockers"]
