import json

from poker_ai.research.expansion_quality_selector import evaluate_expansion_quality_selector


def _write_h2h(path, candidate, baseline, mean, lower95=None, *, n_games=100):
    path.write_text(
        json.dumps(
            {
                "algorithm": "mixed_native_policy_h2h",
                "candidate_checkpoint": candidate,
                "baseline_checkpoint": baseline,
                "mean_candidate_payoff": mean,
                "lower95_candidate_payoff": mean - 0.01 if lower95 is None else lower95,
                "upper95_candidate_payoff": mean + 0.01,
                "n_games": n_games,
                "environment": "poker_ai:full_deck_hu_nlhe",
                "trained_environment_native": True,
                "native_action_projection": False,
                "uses_slumbot_training_data": False,
            }
        ),
        encoding="utf-8",
    )


def test_expansion_quality_selector_selects_supported_confirmed_candidate(tmp_path):
    records = []
    for name, candidate, baseline, mean, lower95 in [
        ("inc_vs_old", "inc.pt", "old.pt", 0.2, 0.18),
        ("new_vs_inc", "new.pt", "inc.pt", 0.05, 0.02),
        ("new_vs_old", "new.pt", "old.pt", 0.3, 0.28),
    ]:
        path = tmp_path / f"{name}.json"
        _write_h2h(path, candidate, baseline, mean, lower95)
        records.append(path)

    metrics = evaluate_expansion_quality_selector(
        support_policies=["old.pt", "inc.pt"],
        candidate_checkpoints=["new.pt"],
        h2h_records=records,
        incumbent_checkpoint="inc.pt",
        min_lower95=0.0,
    )

    assert metrics["passed"] is True
    assert metrics["selected_candidate"] == "new.pt"
    result = metrics["candidate_results"][0]
    assert result["eligible"] is True
    assert result["support_probability"] > 0.0
    assert result["confirmation"]["lower95_candidate_payoff"] == 0.02


def test_expansion_quality_selector_rejects_negative_confirmation(tmp_path):
    records = []
    for name, candidate, baseline, mean, lower95 in [
        ("inc_vs_old", "inc.pt", "old.pt", 0.2, 0.18),
        ("new_vs_inc", "new.pt", "inc.pt", 0.01, -0.02),
        ("new_vs_old", "new.pt", "old.pt", 0.3, 0.28),
    ]:
        path = tmp_path / f"{name}.json"
        _write_h2h(path, candidate, baseline, mean, lower95)
        records.append(path)

    metrics = evaluate_expansion_quality_selector(
        support_policies=["old.pt", "inc.pt"],
        candidate_checkpoints=["new.pt"],
        h2h_records=records,
        incumbent_checkpoint="inc.pt",
        min_lower95=0.0,
    )

    assert metrics["passed"] is False
    assert metrics["selected_candidate"] is None
    assert "candidate_confirmation_lower95_below_threshold" in metrics["candidate_results"][0]["blockers"]


def test_expansion_quality_selector_reports_incomplete_matrix(tmp_path):
    inc_vs_old = tmp_path / "inc_vs_old.json"
    _write_h2h(inc_vs_old, "inc.pt", "old.pt", 0.2, 0.18)

    metrics = evaluate_expansion_quality_selector(
        support_policies=["old.pt", "inc.pt"],
        candidate_checkpoints=["new.pt"],
        h2h_records=[inc_vs_old],
        incumbent_checkpoint="inc.pt",
    )

    assert metrics["passed"] is False
    result = metrics["candidate_results"][0]
    assert "incomplete_expanded_empirical_game" in result["blockers"]
    assert result["empirical_game"]["missing_pairs"] == [["old.pt", "new.pt"], ["inc.pt", "new.pt"]]
