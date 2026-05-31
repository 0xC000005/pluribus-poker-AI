from poker_ai.research.xdo_target_audit import audit_xdo_target_records


def test_audit_xdo_target_records_passes_noncollapsed_positive_holdout():
    records = [
        {
            "root_idx": idx,
            "parent_action": "all_in",
            "oracle_action": action,
            "parent_action_value": -10.0,
            "oracle_action_value": 5.0 + idx,
            "oracle_gap": 15.0 + idx,
            "action_values": {"fold": 0.0, "call": 5.0 + idx, "all_in": -10.0},
            "n_truncated_rollouts": 0,
        }
        for idx, action in enumerate(["fold", "call", "fold", "call"])
    ]

    audit = audit_xdo_target_records(
        records,
        min_targets=4,
        max_top_action_fraction=0.75,
        min_holdout_mean_gap=0.0,
    )

    assert audit["passed"] is True
    assert audit["n_targets"] == 4
    assert audit["max_target_top_action_fraction"] == 0.5
    assert audit["holdout"]["n_targets"] == 2
    assert audit["target_action_counts"] == {"call": 2, "fold": 2}


def test_audit_xdo_target_records_blocks_collapsed_targets():
    records = [
        {
            "root_idx": idx,
            "parent_action": "all_in",
            "oracle_action": "fold",
            "parent_action_value": -1.0,
            "oracle_action_value": 1.0,
            "action_values": {"fold": 1.0, "call": 0.0},
            "n_truncated_rollouts": 0,
        }
        for idx in range(4)
    ]

    audit = audit_xdo_target_records(records, max_top_action_fraction=0.75)

    assert audit["passed"] is False
    assert "target_action_collapse" in audit["blockers"]
    assert audit["max_target_top_action_fraction"] == 1.0


def test_audit_xdo_target_records_blocks_negative_holdout_gap():
    records = [
        {
            "root_idx": 0,
            "parent_action": "call",
            "oracle_action": "raise_0.5",
            "parent_action_value": 1.0,
            "oracle_action_value": 3.0,
            "action_values": {"call": 1.0, "raise_0.5": 3.0},
            "n_truncated_rollouts": 0,
        },
        {
            "root_idx": 1,
            "parent_action": "call",
            "oracle_action": "raise_0.5",
            "parent_action_value": 2.0,
            "oracle_action_value": 1.0,
            "action_values": {"call": 2.0, "raise_0.5": 1.0},
            "n_truncated_rollouts": 0,
        },
    ]

    audit = audit_xdo_target_records(records, min_holdout_mean_gap=0.0)

    assert audit["passed"] is False
    assert "holdout_gap_below_threshold" in audit["blockers"]
    assert audit["holdout"]["mean_oracle_gap"] < 0.0
