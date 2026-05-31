import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import ACTION_TO_INDEX, N_ACTIONS, N_FEATURES
from poker_ai.research.xdo_target_dataset import build_xdo_policy_target_dataset


def _write_records(path: Path) -> None:
    payload = {
        "seed": 20260527,
        "n_worlds": 4,
        "initial_chips": 1000,
        "small_blind": 50,
        "big_blind": 100,
        "rows": [
            {
                "root_idx": 0,
                "hero_cards": [0, 7],
                "parent_action": "fold",
                "oracle_action": "call",
                "parent_action_value": -50.0,
                "oracle_action_value": 25.0,
                "oracle_gap": 75.0,
                "action_values": {"fold": -50.0, "call": 25.0},
                "n_truncated_rollouts": 0,
            },
            {
                "root_idx": 1,
                "hero_cards": [12, 51],
                "parent_action": "call",
                "oracle_action": "fold",
                "parent_action_value": -20.0,
                "oracle_action_value": 0.0,
                "oracle_gap": 20.0,
                "action_values": {"fold": 0.0, "call": -20.0},
                "n_truncated_rollouts": 0,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_xdo_policy_target_dataset_writes_features_masks_and_targets(tmp_path):
    records_json = tmp_path / "records.json"
    output_npz = tmp_path / "xdo_targets.npz"
    _write_records(records_json)

    metrics = build_xdo_policy_target_dataset(
        records_json,
        output_npz,
        min_targets=2,
        max_top_action_fraction=0.75,
        min_holdout_mean_gap=0.0,
    )

    assert metrics["passed"] is True
    assert metrics["n_targets"] == 2
    buffer = PolicyTargetBuffer.from_npz(output_npz)
    assert buffer.features.shape == (2, N_FEATURES)
    assert buffer.legal_masks.shape == (2, N_ACTIONS)
    assert buffer.target_probs[0, ACTION_TO_INDEX["call"]] == 1.0
    assert buffer.target_probs[1, ACTION_TO_INDEX["fold"]] == 1.0
    assert buffer.legal_masks[0, ACTION_TO_INDEX["call"]] == 1.0
    saved = np.load(output_npz, allow_pickle=False)
    assert saved["root_indices"].tolist() == [0, 1]
    assert saved["street_indices"].tolist() == [0, 0]
    assert saved["oracle_gaps"].tolist() == [75.0, 20.0]
    assert saved["oracle_actions"].tolist() == [
        ACTION_TO_INDEX["call"],
        ACTION_TO_INDEX["fold"],
    ]
    assert metrics["target_street_counts"] == {"0": 2}
    assert metrics["whole_game_h2h_ready"] is False
    assert "insufficient_target_streets_for_whole_game_h2h" in metrics["coverage_blockers"]


def test_audit_xdo_target_dataset_cli_writes_npz_and_json(tmp_path):
    records_json = tmp_path / "records.json"
    output_npz = tmp_path / "targets.npz"
    output_json = tmp_path / "targets.json"
    _write_records(records_json)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_xdo_target_dataset.py",
            "--records-json",
            str(records_json),
            "--output",
            str(output_npz),
            "--output-json",
            str(output_json),
            "--min-targets",
            "2",
            "--max-top-action-fraction",
            "0.75",
            "--require-pass",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert json.loads(output_json.read_text())["target_size"] == 2
    assert PolicyTargetBuffer.from_npz(output_npz).size == 2
