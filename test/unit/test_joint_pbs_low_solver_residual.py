import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_joint_pbs_continuation_probe import JointPBSDataset
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.belief_probe import BELIEF_DIM
from train_joint_pbs_low_solver_residual import (
    _load_records_for_labels,
    _residual_features,
    _residual_logits,
    _root_overlap_summary,
)


def _dataset(n_rows: int = 3) -> JointPBSDataset:
    return JointPBSDataset(
        features=np.zeros((n_rows, N_FEATURES), dtype=np.float32),
        policy_features=np.zeros((n_rows, N_FEATURES), dtype=np.float32),
        belief=np.zeros((n_rows, BELIEF_DIM), dtype=np.float32),
        legal_masks=np.ones((n_rows, N_ACTIONS), dtype=np.float32),
        target_probs=np.ones((n_rows, N_ACTIONS), dtype=np.float32) / float(N_ACTIONS),
        policy_weights=np.ones(n_rows, dtype=np.float32),
        hero_values=np.zeros((n_rows, BELIEF_DIM // 2), dtype=np.float32),
        villain_values=np.zeros((n_rows, BELIEF_DIM // 2), dtype=np.float32),
        hero_masks=np.zeros((n_rows, BELIEF_DIM // 2), dtype=np.float32),
        villain_masks=np.zeros((n_rows, BELIEF_DIM // 2), dtype=np.float32),
        hero_value_weights=np.zeros((n_rows, BELIEF_DIM // 2), dtype=np.float32),
        villain_value_weights=np.zeros((n_rows, BELIEF_DIM // 2), dtype=np.float32),
        action_tokens=np.zeros((n_rows, 4), dtype=np.int64),
        action_amounts=np.zeros((n_rows, 4), dtype=np.float32),
        labels=tuple(f"case-{idx}" for idx in range(n_rows)),
    )


def test_residual_features_include_public_private_belief_mask_and_low_policy():
    dataset = _dataset(2)
    low = np.ones((2, N_ACTIONS), dtype=np.float32) / float(N_ACTIONS)

    features = _residual_features(dataset, low)

    assert features.shape == (2, N_FEATURES + 52 + BELIEF_DIM + 2 * N_ACTIONS)


def test_zero_delta_residual_logits_preserve_low_policy_on_legal_actions():
    low = torch.tensor([[0.1, 0.3, 0.6] + [0.0] * (N_ACTIONS - 3)], dtype=torch.float32)
    legal = torch.tensor([[1.0, 1.0, 1.0] + [0.0] * (N_ACTIONS - 3)], dtype=torch.float32)
    delta = torch.zeros((1, N_ACTIONS), dtype=torch.float32)

    probs = torch.softmax(_residual_logits(delta, low, legal), dim=1)

    torch.testing.assert_close(probs[:, :3], low[:, :3], atol=1e-6, rtol=1e-6)
    assert float(probs[:, 3:].sum()) == 0.0


def test_load_records_for_labels_reorders_metadata_by_dataset_labels(tmp_path):
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "cut_records": [
                    {"label": "case-1", "root_label": "root-b"},
                    {"label": "case-0", "root_label": "root-a"},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = _load_records_for_labels(metadata, ("case-0", "case-1"))

    assert [record["label"] for record in records] == ["case-0", "case-1"]


def test_root_overlap_summary_requires_disjoint_public_roots():
    summary = _root_overlap_summary(
        [{"root_label": "root-a"}, {"root_label": "root-b"}],
        [{"root_label": "root-b"}, {"root_label": "root-c"}],
    )

    assert summary["overlap_count"] == 1
    assert summary["overlap_sample"] == ["root-b"]
    assert summary["passed"] is False
