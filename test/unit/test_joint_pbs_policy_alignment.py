import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_joint_pbs_policy_alignment import (  # noqa: E402
    _entropy,
    _group_summary,
    _margin,
    _safe_corr,
    _summarize_numeric_correlations,
)


def test_policy_alignment_entropy_margin_and_corr_helpers():
    legal = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = np.array([0.7, 0.2, 0, 0, 0, 0, 0, 0, 0.1], dtype=np.float32)

    assert _entropy(probs, legal) > 0
    assert _margin(probs, legal) == pytest.approx(0.5)
    assert _safe_corr([0, 1, 2], [0, 2, 4]) == 1.0
    assert _safe_corr([1, 1, 1], [0, 1, 2]) is None


def test_policy_alignment_group_and_correlation_summary():
    records = [
        {
            "action_shape": "cc/b",
            "model_high_l1": 0.2,
            "top1_match": True,
            "low_high_l1": 0.1,
            "pot": 10,
            "bet_count": 1,
            "legal_count": 3,
            "target_entropy": 0.4,
            "target_margin": 0.5,
            "pred_entropy": 0.3,
            "pred_margin": 0.6,
            "pred_confidence": 0.7,
            "target_confidence": 0.8,
            "target_allin_prob": 0.1,
            "pred_allin_prob": 0.2,
            "hero_belief_entropy": 1.0,
            "villain_belief_entropy": 1.1,
        },
        {
            "action_shape": "cc/b",
            "model_high_l1": 0.4,
            "top1_match": False,
            "low_high_l1": 0.2,
            "pot": 20,
            "bet_count": 1,
            "legal_count": 3,
            "target_entropy": 0.5,
            "target_margin": 0.4,
            "pred_entropy": 0.6,
            "pred_margin": 0.2,
            "pred_confidence": 0.6,
            "target_confidence": 0.7,
            "target_allin_prob": 0.2,
            "pred_allin_prob": 0.4,
            "hero_belief_entropy": 1.2,
            "villain_belief_entropy": 1.0,
        },
        {
            "action_shape": "cc/k",
            "model_high_l1": 0.6,
            "top1_match": False,
            "low_high_l1": 0.3,
            "pot": 30,
            "bet_count": 0,
            "legal_count": 2,
            "target_entropy": 0.6,
            "target_margin": 0.3,
            "pred_entropy": 0.7,
            "pred_margin": 0.1,
            "pred_confidence": 0.5,
            "target_confidence": 0.6,
            "target_allin_prob": 0.3,
            "pred_allin_prob": 0.6,
            "hero_belief_entropy": 1.4,
            "villain_belief_entropy": 0.9,
        },
    ]

    groups = _group_summary(records, "action_shape")
    corrs = _summarize_numeric_correlations(records)

    assert groups[0]["key"] == "cc/b"
    assert groups[0]["n"] == 2
    assert groups[0]["top1_match_rate"] == 0.5
    assert corrs
    assert any(row["field"] == "pot" for row in corrs)
