import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_joint_pbs_root_policy import (  # noqa: E402
    _case_slice,
    _legal_uniform,
    _policy_eval_metrics,
)


def test_joint_pbs_root_policy_case_slice_offsets():
    assert _case_slice([0, 1, 2, 3], start_index=1, limit=2) == [1, 2]
    assert _case_slice([0, 1, 2, 3], start_index=2, limit=0) == [2, 3]
    with pytest.raises(ValueError, match="start_index"):
        _case_slice([0, 1], start_index=-1, limit=1)


def test_joint_pbs_root_policy_metrics_require_uniform_improvement():
    legal = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.2, 0, 0, 0, 0, 0, 0, 0.8],
        ],
        dtype=np.float32,
    )
    good = np.array(
        [
            [0.8, 0.2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.25, 0, 0, 0, 0, 0, 0, 0.75],
        ],
        dtype=np.float32,
    )
    bad = _legal_uniform(legal)

    good_metrics = _policy_eval_metrics(
        predictions=good,
        targets=targets,
        legal_masks=legal,
        min_top1_match=1.0,
        max_mean_l1=0.25,
    )
    bad_metrics = _policy_eval_metrics(
        predictions=bad,
        targets=targets,
        legal_masks=legal,
        min_top1_match=1.0,
        max_mean_l1=0.25,
    )

    assert good_metrics["passed"]
    assert good_metrics["policy"]["top1_match_rate"] == 1.0
    assert good_metrics["mean_l1_improvement_vs_uniform"] > 0
    assert not bad_metrics["passed"]
