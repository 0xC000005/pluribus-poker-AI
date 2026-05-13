import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reweight_joint_pbs_by_search_drift import reweight_joint_pbs_by_search_drift  # noqa: E402


def test_search_drift_reweight_multiplies_value_weights_by_root_error(tmp_path):
    joint = tmp_path / "joint.npz"
    meta = tmp_path / "joint.json"
    search = tmp_path / "search.json"
    output = tmp_path / "weighted.npz"
    n = 3
    shape = (n, 4)
    arrays = {
        "features": np.zeros((n, 2), dtype=np.float32),
        "policy_features": np.zeros((n, 2), dtype=np.float32),
        "belief": np.zeros((n, 6), dtype=np.float32),
        "legal_masks": np.ones((n, 3), dtype=np.float32),
        "target_probs": np.ones((n, 3), dtype=np.float32) / 3.0,
        "policy_weights": np.zeros(n, dtype=np.float32),
        "hero_values": np.zeros(shape, dtype=np.float32),
        "villain_values": np.zeros(shape, dtype=np.float32),
        "hero_masks": np.ones(shape, dtype=np.float32),
        "villain_masks": np.ones(shape, dtype=np.float32),
        "hero_value_weights": np.ones(shape, dtype=np.float32) * 2.0,
        "villain_value_weights": np.ones(shape, dtype=np.float32) * 3.0,
        "labels": np.asarray(["root-a-cut0-iter0", "root-a-cut1-iter0", "root-b-cut0-iter0"]),
    }
    np.savez_compressed(joint, **arrays)
    meta.write_text(
        json.dumps(
            {
                "cut_records": [
                    {"label": "root-a-cut0-iter0", "root_label": "root-a"},
                    {"label": "root-a-cut1-iter0", "root_label": "root-a"},
                    {"label": "root-b-cut0-iter0", "root_label": "root-b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    search.write_text(
        json.dumps(
            {
                "mode": "joint_pbs_resolver_successor_cut_ab",
                "records": [
                    {
                        "label": "root-a",
                        "action_l1_drift": 0.5,
                        "action_agreement": False,
                        "cut_applied": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = reweight_joint_pbs_by_search_drift(
        joint_npz=joint,
        metadata_json=meta,
        search_json=search,
        output=output,
    )
    loaded = np.load(output, allow_pickle=False)

    assert metrics["weighted_rows"] == 2
    assert metrics["matched_root_count"] == 1
    np.testing.assert_allclose(loaded["search_consistency_weights"], [2.5, 2.5, 1.0])
    np.testing.assert_allclose(loaded["hero_value_weights"][0], np.full(4, 5.0))
    np.testing.assert_allclose(loaded["villain_value_weights"][1], np.full(4, 7.5))
    np.testing.assert_allclose(loaded["hero_value_weights"][2], np.full(4, 2.0))
