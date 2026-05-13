import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_public_belief_dcvn_leaf_ab import _apply_leaf_predictions
from eval_public_belief_dual_hand_cfv_probe import _load_or_build_dual_dataset
from poker_ai.games.full_deck.state import N_FEATURES
from poker_ai.research.belief_probe import BELIEF_DIM, N_HANDS


def test_apply_leaf_predictions_converts_evs_to_counterfactual_values():
    out_h = np.zeros((1, 2), dtype=np.float32)
    out_v = np.zeros((1, 2), dtype=np.float32)
    pred = np.zeros((2, 1, 4), dtype=np.float32)
    pred[0, 0, 1] = 2.0
    pred[0, 0, 3] = 4.0
    pred[1, 0, 1] = -3.0
    pred[1, 0, 3] = -5.0
    local_to_global = np.asarray([1, 3], dtype=np.int32)
    valid_m = np.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    hero_reach = np.asarray([[0.25, 0.75]], dtype=np.float32)
    villain_reach = np.asarray([[0.4, 0.6]], dtype=np.float32)

    hero, villain = _apply_leaf_predictions(
        out_h=out_h,
        out_v=out_v,
        pred=pred,
        tasks=[0],
        local_to_global=local_to_global,
        valid_m=valid_m,
        hero_reach=hero_reach,
        villain_reach=villain_reach,
    )

    np.testing.assert_allclose(hero[0], [0.8, 3.2])
    np.testing.assert_allclose(villain[0], [-1.875, -3.75])


def test_dual_cfv_dataset_builder_respects_start_index(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        '{"cases": ['
        + ",".join(
            (
                '{"label": "case-%d", "hole_cards": ["Ac", "Kd"], '
                '"board": ["2c", "7d", "Jh", "4s"], '
                '"action_str": "ck/kk/", "client_pos": 0}'
            )
            % idx
            for idx in range(5)
        )
        + "]}",
        encoding="utf-8",
    )
    cfv_cache = tmp_path / "cfv_cache.npz"
    features = np.arange(5 * N_FEATURES, dtype=np.float32).reshape(5, N_FEATURES)
    np.savez_compressed(
        cfv_cache,
        features=features,
        belief=np.arange(5 * BELIEF_DIM, dtype=np.float32).reshape(5, BELIEF_DIM),
        values=np.zeros((5, N_HANDS), dtype=np.float32),
        value_masks=np.zeros((5, N_HANDS), dtype=np.float32),
        labels=np.asarray([f"case-{idx}" for idx in range(5)]),
        records_json=np.asarray("[]"),
    )

    def fake_worker(args):
        case, *_rest = args
        values = np.zeros(N_HANDS, dtype=np.float32)
        masks = np.ones(N_HANDS, dtype=np.float32)
        return values, values, masks, masks, {"label": case.label}

    monkeypatch.setattr(
        "eval_public_belief_dual_hand_cfv_probe._dual_target_worker",
        fake_worker,
    )

    dataset, records, loaded = _load_or_build_dual_dataset(
        cases_json=cases_path,
        cfv_cache=cfv_cache,
        start_index=2,
        limit=2,
        solver_iterations=1,
        solver_backend="cpu",
        value_scale=1.0,
        dual_cache=None,
    )

    assert loaded is False
    assert dataset.labels == ("case-2", "case-3")
    assert [record["label"] for record in records] == ["case-2", "case-3"]
    np.testing.assert_allclose(dataset.features[0], features[2])
