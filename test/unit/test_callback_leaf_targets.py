import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_public_belief_callback_leaf_targets import _leaf_evs_from_numerators


def test_leaf_evs_from_numerators_divides_by_reach_denominators():
    valid = np.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    hero_reach = np.asarray([0.25, 0.75], dtype=np.float32)
    villain_reach = np.asarray([0.4, 0.6], dtype=np.float32)
    hero_den = valid @ villain_reach
    villain_den = valid.T @ hero_reach
    hero_ev = np.asarray([2.0, 4.0], dtype=np.float32)
    villain_ev = np.asarray([-3.0, -5.0], dtype=np.float32)

    hero, villain, hero_mask, villain_mask = _leaf_evs_from_numerators(
        default_hero_values=hero_ev * hero_den,
        default_villain_values=villain_ev * villain_den,
        valid_m=valid,
        hero_reach=hero_reach,
        villain_reach=villain_reach,
    )

    np.testing.assert_allclose(hero, hero_ev)
    np.testing.assert_allclose(villain, villain_ev)
    np.testing.assert_allclose(hero_mask, [1.0, 1.0])
    np.testing.assert_allclose(villain_mask, [1.0, 1.0])


def test_leaf_evs_from_numerators_masks_zero_denominators():
    valid = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    hero_reach = np.asarray([0.0, 1.0], dtype=np.float32)
    villain_reach = np.asarray([0.0, 0.0], dtype=np.float32)

    hero, villain, hero_mask, villain_mask = _leaf_evs_from_numerators(
        default_hero_values=np.asarray([3.0, 4.0], dtype=np.float32),
        default_villain_values=np.asarray([5.0, 6.0], dtype=np.float32),
        valid_m=valid,
        hero_reach=hero_reach,
        villain_reach=villain_reach,
    )

    np.testing.assert_allclose(hero, [0.0, 0.0])
    np.testing.assert_allclose(villain, [0.0, 6.0])
    np.testing.assert_allclose(hero_mask, [0.0, 0.0])
    np.testing.assert_allclose(villain_mask, [0.0, 1.0])
