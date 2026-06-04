"""ReBeL Stage-0 de-risk: the Leduc constant-oracle round-trip (pins the two correctness must-fixes).

Protects: (#1) the CFV normalization convention round-trips numerically; (#2) CFR+ uniform/own-reach
averaging converges. If these regress, the depth-limited solving loop built on top is unsound.
"""
import numpy as np
import pytest

pytest.importorskip("pyspiel")

from poker_ai.rebel.leduc import LeducTree
from poker_ai.rebel.leaf_eval import ExactLeafOracle
from scripts.run_rebel_leduc_roundtrip import max_round1_q_error


@pytest.fixture(scope="module")
def tree():
    return LeducTree()


def test_tree_shape(tree):
    assert tree.n_iset == 936
    assert sum(r == 1 for r in tree.iset_round) == 36   # round-1 (trunk) infosets
    assert sum(r == 2 for r in tree.iset_round) == 900  # round-2 (below the cut)
    assert len(tree.cut_keys()) == 5                    # public cut states


def test_engine_trust_linear_cfr_plus(tree):
    # Linear CFR+ reaching low exact NashConv proves tree + values + regret machinery are exact.
    _, hist = tree.cfr_plus(1000, eval_every=1000, averaging="linear")
    assert hist[-1][1] < 0.01


def test_convention_roundtrip_matches(tree):
    # Normalized convention produced by the oracle and consumed by the trunk must reproduce the
    # exact full-game round-1 q-values to machine precision, for several random strategies.
    oracle = ExactLeafOracle(tree, normalize=True)
    for sd in range(5):
        rng = np.random.default_rng(sd)
        pol = tree.random_policy(rng)
        q_full, _, _, _ = tree.values(pol)
        reaches = tree.cut_reaches(pol)
        leaf_v = oracle.all_leaf_values(reaches, pol)
        q_dl = tree.values_with_leaf(pol, leaf_v)
        assert max_round1_q_error(tree, q_full, q_dl) < 1e-9


def test_negative_control_breaks(tree):
    # The WRONG convention (un-normalized produced, consumed as normalized) must break the round-trip.
    oracle_bad = ExactLeafOracle(tree, normalize=False)
    rng = np.random.default_rng(0)
    pol = tree.random_policy(rng)
    q_full, _, _, _ = tree.values(pol)
    reaches = tree.cut_reaches(pol)
    leaf_v = oracle_bad.all_leaf_values(reaches, pol)
    q_dl = tree.values_with_leaf(pol, leaf_v)
    assert max_round1_q_error(tree, q_full, q_dl) > 1e-3
