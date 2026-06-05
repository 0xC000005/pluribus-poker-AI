"""ReBeL Stage-2 plumbing: the depth-limited trunk solver with a pluggable leaf value function,
and the PBS value net leaf adapter. Fast interface tests (no data generation / training)."""
import numpy as np
import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from poker_ai.rebel.leduc import LeducTree, NCARDS
from poker_ai.rebel.loop import trunk_solve
from poker_ai.rebel.pbs_value_net import PBSValueNet, net_leaf_fn


@pytest.fixture(scope="module")
def tree():
    return LeducTree()


def test_trunk_solve_with_constant_leaf(tree):
    # A trivial constant leaf function must drive a well-formed round-1 strategy (probabilities).
    def leaf_fn(_key, _r0, _r1):
        return np.zeros(NCARDS), np.zeros(NCARDS)

    pol = trunk_solve(tree, leaf_fn, trunk_iters=10, averaging="linear")
    r1 = [i for i in range(tree.n_iset) if tree.iset_round[i] == 1]
    assert set(pol.keys()) == set(r1)
    for i in r1:
        assert pol[i].shape[0] == len(tree.iset_actions[i])
        assert abs(pol[i].sum() - 1.0) < 1e-9
        assert (pol[i] >= -1e-12).all()


def test_net_leaf_fn_shapes(tree):
    keys = sorted(tree.cut_keys())
    net = PBSValueNet(n_keys=len(keys), hidden=16)
    fn = net_leaf_fn(net, keys)
    r0 = np.array([0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
    r1 = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    v0, v1 = fn(keys[0], r0, r1)
    assert v0.shape == (NCARDS,) and v1.shape == (NCARDS,)
    assert np.isfinite(v0).all() and np.isfinite(v1).all()


def test_net_leaf_fn_scale_invariant(tree):
    # Values depend on normalized ranges -> scaling the input ranges must not change outputs.
    keys = sorted(tree.cut_keys())
    net = PBSValueNet(n_keys=len(keys), hidden=16)
    fn = net_leaf_fn(net, keys)
    r0 = np.array([0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
    r1 = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    v0a, v1a = fn(keys[1], r0, r1)
    v0b, v1b = fn(keys[1], r0 * 7.0, r1 * 0.01)
    assert np.allclose(v0a, v0b) and np.allclose(v1a, v1b)
