"""ReBeL step C: river PBS value net plumbing (fast; no exact-leaf target generation)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from poker_ai.rebel.river_pbs_net import RiverPBSNet, train_river_net


def test_river_pbs_net_forward_shape():
    import torch
    H = 50
    net = RiverPBSNet(H, hidden=32)
    out = net(torch.zeros(3, 2 * H))
    assert out.shape == (3, 2 * H)


def test_train_river_net_fits_synthetic_linear():
    # The net must be able to fit a simple synthetic range->CFV map (reach-weighted MAE well below
    # the value scale), confirming the training loop + reach-weighting work.
    rng = np.random.default_rng(0)
    H = 20
    M = rng.standard_normal((H, H))
    Xs, Ys, Ws = [], [], []
    for _ in range(400):
        r0 = rng.dirichlet(np.ones(H)); r1 = rng.dirichlet(np.ones(H))
        ah = M @ r1; av = M.T @ r0
        Xs.append(np.concatenate([r0, r1])); Ys.append(np.concatenate([ah, av]))
        Ws.append(np.concatenate([r0, r1]))
    data = {"X": np.array(Xs, np.float32), "Y": np.array(Ys, np.float32),
            "W": np.array(Ws, np.float32), "H": H}
    _, m = train_river_net(data, hidden=128, epochs=400, seed=0)
    assert m["val_reach_weighted_mae"] < 0.25 * m["value_scale"]
