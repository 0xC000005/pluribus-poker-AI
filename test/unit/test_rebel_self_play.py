"""ReBeL step 3: turn+river self-play loop smoke + component checks (CPU, non-batched, tiny iters)."""
import pytest

pytest.importorskip("torch")

import numpy as np

from poker_ai.rebel.self_play import (
    default_loop_spot, build_turn_solver, CtxRiverNet, ReservoirBuffer,
    sample_targets_from_strategy, sample_targets_all_cuts, _random_strategy,
    train_ctx_net, make_net_leaf_fn, run_self_play,
)


def test_sampler_shapes_and_contract():
    ts = build_turn_solver(default_loop_spot())
    H = ts.n
    rng = np.random.default_rng(0)
    sig = _random_strategy(ts, rng)
    Xo, Yo, Wo, h = sample_targets_from_strategy(ts, sig, river_iters=2, batched=False)
    assert h == H
    assert Xo.shape[0] == Yo.shape[0] == Wo.shape[0] >= 1
    assert Xo.shape[1] == 3 + 2 * H   # ctx(3) | r0n | r1n
    assert Yo.shape[1] == 2 * H       # A_h | A_v
    assert Wo.shape[1] == 2 * H
    # normalized ranges in X sum to ~1 per side
    r0n = Xo[0, 3:3 + H]; r1n = Xo[0, 3 + H:]
    assert abs(r0n.sum() - 1.0) < 1e-4 and abs(r1n.sum() - 1.0) < 1e-4


def test_random_explore_sampler_runs():
    ts = build_turn_solver(default_loop_spot())
    X, Y, W, H = sample_targets_all_cuts(ts, n_strategies=1, seed=1, river_iters=2, batched=False)
    assert X.shape[1] == 3 + 2 * H and Y.shape[1] == 2 * H


def test_reservoir_buffer_caps():
    buf = ReservoirBuffer(cap=5, seed=0)
    buf.add(np.zeros((4, 2), np.float32), np.zeros((4, 2), np.float32), np.zeros((4, 2), np.float32))
    buf.add(np.ones((4, 2), np.float32), np.ones((4, 2), np.float32), np.ones((4, 2), np.float32))
    assert len(buf) == 5
    assert buf.X.shape == (5, 2)


def test_net_leaf_fn_plugs_into_solve():
    ts = build_turn_solver(default_loop_spot())
    net = CtxRiverNet(ts.n)
    hr = np.ones(ts.n, np.float32) / ts.n
    ts.solve(n_iterations=1, hero_range=hr, villain_range=hr, backend="cpu",
             showdown_leaf_fn=make_net_leaf_fn(net, ts))
    assert np.asarray(ts._strategy_sum).shape[0] == ts._tree["n_nodes"]


def test_self_play_loop_smoke_runs_and_is_bounded():
    out, net = run_self_play(n_iters=2, trunk_iters=1, river_iters=2, init_strategies=1,
                             explore_strategies=1, epochs=10, hidden=64, seed=0,
                             batched=False, verbose=False)
    assert net.H == build_turn_solver(default_loop_spot()).n
    h = out["history"]
    assert 1 <= len(h) <= 2
    for rec in h:
        assert np.isfinite(rec["nashconv"]) and rec["nashconv"] >= 0.0
        assert rec["nashconv_pot"] < 1.0           # bounded well under a full pot
    assert np.isfinite(out["best_nashconv"])
    assert out["best_nashconv"] <= h[0]["nashconv"] + 1e-6   # best tracks the minimum seen
