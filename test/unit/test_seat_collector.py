"""Tests for the seat-aware pyspiel collector (frozen-opponent anti-collapse) on small-NLHE."""
import numpy as np
import pytest


def _uniform(obs, legal):
    legal = legal.astype(np.float64)
    return legal / legal.sum(-1, keepdims=True)


def test_seat_collector_self_play_shapes_and_validity():
    pytest.importorskip("pyspiel")
    from poker_ai.rnad.seat_collector import SeatAwarePyspielCollector
    from poker_ai.rnad.small_nlhe import load_small_nlhe
    col = SeatAwarePyspielCollector(load_small_nlhe(), device="cpu")
    rng = np.random.RandomState(0)
    traj = col.collect(_uniform, batch_size=256, trajectory_max=8, rng=rng)  # self-play (no opponent)
    assert traj.obs.shape == (8, 256, col.obs_dim)
    # sampled action legal where valid
    chosen_legal = (traj.action_oh * traj.legal).sum(-1)
    v = traj.valid.bool()
    assert bool((chosen_legal[v] > 0.5).all())
    # rewards zero-sum at every recorded terminal step
    assert float(traj.rewards.sum(-1).abs().max()) < 1e-4


def test_seat_collector_frozen_opponent_marks_only_learner_valid():
    """With a frozen opponent, valid steps must all be the learner's seat (b % P)."""
    pytest.importorskip("pyspiel")
    from poker_ai.rnad.seat_collector import SeatAwarePyspielCollector
    from poker_ai.rnad.small_nlhe import load_small_nlhe
    col = SeatAwarePyspielCollector(load_small_nlhe(), device="cpu")
    rng = np.random.RandomState(1)

    def opp(obs, legal):  # distinct frozen opponent (still uniform here; identity not required)
        return _uniform(obs, legal)

    traj = col.collect(_uniform, batch_size=128, trajectory_max=8, rng=rng, opponent_policy_fn=opp)
    P = col.n_players
    learner_seat = (np.arange(128) % P).astype(np.float32)
    v = traj.valid.numpy().astype(bool)        # [T,B]
    pid = traj.player_id.numpy()               # [T,B]
    # every valid step's acting player == that column's learner seat
    for t in range(v.shape[0]):
        cols = np.flatnonzero(v[t])
        assert np.all(pid[t, cols] == learner_seat[cols])


def test_seat_collector_no_degenerate_policy_rows():
    """Every recorded policy row must be finite and sum to ~1 (no all-zero/NaN rows) — the PPO-NaN root cause."""
    import torch
    pytest.importorskip("pyspiel")
    from poker_ai.rnad.seat_collector import SeatAwarePyspielCollector
    from poker_ai.rnad.small_nlhe import load_small_nlhe
    col = SeatAwarePyspielCollector(load_small_nlhe(), device="cpu")
    rng = np.random.RandomState(2)

    def opp(obs, legal):
        return _uniform(obs, legal)

    traj = col.collect(_uniform, batch_size=256, trajectory_max=8, rng=rng, opponent_policy_fn=opp)
    pol = traj.policy
    assert torch.isfinite(pol).all(), "policy has non-finite entries"
    sums = pol.sum(-1)
    assert torch.all(sums > 0.99) and torch.all(sums < 1.01), "policy rows do not sum to 1"


def test_ppo_loss_finite_on_seat_collector_batch():
    """The PPO loss + all trajectory tensors must be finite on a seat-collector batch (frozen opponent)."""
    import torch
    pytest.importorskip("pyspiel")
    from poker_ai.rnad.seat_collector import SeatAwarePyspielCollector
    from poker_ai.rnad.small_nlhe import load_small_nlhe
    from poker_ai.rnad.ppo import PPOConfig, PPOSolver
    col = SeatAwarePyspielCollector(load_small_nlhe(), device="cpu")
    s = PPOSolver(PPOConfig(batch_size=128, trajectory_max=8, policy_network_layers=(32, 32), seed=1), col, device="cpu")
    rng = s._rng

    def opp(obs, legal):
        return _uniform(obs, legal)

    traj = col.collect(s._policy_fn, 128, 8, rng, opponent_policy_fn=opp)
    for name in ("obs", "legal", "player_id", "valid", "rewards", "action_oh", "policy"):
        assert torch.isfinite(getattr(traj, name)).all(), f"{name} non-finite"
    loss, logs = s.loss_on_trajectory(traj)
    assert torch.isfinite(loss), (loss, logs)
    assert all(np.isfinite(v) for v in logs.values()), logs
