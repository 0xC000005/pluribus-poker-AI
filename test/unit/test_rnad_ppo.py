"""Tests for the PPO arm (poker_ai/rnad/ppo.py) — Arm A of the GO/NO-GO."""
import numpy as np
import pytest
import torch


def _rand_traj(T=6, B=32, A=4, P=2, seed=0):
    from poker_ai.rnad.collector import Trajectory
    rng = np.random.RandomState(seed)
    logits = rng.randn(T, B, A)
    legal = np.ones((T, B, A), dtype=np.float32)
    illegal = rng.rand(T, B, A) < 0.2
    illegal[..., 0] = False
    legal[illegal] = 0.0
    pid = rng.randint(0, P, size=(T, B)).astype(np.float32)
    valid = (rng.rand(T, B) > 0.2).astype(np.float32)
    rewards = np.zeros((T, B, P), dtype=np.float32)
    rewards[-1, :, 0] = rng.randn(B); rewards[-1, :, 1] = -rewards[-1, :, 0]  # zero-sum terminal
    mu = np.exp(logits) * legal; mu /= mu.sum(-1, keepdims=True)
    aoh = np.zeros((T, B, A), dtype=np.float32)
    for t in range(T):
        for b in range(B):
            li = np.where(legal[t, b] > 0)[0]
            aoh[t, b, rng.choice(li)] = 1.0
    tt = lambda x: torch.tensor(x, dtype=torch.float32)
    return Trajectory(obs=tt(rng.randn(T, B, 8)), legal=tt(legal), player_id=tt(pid),
                      valid=tt(valid), rewards=tt(rewards), action_oh=tt(aoh), policy=tt(mu))


def test_ppo_loss_finite_and_grads_flow():
    from poker_ai.rnad.ppo import compute_ppo_loss
    from poker_ai.rnad.network import RNaDNetwork
    d = _rand_traj()
    net = RNaDNetwork(8, 4, (16, 16))
    pi, v, log_pi, _ = net(d.obs, d.legal)
    loss, logs = compute_ppo_loss(pi, log_pi, v, valid=d.valid, player_id=d.player_id, legal=d.legal,
                                  action_oh=d.action_oh, behavior_policy=d.policy, rewards=d.rewards,
                                  num_players=2, clip_epsilon=0.2, trinal_clip=3.0, value_coef=0.5,
                                  entropy_coef=0.1)
    assert torch.isfinite(loss), loss
    loss.backward()
    g = [p.grad for p in net.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
    assert logs["entropy"] >= 0.0
    assert logs["mean_ratio"] > 0.0


def test_ppo_value_head_fits_returns():
    """A few grad steps on a FIXED batch should reduce the value loss (critic can fit returns)."""
    from poker_ai.rnad.ppo import compute_ppo_loss
    from poker_ai.rnad.network import RNaDNetwork
    d = _rand_traj(seed=3)
    net = RNaDNetwork(8, 4, (32, 32))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)

    def vloss():
        pi, v, log_pi, _ = net(d.obs, d.legal)
        _, logs = compute_ppo_loss(pi, log_pi, v, valid=d.valid, player_id=d.player_id, legal=d.legal,
                                   action_oh=d.action_oh, behavior_policy=d.policy, rewards=d.rewards,
                                   num_players=2, clip_epsilon=0.2, trinal_clip=3.0, value_coef=0.5,
                                   entropy_coef=0.0)
        return logs["value_loss"]

    v0 = vloss()
    for _ in range(60):
        opt.zero_grad()
        pi, v, log_pi, _ = net(d.obs, d.legal)
        loss, _ = compute_ppo_loss(pi, log_pi, v, valid=d.valid, player_id=d.player_id, legal=d.legal,
                                   action_oh=d.action_oh, behavior_policy=d.policy, rewards=d.rewards,
                                   num_players=2, clip_epsilon=0.2, trinal_clip=3.0, value_coef=1.0,
                                   entropy_coef=0.0)
        loss.backward(); opt.step()
    v1 = vloss()
    assert v1 < v0 * 0.8, (v0, v1)


class _StubCol:
    """Minimal collector stub so PPOSolver can be built without pyspiel (deps-free)."""
    is_gpu = False
    n_players = 2
    n_actions = 4
    obs_dim = 8


def _traj_with_dead_rows(T=7, B=64, A=4, P=2, seed=0):
    """A trajectory whose padding rows carry an ALL-ZERO legal mask + valid=0 (the PPO-NaN root cause).

    Mirrors the seat-aware collector: live learner rows have a legal mask and an action; dead rows
    (opponent/terminal padding) are fully masked. The terminal payoff is repeated on every trailing
    terminal step (so rewards[-1] is the per-game return), matching the real collector contract.
    """
    from poker_ai.rnad.collector import Trajectory
    rng = np.random.RandomState(seed)
    legal = np.zeros((T, B, A), dtype=np.float32)
    valid = np.zeros((T, B), dtype=np.float32)
    aoh = np.zeros((T, B, A), dtype=np.float32)
    pol = np.full((T, B, A), 1.0 / A, dtype=np.float32)
    pid = (rng.randint(0, P, size=(T, B))).astype(np.float32)
    rewards = np.zeros((T, B, P), dtype=np.float32)
    for b in range(B):
        n_live = rng.randint(1, T - 1)                 # a few real decisions, rest are dead padding
        payoff = float(rng.choice([-4, -2, 2, 4]))
        for t in range(n_live):
            mask = np.ones(A, dtype=np.float32)
            mask[rng.rand(A) < 0.3] = 0.0
            mask[0] = 1.0                              # always >=1 legal
            legal[t, b] = mask
            valid[t, b] = 1.0
            li = np.where(mask > 0)[0]
            aoh[t, b, rng.choice(li)] = 1.0
            pol[t, b] = mask / mask.sum()
        for t in range(n_live, T):                      # trailing terminal rows: all-zero legal, repeated payoff
            rewards[t, b, 0] = payoff
            rewards[t, b, 1] = -payoff
    tt = lambda x: torch.tensor(x, dtype=torch.float32)
    return Trajectory(obs=tt(rng.randn(T, B, 8)), legal=tt(legal), player_id=tt(pid),
                      valid=tt(valid), rewards=tt(rewards), action_oh=tt(aoh), policy=tt(pol))


def test_ppo_step_keeps_params_finite_with_dead_rows():
    """REGRESSION (PPO-NaN): an optimizer step over a trajectory with all-zero-legal dead rows must
    keep every network parameter finite. Before the fix the NaN from a fully-masked row poisoned all
    params on the first step. Deps-free (no pyspiel)."""
    from poker_ai.rnad.ppo import PPOConfig, PPOSolver
    traj = _traj_with_dead_rows()
    assert bool((traj.legal.sum(-1) == 0).any()), "test must contain all-zero-legal rows"
    s = PPOSolver(PPOConfig(batch_size=64, trajectory_max=7, policy_network_layers=(16, 16),
                            learning_rate=0.01, ppo_epochs=3, seed=1), _StubCol(), device="cpu")
    assert all(torch.isfinite(p).all() for p in s.net.parameters())
    for _ in range(3):
        s.optimizer.zero_grad(set_to_none=True)
        loss, logs = s.loss_on_trajectory(traj)
        assert torch.isfinite(loss), (loss, logs)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in s.net.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(s.net.parameters(), s.config.clip_gradient)
        s.optimizer.step()
        assert all(torch.isfinite(p).all() for p in s.net.parameters()), "params went non-finite"


def test_ppo_return_is_terminal_not_summed():
    """REGRESSION (value-target inflation): the MC return must be the per-game terminal payoff (taken
    once), not summed over the repeated trailing terminal steps."""
    from poker_ai.rnad.ppo import compute_ppo_loss
    from poker_ai.rnad.network import RNaDNetwork
    traj = _traj_with_dead_rows(seed=5)
    net = RNaDNetwork(8, 4, (16, 16))
    pi, v, log_pi, _ = net(traj.obs, traj.legal)
    # value_loss uses G_actor = rewards[-1]; |payoff| <= 4, so value_loss must be O(payoff^2), not the
    # ~7x-inflated value the summed return produced. Bound it well below the summed-return regime.
    safe_legal = torch.where(traj.legal.sum(-1, keepdim=True) > 0, traj.legal, torch.ones_like(traj.legal))
    pi, v, log_pi, _ = net(traj.obs, safe_legal)
    _, logs = compute_ppo_loss(pi, log_pi, v, valid=traj.valid, player_id=traj.player_id, legal=safe_legal,
                               action_oh=traj.action_oh, behavior_policy=traj.policy, rewards=traj.rewards,
                               num_players=2, clip_epsilon=0.2, trinal_clip=3.0, value_coef=0.5,
                               entropy_coef=0.0)
    assert np.isfinite(logs["value_loss"])
    # untrained value head ~0, so value_loss ~ mean(G^2) with |G|<=4 -> < 64; the summed return (|G| up
    # to ~28) would push this far higher. Assert the terminal-return regime.
    assert logs["value_loss"] < 64.0, logs["value_loss"]


def test_ppo_solver_runs_on_small_nlhe():
    pytest.importorskip("pyspiel")
    from poker_ai.rnad.ppo import PPOConfig, PPOSolver
    from poker_ai.rnad.collector import LeducTreeCollector
    from poker_ai.rnad.small_nlhe import load_small_nlhe
    col = LeducTreeCollector(load_small_nlhe(), device="cpu")
    s = PPOSolver(PPOConfig(batch_size=128, trajectory_max=8, policy_network_layers=(32, 32),
                            ppo_epochs=2, seed=1), col, device="cpu")
    for _ in range(3):
        logs = s.step()
    assert np.isfinite(logs["loss"])
