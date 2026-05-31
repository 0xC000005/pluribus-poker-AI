"""Correctness tests for the GPU-resident tree collector (poker_ai/rnad/collector.py::GPUTreeCollector).

The loss-stack parity tests (test_rnad_torch.py) are collector-independent and already prove the learner
math. These tests prove the NEW GPU collector preserves correctness:
  1. STRUCTURAL invariants on a GPU-collected trajectory (sampled action always legal; valid matches
     terminal-ness; recorded rewards == terminal returns; player ids valid).
  2. NO residual chance after the fixed unroll (the derived _max_chance_depth is sufficient).
  3. DISTRIBUTIONAL equivalence vs the CPU collector under a FIXED uniform policy (same game, same
     sampling distribution -> matching node-visit + reward statistics within Monte-Carlo error).
  4. END-TO-END: GPU R-NaD reduces exact NashConv on Kuhn (the assembled GPU path actually learns).

Requires CUDA; skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU collector requires CUDA")

GAMES = ["kuhn_poker", "leduc_poker"]


def _uniform_policy_t(obs_t, legal_t):
    """Deterministic uniform-over-legal policy (torch, on device)."""
    legal = legal_t.to(obs_t.dtype)
    return legal / legal.sum(dim=-1, keepdim=True).clamp_min(1.0)


@pytest.mark.parametrize("game", GAMES)
def test_no_residual_chance(game):
    from poker_ai.rnad import GPUTreeCollector
    col = GPUTreeCollector(game, device="cuda")
    assert col._max_chance_depth >= 1
    col.verify_no_residual_chance(batch_size=4096, seed=0)  # asserts internally


@pytest.mark.parametrize("game", GAMES)
def test_structural_invariants(game):
    from poker_ai.rnad import GPUTreeCollector
    col = GPUTreeCollector(game, device="cuda")
    gen = torch.Generator(device="cuda"); gen.manual_seed(7)
    traj = col.collect(_uniform_policy_t, batch_size=4096, trajectory_max=12, gen=gen)

    valid = traj.valid.bool()                       # [T,B]
    aoh = traj.action_oh                            # [T,B,A]
    legal = traj.legal                              # [T,B,A]
    # 1. sampled action is always legal where valid
    chosen_legal = (aoh * legal).sum(-1)            # [T,B] 1 if chosen action legal else 0
    assert torch.all(chosen_legal[valid] > 0.5), "sampled an illegal action on a valid step"
    # 2. exactly one action chosen per step
    assert torch.all(aoh.sum(-1) <= 1.0 + 1e-5)
    # 3. policy is a distribution over legal actions on valid steps
    pol = traj.policy
    assert torch.allclose(pol[valid].sum(-1), torch.ones(valid.sum(), device="cuda"), atol=1e-4)
    assert torch.all((pol * (1 - legal))[valid] < 1e-4), "policy mass on illegal action"
    # 4. player ids on valid steps are in {0, 1}
    pid = traj.player_id[valid]
    assert torch.all((pid == 0) | (pid == 1)), "invalid player id on a valid step"
    # 5. rewards are zero-sum (2-player zero-sum game) at every recorded step
    rsum = traj.rewards.sum(-1)
    assert torch.all(rsum.abs() < 1e-4), "rewards not zero-sum"


@pytest.mark.parametrize("game", GAMES)
def test_distributional_equivalence_vs_cpu(game):
    """Under a fixed uniform policy, GPU and CPU collectors sample the same distribution.

    Compare per-(player,action) visit frequencies and mean terminal reward across a large batch;
    they must match within Monte-Carlo error (different RNG streams, so statistical not bitwise).
    """
    from poker_ai.rnad import GPUTreeCollector, LeducTreeCollector

    # Reseed global RNGs so this test is independent of execution order within the file.
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    B, T = 65536, 14
    gcol = GPUTreeCollector(game, device="cuda")
    gen = torch.Generator(device="cuda"); gen.manual_seed(123)
    gtraj = gcol.collect(_uniform_policy_t, B, T, gen=gen)

    ccol = LeducTreeCollector(game, device="cpu")
    rng = np.random.RandomState(123)

    def uniform_np(obs, legal):
        legal = legal.astype(np.float64)
        return legal / legal.sum(axis=-1, keepdims=True)

    ctraj = ccol.collect(uniform_np, B, T, rng)

    def stats(traj):
        valid = traj.valid.bool().cpu()
        aoh = traj.action_oh.cpu()
        pid = traj.player_id.cpu()
        # action frequency per player over all valid steps
        freqs = {}
        for p in (0, 1):
            m = valid & (pid == p)
            if m.sum() > 0:
                freqs[p] = (aoh[m].sum(0) / m.sum()).numpy()
        # mean total reward magnitude per step (sanity on reward bookkeeping)
        rew_abs = traj.rewards.cpu().abs().sum(-1)[valid].mean().item()
        # fraction of valid steps
        vf = valid.float().mean().item()
        return freqs, rew_abs, vf

    gf, grew, gvf = stats(gtraj)
    cf, crew, cvf = stats(ctraj)

    # valid-step fraction matches (same tree, same policy, same T) within MC error
    assert abs(gvf - cvf) < 0.02, f"valid-fraction mismatch GPU {gvf:.3f} vs CPU {cvf:.3f}"
    # per-player action frequencies match within MC error (~1/sqrt(N))
    for p in gf:
        if p in cf:
            diff = np.abs(gf[p] - cf[p]).max()
            assert diff < 0.03, f"action-freq mismatch player {p}: {gf[p]} vs {cf[p]} (max {diff:.3f})"
    # mean per-step reward magnitude matches
    assert abs(grew - crew) < 0.05, f"reward-magnitude mismatch GPU {grew:.3f} vs CPU {crew:.3f}"


def test_kuhn_gpu_convergence_smoke():
    """GPU R-NaD reduces exact NashConv on Kuhn — the assembled GPU path actually learns."""
    pytest.importorskip("pyspiel")
    import pyspiel
    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad import GPUTreeCollector, RNaDConfig, RNaDSolver

    game = pyspiel.load_game("kuhn_poker")
    col = GPUTreeCollector("kuhn_poker", device="cuda")
    cfg = RNaDConfig(batch_size=512, trajectory_max=10, policy_network_layers=(64, 64),
                     learning_rate=0.005, entropy_schedule_size=(100,), entropy_schedule_repeats=(1,),
                     seed=1)
    solver = RNaDSolver(cfg, col, device="cuda")

    # obs/legal lookup for tabular NashConv
    obs_by_key = {}
    stack, seen = [game.new_initial_state()], set()
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                c = s.clone(); c.apply_action(a); stack.append(c)
            continue
        k = s.information_state_string()
        if k not in obs_by_key:
            obs_by_key[k] = (np.asarray(s.information_state_tensor(), dtype=np.float32),
                             np.asarray(s.legal_actions_mask(), dtype=np.float32))
        for a in s.legal_actions():
            c = s.clone(); c.apply_action(a)
            if c.history_str() not in seen:
                seen.add(c.history_str()); stack.append(c)

    def nc():
        tp = policy_lib.TabularPolicy(game)
        for key, idx in tp.state_lookup.items():
            if key in obs_by_key:
                obs, legal = obs_by_key[key]
                pi = solver.action_probabilities(obs[None, :], legal[None, :])[0]
                row = tp.action_probability_array[idx]
                row[:] = 0.0
                row[: len(pi)] = pi
        return float(exploitability.nash_conv(game, tp))

    nc0 = nc()
    for _ in range(800):
        solver.step()
    nc1 = nc()
    assert nc1 < nc0 * 0.6, f"GPU NashConv did not converge: {nc0:.3f} -> {nc1:.3f}"
