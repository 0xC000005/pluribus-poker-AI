"""Parity / invariant tests for the GPU R-NaD collector (CUDANativeRNaDCollector).

CUDA-guarded. The GPU collector must match the CPU CompiledNativeRNaDCollector SEMANTICS
(deck RNG differs by construction, so equivalence is distributional, not bitwise — same policy
as GPUTreeCollector). Run with the numba nvvm path on LD_LIBRARY_PATH (see CLAUDE.md).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()
try:
    from numba import cuda as _nbcuda

    _HAS_NUMBA_CUDA = _nbcuda.is_available()
except Exception:  # pragma: no cover
    _HAS_NUMBA_CUDA = False

pytestmark = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_NUMBA_CUDA), reason="CUDA + numba.cuda required"
)


def _uniform_pf_t(obs, legal):
    return legal / legal.sum(-1, keepdim=True).clamp(min=1.0)


def _uniform_pf_np(obs, legal):
    s = legal.sum(-1, keepdims=True)
    s[s == 0] = 1.0
    return legal / s


def _make_collector(**kw):
    from poker_ai.rnad.cuda_collector import CUDANativeRNaDCollector

    return CUDANativeRNaDCollector(seed=42, initial_chips=1000, device="cuda", **kw)


def test_structural_invariants():
    col = _make_collector()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(123)
    B, T = 256, 24
    tr = col.collect(_uniform_pf_t, B, T, gen)

    assert tr.obs.shape == (T, B, 126)
    assert tr.legal.shape == (T, B, 9)
    assert tr.player_id.shape == (T, B)
    assert tr.valid.shape == (T, B)
    assert tr.rewards.shape == (T, B, 2)
    assert tr.action_oh.shape == (T, B, 9)
    assert tr.policy.shape == (T, B, 9)
    for v in tr.__dict__.values():
        assert v.is_cuda and v.dtype == torch.float32

    assert bool(((tr.valid == 0) | (tr.valid == 1)).all())
    assert bool(((tr.legal == 0) | (tr.legal == 1)).all())

    vm = tr.valid > 0.5
    assert bool((tr.action_oh[vm].sum(-1) == 1).all())
    assert bool((tr.action_oh[vm].max(-1).values == 1).all())
    ps = tr.policy[vm].sum(-1)
    assert torch.allclose(ps, torch.ones_like(ps), atol=1e-4)
    assert float((tr.policy * (tr.legal <= 0))[vm].sum()) == pytest.approx(0.0, abs=1e-5)
    assert bool(((tr.player_id[vm] == 0) | (tr.player_id[vm] == 1)).all())

    # rewards: zero-sum per game, nonzero at <=1 step per game.
    assert float(tr.rewards.sum(-1).abs().max()) == pytest.approx(0.0, abs=1e-4)
    nz = (tr.rewards.abs().sum(-1) > 1e-9).sum(0)
    assert int(nz.max()) <= 1

    assert col.last_metrics["illegal_records"] == 0
    assert col.last_metrics["needs_python_showdown"] == 0
    assert col.last_metrics["steps"] > 0


def test_seeded_reproducibility():
    col = _make_collector()
    gen = torch.Generator(device="cuda")
    B, T = 128, 16

    gen.manual_seed(777)  # FIX: re-seed before EACH collect — gen is stateful.
    a = col.collect(_uniform_pf_t, B, T, gen)
    gen.manual_seed(777)
    b = col.collect(_uniform_pf_t, B, T, gen)

    for k in a.__dict__:
        assert torch.equal(getattr(a, k), getattr(b, k)), f"field {k} not reproducible"


def test_distributional_equivalence_vs_cpu():
    from poker_ai.research.native_rnad import CompiledNativeRNaDCollector

    B, T = 2048, 16
    gcol = _make_collector()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(42)
    g = gcol.collect(_uniform_pf_t, B, T, gen)

    # NOTE: CPU draws rng.randint(0, 2**30-1) (exclusive) vs GPU torch.randint(0, 2**30);
    # deck RNG algorithms differ entirely -> distributional comparison only, never bitwise.
    ccol = CompiledNativeRNaDCollector(
        collector_batch_size=256, initial_chips=1000, seed=42, device="cpu"
    )
    rng = np.random.RandomState(42)
    c = ccol.collect(_uniform_pf_np, B, T, rng)

    gv, cv = g.valid > 0.5, c.valid > 0.5
    assert abs(float(g.valid.mean()) - float(c.valid.mean())) < 0.05
    # Acting-seat distribution is naturally asymmetric in HU (~0.44: SB acts first preflop, so
    # seat 0 gets more decisions). The parity check is GPU == CPU, not == 0.5.
    assert abs(
        float(g.player_id[gv].float().mean()) - float(c.player_id[cv].float().mean())
    ) < 0.04

    gf = g.action_oh[gv].mean(0).cpu()
    cf = c.action_oh[cv].mean(0).cpu()
    assert float((gf - cf).abs().max()) < 0.05

    # reward magnitude (mean abs terminal payoff) in the same ballpark.
    gr = float(g.rewards[g.rewards.abs().sum(-1) > 1e-9].abs().mean())
    cr = float(c.rewards[c.rewards.abs().sum(-1) > 1e-9].abs().mean())
    assert abs(gr - cr) < 0.1


def test_first_step_feature_invariants():
    col = _make_collector()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(5)
    B = 64
    tr = col.collect(_uniform_pf_t, B, 1, gen)
    obs0 = tr.obs[0]  # [B,126], all live at t=0
    assert bool((obs0[:, 0:52].sum(-1) == 2.0).all())  # exactly 2 hole cards
    assert bool((obs0[:, 52:104].sum(-1) == 0.0).all())  # no community preflop
    assert bool((obs0[:, 104] == 1.0).all())  # preflop round one-hot
    assert bool((obs0[:, 104:108].sum(-1) == 1.0).all())
    assert bool((obs0[:, 113] == 0.0).all())  # n_raises=0 at start
    assert bool((tr.legal[0, :, 0] == 1.0).all())  # fold legal
    assert bool((tr.legal[0, :, 1] == 1.0).all())  # call legal


def test_end_to_end_rnad_solver_gpu():
    from poker_ai.rnad.solver import RNaDConfig, RNaDSolver

    cfg = RNaDConfig(
        batch_size=256, trajectory_max=24, policy_network_layers=(128, 128),
        entropy_schedule_size=(1000,), entropy_schedule_repeats=(1,), seed=11,
    )
    col = _make_collector()
    solver = RNaDSolver(cfg, col, device="cuda")
    losses = [solver.step()["loss"] for _ in range(3)]
    assert all(np.isfinite(x) for x in losses)
    assert col.last_metrics["illegal_records"] == 0
    assert col.last_metrics["needs_python_showdown"] == 0


class _UniformAdapter:
    """Minimal opponent adapter: uniform over legal actions (matches the _adapter contract)."""

    def probs(self, features, legal, device):
        legal = np.asarray(legal, dtype=np.float32)
        s = legal.sum()
        return legal / s if s > 0 else np.full_like(legal, 1.0 / legal.shape[0])


def test_population_opponent_path():
    col = _make_collector(
        opponent_policies=[_UniformAdapter(), _UniformAdapter()],
        opponent_meta_strategy=[0.5, 0.5],
    )
    gen = torch.Generator(device="cuda")
    gen.manual_seed(99)
    B, T = 512, 24
    tr = col.collect(_uniform_pf_t, B, T, gen)

    m = col.last_metrics
    assert m["train_opponent_mode"] == "population"
    assert m["population_opponent_size"] == 2
    assert m["opponent_controlled_steps"] > 0
    assert m["learner_controlled_steps"] > 0
    assert m["illegal_records"] == 0
    # invariants still hold with opponents in the loop.
    vm = tr.valid > 0.5
    ps = tr.policy[vm].sum(-1)
    assert torch.allclose(ps, torch.ones_like(ps), atol=1e-4)
    assert float(tr.rewards.sum(-1).abs().max()) == pytest.approx(0.0, abs=1e-4)
