"""CUDA parity gate for the HUNL population solver (P2 bring-up).

Gates the CUDA float64 path of ``population_solver.solve_population`` against
the CPU float64 reference (itself gated against the trusted
``fast_cfr.solve_cfr`` by test_hunl_population_parity.py) on a fixed spot set:
3 river boards x 3 (pot, stack) configs -- including one all-in-truncated
short-stack tree -- with every topology bucket fused at B=6 (> 1), 250 CFR
iterations, float64 (the mandated production dtype).

EMPIRICALLY JUSTIFIED TOLERANCES (measured 2026-06-10, RTX 3070 Ti,
torch 2.12.0+cu130; recorded in autoresearch-session/rebel/p2_cuda_bringup.json
section "parity"). The historical ~1e-3 cuda-vs-cpu band came from float32
index_add_/atomics nondeterminism; at float64 the measured drift on this exact
workload is:

    cuda64 vs cpu64:  avg-strategy L-inf 4.7e-10, value L-inf 1.5e-7 chips
                      (3.1e-11 of pot); zero-sum identity residual 5.5e-12
    cuda64 run-to-run: avg-strategy L-inf 3.2e-11, value L-inf 6.6e-9 chips

Gates are set ~200-600x above the measured drift (and still ~4 orders below
any decision-relevant scale): avg-strategy <= 1e-7, values <= 1e-4 chips,
identity residual <= 1e-9.

The float32 CUDA variant is EXPERIMENTAL (profiling only -- float64 stays
mandatory per the committed 0c contract): it diverges to a different
near-equilibrium point (measured avg-strategy L-inf 0.235, value 2.1e-2 of
pot at 250 iters). Its separate gate below is a loose trajectory-sanity band,
not an exactness claim.

Auto-skips when CUDA is unavailable.
"""
import numpy as np
import pytest
import torch

from poker_ai.rebel.hunl import subgame_spec as sgs
from poker_ai.rebel.hunl import population_solver as pop
from poker_ai.rebel import turn_river as trv

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable")

N_ITERATIONS = 250
B_MAX = 32  # each 6-element bucket solves as ONE fused chunk

# measured-drift-derived gates (see module docstring)
TOL_AVG_STRATEGY = 1e-7
TOL_VALUE_CHIPS = 1e-4
TOL_IDENTITY = 1e-9
TOL_F32_VALUE_OVER_POT = 0.1  # EXPERIMENTAL float32 band (measured 2.1e-2)


def _board(*cards):
    return tuple(trv.parse_card(c) for c in cards)


RIVER_BOARDS = [
    _board("Ah", "Kd", "7c", "2s", "Jh"),
    _board("Qs", "Js", "9s", "3d", "2c"),
    _board("8c", "8d", "Kh", "4s", "4d"),
]

CONFIGS = {
    "mid_nn75": dict(pot=400, stack0=400, stack1=400, first_to_act=0),
    "census_p95_nn411": dict(pot=4850, stack0=17575, stack1=17575, first_to_act=1),
    "allin_trunc_nn10": dict(pot=3000, stack0=150, stack1=600, first_to_act=1),
}


def _make_specs():
    specs = []
    for cfg in CONFIGS.values():
        for bi, board in enumerate(RIVER_BOARDS):
            for s in range(2):
                rng = np.random.default_rng(1000 + 97 * bi + s)
                r0 = rng.random(sgs.N_GLOBAL_HANDS)
                r1 = rng.random(sgs.N_GLOBAL_HANDS)
                specs.append(sgs.SubgameSpec(
                    street="river", board=board, pot=cfg["pot"],
                    stack0=cfg["stack0"], stack1=cfg["stack1"],
                    first_to_act=cfg["first_to_act"],
                    r0=r0 / r0.sum(), r1=r1 / r1.sum()))
    return specs


def _solve(specs, device, dtype):
    return pop.solve_population(specs, n_iterations=N_ITERATIONS,
                                dtype=dtype, device=device, b_max=B_MAX)


@pytest.fixture(scope="module")
def runs():
    specs = _make_specs()
    cpu64 = _solve(specs, "cpu", torch.float64)
    cuda64 = _solve(specs, "cuda", torch.float64)
    return specs, cpu64, cuda64


def _max_diffs(res_a, res_b):
    s = v = 0.0
    for a, b in zip(res_a, res_b, strict=True):
        s = max(s, float(np.max(np.abs(a.avg_strategy - b.avg_strategy))))
        v = max(v, float(np.max(np.abs(a.v0 - b.v0))),
                float(np.max(np.abs(a.v1 - b.v1))))
    return s, v


def test_spot_set_has_fused_buckets_and_allin_truncation(runs):
    specs, cpu64, _ = runs
    from collections import Counter
    sizes = Counter(r.topology_key for r in cpu64)
    assert len(sizes) == 3  # three distinct topology buckets
    assert all(b == 6 for b in sizes.values())  # every bucket fused at B>1
    # the short-stack config really produces an all-in-truncated tree
    import fast_cfr
    spec = next(s for s in specs if s.stack0 == 150)
    tree = spec.tree()
    assert any(
        tree["terminal_type"][i] == fast_cfr.T_SHOWDOWN
        and (int(tree["stacks_h"][i]) == 0) != (int(tree["stacks_v"][i]) == 0)
        for i in range(tree["n_nodes"]))


def test_cuda64_avg_strategy_and_values_match_cpu64(runs):
    _, cpu64, cuda64 = runs
    d_strat, d_val = _max_diffs(cuda64, cpu64)
    print(f"\n[CUDA parity] cuda64 vs cpu64: avg-strategy L-inf {d_strat:.3e} "
          f"(tol {TOL_AVG_STRATEGY:.0e}), value L-inf {d_val:.3e} chips "
          f"(tol {TOL_VALUE_CHIPS:.0e})")
    assert d_strat <= TOL_AVG_STRATEGY
    assert d_val <= TOL_VALUE_CHIPS


def test_cuda64_zero_sum_identity(runs):
    _, _, cuda64 = runs
    worst = 0.0
    for res in cuda64:
        spec = res.spec
        valid64 = sgs.local_valid_matrix(spec.board).astype(np.float64)
        r0l, r1l = spec.local_ranges()
        lhs = float(r0l @ res.v0 + r1l @ res.v1)
        rhs = float(spec.pot * (r0l @ valid64 @ r1l))
        worst = max(worst, abs(lhs - rhs))
    print(f"\n[CUDA parity] zero-sum identity residual: {worst:.3e} "
          f"(tol {TOL_IDENTITY:.0e})")
    assert worst <= TOL_IDENTITY


def test_cuda32_experimental_band(runs):
    """EXPERIMENTAL float32 variant -- separately gated, NOT production.

    float32 on CUDA lands on a different near-equilibrium point (trajectory
    divergence, not atomics noise); this loose band only documents that it
    stays in the same value neighborhood. float64 remains mandatory.
    """
    specs, cpu64, _ = runs
    cuda32 = _solve(specs, "cuda", torch.float32)
    worst = 0.0
    for a, b in zip(cuda32, cpu64, strict=True):
        dv = max(float(np.max(np.abs(a.v0 - b.v0))),
                 float(np.max(np.abs(a.v1 - b.v1))))
        worst = max(worst, dv / a.spec.pot)
    print(f"\n[CUDA parity/EXPERIMENTAL] float32 value L-inf/pot: {worst:.3e} "
          f"(band {TOL_F32_VALUE_OVER_POT})")
    assert worst <= TOL_F32_VALUE_OVER_POT
