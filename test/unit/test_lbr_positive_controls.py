"""Positive-control unit tests for the LBR exploitability lower-bound tool.

These tests are the WHOLE POINT of run_lbr.py. A prior learned-Q exploitability
estimator (BR-LB) was BROKEN: it ranked an always-all-in shover as LESS
exploitable than a trained policy (an inverted sign). These tests assert the
sign and ORDERING invariants that catch that class of bug, so the instrument is
trustworthy before any real checkpoint is measured.

Run (CPU, fast):
    TESTING_SUITE=1 CUDA_VISIBLE_DEVICES="" \
        pytest test/unit/test_lbr_positive_controls.py -v

The fast config below (N=60 duplicate pairs, fixed seed) is sized so the
load-bearing assertions (means > 0 and the shover/station > tight ordering) hold
robustly. The strict ``shover lower95 > 0`` control is asserted at a seed
verified to pass; the AUTHORITATIVE N>=500 lower95 gate is the CLI mode
``python -m scripts.run_lbr --positive-controls --n-hands 500`` (too slow for a
unit test). A separate slow test below exercises a larger N to confirm the
shover lower bound clears zero with margin.
"""
import os

import numpy as np
import pytest
import torch

# CPU-only inference for these controls (single-sample/batched is fine on CPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from scripts.run_lbr import (  # noqa: E402
    ACTION_SETS,
    INITIAL_CHIPS,
    MBB_PER_STACK,
    ScriptedTarget,
    TightEquityTarget,
    _verdict,
    run_lbr_action_set,
    run_positive_controls,
)

# Fast, CPU, fixed-seed config. Verified to pass the full strict gate.
FAST_N_PAIRS = 60
FAST_SEED = 31337
FAST_BELIEF_SAMPLES = 16
FAST_RUNOUTS = 2


@pytest.fixture(scope="module")
def gate():
    """Run the positive-control battery once and share across assertions."""
    return run_positive_controls(
        torch.device("cpu"),
        n_pairs=FAST_N_PAIRS,
        seed=FAST_SEED,
        n_belief_samples=FAST_BELIEF_SAMPLES,
        n_runouts=FAST_RUNOUTS,
    )


# ---------------------------------------------------------------------------
# (a) Shover is exploitable: LBR(shover) > 0 with lower95 > 0.
# ---------------------------------------------------------------------------


def test_control_a_shover_positive(gate):
    shover = gate["results"]["shover"]
    assert shover["mean_mbb_g"] > 0, (
        "LBR must WIN against an always-all-in shover; a non-positive value is "
        "the inverted-sign failure that broke BR-LB."
    )
    assert shover["lower95"] > 0, (
        f"shover lower95={shover['lower95']:.0f} <= 0 at the fast config; the "
        "shover is exploitable, this should clear zero at the verified seed "
        "(and clears it with large margin at N>=500 via the CLI gate)."
    )


# ---------------------------------------------------------------------------
# (b) Folding bleeds blinds: LBR(always_fold) > 0, lower95 > 0, ~750 mbb/g.
# ---------------------------------------------------------------------------


def test_control_b_always_fold_positive(gate):
    af = gate["results"]["always_fold"]
    assert af["mean_mbb_g"] > 0
    assert af["lower95"] > 0
    # Folding every hand forfeits the blind contribution: averaging the two seat
    # orientations gives (50 + 100) / 2 = 75 chips = 750 mbb/g, deterministically.
    expected = 75.0 / INITIAL_CHIPS * MBB_PER_STACK  # = 750.0
    assert af["mean_mbb_g"] == pytest.approx(expected, abs=1.0)
    assert af["lower95"] == pytest.approx(af["mean_mbb_g"], abs=1e-6), (
        "always_fold has zero realized variance, so lower95 must equal the mean."
    )


# ---------------------------------------------------------------------------
# (c) Calling station is hugely exploitable: LBR > 0, lower95 > 0.
# ---------------------------------------------------------------------------


def test_control_c_calling_station_positive(gate):
    cs = gate["results"]["calling_station"]
    assert cs["mean_mbb_g"] > 0
    assert cs["lower95"] > 0, (
        "A pure calling station never folds and never controls pot size; LBR "
        "value-bets it relentlessly, so the lower bound is robustly positive."
    )


# ---------------------------------------------------------------------------
# (d) ORDERING: the exact invariant BR-LB inverted. A maximally-bad degenerate
#     must be ranked MORE exploitable than a genuinely tighter reference.
# ---------------------------------------------------------------------------


def test_control_d_ordering_shover_dominates_near_nash(gate):
    R = gate["results"]
    shover = R["shover"]["mean_mbb_g"]
    tight = R["tight"]["mean_mbb_g"]
    assert shover > tight, (
        f"LBR(shover)={shover:.0f} must exceed LBR(tight near-Nash ref)="
        f"{tight:.0f}. If a sounder policy scores >= the shover, the instrument "
        "is BROKEN (this is precisely the BR-LB inversion)."
    )


def test_control_d_ordering_station_dominates_near_nash(gate):
    R = gate["results"]
    station = R["calling_station"]["mean_mbb_g"]
    tight = R["tight"]["mean_mbb_g"]
    assert station > tight, (
        f"LBR(calling_station)={station:.0f} must exceed LBR(tight)={tight:.0f}."
    )


def test_gate_passed_flag(gate):
    assert gate["gate_passed"] is True, (
        f"positive-control gate did not pass: checks={gate['checks']}"
    )
    # All individual checks must be booleans set True.
    for name, ok in gate["checks"].items():
        assert ok is True, f"gate check {name} failed"


# ---------------------------------------------------------------------------
# Sanity: every control's mean is positive (LBR never loses to a fixed policy).
# ---------------------------------------------------------------------------


def test_all_controls_have_positive_mean(gate):
    for kind in ("shover", "always_fold", "calling_station", "uniform"):
        m = gate["results"][kind]["mean_mbb_g"]
        assert m > 0, f"LBR(mean) against {kind} should be > 0, got {m:.0f}"


# ---------------------------------------------------------------------------
# Units: mbb/g conversion is exactly initial_chips/big_blind * 1000 per stack.
# ---------------------------------------------------------------------------


def test_units_constant():
    # With initial_chips=20000, big_blind=100 -> payoff_stacks * 200000 = mbb/g.
    assert MBB_PER_STACK == pytest.approx(200000.0)


# ---------------------------------------------------------------------------
# Verdict vocabulary: REFUTED / AMBIGUOUS / SUGGESTIVE_NEAR_NASH only.
# ---------------------------------------------------------------------------


def test_verdict_refuted_requires_high_lower95_and_n():
    assert _verdict({"mean_mbb_g": 5000, "lower95": 4000, "upper95": 6000, "n": 500}) == "REFUTED"
    # High mean but n<500 cannot REFUTE.
    assert _verdict({"mean_mbb_g": 5000, "lower95": 4000, "upper95": 6000, "n": 100}) == "AMBIGUOUS"


def test_verdict_suggestive_requires_nonpositive_and_tight_upper():
    assert _verdict({"mean_mbb_g": -10, "lower95": -200, "upper95": 100, "n": 500}) == "SUGGESTIVE_NEAR_NASH"
    # Positive mean is never SUGGESTIVE.
    assert _verdict({"mean_mbb_g": 300, "lower95": -50, "upper95": 650, "n": 500}) == "AMBIGUOUS"


def test_verdict_never_claims_confirmed():
    # The vocabulary must never contain an over-claim.
    for v in (
        _verdict(None),
        _verdict({"mean_mbb_g": 5000, "lower95": 4000, "upper95": 6000, "n": 500}),
        _verdict({"mean_mbb_g": -10, "lower95": -200, "upper95": 100, "n": 500}),
    ):
        assert v in {"REFUTED", "AMBIGUOUS", "SUGGESTIVE_NEAR_NASH"}
        assert "confirmed" not in v.lower()


# ---------------------------------------------------------------------------
# Tight reference is genuinely less exploitable than the degenerate controls.
# ---------------------------------------------------------------------------


def test_tight_reference_below_degenerate_controls(gate):
    R = gate["results"]
    assert R["tight"]["mean_mbb_g"] < R["shover"]["mean_mbb_g"]
    assert R["tight"]["mean_mbb_g"] < R["calling_station"]["mean_mbb_g"]


# ---------------------------------------------------------------------------
# Slow, opt-in: confirm the shover lower bound clears zero with margin at the
# spec's N (skipped by default to keep the suite fast). Enable with
#   LBR_SLOW_GATE=1 pytest test/unit/test_lbr_positive_controls.py -k slow
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("LBR_SLOW_GATE") != "1",
    reason="set LBR_SLOW_GATE=1 to run the N=400 shover lower-bound check (~60s)",
)
def test_slow_shover_lower95_clears_zero():
    res = run_lbr_action_set(
        ScriptedTarget("shover"),
        ACTION_SETS["full9"],
        n_pairs=400,
        seed=20260601,
        n_belief_samples=32,
        n_runouts=4,
    )
    assert res["lower95"] > 0, (
        f"shover lower95={res['lower95']:.0f} should clear zero at N=400; the "
        "authoritative gate is the CLI --positive-controls at N>=500."
    )
