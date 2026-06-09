"""Parity gates P-A..P-F for the lazy HUNL population solver (p1_design.json STEP 6).

Ground truth = ``scripts/fast_cfr.py::solve_cfr`` (the trusted float32 CPU reference)
plus ``poker_ai/rebel/turn_river.py``'s verified extraction/convention machinery
(``subgame_value_pass``, ``turn_leaf_river_cfv`` and the net-from-river ->
net-from-turn counterfactual offset of ``make_exact_river_showdown_fn`` L241-242).

Gates (fixed seeds, CPU-only -- run with CUDA_VISIBLE_DEVICES=""):
  P-A  B=1 kernel exactness vs solve_cfr (avg-strategy L-inf <= 1e-4 vs the float32
       reference) + the zero-sum convention identity
       sum(r0*v0) + sum(r1*v1) == pot * (r0 @ valid @ r1)  to <= 1e-9 in float64.
  P-B  batched B>1 vs B=1 per-element exactness (< 1e-9, same dtype), incl. b_max
       chunking invariance and mixed-topology bucket routing.
  P-C  batched root value pass vs turn_river.subgame_value_pass per element.
  P-D  terminal_eval rank/W/L/T/valid parity vs solver.py's matrices on >= 3 real
       river boards (exact equality).
  P-E  (i) short-stack all-in-truncated tree exactness; (ii) turn river-deal cut
       leaves: batched exact-river injection + convention offset vs the verified
       single-spot path turn_leaf_river_cfv.
  P-F  gadget-root sanity: follow/terminate probabilities bounded, extreme
       terminate values drive follow -> 0 / follow -> 1, and the follow->1 limit
       recovers the ungated solve.

float32-vs-float64 drift on real HUNL trees is measured inside P-A and printed as
a recorded datum (run pytest with -s to see the numbers).
"""
import itertools

import numpy as np
import pytest
import torch

from poker_ai.rebel.hunl import tree_builder as tb  # wires scripts/ into sys.path
from poker_ai.rebel.hunl import subgame_spec as sgs
from poker_ai.rebel.hunl import terminal_eval as tev
from poker_ai.rebel.hunl import population_solver as pop

import fast_cfr  # noqa: E402  (protected surface; imported, never edited)
import solver as slv  # noqa: E402  (protected surface; imported, never edited)
from poker_ai.rebel import turn_river as trv  # noqa: E402

DEVICE = "cpu"
DTYPE = torch.float64


def _board(*cards):
    return tuple(trv.parse_card(c) for c in cards)


RIVER_BOARDS = [
    _board("Ah", "Kd", "7c", "2s", "Jh"),
    _board("Qs", "Js", "9s", "3d", "2c"),
    _board("8c", "8d", "Kh", "4s", "4d"),
]
TURN_BOARD = _board("Ah", "Kd", "7c", "2s")

# Small-chip spots keep the gates fast AND keep the float64 identity assert at
# its design threshold (1e-9 absolute at chips scale).
PA_CONFIG = dict(pot=400, stack0=400, stack1=400, first_to_act=0)
PE_CONFIG = dict(pot=3000, stack0=150, stack1=600, first_to_act=1)  # all-in truncated
PF_CONFIG = dict(pot=3000, stack0=150, stack1=150, first_to_act=1)


def _global_ranges(seed):
    """Fixed-seed global [1326] ranges, float32-representable (so the float32
    reference and the float64 kernel consume bit-identical inputs)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        r = rng.random(sgs.N_GLOBAL_HANDS)
        r = (r / r.sum()).astype(np.float32).astype(np.float64)
        out.append(r)
    return out[0], out[1]


def _make_spec(board, config, seed, street="river"):
    r0, r1 = _global_ranges(seed)
    return sgs.SubgameSpec(
        street=street,
        board=tuple(board),
        pot=config["pot"],
        stack0=config["stack0"],
        stack1=config["stack1"],
        first_to_act=config["first_to_act"],
        r0=r0,
        r1=r1,
    )


def _avg64(strategy_sum):
    """float64 twin of turn_river._average_strategy_array (zero rows stay zero)."""
    s = np.asarray(strategy_sum, dtype=np.float64)
    denom = s.sum(axis=1, keepdims=True)
    return np.divide(s, denom, out=np.zeros_like(s), where=denom > 1e-12)


def _reference_solve(spec, n_iterations):
    """Trusted float32 reference on the identical tree/ranges/matrices."""
    tree = tb.build_street_tree(
        spec.pot, spec.stack0, spec.stack1, spec.first_to_act, spec.street)
    win, lose, tie, valid = tev.showdown_matrices(spec.board)
    r0l, r1l = spec.local_ranges()
    regret, strat = fast_cfr.solve_cfr(
        tree, spec.n_hands, win, lose, tie, valid,
        spec.pot, spec.stack0, spec.stack1,
        n_iterations=n_iterations,
        hero_range=r0l.astype(np.float32),
        villain_range=r1l.astype(np.float32),
    )
    return tree, valid, regret, strat


# ---------------------------------------------------------------------------
# Index-map / spec sanity (underpins P-D's conflict-gather claim)
# ---------------------------------------------------------------------------

def test_index_maps_round_trip():
    board = RIVER_BOARDS[0]
    hands = sgs.local_hands(board)
    remaining = sorted(set(range(52)) - set(board))
    assert hands == list(itertools.combinations(remaining, 2))
    assert len(hands) == sgs.RIVER_H == 1081
    assert len(sgs.local_hands(TURN_BOARD)) == sgs.TURN_H == 1128
    assert len(sgs.GLOBAL_HANDS) == sgs.N_GLOBAL_HANDS == 1326

    l2g = sgs.local_to_global(board)
    g2l = sgs.global_to_local(board)
    assert all(sgs.GLOBAL_HANDS[l2g[i]] == h for i, h in enumerate(hands))
    assert np.array_equal(g2l[l2g], np.arange(len(hands)))
    off_board_mask = np.ones(sgs.N_GLOBAL_HANDS, dtype=bool)
    off_board_mask[l2g] = False
    assert np.all(g2l[off_board_mask] == -1)

    # scatter/gather round trip
    vec = np.arange(len(hands), dtype=np.float64)
    g = sgs.scatter_global(board, vec)
    assert np.array_equal(sgs.gather_local(board, g), vec)

    # global conflict matrix: shared-card pairs are invalid
    valid = sgs.global_valid_matrix()
    i_ab = sgs.hand_index(0, 1)
    j_bc = sgs.hand_index(1, 2)
    k_cd = sgs.hand_index(2, 3)
    assert valid[i_ab, i_ab] == 0.0
    assert valid[i_ab, j_bc] == 0.0  # share card 1
    assert valid[i_ab, k_cd] == 1.0


# ---------------------------------------------------------------------------
# P-D: terminal_eval parity vs solver.py's matrices (exact)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("board_index", [0, 1, 2])
def test_pd_terminal_eval_matches_solver_matrices(board_index):
    board = RIVER_BOARDS[board_index]
    win, lose, tie, valid = tev.showdown_matrices(board)
    ranks = tev.board_rank_vector(board)
    assert ranks.dtype == np.int32 and ranks.shape == (sgs.RIVER_H,)

    ref = slv.StreetSolver(list(board), 200, 400, 400, True)
    np.testing.assert_array_equal(valid, ref.valid)
    np.testing.assert_array_equal(win, ref.win_m)
    np.testing.assert_array_equal(lose, ref.lose_m)
    np.testing.assert_array_equal(tie, ref.tie_m)


# ---------------------------------------------------------------------------
# P-A: B=1 kernel exactness + zero-sum convention identity
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pa_run():
    spec = _make_spec(RIVER_BOARDS[0], PA_CONFIG, seed=7)
    res = pop.solve_population(
        [spec], n_iterations=30, dtype=DTYPE, device=DEVICE)[0]
    return spec, res


def test_pa_b1_kernel_exactness_vs_solve_cfr(pa_run):
    spec, res = pa_run
    _, _, _, ref_strat = _reference_solve(spec, n_iterations=30)
    drift = float(np.max(np.abs(res.avg_strategy - _avg64(ref_strat))))
    print(f"\n[P-A] avg-strategy L-inf, float64 kernel vs float32 solve_cfr: {drift:.3e}")
    assert drift <= 1e-4


def test_pa_zero_sum_convention_identity(pa_run):
    spec, res = pa_run
    valid64 = sgs.local_valid_matrix(spec.board).astype(np.float64)
    r0l, r1l = spec.local_ranges()
    lhs = float(r0l @ res.v0 + r1l @ res.v1)
    rhs = float(spec.pot * (r0l @ valid64 @ r1l))
    resid = abs(lhs - rhs)
    print(f"\n[P-A] zero-sum identity residual: {resid:.3e} (lhs={lhs:.9f} rhs={rhs:.9f})")
    assert resid <= 1e-9


# ---------------------------------------------------------------------------
# P-B: batching invariance (B>1 vs B=1, chunking, mixed-topology routing)
# ---------------------------------------------------------------------------

def test_pb_batched_matches_b1_per_element():
    specs = [_make_spec(RIVER_BOARDS[i], PA_CONFIG, seed=20 + i) for i in range(3)]
    specs.append(_make_spec(RIVER_BOARDS[0], PA_CONFIG, seed=31))
    specs.append(_make_spec(RIVER_BOARDS[1], PE_CONFIG, seed=32))  # second topology bucket

    batched = pop.solve_population(specs, n_iterations=12, b_max=8,
                                   dtype=DTYPE, device=DEVICE)
    chunked = pop.solve_population(specs, n_iterations=12, b_max=2,
                                   dtype=DTYPE, device=DEVICE)
    assert len({r.topology_key for r in batched}) == 2  # two real topology buckets

    max_strat = 0.0
    max_val = 0.0
    for i, spec in enumerate(specs):
        solo = pop.solve_population([spec], n_iterations=12,
                                    dtype=DTYPE, device=DEVICE)[0]
        for other in (batched[i], chunked[i]):
            assert other.topology_key == solo.topology_key
            max_strat = max(max_strat, float(np.max(np.abs(
                other.avg_strategy - solo.avg_strategy))))
            max_val = max(max_val, float(np.max(np.abs(other.v0 - solo.v0))),
                          float(np.max(np.abs(other.v1 - solo.v1))))
    print(f"\n[P-B] batched/chunked vs B=1: avg-strategy max |diff| {max_strat:.3e}, "
          f"value max |diff| {max_val:.3e}")
    assert max_strat < 1e-9
    assert max_val < 1e-9


# ---------------------------------------------------------------------------
# P-C: batched root value pass vs turn_river.subgame_value_pass
# ---------------------------------------------------------------------------

def test_pc_value_pass_matches_subgame_value_pass(pa_run):
    spec, res = pa_run
    solver = slv.StreetSolver(
        list(spec.board), spec.pot, spec.stack0, spec.stack1,
        hero_first=(spec.first_to_act == 0))
    # the lazily built tree is shape-identical to the trusted solver's tree
    assert tb.topology_key(solver._tree) == res.topology_key
    assert [tuple(h) for h in solver.hands] == sgs.local_hands(spec.board)

    r0l, r1l = spec.local_ranges()
    ref_v0, ref_v1 = trv.subgame_value_pass(solver, res.avg_strategy, r0l, r1l)
    diff = max(float(np.max(np.abs(res.v0 - ref_v0))),
               float(np.max(np.abs(res.v1 - ref_v1))))
    print(f"\n[P-C] value pass vs subgame_value_pass max |diff|: {diff:.3e}")
    assert diff <= 1e-8


# ---------------------------------------------------------------------------
# P-E (i): short-stack all-in-truncated tree
# ---------------------------------------------------------------------------

def test_pe_short_stack_allin_truncated_tree():
    spec = _make_spec(RIVER_BOARDS[1], PE_CONFIG, seed=40)
    tree = tb.build_street_tree(
        spec.pot, spec.stack0, spec.stack1, spec.first_to_act, spec.street)
    # all-in truncation is actually present: a showdown terminal where exactly
    # one player is felted (acting==0 short-circuit / call-clip paths)
    truncated = [
        i for i in range(tree["n_nodes"])
        if tree["terminal_type"][i] == fast_cfr.T_SHOWDOWN
        and (int(tree["stacks_h"][i]) == 0) != (int(tree["stacks_v"][i]) == 0)
    ]
    assert truncated, "expected an all-in-truncated showdown terminal"

    res = pop.solve_population([spec], n_iterations=30, dtype=DTYPE, device=DEVICE)[0]
    _, valid, _, ref_strat = _reference_solve(spec, n_iterations=30)
    drift = float(np.max(np.abs(res.avg_strategy - _avg64(ref_strat))))

    r0l, r1l = spec.local_ranges()
    lhs = float(r0l @ res.v0 + r1l @ res.v1)
    rhs = float(spec.pot * (r0l @ valid.astype(np.float64) @ r1l))
    resid = abs(lhs - rhs)
    print(f"\n[P-E/short-stack] avg-strategy L-inf vs solve_cfr: {drift:.3e}, "
          f"identity residual: {resid:.3e}")
    assert drift <= 1e-4
    assert resid <= 1e-9


# ---------------------------------------------------------------------------
# P-E (ii): turn river-deal cut leaves -- batched exact-river injection and the
# net-from-river -> net-from-turn counterfactual offset, vs the verified path
# ---------------------------------------------------------------------------

def test_pe_turn_river_deal_leaf_and_offset_parity():
    pot, s0, s1, first = 800, 500, 500, 0
    river_iters = 12
    tree = tb.build_street_tree(pot, s0, s1, first, "turn")
    turn_hands = sgs.local_hands(TURN_BOARD)
    n = len(turn_hands)

    rng = np.random.default_rng(55)
    hr = rng.random(n).astype(np.float32).astype(np.float64) * 0.5
    vr = rng.random(n).astype(np.float32).astype(np.float64) * 0.5

    show = tree["showdown_idx"]
    sh = tree["stacks_h"]
    sv = tree["stacks_v"]
    potN = tree["pot"]
    # one all-in river-deal cut (trivial river subgames -> near-exact parity) and
    # one betting-continuation cut (river betting; CFR float32-vs-float64 band)
    allin_t = next(int(i) for i in show if int(sh[i]) == 0 and int(sv[i]) == 0)
    bet_t = next(int(i) for i in show if 0 < int(sh[i]) < s0)

    refs = {
        term: trv.turn_leaf_river_cfv(
            list(TURN_BOARD), int(potN[term]), int(sh[term]), int(sv[term]), True,
            turn_hands, hr, vr, river_iters=river_iters, backend="cpu")
        for term in (allin_t, bet_t)
    }

    # The batched leaf HOOK = exact-river population injection + the
    # net-from-turn offset exactly as make_exact_river_showdown_fn L241-242
    # applies it on the verified single path.
    valid64 = sgs.local_valid_matrix(TURN_BOARD).astype(np.float64)
    leaf = tev.make_exact_river_leaf(
        n_iterations=river_iters, dtype=DTYPE, device=DEVICE)
    spec = sgs.SubgameSpec(
        street="turn", board=TURN_BOARD, pot=pot, stack0=s0, stack1=s1,
        first_to_act=first,
        r0=sgs.scatter_global(TURN_BOARD, hr), r1=sgs.scatter_global(TURN_BOARD, vr))
    show_sel = np.asarray([allin_t, bet_t], dtype=np.int64)
    ctx = {
        "iteration": None,
        "specs": [spec],
        "trees": [tree],
        "showdown_indices": show_sel,
        "hero_reach": hr[np.newaxis, np.newaxis, :].repeat(2, axis=1),
        "villain_reach": vr[np.newaxis, np.newaxis, :].repeat(2, axis=1),
        "default_hero_values": np.zeros((1, 2, n)),
        "default_villain_values": np.zeros((1, 2, n)),
        "valid_ms": [sgs.local_valid_matrix(TURN_BOARD)],
        "pot_starts": np.asarray([pot], dtype=np.float64),
        "hero_stack_starts": np.asarray([s0], dtype=np.float64),
        "villain_stack_starts": np.asarray([s1], dtype=np.float64),
        "terminal_pots": potN[show_sel][np.newaxis, :],
        "terminal_stacks_h": sh[show_sel][np.newaxis, :],
        "terminal_stacks_v": sv[show_sel][np.newaxis, :],
    }
    hook_h, hook_v = leaf(ctx)
    for k, term in enumerate((allin_t, bet_t)):
        hi_cut = s0 - int(sh[term])
        vi_cut = s1 - int(sv[term])
        h_off = hi_cut * (vr @ valid64.T)
        v_off = vi_cut * (hr @ valid64)
        # raw net-from-river parity (offset added back) vs turn_leaf_river_cfv
        d_raw = max(float(np.max(np.abs((hook_h[0, k] + h_off) - refs[term][0]))),
                    float(np.max(np.abs((hook_v[0, k] + v_off) - refs[term][1]))))
        # offset-applied (net-from-turn-start) parity, the convention trap
        d_off = max(float(np.max(np.abs(hook_h[0, k] - (refs[term][0] - h_off)))),
                    float(np.max(np.abs(hook_v[0, k] - (refs[term][1] - v_off)))))
        tol = 1e-6 if term == allin_t else 1e-3 * float(potN[term])
        kind = "all-in cut" if term == allin_t else f"betting cut (pot_cut={int(potN[term])})"
        print(f"\n[P-E/turn-leaf] {kind}: net-from-river max |diff| {d_raw:.3e}, "
              f"offset-applied max |diff| {d_off:.3e} (tol {tol:.3e})")
        assert d_raw <= tol, f"terminal {term}: raw leaf diff {d_raw:.3e} > {tol:.3e}"
        assert d_off <= tol, f"terminal {term}: offset-applied diff {d_off:.3e} > {tol:.3e}"


# ---------------------------------------------------------------------------
# P-F: gadget root sanity (follow/terminate bounds + extremes)
# ---------------------------------------------------------------------------

def test_pf_gadget_root_bounds_and_extremes():
    spec = _make_spec(RIVER_BOARDS[2], PF_CONFIG, seed=60)
    n = spec.n_hands
    iters = 60
    base = pop.solve_population([spec], n_iterations=iters,
                                dtype=DTYPE, device=DEVICE)[0]
    assert base.gadget is None

    hi = pop.solve_population(
        [spec], n_iterations=iters, dtype=DTYPE, device=DEVICE,
        gadget_player=1, gadget_opp_cfv=[np.full(n, 1e6)])[0]
    lo = pop.solve_population(
        [spec], n_iterations=iters, dtype=DTYPE, device=DEVICE,
        gadget_player=1, gadget_opp_cfv=[np.full(n, -1e6)])[0]

    for res in (hi, lo):
        follow = res.gadget["follow_final"]
        assert follow.shape == (n,)
        assert np.all(follow >= 0.0) and np.all(follow <= 1.0)

    reachable = hi.gadget["den"] > 1e-12
    assert reachable.any()
    f_hi = float(hi.gadget["follow_final"][reachable].max())
    f_lo = float(lo.gadget["follow_final"][reachable].min())
    print(f"\n[P-F] follow prob -- huge terminate value: max {f_hi:.3e} (want ~0); "
          f"awful terminate value: min {f_lo:.6f} (want ~1)")
    assert f_hi < 0.05
    assert f_lo > 0.95

    # follow->1 limit recovers the ungated solve. The iteration-1 gf=0.5
    # transient (iig_gadget's no-regret default) perturbs the CFR trajectory,
    # so per-hand strategies land on a different point of the near-equilibrium
    # set (equilibrium-selection noise; loose band only) -- the recovery claim
    # is on the VALUES, which must match to a small fraction of the pot.
    strat_drift = float(np.max(np.abs(lo.avg_strategy - base.avg_strategy)))
    val_drift = max(float(np.max(np.abs(lo.v0 - base.v0))),
                    float(np.max(np.abs(lo.v1 - base.v1))))
    print(f"[P-F] follow->1 vs ungated: value L-inf {val_drift:.3e} "
          f"(tol {2e-3 * spec.pot:.1f} chips), avg-strategy L-inf {strat_drift:.3e}")
    assert val_drift <= 2e-3 * spec.pot
    assert strat_drift <= 0.15  # equilibrium-selection band, not an exactness gate
