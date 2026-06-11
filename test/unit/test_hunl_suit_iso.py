"""Suit-isomorphism canonicalization tests (river PBS net v1).

Suit relabeling is a LOSSLESS game symmetry of HUNL (P0b bundle: allowed as
exact symmetry, not abstraction): permuting the four suits by any of the 24
bijections maps boards to boards and the 1326-hand index to itself, leaves
pots/stacks/actor order untouched, and maps exact subgame solutions to exact
subgame solutions. ``poker_ai/rebel/hunl/suit_iso.py`` implements the
canonical-suit mapping; these tests pin its three contracts:

(1) IDEMPOTENCY  -- the canonical map is a projection: canonicalizing a
    canonical (board, ranges) input returns the identical canonical form.
(2) ROUND-TRIP EXACTNESS -- permuting range / CFV vectors and inverting is
    bit-identical (pure index permutation, no arithmetic), and the induced
    hand permutation is consistent with the board's local hand sets.
(3) SOUNDNESS -- for >= 3 pairs of suit-isomorphic river subgames (same spec
    up to suit relabeling, asymmetric Dirichlet ranges), solving BOTH with the
    parity-gated population solver and mapping the V* targets into canonical
    space must agree. Canonical INPUTS are bit-identical by construction
    (asserted exactly -- this is what licenses post-hoc canonicalization of
    stored training rows). Canonical-mapped V* TARGETS are NOT bit-identical:
    the two isomorphic solves run in different local-hand index orders, so
    f64 GEMM reduction order differs in the last ulp and accumulates over CFR
    iterations. MEASURED (2026-06-10, torch CPU f64, 120 iters, the 3 cases
    below): L-inf 2.3e-11 chips worst-case (~1e-15 of pot). The CPU gate is
    set ~400x above the measured drift (the repo's parity-gate convention,
    cf. test_hunl_cuda_parity.py) and still 4+ orders below the documented
    CUDA f64 band (1e-4 chips), which gates the CUDA variant.
"""
import numpy as np
import pytest
import torch

from poker_ai.rebel.hunl import population_solver as pop
from poker_ai.rebel.hunl import subgame_spec as sgs
from poker_ai.rebel.hunl import suit_iso as siso

N_ITERATIONS = 120           # symmetry holds at ANY iteration count
TOL_SOUNDNESS_CHIPS = 1e-8   # f64 CPU: measured 2.3e-11 chips (GEMM order)
TOL_CUDA_CHIPS = 1e-4        # documented f64 CUDA band (test_hunl_cuda_parity)

# (board, suit perm, (pot, eff_stack)) soundness pairs: distinct tree
# topologies (nn = 9 / 15 / 45), full-derangement suit perms, one
# flush-textured board (3 hearts) to exercise suit-asymmetric strength.
SOUNDNESS_CASES = [
    ((2, 11, 25, 38, 51), (1, 2, 3, 0), (20000, 1000)),    # nn=9
    ((6, 10, 18, 30, 46), (3, 2, 0, 1), (20000, 5000)),    # nn=15
    ((14, 22, 34, 45, 49), (2, 3, 1, 0), (12000, 8000)),   # nn=45, 3 hearts
]


def _random_board(rng):
    return tuple(sorted(rng.choice(52, size=5, replace=False).tolist()))


def _dirichlet_pair(board, rng, a0=0.4, a1=2.5):
    """Asymmetric global range pair supported on the board's local hands."""
    n_local = len(sgs.local_hands(board))
    r0 = sgs.scatter_global(board, rng.dirichlet(np.full(n_local, a0)))
    r1 = sgs.scatter_global(board, rng.dirichlet(np.full(n_local, a1)))
    return r0, r1


# ---------------------------------------------------------------------------
# (0) the permutation group plumbing itself
# ---------------------------------------------------------------------------

def test_suit_perms_are_the_full_group():
    assert len(siso.SUIT_PERMS) == 24
    assert len(set(siso.SUIT_PERMS)) == 24
    assert siso.IDENTITY_PERM in siso.SUIT_PERMS
    for perm in siso.SUIT_PERMS:
        inv = siso.invert_perm(perm)
        assert tuple(perm[inv[s]] for s in range(4)) == (0, 1, 2, 3)
        assert tuple(inv[perm[s]] for s in range(4)) == (0, 1, 2, 3)


def test_hand_perm_is_a_permutation_and_matches_cards():
    rng = np.random.default_rng(7)
    for perm in siso.SUIT_PERMS:
        hp = siso.hand_perm(perm)
        assert hp.shape == (sgs.N_GLOBAL_HANDS,)
        assert np.array_equal(np.sort(hp), np.arange(sgs.N_GLOBAL_HANDS))
    # spot-check the card-level definition on random hands
    for _ in range(200):
        c0, c1 = rng.choice(52, size=2, replace=False)
        i = sgs.hand_index(c0, c1)
        perm = siso.SUIT_PERMS[int(rng.integers(0, 24))]
        j = sgs.hand_index(siso.permute_card(int(c0), perm),
                           siso.permute_card(int(c1), perm))
        assert siso.hand_perm(perm)[i] == j


def test_permute_card_preserves_rank():
    for perm in siso.SUIT_PERMS:
        for card in range(52):
            mapped = siso.permute_card(card, perm)
            assert mapped // 4 == card // 4          # rank invariant
            assert mapped % 4 == perm[card % 4]      # suit relabeled


# ---------------------------------------------------------------------------
# (1) canonical map idempotent
# ---------------------------------------------------------------------------

def test_canonical_board_idempotent_random():
    rng = np.random.default_rng(11)
    for _ in range(300):
        board = _random_board(rng)
        canon, perm = siso.canonicalize(board)
        assert siso.permute_board(board, perm) == canon
        canon2, _ = siso.canonicalize(canon)
        assert canon2 == canon
        # canonical board is the orbit minimum
        for p in siso.SUIT_PERMS:
            assert siso.permute_board(board, p) >= canon


def test_canonical_form_idempotent_with_ranges():
    """Full (board, r0, r1) canonical FORM is a fixed point, including on
    suit-symmetric boards where range tie-breaking decides the perm."""
    rng = np.random.default_rng(13)
    boards = [_random_board(rng) for _ in range(20)]
    boards += [
        (0, 4, 8, 24, 48),     # monotone clubs (suit-degenerate)
        (0, 1, 8, 9, 16),      # two suits with identical board rank sets
        (3, 7, 11, 15, 19),    # monotone spades
    ]
    for board in boards:
        r0, r1 = _dirichlet_pair(board, rng)
        canon, perm = siso.canonicalize(board, r0, r1)
        r0c = siso.permute_global(r0, perm)
        r1c = siso.permute_global(r1, perm)
        canon2, perm2 = siso.canonicalize(canon, r0c, r1c)
        assert canon2 == canon
        assert np.array_equal(siso.permute_global(r0c, perm2), r0c)
        assert np.array_equal(siso.permute_global(r1c, perm2), r1c)


def test_canonicalize_spec_idempotent():
    rng = np.random.default_rng(17)
    board = _random_board(rng)
    r0, r1 = _dirichlet_pair(board, rng)
    spec = sgs.SubgameSpec(street="river", board=board, pot=12000,
                           stack0=8000, stack1=8000, first_to_act=1,
                           r0=r0, r1=r1)
    canon_spec, perm = siso.canonicalize_spec(spec)
    assert canon_spec.board == siso.permute_board(board, perm)
    assert (canon_spec.pot, canon_spec.stack0, canon_spec.stack1,
            canon_spec.first_to_act) == (spec.pot, spec.stack0, spec.stack1,
                                         spec.first_to_act)
    canon2, perm2 = siso.canonicalize_spec(canon_spec)
    assert canon2.board == canon_spec.board
    assert np.array_equal(canon2.r0, canon_spec.r0)
    assert np.array_equal(canon2.r1, canon_spec.r1)


# ---------------------------------------------------------------------------
# (2) round-trip permutation exactness on ranges and CFVs
# ---------------------------------------------------------------------------

def test_round_trip_bit_exact_ranges_and_cfvs():
    rng = np.random.default_rng(19)
    for perm in siso.SUIT_PERMS:
        inv = siso.invert_perm(perm)
        # float64 range-like vector
        r = rng.dirichlet(np.full(sgs.N_GLOBAL_HANDS, 0.7))
        assert np.array_equal(siso.permute_global(siso.permute_global(r, perm), inv), r)
        # float32 CFV-like vector with negatives and exact zeros
        v = (rng.standard_normal(sgs.N_GLOBAL_HANDS) * 1e4).astype(np.float32)
        v[rng.integers(0, sgs.N_GLOBAL_HANDS, size=200)] = 0.0
        rt = siso.permute_global(siso.permute_global(v, perm), inv)
        assert rt.dtype == v.dtype
        assert np.array_equal(rt, v)
        # 2-D batch round trip
        m = rng.standard_normal((5, sgs.N_GLOBAL_HANDS))
        assert np.array_equal(siso.permute_global(siso.permute_global(m, perm), inv), m)


def test_permute_global_consistent_with_local_hand_sets():
    """Scattered local values follow the board through the permutation: the
    value of local hand (c0, c1) on board B lands exactly on hand
    (perm(c0), perm(c1)) of board perm(B)."""
    rng = np.random.default_rng(23)
    for _ in range(20):
        board = _random_board(rng)
        perm = siso.SUIT_PERMS[int(rng.integers(0, 24))]
        v_local = rng.standard_normal(len(sgs.local_hands(board)))
        g = sgs.scatter_global(board, v_local)
        gp = siso.permute_global(g, perm)
        board_p = siso.permute_board(board, perm)
        for k, (c0, c1) in enumerate(sgs.local_hands(board)):
            j = sgs.hand_index(siso.permute_card(c0, perm),
                               siso.permute_card(c1, perm))
            assert gp[j] == v_local[k]
        # off-board (conflicting) hands stay exactly zero on the new board
        g2l = sgs.global_to_local(board_p)
        assert np.all(gp[g2l < 0] == 0.0)


# ---------------------------------------------------------------------------
# (3) THE SOUNDNESS TEST: suit-isomorphic subgames solve to the same
#     canonical-mapped V* targets
# ---------------------------------------------------------------------------

def _soundness_pair(board, perm, pot, eff, seed):
    """Build (spec, isomorphic spec) with asymmetric ranges; solve both."""
    rng = np.random.default_rng(seed)
    r0, r1 = _dirichlet_pair(board, rng)
    spec_a = sgs.SubgameSpec(street="river", board=board, pot=pot,
                             stack0=eff, stack1=eff, first_to_act=1,
                             r0=r0, r1=r1)
    spec_b = sgs.SubgameSpec(street="river",
                             board=siso.permute_board(board, perm), pot=pot,
                             stack0=eff, stack1=eff, first_to_act=1,
                             r0=siso.permute_global(r0, perm),
                             r1=siso.permute_global(r1, perm))
    return spec_a, spec_b


def _canonical_targets(spec, result):
    _, perm = siso.canonicalize(spec.board, spec.r0, spec.r1)
    return (siso.permute_global(result.v0_global(), perm),
            siso.permute_global(result.v1_global(), perm))


def _run_soundness(device):
    diffs = []
    for case_idx, (board, perm, (pot, eff)) in enumerate(SOUNDNESS_CASES):
        spec_a, spec_b = _soundness_pair(board, perm, pot, eff,
                                         seed=1000 + case_idx)
        # canonical INPUTS must coincide bit-exactly (this is what makes
        # post-hoc canonicalization of stored training rows valid)
        canon_a, pa = siso.canonicalize(spec_a.board, spec_a.r0, spec_a.r1)
        canon_b, pb = siso.canonicalize(spec_b.board, spec_b.r0, spec_b.r1)
        assert canon_a == canon_b
        assert np.array_equal(siso.permute_global(spec_a.r0, pa),
                              siso.permute_global(spec_b.r0, pb))
        assert np.array_equal(siso.permute_global(spec_a.r1, pa),
                              siso.permute_global(spec_b.r1, pb))

        res_a = pop.solve_population(
            [spec_a], n_iterations=N_ITERATIONS, dtype=torch.float64,
            device=device, b_max=1, compute_value_pass=True)[0]
        res_b = pop.solve_population(
            [spec_b], n_iterations=N_ITERATIONS, dtype=torch.float64,
            device=device, b_max=1, compute_value_pass=True)[0]
        v0a, v1a = _canonical_targets(spec_a, res_a)
        v0b, v1b = _canonical_targets(spec_b, res_b)
        d = max(float(np.abs(v0a - v0b).max()), float(np.abs(v1a - v1b).max()))
        diffs.append({"case": case_idx, "board": board, "perm": perm,
                      "pot": pot, "eff": eff,
                      "n_nodes": int(spec_a.tree()["n_nodes"]),
                      "linf_chips": d, "linf_over_pot": d / pot})
    return diffs


def test_soundness_isomorphic_solves_cpu_f64():
    diffs = _run_soundness("cpu")
    for d in diffs:
        print(f"[suit-iso soundness cpu64] case {d['case']} nn={d['n_nodes']} "
              f"pot={d['pot']}: L-inf {d['linf_chips']:.3e} chips "
              f"({d['linf_over_pot']:.3e} of pot)")
    worst = max(d["linf_chips"] for d in diffs)
    assert worst <= TOL_SOUNDNESS_CHIPS, (
        f"suit-isomorphic solves diverge {worst:.3e} chips in canonical space "
        f"(tolerance {TOL_SOUNDNESS_CHIPS:.1e})")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_soundness_isomorphic_solves_cuda_f64_band():
    diffs = _run_soundness("cuda")
    for d in diffs:
        print(f"[suit-iso soundness cuda64] case {d['case']} nn={d['n_nodes']} "
              f"pot={d['pot']}: L-inf {d['linf_chips']:.3e} chips "
              f"({d['linf_over_pot']:.3e} of pot)")
    worst = max(d["linf_chips"] for d in diffs)
    assert worst <= TOL_CUDA_CHIPS, (
        f"suit-isomorphic CUDA solves diverge {worst:.3e} chips "
        f"(documented band {TOL_CUDA_CHIPS:.1e})")
