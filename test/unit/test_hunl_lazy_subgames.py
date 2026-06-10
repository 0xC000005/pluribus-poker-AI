"""Tests for the lazy HUNL subgame substrate steps 7-8 (p1_design.json).

Covers ``poker_ai/rebel/hunl/lazy_subgames.py`` (SubgameFactory + PopulationQueue
+ the census-arithmetic ``b_max_for``) and ``poker_ai/rebel/hunl/targets.py``
(per-belief V* river target emission for the P2/P4 PBS net).

  (1) expand_to_river card-removal masking parity vs the reference mapping of
      ``turn_river.turn_leaf_river_cfv`` (the exact loop terminal_eval's
      ``exact_river_cfv_population`` shares) -- exact equality.
  (2) PopulationQueue flush correctness: solving via the queue (bucketed,
      chunked at B_max) is BIT-IDENTICAL on CPU to solving the same specs
      directly via ``solve_population`` (the P-B property).
  (3) a tiny ``generate_river_targets`` run (2 beliefs x 2 boards, low iters)
      produces rows with the documented layout/weights and v0/v1 matching a
      direct solve.
  (4) no-global-enumeration guard: the factory never builds more trees/specs
      than requested (spied via build_street_tree's LRU cache stats).
  (5) ``b_max_for`` reproduces the committed census B_max table
      (autoresearch-session/rebel/hunl_topology_census.json) exactly.

Fixed seeds, CPU-only -- run with CUDA_VISIBLE_DEVICES="".
"""
import itertools

import numpy as np
import pytest
import torch

from poker_ai.rebel.hunl import tree_builder as tb  # wires scripts/ into sys.path
from poker_ai.rebel.hunl import subgame_spec as sgs
from poker_ai.rebel.hunl import lazy_subgames as lzs
from poker_ai.rebel.hunl import population_solver as pop
from poker_ai.rebel.hunl import targets as tgt

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

PA_CONFIG = dict(pot=400, stack0=400, stack1=400, first_to_act=0)
PE_CONFIG = dict(pot=3000, stack0=150, stack1=600, first_to_act=1)  # all-in truncated
TURN_CONFIG = dict(pot=800, stack0=500, stack1=500, first_to_act=0)


def _global_ranges(seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        r = rng.random(sgs.N_GLOBAL_HANDS)
        r = (r / r.sum()).astype(np.float32).astype(np.float64)
        out.append(r)
    return out[0], out[1]


def _make_spec(board, config, seed, street="river", factory=None):
    r0, r1 = _global_ranges(seed)
    maker = factory.from_state if factory is not None else sgs.SubgameSpec
    return maker(
        street=street,
        board=tuple(board),
        pot=config["pot"],
        stack0=config["stack0"],
        stack1=config["stack1"],
        first_to_act=config["first_to_act"],
        r0=r0,
        r1=r1,
    )


# ---------------------------------------------------------------------------
# (1) expand_to_river: card-removal masking parity vs the reference mapping
# ---------------------------------------------------------------------------

def test_expand_to_river_masking_parity():
    factory = lzs.SubgameFactory()
    n_turn = len(sgs.local_hands(TURN_BOARD))
    rng = np.random.default_rng(11)
    hr = rng.random(n_turn) * 0.5
    vr = rng.random(n_turn) * 0.5
    turn_spec = factory.from_state(
        street="turn", board=TURN_BOARD, r0=sgs.scatter_global(TURN_BOARD, hr),
        r1=sgs.scatter_global(TURN_BOARD, vr), **TURN_CONFIG)

    tree = turn_spec.tree()
    sh, sv, potN = tree["stacks_h"], tree["stacks_v"], tree["pot"]
    # a betting-continuation river-deal cut (pot/stacks differ from entry)
    leaf_node = next(
        int(i) for i in tree["showdown_idx"]
        if 0 < int(sh[i]) < TURN_CONFIG["stack0"])
    specs = factory.expand_to_river(turn_spec, leaf_node)

    # exactly the 48 runouts, in ascending-card order, at the cut's chips
    runouts = [c for c in range(52) if c not in TURN_BOARD]
    assert len(specs) == len(runouts) == 48
    for spec, r in zip(specs, runouts):
        assert spec.street == "river"
        assert spec.board == sgs.board_key(TURN_BOARD) + (r,)
        assert spec.pot == int(potN[leaf_node])
        assert spec.stack0 == int(sh[leaf_node])
        assert spec.stack1 == int(sv[leaf_node])
        assert spec.first_to_act == turn_spec.first_to_act

    # card-removal range masking == turn_leaf_river_cfv's exact mapping:
    # restrict turn reaches to hands not containing the runout, mapped onto the
    # river subgame's hand order (StreetSolver enumeration).
    turn_hands = sgs.local_hands(TURN_BOARD)
    turn_idx = {tuple(h): i for i, h in enumerate(turn_hands)}
    for spec, r in zip(specs, runouts):
        board5 = list(TURN_BOARD) + [r]
        remaining = sorted(set(range(52)) - set(board5))
        river_hands = list(itertools.combinations(remaining, 2))
        rh2i = {h: i for i, h in enumerate(river_hands)}
        hr_ref = np.zeros(len(river_hands))
        vr_ref = np.zeros(len(river_hands))
        for h, ti in turn_idx.items():
            if r in h:
                continue
            ri = rh2i.get(h)
            if ri is None:
                continue
            hr_ref[ri] = hr[ti]
            vr_ref[ri] = vr[ti]
        r0l, r1l = spec.local_ranges()
        np.testing.assert_array_equal(r0l, hr_ref)
        np.testing.assert_array_equal(r1l, vr_ref)

    # explicit cut-reach override (global index) is honored
    hr2 = rng.random(n_turn)
    r0_g = sgs.scatter_global(TURN_BOARD, hr2)
    specs2 = factory.expand_to_river(turn_spec, leaf_node, r0=r0_g)
    np.testing.assert_array_equal(specs2[0].r0, r0_g)
    np.testing.assert_array_equal(specs2[0].r1, turn_spec.r1)

    # the shared expansion is what terminal_eval's population leaf consumes
    direct = lzs.river_runout_specs(
        TURN_BOARD, int(potN[leaf_node]), int(sh[leaf_node]), int(sv[leaf_node]),
        turn_spec.first_to_act, turn_spec.r0, turn_spec.r1)
    assert [s.board for s in direct] == [s.board for s in specs]

    # non-showdown node is rejected
    with pytest.raises(ValueError):
        factory.expand_to_river(turn_spec, 0)
    # river specs cannot be expanded
    with pytest.raises(ValueError):
        factory.expand_to_river(specs[0], leaf_node)


# ---------------------------------------------------------------------------
# (2) PopulationQueue: bucketed/chunked flushing == direct solve, bit-identical
# ---------------------------------------------------------------------------

def test_population_queue_flush_bit_identical():
    specs = [_make_spec(RIVER_BOARDS[i % 3], PA_CONFIG, seed=70 + i) for i in range(4)]
    specs.append(_make_spec(RIVER_BOARDS[1], PE_CONFIG, seed=80))  # second bucket
    iters = 10

    direct = pop.solve_population(specs, n_iterations=iters, b_max=2,
                                  dtype=DTYPE, device=DEVICE)
    assert len({r.topology_key for r in direct}) == 2

    queue = lzs.PopulationQueue(n_iterations=iters, b_max=2,
                                dtype=DTYPE, device=DEVICE)
    flushed = []
    flush_sizes = []
    for spec in specs:
        out = queue.add(spec)
        if out:
            flush_sizes.append(len(out))
        flushed.extend(out)
    assert queue.pending == 1            # the lone PE spec
    tail = queue.flush_all()
    assert queue.pending == 0
    assert queue.flush_all() == []       # idempotent drain
    flushed.extend(tail)
    assert flush_sizes == [2, 2]         # bucket flushed exactly at b_max
    assert len(flushed) == len(specs)

    by_spec = {id(r.spec): r for r in flushed}
    for spec, ref in zip(specs, direct):
        got = by_spec[id(spec)]
        assert got.topology_key == ref.topology_key
        # bit-identical on CPU: identical kernel calls on identical chunks
        np.testing.assert_array_equal(got.v0, ref.v0)
        np.testing.assert_array_equal(got.v1, ref.v1)
        np.testing.assert_array_equal(got.avg_strategy, ref.avg_strategy)
        np.testing.assert_array_equal(got.strategy_sum, ref.strategy_sum)
        np.testing.assert_array_equal(got.regret_sum, ref.regret_sum)
    print(f"\n[queue] {len(specs)} specs, 2 buckets, flush sizes {flush_sizes} + "
          f"drain {len(tail)}: bit-identical to direct solve_population")

    # memory-derived B_max: with no override the bucket threshold comes from
    # the census arithmetic applied to the spec's own tree
    auto_queue = lzs.PopulationQueue(n_iterations=iters, budget_gb=6.5,
                                     dtype=DTYPE, device=DEVICE)
    spec = specs[0]
    tree = spec.tree()
    n_nodes = int(tree["n_nodes"])
    n_dec = int(np.sum(np.asarray(tree["terminal_type"]) == tb.fast_cfr.T_DECISION))
    expected = lzs.b_max_for("river", n_nodes, 6.5, n_decision=n_dec)
    assert auto_queue.b_max_for_spec(spec) == expected
    # turn specs without a leaf hook fail fast at enqueue time
    with pytest.raises(ValueError):
        auto_queue.add(_make_spec(TURN_BOARD, TURN_CONFIG, seed=81, street="turn"))


# ---------------------------------------------------------------------------
# (5) b_max_for reproduces the committed census B_max table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "street,n_nodes,n_decision,expected",
    [
        # autoresearch-session/rebel/hunl_topology_census.json b_max_table rows
        ("river", 21, 8, 154),
        ("river", 411, 138, 51),
        ("river", 1257, 420, 21),
        ("turn", 279, 94, 62),
        ("turn", 1125, 376, 22),
        ("turn", 1257, 420, 20),
    ],
)
def test_b_max_for_matches_census_table(street, n_nodes, n_decision, expected):
    assert lzs.b_max_for(street, n_nodes, 6.5, n_decision=n_decision) == expected


def test_b_max_for_bounds():
    # conservative default (n_decision=n_nodes) never exceeds the exact value
    exact = lzs.b_max_for("river", 411, 6.5, n_decision=138)
    conservative = lzs.b_max_for("river", 411, 6.5)
    assert 1 <= conservative <= exact
    # never returns less than one solvable element
    assert lzs.b_max_for("river", 1257, 0.001) == 1
    with pytest.raises(ValueError):
        lzs.b_max_for("flop", 100, 6.5)


# ---------------------------------------------------------------------------
# (3) generate_river_targets: tiny end-to-end run
# ---------------------------------------------------------------------------

def test_generate_river_targets_tiny():
    boards = [RIVER_BOARDS[0], RIVER_BOARDS[1]]
    n_beliefs = 2
    iters = 8
    seed = 123
    scale = 20000.0
    config = (PA_CONFIG["pot"], PA_CONFIG["stack0"], PA_CONFIG["stack1"],
              PA_CONFIG["first_to_act"])

    def sampler(_rng):
        return config

    queue = lzs.PopulationQueue(n_iterations=iters, b_max=4,
                                dtype=DTYPE, device=DEVICE)
    rng = np.random.default_rng(seed)
    X, Y, W = tgt.generate_river_targets(n_beliefs, boards, sampler, rng, queue)
    assert queue.pending == 0

    n_rows = n_beliefs * len(boards)
    assert X.shape == (n_rows, tgt.X_DIM) and X.dtype == np.float32
    assert Y.shape == (n_rows, tgt.Y_DIM) and Y.dtype == np.float32
    assert W.shape == (n_rows, tgt.Y_DIM) and W.dtype == np.float32
    assert tgt.X_DIM == 52 + 2 + 2 * sgs.N_GLOBAL_HANDS
    assert tgt.Y_DIM == 2 * sgs.N_GLOBAL_HANDS

    # rebuild the identical specs (same rng convention as the driver:
    # a0, a1 ~ choice({0.3, 1.0, 3.0}), Dirichlet over the board's local hands)
    rng2 = np.random.default_rng(seed)
    expected_specs = []
    for board in boards:
        H = len(sgs.local_hands(board))
        for _ in range(n_beliefs):
            a0 = float(rng2.choice([0.3, 1.0, 3.0]))
            a1 = float(rng2.choice([0.3, 1.0, 3.0]))
            r0l = rng2.dirichlet(np.full(H, a0))
            r1l = rng2.dirichlet(np.full(H, a1))
            pot, s0, s1, first = sampler(rng2)
            expected_specs.append(sgs.SubgameSpec(
                street="river", board=board, pot=pot, stack0=s0, stack1=s1,
                first_to_act=first,
                r0=sgs.scatter_global(board, r0l),
                r1=sgs.scatter_global(board, r1l)))

    max_v_diff = 0.0
    for i, spec in enumerate(expected_specs):
        l2g = sgs.local_to_global(spec.board)
        board_onehot = X[i, :52]
        assert set(np.flatnonzero(board_onehot)) == set(spec.board)
        assert np.all(board_onehot[list(spec.board)] == 1.0)
        np.testing.assert_allclose(X[i, 52], spec.pot / scale, rtol=1e-6)
        np.testing.assert_allclose(
            X[i, 53], min(spec.stack0, spec.stack1) / scale, rtol=1e-6)

        # ranges/weights: normalized over local support, zero off-board, and
        # the weight rows equal the x range slices (the _row convention)
        r0n = np.float32(spec.r0 / spec.r0.sum())
        r1n = np.float32(spec.r1 / spec.r1.sum())
        x_r0 = X[i, 54:54 + sgs.N_GLOBAL_HANDS]
        x_r1 = X[i, 54 + sgs.N_GLOBAL_HANDS:]
        np.testing.assert_array_equal(x_r0, r0n)
        np.testing.assert_array_equal(x_r1, r1n)
        np.testing.assert_array_equal(W[i, :sgs.N_GLOBAL_HANDS], x_r0)
        np.testing.assert_array_equal(W[i, sgs.N_GLOBAL_HANDS:], x_r1)
        off_board = np.setdiff1d(np.arange(sgs.N_GLOBAL_HANDS), l2g)
        assert np.all(x_r0[off_board] == 0.0)
        np.testing.assert_allclose(x_r0.sum(), 1.0, atol=1e-5)

        # targets match a direct B=1 solve of the same spec
        ref = pop.solve_population([spec], n_iterations=iters,
                                   dtype=DTYPE, device=DEVICE)[0]
        y0 = Y[i, :sgs.N_GLOBAL_HANDS]
        y1 = Y[i, sgs.N_GLOBAL_HANDS:]
        d = max(float(np.max(np.abs(y0 - np.float32(ref.v0_global())))),
                float(np.max(np.abs(y1 - np.float32(ref.v1_global())))))
        max_v_diff = max(max_v_diff, d)
        assert np.all(y0[off_board] == 0.0)
        assert np.all(y1[off_board] == 0.0)
    print(f"\n[targets] {n_rows} rows; max |Y - direct-solve V*| (float32): "
          f"{max_v_diff:.3e} chips (pot {PA_CONFIG['pot']})")
    assert max_v_diff <= 1e-3


# ---------------------------------------------------------------------------
# (4) no-global-enumeration guard
# ---------------------------------------------------------------------------

def test_no_global_enumeration_guard():
    tb.street_tree_cache_clear()
    factory = lzs.SubgameFactory()

    # spec construction is fully lazy: no tree builds at all
    spec_a = _make_spec(RIVER_BOARDS[0], PA_CONFIG, seed=90, factory=factory)
    spec_b = _make_spec(RIVER_BOARDS[1], PA_CONFIG, seed=91, factory=factory)
    spec_c = _make_spec(RIVER_BOARDS[1], PE_CONFIG, seed=92, factory=factory)
    assert tb.street_tree_cache_info().misses == 0

    # expansion builds exactly ONE turn tree; the 48 river specs stay lazy
    turn_spec = _make_spec(TURN_BOARD, TURN_CONFIG, seed=93, street="turn",
                           factory=factory)
    tree = turn_spec.tree()
    leaf_node = next(
        int(i) for i in tree["showdown_idx"]
        if 0 < int(tree["stacks_h"][i]) < TURN_CONFIG["stack0"])
    expanded = factory.expand_to_river(turn_spec, leaf_node)
    assert tb.street_tree_cache_info().misses == 1

    # bucketing all 48 runouts builds ONE shared river tree (board-free shape)
    assert len({s.bucket_key() for s in expanded}) == 1
    assert tb.street_tree_cache_info().misses == 2

    # bucketing the 3 hand-made specs adds exactly their 2 distinct configs
    for s in (spec_a, spec_b, spec_c):
        s.bucket_key()
    info = tb.street_tree_cache_info()
    assert info.misses == 4
    assert info.currsize == 4

    # the factory built exactly what was requested -- nothing global
    assert factory.n_specs_built == 4 + len(expanded)
    print(f"\n[guard] {factory.n_specs_built} specs requested -> "
          f"{info.misses} tree builds (census river config space: 9030)")
