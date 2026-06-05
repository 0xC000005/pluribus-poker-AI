"""ReBeL real-game step 0: turn+river harness foundation (card parsing, spot, turn tree, cut nodes)."""
import pytest

pytest.importorskip("torch")

import numpy as np

from poker_ai.rebel.turn_river import (
    parse_card, card_str, default_spot, build_turn_solver, showdown_cut_indices,
    _average_strategy_array, subgame_value_pass, turn_leaf_river_cfv,
)


def test_card_roundtrip():
    for s in ("2c", "Ah", "Kd", "7c", "2s", "Ts", "9h"):
        assert card_str(parse_card(s)) == s
    # convention: card = rank*4 + suit, ranks 23456789TJQKA, suits cdhs
    assert parse_card("2c") == 0
    assert parse_card("Ah") == 12 * 4 + 2


def test_default_spot():
    spot = default_spot()
    assert len(spot.board) == 4 and len(set(spot.board)) == 4
    assert spot.board_str == "Ah Kd 7c 2s"
    assert spot.pot == 2000 and spot.hero_stack == 8000  # 20bb / 80bb at 100 chips/bb


def test_turn_tree_and_cut_nodes():
    ts = build_turn_solver(default_spot())
    assert ts.n == 1128  # C(48,2) hands on a 4-card board
    cuts = showdown_cut_indices(ts)
    assert len(cuts) > 0
    nodes = ts._tree["all_nodes"]
    # every cut node is a showdown terminal; fold terminals are excluded
    for i in cuts:
        assert getattr(nodes[i], "terminal_type", None) == "showdown"


def test_subgame_value_pass_convention_identity():
    # The per-hand counterfactual value pass must satisfy, for ANY strategy:
    #   sum(hr*hcfv) + sum(vr*vcfv) == pot * (hr @ valid @ vr)
    # (every terminal has hero_val(a,b)+villain_val(b,a)=pot_start). Verifies the extractor.
    import solver as S
    spot = default_spot()
    river = [c for c in range(52) if c not in spot.board][0]
    board5 = spot.board + [river]
    full = S.StreetSolver(board5, 400, 400, 400, True)
    avail = [c for c in range(52) if c not in board5]
    disjoint = [(avail[2 * k], avail[2 * k + 1]) for k in range(14)]
    active = sorted(full.hand_to_idx[tuple(sorted(h))] for h in disjoint)
    POT, HS, VS = 400, 400, 400
    rs = S.StreetSolver(board5, POT, HS, VS, True, active_indices=active)
    n = rs.n
    hr = np.ones(n, np.float32) / n
    vr = np.ones(n, np.float32) / n
    rs.solve(n_iterations=300, hero_range=hr, villain_range=vr, backend="cpu")
    avg = _average_strategy_array(rs._strategy_sum)
    hcfv, vcfv = subgame_value_pass(rs, avg, hr.astype(np.float64), vr.astype(np.float64))
    lhs = float(np.dot(hr, hcfv) + np.dot(vr, vcfv))
    rhs = POT * float(hr @ rs.valid @ vr)
    assert abs(lhs - rhs) < 1e-2


def test_turn_leaf_river_cfv_convention_identity():
    # Runout-averaging + turn<->river hand-index mapping must preserve the convention identity:
    #   sum(hr*ch) + sum(vr*cv) == P_cut * (hr @ valid_turn @ vr)
    # Use an all-in cut (stacks 0 -> river subgames are single showdown nodes) so it runs fast.
    import numpy as np
    import solver as S
    turn_board = [50, 45, 20, 3]  # Ah Kd 7c 2s
    POT = 600
    ts = S.StreetSolver(turn_board, POT, 0, 0, True)  # all-in cut: 0 stacks
    n = ts.n
    hr = np.ones(n) / n
    vr = np.ones(n) / n
    ch, cv = turn_leaf_river_cfv(turn_board, POT, 0, 0, True, ts.hands, hr, vr, river_iters=1)
    lhs = float(np.dot(hr, ch) + np.dot(vr, cv))
    rhs = POT * float(hr @ ts.valid @ vr)
    assert abs(lhs - rhs) < 2.0


def test_river_leaf_convention_offset_matches_turn_equity():
    # At an all-in cut (river = showdown only), the offset-corrected exact-river leaf must equal the
    # turn solver's own averaged-equity terminal value (net-from-turn-start). Verifies the offset.
    import numpy as np
    import solver as S
    from poker_ai.rebel.turn_river import turn_leaf_river_cfv
    turn_board = [50, 45, 20, 3]
    P_TS, HS0, VS0 = 300, 400, 400
    ts = S.StreetSolver(turn_board, P_TS, HS0, VS0, True)
    n = ts.n
    hr = np.ones(n) / n
    vr = np.ones(n) / n
    P_cut = P_TS + HS0 + VS0  # both all-in on the turn
    ch, cv = turn_leaf_river_cfv(turn_board, P_cut, 0, 0, True, ts.hands, hr, vr, river_iters=1)
    ch_turn = ch - HS0 * (vr @ ts.valid.T)
    cv_turn = cv - VS0 * (hr @ ts.valid)
    win, lose, tie = ts.win_m, ts.lose_m, ts.tie_m
    h_ref = (P_TS + VS0) * (vr @ win.T) + (-HS0) * (vr @ lose.T) + ((P_TS + VS0 - HS0) / 2) * (vr @ tie.T)
    v_ref = (P_TS + HS0) * (hr @ lose) + (-VS0) * (hr @ win) + ((P_TS + HS0 - VS0) / 2) * (hr @ tie)
    assert float(np.max(np.abs(ch_turn - h_ref))) < 1e-2
    assert float(np.max(np.abs(cv_turn - v_ref))) < 1e-2


def test_exact_river_showdown_fn_integration():
    # The exact-river showdown_leaf_fn drives solve_cfr end-to-end and yields a valid strategy.
    # Short-stack spot (few showdown leaves, tiny river trees) + minimal iters to stay fast.
    import numpy as np
    from poker_ai.rebel.turn_river import (
        TurnSpot, build_turn_solver, make_exact_river_showdown_fn, parse_card,
    )
    spot = TurnSpot(board=[parse_card(c) for c in ("Ah", "Kd", "7c", "2s")],
                    pot=2000, hero_stack=100, villain_stack=100, hero_first=True)
    ts = build_turn_solver(spot)
    n = ts.n
    hr = np.ones(n, np.float32) / n
    vr = np.ones(n, np.float32) / n
    sdfn = make_exact_river_showdown_fn(ts, river_iters=2)
    ts.solve(n_iterations=1, hero_range=hr, villain_range=vr, backend="cpu", showdown_leaf_fn=sdfn)
    strat = ts.get_strategy(ts.hands[0])
    assert abs(sum(strat.values()) - 1.0) < 1e-6


def test_turn_leaf_river_cfv_batched_matches_cpu():
    # GPU-batched 44-runout solve must satisfy the same convention identity as the CPU path.
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        import pytest as _pt
        _pt.skip("CUDA not available")
    import solver as S
    from poker_ai.rebel.turn_river import turn_leaf_river_cfv_batched
    tb = [50, 45, 20, 3]
    POT, HS, VS = 600, 200, 200
    ts = S.StreetSolver(tb, POT, HS, VS, True)
    n = ts.n
    hr = np.ones(n) / n
    vr = np.ones(n) / n
    ch, cv = turn_leaf_river_cfv_batched(tb, POT, HS, VS, True, ts.hands, hr, vr, river_iters=50)
    lhs = float(np.dot(hr, ch) + np.dot(vr, cv))
    rhs = POT * float(hr @ ts.valid @ vr)
    assert abs(lhs - rhs) < 2.0
