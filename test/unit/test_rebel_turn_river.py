"""ReBeL real-game step 0: turn+river harness foundation (card parsing, spot, turn tree, cut nodes)."""
import pytest

pytest.importorskip("torch")

from poker_ai.rebel.turn_river import (
    parse_card, card_str, default_spot, build_turn_solver, showdown_cut_indices,
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
