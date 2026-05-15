from poker_ai.poker.deck import default_include_suits


def test_default_deck_suit_order_is_deterministic():
    assert default_include_suits == sorted(default_include_suits)
