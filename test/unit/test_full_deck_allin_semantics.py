from poker_ai.deep_cfr.fast_state import new_fast_game
from poker_ai.games.full_deck.state import new_game


def test_heads_up_allin_requires_opponent_response():
    state = new_game(2, initial_chips=1000)

    after_allin = state.apply_action("all_in")

    assert after_allin.is_terminal is False
    assert after_allin.player_i == 1
    assert "call" in after_allin.legal_actions
    assert "fold" in after_allin.legal_actions


def test_heads_up_allin_fold_awards_only_opponent_blind_net():
    state = new_game(2, initial_chips=1000)

    after_allin = state.apply_action("all_in")
    folded = after_allin.apply_action("fold")

    assert folded.is_terminal is True
    assert folded.payout[0] == 100
    assert folded.payout[1] == -100


def test_heads_up_allin_call_reaches_called_showdown():
    state = new_game(2, initial_chips=1000)

    after_allin = state.apply_action("all_in")
    called = after_allin.apply_action("call")

    assert called.is_terminal is True
    assert sum(called.payout.values()) == 0
    assert sorted(abs(value) for value in called.payout.values()) in ([0, 0], [1000, 1000])


def test_fast_heads_up_allin_requires_opponent_response():
    state = new_fast_game(2, initial_chips=1000)

    state.apply_action(8)

    assert state.is_terminal is False
    assert state.current_player_i == 1
    assert 0 in state.legal_actions
    assert 1 in state.legal_actions


def test_fast_heads_up_allin_call_reaches_called_showdown():
    state = new_fast_game(2, initial_chips=1000)

    state.apply_action(8)
    state.apply_action(1)

    assert state.is_terminal is True
    assert sum(state.payout.values()) == 0
    assert sorted(abs(value) for value in state.payout.values()) in ([0, 0], [1000, 1000])
