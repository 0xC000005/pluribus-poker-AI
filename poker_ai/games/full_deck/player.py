from poker_ai.poker.player import Player
from poker_ai.poker.pot import Pot


class PokerPlayer(Player):
    """Player for full-deck poker games."""

    def __init__(self, player_i: int, initial_chips: int, pot: Pot):
        super().__init__(
            name=f"player_{player_i}", initial_chips=initial_chips, pot=pot,
        )
        self.is_turn = False
