## Visualisation Code

This is a legacy visualisation app for `ShortDeckPokerState`. It is not wired
to the current full-deck Deep CFR or Slumbot training path. Use it only when
working on the old short-deck/terminal stack.

### Run

Build the Vue frontend first:

```bash
cd applications/visualisation/frontend
npm install
npm run build
```

Then create a `PokerPlot` instance from Python and send it short-deck states:

```python
from plot import PokerPlot
from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.games.short_deck.state import ShortDeckPokerState
from poker_ai.poker.pot import Pot

pot = Pot()
players = [
    ShortDeckPokerPlayer(player_i=i, initial_chips=10000, pot=pot)
    for i in range(6)
]
state = ShortDeckPokerState(players=players, pickle_dir="../../research/blueprint_algo/")

plot = PokerPlot()
plot.update_state(state)
```

For current engine work, prefer the parity tests and Slumbot diagnostics under
`scripts/`.
