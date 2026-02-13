# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pluribus Poker AI — poker AI using Counterfactual Regret Minimization (CFR). Two training pipelines:
- **Deep CFR** (`poker_ai/deep_cfr/`) — Full 52-card deck, neural network-based (recommended)
- **Legacy tabular MCCFR** (`poker_ai/ai/` + `poker_ai/clustering/`) — 20-card Short Deck only

## Common Commands

### Install (development)
```bash
pip install -e .
```

### Run Tests
```bash
# All tests
TESTING_SUITE=1 pytest

# Single test file
TESTING_SUITE=1 pytest test/unit/test_card.py

# Single test
TESTING_SUITE=1 pytest test/functional/test_engine.py::test_hand -v
```
`TESTING_SUITE=1` is required — it disables multiprocessing Manager initialization in the Agent class, which would otherwise hang in test contexts.

### Deep CFR Training (full deck)
```bash
poker_ai train-deep-cfr --n-iterations 200 --n-traversals 500 --device auto
poker_ai train-deep-cfr --resume ./models/deep_cfr_final.pt  # resume training
```

### Legacy CLI Commands (short deck only)
```bash
poker_ai cluster              # Build card abstraction lookup tables
poker_ai train start           # Start tabular MCCFR training
poker_ai play                  # Play against trained agent in terminal
```

## Architecture

### Deep CFR Pipeline (full deck, recommended)

- **`poker_ai/games/full_deck/state.py`** — `PokerState` class for any deck size. No LUT dependency. Provides `to_feature_vector()` (126-dim float32) for neural net input and `apply_action()` for tree traversal.
- **`poker_ai/deep_cfr/networks.py`** — `ValueNetwork` MLP (~99K params). Maps state features → per-action advantage/regret values.
- **`poker_ai/deep_cfr/buffer.py`** — `ReservoirBuffer` with reservoir sampling. Stores (features, iteration, advantages) training samples.
- **`poker_ai/deep_cfr/deep_cfr.py`** — `DeepCFRTrainer` implementing Single Deep CFR (SD-CFR). External-sampling MCCFR with neural regret storage. Key functions: `traverse()` (game tree walk), `train_value_network()` (retrained from scratch each iteration), `regret_match()`.
- **`poker_ai/deep_cfr/trainer.py`** — CLI entry point (`train_deep_cfr`). Training loop with progress bars, checkpointing, and evaluation vs random.

### Legacy Pipeline (short deck)

1. **Clustering** (`poker_ai/clustering/`) — k-means card abstraction into LUTs
2. **Training** (`poker_ai/ai/`) — Tabular External Sampling MCCFR
3. **Playing** (`poker_ai/terminal/`) — Terminal UI

### Shared Engine

- **`poker_ai/poker/`** — Base poker mechanics: Card, Deck, Dealer, PokerEngine, hand evaluation (bit-encoded lookup tables). Supports full 52-card deck and N players.
- **`poker_ai/poker/evaluation/`** — Cactus Kev hand evaluator. Handles 5, 6, 7 card hands for all 13 ranks.

### Entry Points

- CLI: `bin/poker_ai` → `poker_ai.cli.runner.cli()`
- Deep CFR: `poker_ai train-deep-cfr [OPTIONS]`
- Python: `from poker_ai.games.full_deck import new_game, PokerState`

## Key Design Decisions

- **SD-CFR**: Single value network (no separate averaging network). Better exploitability and simpler than full Deep CFR.
- **Retrain from scratch**: Value network is reinitialized each CFR iteration, not fine-tuned. Converges better per the paper.
- **Reservoir sampling**: Buffer uses reservoir sampling, not sliding window. Sliding window causes convergence stall.
- **Feature vector**: 126-dim encoding (52 hole cards + 52 community cards + 4 round one-hot + 6 scalars + 12 action history).
- **Gradient clipping**: Max norm 1.0 to prevent NaN explosion during training.

## Development Notes

- Git flow: feature branches off `develop`, PRs into `develop`.
- `TESTING_SUITE=1` env var required for tests (disables multiprocessing Manager).
- Test suite: `test/unit/` (component) + `test/functional/` (integration).
- The `.venv` uses Python 3.13. Use `python -m pip install` to install packages into it.
