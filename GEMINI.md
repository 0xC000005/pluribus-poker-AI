# Gemini CLI Project Context: Poker AI

This repository's active goal is a full-deck poker AI that can train by
self-play and play heads-up no-limit Texas hold'em against Slumbot. The current
main path is Deep CFR with CUDA traversal, a 9-action abstraction, and
turn/river range-vs-range CFR+ solving. Legacy short-deck tabular MCCFR,
clustering, terminal play, and visualisation code are retained for reference.

## Main Architecture

- `poker_ai/deep_cfr/`: active learning stack. `networks.py` defines the
  advantage/policy-head MLP, `buffer.py` defines reservoir replay, and
  `deep_cfr.py` contains the reference training loop.
- `poker_ai/deep_cfr/cuda/`: GPU trainer and Numba kernels. This is the
  preferred path for compute-heavy experiments on CUDA hosts.
- `poker_ai/deep_cfr/fast_state.py` and `fast_traverse.py`: CPU-optimized
  traversal path for comparison or CPU-only machines.
- `poker_ai/games/full_deck/state.py`: canonical full 52-card game state and
  feature encoder.
- `scripts/train_slumbot_2p.py`: Slumbot-aligned two-player, 200BB training.
- `scripts/play_slumbot.py`: Slumbot API adapter plus optional turn/river
  solver.
- `scripts/solver.py` and `scripts/fast_cfr.py`: range-vs-range CFR+ solver.

## Current Contract

- Action space: 9 actions: fold, check/call, six pot-fraction raises
  `(0.25, 0.5, 0.75, 1.0, 1.5, 2.0)`, and all-in.
- Feature vector: 126 floats: full-card one-hots, street one-hot, scalar stack
  and pot features, and action-history slots.
- `ValueNetwork` masks the engineered history slots internally. If action or
  feature logic changes, keep CPU, fast, CUDA, and Slumbot paths in parity.

## Commands

Install:

```bash
pip install -e .
```

Fast unit checks:

```bash
pytest -q test/unit/test_network_mask.py test/unit/test_slumbot_mapping.py test/unit/test_legal_mask_parity.py
```

GPU Deep CFR:

```bash
python scripts/run_gpu_deep_cfr.py --n-players 2 --n-iterations 50 --n-traversals 400 --save-path ./models
```

Slumbot training:

```bash
python scripts/train_slumbot_2p.py --n-iterations 1000 --n-traversals 10000 --hidden-dim 512 --n-layers 4
```

Slumbot play:

```bash
python scripts/play_slumbot.py --model models/slumbot_2p_iter1000.pt --hands 300 --greedy
```

## Development Rules

- Treat full-deck Deep CFR and Slumbot parity as the source of truth.
- Avoid new hand-crafted poker heuristics; prefer learned policies, regret
  matching, and principled search.
- For performance work, report `iters/hour`, `samples/sec`, and
  `train sec/iter` with the exact command.
- Do not commit generated checkpoints from `models/` or lookup-table artifacts
  from `research/`.
