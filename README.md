# Poker AI

This repository is currently focused on building a full-deck poker AI that can
train from self-play and play heads-up no-limit Texas hold'em against Slumbot.
The active engine is Deep CFR with a 9-action abstraction, CUDA traversal, and
turn/river range-vs-range CFR+ solving. Older short-deck tabular MCCFR and
clustering code remains in the tree for reference, but it is not the
recommended path for new work.

## Current Engine

- `poker_ai/deep_cfr/` contains the active Deep CFR stack: neural networks,
  reservoir buffers, CPU traversal, fast numpy traversal, and CUDA traversal.
- `poker_ai/deep_cfr/cuda/` contains the Numba CUDA trainer and kernels used to
  move self-play traversal, legal masks, regret matching, action sampling, and
  propagation onto the GPU.
- `poker_ai/games/full_deck/` is the canonical full 52-card game state and
  feature encoder used for correctness and parity checks.
- `scripts/train_slumbot_2p.py` trains a 2-player, 200BB model for Slumbot.
- `scripts/play_slumbot.py` maps the learned 9-action policy to Slumbot's API
  and optionally uses the turn/river solver.
- `scripts/solver.py` and `scripts/fast_cfr.py` implement the turn/river
  range-vs-range CFR+ solver. Live Slumbot solving prunes low-probability
  hands using the learned range tracker before CFR.
- `scripts/poker_resolver_benchmark.py` runs fixed public-state diagnostics for
  blueprint-vs-resolver legality, latency, action drift, and learned-advantage
  proxies before spending live Slumbot hands.

## Action And Feature Contract

The current action space has 9 actions:

```text
0 fold
1 check/call
2..7 raise 0.25x, 0.5x, 0.75x, 1.0x, 1.5x, 2.0x pot
8 all-in
```

The model input is a 126-dimensional full-deck feature vector:

```text
52 hole-card one-hots
52 board-card one-hots
4 street one-hots
6 scalar game features
12 action-history summary features
```

The `ValueNetwork` masks the engineered history slots internally. Keep feature
and legal-mask behavior aligned across `full_deck/state.py`,
`deep_cfr/fast_state.py`, CUDA kernels, and `scripts/play_slumbot.py`.

## Setup

Use Python 3.13 or the project-managed environment when available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For CUDA traversal, ensure PyTorch, Numba, and the CUDA NVVM runtime are visible
in the active environment.

## Training

CPU-friendly Deep CFR:

```bash
poker_ai train-fast-deep-cfr --n-iterations 200 --n-traversals 500
```

GPU Deep CFR, recommended when CUDA is available:

```bash
python scripts/run_gpu_deep_cfr.py \
  --n-players 2 \
  --n-iterations 50 \
  --n-traversals 400 \
  --hidden-dim 256 \
  --n-training-steps 1500 \
  --save-path ./models
```

Slumbot-aligned heads-up training:

```bash
python scripts/train_slumbot_2p.py \
  --n-iterations 1000 \
  --n-traversals 10000 \
  --hidden-dim 512 \
  --n-layers 4
```

Autoresearch GPU training gate, which emits JSON throughput metrics and a
candidate checkpoint path. Use periodic saves plus auto-compare for longer
runs so the workflow evaluates intermediate checkpoints against the incumbent:

```bash
python scripts/poker_autoresearch.py enqueue-train \
  --n-iterations 50 \
  --n-traversals 4000 \
  --n-training-steps 1500 \
  --prefix candidate_gpu \
  --save-every 25 \
  --auto-compare \
  --compare-strategy-source regret
```

Use `--compare-strategy-source policy-head` only when both candidate and
incumbent checkpoints contain trained `policy_head` weights; legacy incumbents
must stay on `regret` comparisons.

## Autoresearch Governance

Use the autoresearch workflow for unattended experiments and methodology
changes. It records resumable state under `autoresearch-session/` and appends
concise entries to `RESEARCH_LOG.md`.

```bash
python scripts/poker_autoresearch.py init
python scripts/poker_autoresearch.py status
python scripts/poker_autoresearch.py continuous --sleep-seconds 30
```

Before changing a method, evaluation protocol, promotion rule, or persistent
research knob, queue a methodology review. Completing the review requires
independent-verifier notes, related work with source URLs, a benchmark-hacking
audit, and a decision file:

```bash
python scripts/poker_autoresearch.py enqueue-review \
  --subject "New search objective" \
  --trigger method_change \
  --claim "The objective should improve Slumbot transfer."
python scripts/poker_methodology_review.py \
  --review-dir autoresearch-session/poker_reviews/<review_id> \
  --require-complete
```

Register persistent knobs through `add-knob`, not ad hoc config drift. Each
knob needs one mechanism, one default, one failure class, and a removal
criterion; broad sweeps are intentionally rejected.

Evaluation harnesses, Slumbot adapters, solver benchmarks, promotion logic,
parsers, seed lists, and parity tests are protected surfaces. Audit them before
keeping a candidate or committing methodology changes:

```bash
python scripts/poker_objective_audit.py --base-ref HEAD
```

Before spending Slumbot confirmation hands on a candidate, queue the local
falsification ladder. It runs objective-drift audit, duplicate-swapped
incumbent comparison, and fixed-state resolver diagnostics:

```bash
python scripts/poker_autoresearch.py enqueue-falsification \
  --candidate models/candidate.pt \
  --mechanism "search-distilled policy targets reduce Slumbot transfer loss" \
  --max-resolver-cases 3
```

## Playing Slumbot

```bash
python scripts/play_slumbot.py \
  --model models/slumbot_2p_iter1000.pt \
  --hands 300 \
  --greedy
```

Use `--no-solver` to disable turn/river solving and `--no-allin` for lower
variance diagnostics. Use `--strategy-source policy-head` to test the trained
policy head instead of regret matching the advantage head for non-solver
decisions and range tracking. Use `--solver-backend torch-cuda` to force the
experimental torch/CUDA CFR+ backend for benchmarking; the default `auto`
backend stays on the measured-fast CPU solver until the CFR recurrence is
fused.

## Resolver Diagnostics

Run the fixed turn/river benchmark before promoting a checkpoint to live
Slumbot evaluation:

```bash
python scripts/poker_resolver_benchmark.py \
  --checkpoint models/slumbot_2p_iter1000.pt \
  --solver-iterations 25 \
  --solver-backend auto
```

The same check is available as an autoresearch gate:

```bash
python scripts/poker_autoresearch.py gate eval-resolver-fixed-states
```

## Testing

Fast checks for current Deep CFR and Slumbot parity:

```bash
pytest -q \
  test/unit/test_network_mask.py \
  test/unit/test_slumbot_mapping.py \
  test/unit/test_legal_mask_parity.py
```

Feature parity diagnostic:

```bash
python scripts/test_feature_encoding.py
```

CUDA integration checks, when a CUDA device is available:

```bash
pytest -q test/unit/test_gpu_optimizations.py
```

## Legacy Components

The following components are retained for historical compatibility and should
not be used as the main Slumbot path:

- `poker_ai/ai/` tabular MCCFR.
- `poker_ai/clustering/` short-deck card abstraction.
- `poker_ai/games/short_deck/`.
- `poker_ai/terminal/` console play against legacy agents.
- `applications/visualisation/` old short-deck visualisation.

## Development Notes

- Keep generated checkpoints in `models/`; do not commit large binary artifacts.
- For action-space, feature, legality, or Slumbot mapping changes, update parity
  tests in `test/unit/`.
- For trainer performance changes, report `iters/hour`, `samples/sec`, and
  `train sec/iter` with the exact command and model size.
- Prefer learned policy/value networks plus search over hand-written poker
  rules. CFR search and safe action translation are acceptable; ad hoc strategic
  rules are not.
