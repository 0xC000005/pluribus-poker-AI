# Gemini CLI Project Context: Poker AI

This repository's active goal is a full-deck poker AI that can train by
self-play and play heads-up no-limit Texas hold'em against Slumbot. The current
main path is Deep CFR with CUDA traversal, a 9-action abstraction, and
turn/river range-vs-range CFR+ solving. Legacy short-deck tabular MCCFR,
clustering, terminal play, and visualisation code are retained for reference.

## Main Architecture

- `poker_ai/deep_cfr/`: active learning stack. `networks.py` defines the
  advantage/policy-head MLP, `policy_targets.py` defines optional
  search-distilled policy target buffers, `buffer.py` defines reservoir replay,
  and `deep_cfr.py` contains the reference training loop.
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
- `scripts/poker_resolver_benchmark.py`: fixed public-state resolver benchmark
  for legality, latency, blueprint drift, and learned-advantage proxies.
- `scripts/build_search_targets.py`: builds bounded resolver policy targets for
  optional search-consistency training.
- `scripts/build_policy_calibration_targets.py` and
  `scripts/train_policy_head_calibration.py`: collect learned self-play policy
  targets and train only the policy head for range-likelihood diagnostics.

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
python scripts/play_slumbot.py --model models/slumbot_2p_iter1000.pt --hands 300 --greedy --strategy-source policy-head --solver-backend auto
```

`--solver-backend torch-cuda` exists for benchmarking, but the default `auto`
backend currently stays on CPU because the torch backend is not fused yet.

Resolver diagnostics:

```bash
python scripts/poker_resolver_benchmark.py --checkpoint models/slumbot_2p_iter1000.pt --solver-iterations 25 --solver-backend auto
python scripts/poker_autoresearch.py gate eval-resolver-fixed-states
```

Search-consistency targets:

```bash
python scripts/build_search_targets.py --output autoresearch-session/search_targets/sampled_turn_river_train.npz --sampled-cases 64 --seed 20260512 --solver-iterations 25 --solver-backend auto
python scripts/poker_autoresearch.py enqueue-train --n-iterations 50 --n-traversals 4000 --search-targets autoresearch-session/search_targets/sampled_turn_river_train.npz --search-target-weight 0.05 --prefix search_consistency --save-every 25 --auto-compare
python scripts/eval_search_targets.py --checkpoint models/candidate.pt --targets autoresearch-session/search_targets/sampled_turn_river_holdout.npz --strategy-source policy-head
```

Policy-head calibration diagnostics:

```bash
python scripts/build_policy_calibration_targets.py --checkpoint models/control.pt --output autoresearch-session/policy_calibration/calib_targets.npz --n-targets 4096 --strategy-source regret
python scripts/train_policy_head_calibration.py --checkpoint models/control.pt --targets autoresearch-session/policy_calibration/calib_targets.npz --output autoresearch-session/policy_calibration/calibrated.pt --n-steps 600
python scripts/diagnose_range_tracker.py --checkpoint autoresearch-session/policy_calibration/calibrated.pt --cases-json autoresearch-session/search_targets/reachable_policyhead_holdout_16x5.cases.json --strategy-source policy-head
```

Autoresearch governance:

```bash
python scripts/poker_autoresearch.py enqueue-review --subject "New search objective" --trigger method_change --claim "The objective should improve Slumbot transfer."
python scripts/poker_methodology_review.py --review-dir autoresearch-session/poker_reviews/<review_id> --require-complete
python scripts/poker_objective_audit.py --base-ref HEAD
python scripts/poker_autoresearch.py enqueue-falsification --candidate models/candidate.pt --mechanism "search-distilled policy targets reduce Slumbot transfer loss" --max-resolver-cases 3
python scripts/poker_autoresearch.py add-knob --name search_target_mix --default 0.0 --failure-class search_quality --mechanism "Test whether search-distilled targets reduce live transfer loss." --rationale "One variable isolates the target mechanism." --removal-criterion "Retire if Slumbot transfer remains negative after confirmation."
```

## Development Rules

- Treat full-deck Deep CFR and Slumbot parity as the source of truth.
- Avoid new hand-crafted poker heuristics; prefer learned policies, regret
  matching, and principled search.
- Search-consistency targets must come from resolver/search output under the
  legal mask; do not encode manual no-all-in or street-specific rules.
- Policy-head calibration is a diagnostic for learned range-likelihood
  calibration. Improved behavior-cloning loss or range dispersion is not a
  promotion claim without head-to-head, resolver, and Slumbot evidence.
- Before method, evaluation, promotion, or persistent-knob changes, complete a
  methodology review with independent verification, related work, and a
  benchmark-hacking audit.
- Treat evaluation scripts, Slumbot adapters, solver benchmarks, promotion
  logic, parsers, seed lists, and parity tests as protected surfaces.
- Queue the falsification ladder before Slumbot confirmation; it is the local
  counter-test stage, not a promotion by itself.
- Add research knobs only through `add-knob`; keep one mechanism and one
  primary variable, and avoid broad sweeps.
- For performance work, report `iters/hour`, `samples/sec`, and
  `train sec/iter` with the exact command.
- Batch commits by research objective; do not commit every individual file edit
  or commit from continuous autoresearch mode.
- Do not commit generated checkpoints from `models/` or lookup-table artifacts
  from `research/`.
