# Poker Autoresearch Current Findings

Date: 2026-05-10

## Incumbent

- Checkpoint: `models/slumbot_2p_iter1000.pt`
- Iteration: 1000
- Shape: 4 layers, hidden dim 512
- Track: heads-up, full-deck, Slumbot-stack training

## Completed Gates

- `tier0`: passed legal-mask, network-mask, Slumbot mapping tests, and 10/10
  Slumbot feature-encoding parity checks under CUDA simulation.
- `eval-local`: passed 32-game fixed-seed local random-opponent smoke.
- `eval-local-confidence`: passed 1,000-game local random-opponent baseline.
- `eval-local-multiseed`: passed 3 seeds x 1,000 games.
- `eval-incumbent-self-compare`: passed incumbent-vs-itself comparison and
  emitted non-promotable local comparison blockers.
- `eval-head-to-head-self-compare`: passed duplicate-swapped model-vs-model
  self-comparison.
- `slumbot-smoke`: passed live API integration with solver/all-in disabled.
- `slumbot-solver-smoke`: passed live API integration with turn/river solver
  enabled and all-in disabled.

## Metric Snapshot

Local random baseline:

- 3,000 total local games across three seeds.
- Average: `+319.23` chips/hand.
- Lower 95% across seeds: `+233.72` chips/hand.

Live Slumbot diagnostic smokes:

- No-solver, no-all-in, 5 hands: `-80` chips/hand.
- Solver enabled, no-all-in, 10 hands: `-443` chips/hand.
- Solver smoke runtime: `84.98` seconds for 10 hands.
- Runtime-metric no-solver smoke: `-308` chips/hand over 5 hands,
  `3.669` elapsed seconds, `0.734` seconds/hand.

Local incumbent comparison:

- Incumbent self-compare: `0.0` delta chips/hand over 3 seeds x 500 games.
- Promotion status: `promotable=false`.
- Promotion blockers: `local_random_comparison_is_not_strategy_quality`,
  `requires_incumbent_or_slumbot_confidence_gate`.

Local head-to-head self-comparison:

- Duplicate-swapped self-compare: `0.0` chips/hand over 1,200 local games
  (3 seeds x 200 duplicate pairs).
- Promotion status: `promotable=false`.
- Promotion blocker: `local_head_to_head_requires_slumbot_confirmation`.

Candidate checkpoint comparison:

- `iter700` vs `iter1000`: `-49.32` chips/hand, lower95 `-94.37`
  over 3,000 duplicate-swapped local games.
- `iter800` vs `iter1000`: `-11.75` chips/hand, lower95 `-103.79`
  over 3,000 duplicate-swapped local games.
- `iter900` vs `iter1000`: `-9.50` chips/hand, lower95 `-27.58`
  over 3,000 duplicate-swapped local games.
- Larger `iter900` vs `iter1000`: `+6.14` chips/hand, lower95 `+0.46`
  over 12,000 duplicate-swapped local games.

Sparse live candidate smoke:

- `iter900`, no-solver, no-all-in, 10 hands: `-1138` chips/hand,
  CI `1982`, `0.694` seconds/hand. This is too noisy for ranking, but it
  reinforces that local head-to-head gains do not yet prove Slumbot transfer.

Slumbot diagnostic instrumentation:

- Live Slumbot summaries now include policy/solver/fallback decision counts,
  action mix, increment mix, parse/API error counts, and action-mapping drift.
  These fields are parsed into gate JSON so future smokes can distinguish
  poor strategy from adapter or mapping drift.

## Diagnosis

Primary failure class: `distribution_shift`.

The current model is not validated by random-opponent local wins. The local
random baseline is positive, but Slumbot smokes are negative even with the
diagnostic no-all-in setting. Five to ten hands are not statistically strong,
but they are enough to show that the live evaluation path is working and that
random-opponent performance is too weak as the main promotion metric.

Secondary failure class: `search_quality`.

The solver-enabled smoke took about 8.5 seconds per hand at the current 10-hand
setting. This is acceptable for a diagnostic smoke but too slow to use casually
inside every unattended iteration. Slumbot search needs separate latency,
cache, and quality gates.

Resolved workflow issue: `rules_parity`.

The feature-encoding parity script was stale for heads-up postflop order. It
assumed SB-first postflop in several scenarios, but the current fast/CUDA game
uses `[1, 0]` postflop order, matching BB-first heads-up poker and Slumbot.
After updating those scenarios and running under `NUMBA_ENABLE_CUDASIM=1`, the
script passes all 10 checks and is now part of Tier 0.

## Next Workflow Moves

1. Use Slumbot only for sparse live checks until the local incumbent-comparison
   path and promotion blockers are stable.
2. Candidate checkpoint comparisons are now queueable with
   `python scripts/poker_autoresearch.py enqueue-compare --candidate <path>`.
   Use `--head-to-head` for the stronger duplicate-swapped model-vs-model gate.
3. Sparse live Slumbot smokes are now queueable with
   `python scripts/poker_autoresearch.py enqueue-slumbot --model <path>`.
4. Investigate why no-solver and solver Slumbot smokes remain negative despite
   strong local-random metrics by comparing the new action diagnostics across
   incumbent and candidate checkpoints.
