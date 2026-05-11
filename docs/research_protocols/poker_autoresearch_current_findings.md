# Poker Autoresearch Current Findings

Date: 2026-05-11

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
- `eval-resolver-fixed-states`: passed fixed public-state resolver legality
  and latency diagnostics for CPU, experimental torch-CUDA, and default auto
  backend selection.

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
- Policy-head sampled solver smoke, no-all-in, 50 hands: `-432` chips/hand,
  CI `1576`, zero parse/API errors, mapping drift mean `0.001`, and
  `5.249` seconds/hand after learned range pruning.

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
- Pre-min-raise-fix paired 20-hand no-solver smokes showed the issue clearly:
  `iter1000` produced 26 bet increments with mapping drift mean `0.119` and
  max `0.911`, while `iter900` was fold-heavy with zero mapping drift.
- Live solver diagnostics now emit solver call count, mean/max solver latency,
  solver cache hits, active hand count after range pruning, full hand count,
  and mean prune ratio. These fields are parsed into autoresearch JSON for
  future performance gates.

Resolved workflow issue: `action_mapping`.

The training and live masks exposed under-minimum raise buckets. Example: SB
first preflop marked the `0.5x` bucket legal even though it would spend only
125 chips from stack and must be clamped to a legal raise-to-200 Slumbot action.
This made the selected bucket differ from the executed live action. CPU,
fast-state, CUDA, Slumbot play, and range-tracker masks now require fractional
raise amounts to cover the no-limit minimum raise contribution before the
bucket is legal.

Post-fix sparse live diagnostics:

- `iter1000`, no-solver, no-all-in, 20 hands: mapping drift mean dropped from
  `0.119` to `0.002` and max from `0.911` to `0.013`; result was still noisy
  and negative at `-1186 +/- 1913` chips/hand.
- `iter900`, no-solver, no-all-in, 20 hands: mapping drift mean `0.001`, max
  `0.013`; result was `-2194 +/- 2391` chips/hand.
- Read: the adapter drift is fixed; old checkpoints remain stale/weak under
  corrected legality and should not be promoted without retraining.

GPU training readiness:

- Installed a CUDA NVVM wheel in the active Python environment and added the
  missing `libnvvm.so.4` compatibility symlink so Numba can discover NVVM.
- Added `scripts/cuda_env.py` so training entrypoints set `CUDA_HOME` and
  `NUMBA_FORCE_CUDA_CC=8.6` inside Python before importing Numba. Shell-level
  env overrides made CUDA invisible here.
- One-iteration corrected-legality Slumbot GPU training smoke passed with
  `device=cuda`, `traverse=2.3s`, `train=0.6s`, and saved
  `models/autoresearch_minraise_smoke/corrected_minraise_final.pt` without
  overwriting existing checkpoints.
- A larger 10M-buffer run failed at iteration 13 with CUDA OOM because the GPU
  replay cache consumed most of the 8GB card. The trainer now skips the GPU
  replay cache when its estimated tensor footprint would exceed a safe fraction
  of free GPU memory, falling back to host sampling instead of crashing.

Live solver backend status:

- The torch-CUDA street-solver backend is implemented but remains
  experimental. Clean A/B showed it is slower than NumPy CPU for the current
  Python-driven CFR recurrence: fixed resolver average latency was about
  `4174 ms` with torch-CUDA versus `2360 ms` with CPU, and direct river tests
  were roughly 2x slower on torch-CUDA across 25, 100, and 250 iterations.
- Default `--solver-backend auto` intentionally stays on CPU until the CFR
  recurrence is fused into coarse GPU kernels or matrix/sparse operations.
  Forcing torch-CUDA is useful only as a regression benchmark today.
- Learned range pruning is the current confirmed live-search speed path. In a
  concentrated-range river benchmark, pruning reduced a 1081-hand solve to 2
  active hands and cut wall time from about `2.03s` to `0.53s`.

## Diagnosis

Primary failure class: `distribution_shift`.

The current model is not validated by random-opponent local wins. The local
random baseline is positive, but Slumbot smokes are negative even with the
diagnostic no-all-in setting. Five to ten hands are not statistically strong,
but they are enough to show that the live evaluation path is working and that
random-opponent performance is too weak as the main promotion metric.

Secondary failure class: `search_quality`.

The original solver-enabled smoke took about 8.5 seconds per hand at the
10-hand setting. Policy-head sampled play with learned range pruning reduced
the latest 50-hand solver smoke to `5.249` seconds/hand, but this is still too
slow to use casually inside every unattended iteration. Slumbot search needs
separate latency, cache, active-hand-count, and quality gates.

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
5. Train a fresh corrected-legality checkpoint and compare it locally before
   spending more Slumbot hands.
6. Do not spend engineering time forcing the current torch-CUDA solver path for
   live play; the principled GPU step is a fused CFR backend following the
   matrix/sparse-operator direction, with range pruning and caching retained.
