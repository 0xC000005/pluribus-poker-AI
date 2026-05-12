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
- Policy-head sampled solver smoke with direct solver performance metrics,
  no-all-in, 30 hands: `-1272` chips/hand, CI `1845`, `5.416` seconds/hand,
  12 solver calls, mean solver latency `12229.1 ms`, mean active hands
  `1017.2/1108.4`, prune ratio `0.9178`, and no parse/API errors.

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
- Autoresearch gates now prefer the repository `.venv/bin/python` (falling back
  to `CONDA_PREFIX/bin/python`) because a stale `VIRTUAL_ENV` from another repo
  can expose Torch CUDA while missing the pip NVVM package needed by Numba.
- One-iteration corrected-legality Slumbot GPU training smoke passed with
  `device=cuda`, `traverse=2.3s`, `train=0.6s`, and saved
  `models/autoresearch_minraise_smoke/corrected_minraise_final.pt` without
  overwriting existing checkpoints.
- JSON-emitting autoresearch GPU train gate passed after the Python selection
  fix: `device=cuda`, `avg_iter_seconds=4.114`, `iters_per_hour=875.087`,
  `traversals_per_second=2.43`, checkpoint
  `models/autoresearch_gpu_20260511T162934Z/gpu_smoke_fixed_final.pt`.
- Modest GPU candidate smoke passed end to end: 5 iterations, 400 traversals,
  100 training steps, `avg_iter_seconds=1.322`, `iters_per_hour=2722.91`,
  `traversals_per_second=302.458`, checkpoint
  `models/autoresearch_gpu_20260511T163126Z/gpu_candidate_smoke_final.pt`.
  The follow-up 400-game duplicate-swapped comparison was not promotable:
  `avg_chips_per_hand=-99.5525`, lower95 `-390.294`.
- Larger GPU candidate run passed: 20 iterations, 2000 traversals, 500
  training steps, `avg_iter_seconds=2.736`, `iters_per_hour=1315.781`,
  `traversals_per_second=730.962`, checkpoint
  `models/autoresearch_gpu_20260511T163408Z/gpu_candidate_20x2k_final.pt`.
  The 3,000-game duplicate-swapped comparison against the incumbent was still
  negative: `avg_chips_per_hand=-66.27`, lower95 `-111.607`.
- Incumbent resume gate now loads legacy checkpoints, but the first resumed
  run was not a true continuation because existing checkpoints do not store
  replay buffers and Deep CFR retrains from replay. The 50-iteration resumed
  run reached iteration 1050 with `avg_iter_seconds=6.148` and
  `traversals_per_second=325.286`, but comparison was negative:
  `avg_chips_per_hand=-106.451`, lower95 `-183.261`.
- Fresh 4x512 GPU run reached 100 iterations with 2000 traversals and 1000
  training steps per iteration in `751.758s` (`avg_iter_seconds=7.518`,
  `traversals_per_second=266.043`). The 3,000-game duplicate-swapped local
  comparison was close but not promotable: `avg_chips_per_hand=15.758`,
  lower95 `-9.706`.
- Extended evaluation of the same 100-iteration checkpoint over 6,000
  duplicate-swapped games was not promotable: `avg_chips_per_hand=-9.883`,
  lower95 `-67.258`.
- Fresh 4x512 GPU run at 200 iterations with the same per-iteration budget
  completed in `1497.888s` (`avg_iter_seconds=7.489`,
  `traversals_per_second=267.043`) but regressed locally:
  `avg_chips_per_hand=-23.961`, lower95 `-76.519`.
- Autoresearch GPU training now supports periodic `--save-every` checkpoints
  and `--auto-compare`, which queues incumbent head-to-head comparisons for
  every emitted checkpoint after the training gate passes. This is now the
  preferred unattended mode because current results are non-monotonic across
  training length.
- Local evaluation can now choose `--strategy-source regret` or
  `--strategy-source policy-head`. Policy-head evaluation is guarded so legacy
  checkpoints without trained `policy_head` weights are rejected instead of
  silently using a randomly initialized head. The current recorded incumbent
  `models/slumbot_2p_iter1000.pt` is legacy, so policy-head comparisons require
  first promoting or explicitly selecting a policy-head-capable incumbent.
- Reducing value-network optimization to 500 training steps improved throughput
  but did not improve strategy quality. The 100-iteration 4x512 run completed
  in `616.236s` (`avg_iter_seconds=6.162`, `iters_per_hour=584.201`,
  `traversals_per_second=324.551`). Against the legacy incumbent under regret
  matching, `iter_50` was a near miss (`avg_chips_per_hand=3.268`, lower95
  `-6.541` over 18,000 games), while `iter_100`/final regressed
  (`avg_chips_per_hand=-14.295`, lower95 `-21.486`). This suggests optimizer
  step count is not the main remaining strategy-quality bottleneck.
- Policy-head local comparison produced a strong positive signal between two
  newer policy-head-capable checkpoints: `trainsteps2k_4x512_100x2k_final.pt`
  beat `fresh_4x512_175x2k_curve_final.pt` by `104.361` chips/hand with lower95
  `61.924` over 18,000 duplicate-swapped games. This should be treated as a
  policy-head diagnostic, not promotion evidence against the legacy incumbent.
- Sparse Slumbot smokes did not confirm that local policy-head signal. The same
  `trainsteps2k_4x512_100x2k_final.pt` checkpoint produced a noisy positive
  50-hand sampled policy-head smoke (`avg_chips_per_hand=366`, CI `336`), but
  the 300-hand confirmation was negative (`avg_chips_per_hand=-677`, CI `569`,
  no solver/no all-in). The legacy incumbent under the same 300-hand sampled
  no-solver/no-all-in protocol was also negative (`avg_chips_per_hand=-393`, CI
  `329`), so the live protocol remains harsh and noisy, but the candidate still
  fails transfer at this sample size.
- Queueing two one-off gates inside the same second exposed a workflow bug:
  timestamp-only gate names collided and one Slumbot smoke overwrote another.
  One-off gate creation now uniquifies names with numeric suffixes when needed.
- The 175-iteration checkpointed curve still failed extended confirmation:
  the final checkpoint moved from a near-miss 3,000-game comparison to
  `avg_chips_per_hand=-17.650`, lower95 `-31.220` over 6,000 games.
- Training logs showed frequent traversal pool exhaustion with the default
  `500` slots/traversal assumption, often reporting more than 300% pool use.
  This demotes traverser nodes into sampled actions and can discard regret
  samples. A high-fidelity `2,500` slots/traversal mode preserved more fork
  space but slowed training to `25.054s/iter` and did not improve the 50-iter
  local comparison, so it is available as an explicit experiment knob rather
  than the default.
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
- The latest live Slumbot ranges were still diffuse, so threshold pruning kept
  most hands (`0.9178` mean prune ratio). The practical read is that better
  learned range concentration or a fused GPU CFR backend is required for a
  larger live-search speedup.
- Treat old checkpoint resume as a compute/path smoke, not a strategy
  improvement method, until either replay buffers are serialized or the trainer
  intentionally supports warm-started value-network updates.

Workflow governance status:

- Autoresearch state now carries explicit commit, review, and knob policies.
  Existing sessions are migrated by `python scripts/poker_autoresearch.py init`
  without clearing history.
- Methodology-review bundles are queueable with `enqueue-review` and validated
  by `scripts/poker_methodology_review.py --require-complete`. A completed
  review must include independent-verifier findings, related work with source
  URLs, benchmark-hacking audit, and a decision file. `team_review.md` maps
  those files to research lead, verifier, literature scout, and benchmark
  auditor roles for sub-agent use.
- Objective-drift audit now blocks protected evaluation-surface changes unless
  a completed review exists. Protected surfaces include evaluation harnesses,
  Slumbot adapters, solver benchmarks, promotion logic, parsers, seed lists,
  and parity tests.
- The POPPER-inspired falsification ladder is now the expected local
  counter-test before Slumbot confidence spend. It runs objective-drift audit,
  duplicate-swapped incumbent comparison, and fixed-state resolver diagnostics.
- Persistent knobs are registered through `add-knob`, require a mechanism and
  removal criterion, and reject broad sweep-shaped defaults. This keeps the
  workflow focused on falsifying mechanisms instead of benchmark tuning.
- New research-log entries record a metrics artifact path plus key metrics
  instead of pasting full raw command JSON into `RESEARCH_LOG.md`.

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
The first direct solver-performance gate showed the main live bottleneck:
solver calls averaged `12.2s` and range pruning kept most possible hands.

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
   Use `--head-to-head` for the stronger duplicate-swapped model-vs-model gate,
   and reserve `--strategy-source policy-head` for checkpoints with trained
   policy heads on both sides.
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
7. Before changing the training objective, evaluation ladder, promotion rule,
   or persistent knob set, run `enqueue-review`, complete related-work review,
   and commit only after implementation, tests, docs, and log updates are
   batched into one research-objective change.
8. Before keeping any candidate that touched protected evaluation surfaces, run
   `python scripts/poker_objective_audit.py --base-ref HEAD` and require a
   completed review bundle if the audit reports protected hits.
9. Before Slumbot confirmation, queue
   `python scripts/poker_autoresearch.py enqueue-falsification --candidate <path>
   --mechanism "<mechanism>"` and close the falsification cycle through the
   normal autoresearch log.
