# Poker Autoresearch Current Findings

Date: 2026-05-13

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
- Restored-history 200x2k 4x512 checkpoint, solver enabled, 100 hands:
  `+31` chips/hand, CI `678`, `10.175` seconds/hand, 45 solver calls, mean
  solver latency `21481 ms`, and zero parse/API errors. This is positive but
  too noisy and too slow for confirmation.
- Same checkpoint, no solver, 100 hands: `+255` chips/hand, CI `260`,
  `0.529` seconds/hand. The 500-hand no-solver follow-up reversed to `-45`
  chips/hand, CI `240`, `0.484` seconds/hand. Treat the 100-hand win as noise.

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
  every emitted checkpoint after the training gate passes. Auto-comparisons
  now require a positive lower 95% bound; finite negative bounds are rejection
  evidence, not pass conditions. This is now the preferred unattended mode
  because current results are non-monotonic across training length.
- Restored-history observation is now the default for new Deep CFR checkpoints.
  A toy 5-iteration A/B beat the masked-history compatibility control locally,
  but fixed-state resolver drift did not improve, so it was not a Slumbot
  candidate. Scaling the same mechanism to 200 iterations, 2000 traversals, and
  1000 training steps with a 4x512 net produced the first restored-history
  local pass: `avg_iter_seconds=8.255`, `iters_per_hour=436.079`,
  `traversals_per_second=242.264`; checkpoints at 50, 100, and 150 iterations
  failed incumbent H2H, while `iter_200`/final passed with
  `avg_chips_per_hand=57.080`, lower95 `49.961` over 3000 duplicate-swapped
  games.
- The restored-history 200x2k final checkpoint passed the full local
  falsification ladder: objective audit clean except `RESEARCH_LOG.md`, H2H
  lower95 positive, fixed-state resolver legality clean, mean resolver action
  L1 drift `1.375`, policy-head drift `0.632`, and no illegal fixed cases.
  The resolver benchmark remains diagnostic only; live Slumbot confirmation is
  still inconclusive and solver latency is the current practical bottleneck.
- Extending the same restored-history 4x512 setup to 400 iterations did not
  improve the local gate. Training slowed to `avg_iter_seconds=9.363`,
  `iters_per_hour=384.501`, and `traversals_per_second=213.611`. Checkpoints at
  100 and 200 iterations failed (`-52.790`, lower95 `-90.513`; `-1.202`,
  lower95 `-45.549`), `iter_300` was a near miss (`+12.702`, lower95 `-1.445`),
  and `iter_400`/final failed with high variance (`+27.073`, lower95
  `-96.267`). Do not treat horizon-only scaling beyond 200 iterations as the
  current mechanism; the next bottleneck is transfer/search quality.
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
- The corrected falsification ladder now rejects the same near-miss checkpoint
  because candidate-vs-incumbent lower95 is negative (`-15.59` chips/hand).
  The old behavior treated a successful evaluator process as a pass even when
  the statistical lower bound failed; this is now blocked before Slumbot spend.
- Fixed turn/river resolver diagnostics show a clearer mechanism target than
  more blind optimizer-step tuning. The near-miss candidate has mean
  blueprint-vs-solver action L1 drift `1.597`, policy-head drift `0.682`, and
  all-in selection on `2/4` fixed cases; the incumbent has mean action drift
  `1.369`, policy-head drift `0.648`, and all-in selection on `2/4` fixed
  cases. The diagnostics pass legality/latency mechanics, but strategy drift is
  high.
- Methodology review
  `20260512T193213Z-search-consistency-training-objective` approved a narrow
  next step: generate bounded resolver targets for sampled turn/river public
  states and train the existing policy head toward those targets under the
  legal mask. Fixed resolver cases remain diagnostics, not a training target;
  no manual no-all-in rule should be added.
- Search-consistency plumbing is now implemented behind an explicit opt-in
  weight. `scripts/build_search_targets.py` can generate resolver policy-target
  `.npz` files, `PolicyTargetBuffer` enforces legal-mask normalization, and GPU
  training accepts `--search-targets`, `--search-target-weight`, and
  `--search-target-batch-size`. A one-iteration CUDA smoke with four fixed
  resolver targets passed and recorded `search_target_size=4`; this validates
  the path, not strategy quality.
- Sampled search-target generation now supports train/held-out turn-river
  public states with recorded seeds and emitted case JSON. A tiny controlled
  A/B (`10` iterations, `500` traversals, `50` training steps, `hidden_dim=128`)
  showed the target path is learnable but not promotable: held-out policy-target
  L1 improved only from `0.349` to `0.330`, policy-head resolver drift improved
  from `0.455` to `0.441`, but blueprint resolver drift worsened from `1.301`
  to `1.386` and policy-head all-in rate rose to `0.875` against a holdout
  target all-in rate of `0.375`. Treat this as a failed search-quality cycle
  and diagnose target quality/action-bucket legality before scaling it.
- The first diagnosis found a solver action-space parity bug: the street solver
  clamped under-minimum fractional raise buckets up to the minimum raise, while
  the training/Slumbot legal masks remove those buckets. The solver now omits
  under-minimum fractional actions instead. On the same 8-case sampled holdout,
  resolver `illegal_case_count` dropped from `2` to `0`; control/target resolver
  benchmarks both pass legality, though strategy drift remains too high for
  promotion.
- A corrected search-target A/B with 32 train and 16 held-out sampled states at
  10 solver iterations confirmed the next blocker is target distribution
  quality, not plumbing. Target training learned the targets (`top1` `0.875`
  on holdout) and improved resolver drift (`1.172` to `0.997`), but the held-out
  target set was `0.875` all-in and the policy head selected all-in on every
  held-out case. Local duplicate-swapped transfer was negative versus control
  under regret (`-10.872`, lower95 `-99.501`) and policy-head play (`-51.873`,
  lower95 `-185.978`). Do not scale random public-state/uniform-range targets;
  move target generation toward gameplay-distributed public states and learned
  public-belief/range inputs.
- The gameplay-distributed search-consistency scaleup used resolver targets
  generated from restored-history blueprint self-play
  (`32` train targets, `16` held-out targets) and trained a 100-iteration,
  2000-traversal, 4x512 checkpoint with `search_target_weight=0.05`. Training
  completed on CUDA at `avg_iter_seconds=10.134`, `iters_per_hour=355.235`,
  and `traversals_per_second=197.351`. The candidate memorized its tiny
  training target set (`mean_l1=0.0348`, `top1_match_rate=0.9375`) but did not
  generalize on held-out targets (`mean_l1=1.1369`, `top1_match_rate=0.0625`),
  slightly worse than the restored-history baseline (`mean_l1=1.1247`,
  `top1_match_rate=0.0625`). Local regret-source comparison was mean-positive
  but not promotable: `iter_50` `+3.046`, lower95 `-39.915`; `iter_100`/final
  `+28.691`, lower95 `-22.832` over 3000 duplicate-swapped games. The trained
  policy head itself was weak against the incumbent regret source
  (`-61.756`, lower95 `-130.252`) and weaker than its own regret source
  (`-32.194`, lower95 `-174.269`). Read: naive small-batch resolver-target
  distillation is overfitting and should not be scaled by weight or horizon
  alone.
- Related work reinforces this diagnosis. Deep CFR trains an average strategy
  network from a broad strategy memory, not a tiny fixed resolver-target set,
  and SD-CFR argues that avoiding a separate average-strategy network can reduce
  approximation error. ReBeL and Student of Games both point to the stronger
  mechanism: train and search over public belief states with value targets from
  search during self-play, using the same search distribution at training and
  inference. The next principled mechanism is therefore public-belief/value
  learning with search-generated PBS targets, not more ad hoc target weighting.
- A public-belief probe now exists for this mechanism. It appends raw learned
  hero/villain range distributions over all 1326 card combos to the existing
  feature vector and compares a supervised feature-only probe to a
  feature-plus-belief probe. On the tiny `32/16` restored-history split, belief
  inputs still overfit and worsened held-out fit (`mean_l1` `1.2586` vs
  `1.1719`). On a larger `128/64` gameplay-distributed split, the belief probe
  gave a small but not promotion-grade signal: L1 improved in all three probe
  seeds (`+0.0114`, `+0.0404`, `+0.0460`), KL improved in two of three
  (`-0.0337`, `+0.0082`, `+0.0320`), and the strict pass criterion passed in
  `2/3` seeds. Read: raw public belief is useful enough to keep investigating,
  but the effect is too small and noisy to wire into mainline Deep CFR without a
  real value-target probe or larger search distribution.
- The gameplay-distributed target sampler had a target-distribution bug: it
  captured the first eligible postflop hero decision, so all recorded
  `restored200_blueprint_*` train/holdout targets were turn cases. The sampler
  now selects the next desired target street round-robin via
  `--blueprint-target-streets`. River-only smoke generated `4/4` river targets,
  and a mixed-street smoke generated `2/2` turn and `2/2` river targets. Old all-turn
  target artifacts remain useful only as diagnostics.
- On corrected round-robin turn/river target generation, the public-belief
  probe signal strengthened. A `64/32` mixed-street split passed the strict
  criterion in all three probe seeds: L1 deltas were `+0.0304`, `+0.0794`, and
  `+0.1295`; KL deltas were `+0.0165`, `+0.0221`, and `+0.0044`; top-1 match
  improved by `+0.0938`, `+0.1875`, and `+0.1875`. Read: learned public belief
  is the next principled model input to test, preferably through a PBS value
  target pathway rather than another small action-policy distillation patch.
- A first scalar public-belief value probe did not confirm that direction as a
  direct raw-input value head. The new probe computes searched hero EV targets
  from solved turn/river subgames, compares feature-only regression to
  feature-plus-raw-belief regression, and caches the expensive EV labels for
  repeated seeds. On the same corrected `64/32` mixed-street split, raw belief
  failed `0/3` seeds and worsened held-out MAE/RMSE: mean MAE delta `-0.2493`
  and mean RMSE delta `-0.3298` in stack-scaled units. Read: do not wire the
  raw 2652-dimensional public-belief vector directly into the main scalar
  value path from this small target set. Either scale gameplay-distributed
  value targets first or test a learned/compressed belief encoder as a bounded
  representation probe.
- A follow-up counterfactual-value vector probe used the more principled
  DeepStack/ReBeL-shaped target: public features with private-card slots zeroed
  and one searched hero-CFV vector over all 1326 private hands per public state.
  This greatly reduced the penalty versus scalar EV but still failed all three
  seeds on the corrected `64/32` mixed-street split: mean MAE delta `-0.0399`,
  mean RMSE delta `-0.0627`, with `70,688` train hand-value labels and `35,344`
  holdout labels. Read: CFV-vector value learning is the right target shape,
  but raw public-belief input is not yet promotable at this data scale.
- Doubling the corrected CFV split to `128/64` did not fix this; it failed
  `0/3` seeds with mean MAE delta `-0.0594` and mean RMSE delta `-0.0935`
  despite `141,376` train hand-value labels and `70,688` holdout labels. Read:
  simply adding a small amount of supervised CFV data is not the next lever.
  The next belief-value test should change representation learning or
  regularization, not target count alone.
- A shared per-hand CFV probe is the first positive belief-value representation
  signal. Instead of predicting a 1326-dimensional vector from one public-state
  row, it trains on `(public state, hero hand)` pairs with shared public,
  hand-card, and belief encoders. On the corrected `64/32` split it passed
  `3/3` seeds with mean MAE delta `+0.0183` and mean RMSE delta `+0.0126`.
  The independently generated `128/64` split initially failed (`0/3`, mean MAE
  delta `-0.0193`, mean RMSE delta `-0.0675`), but verifier diagnostics traced
  this to small-split river CFV distribution mismatch rather than a clean
  architecture falsification. A fixed `128/64` split stratified by street and
  searched-CFV mean bins passed `3/3` seeds, with mean MAE delta `+0.0573` and
  mean RMSE delta `+0.0633`. Read: hand-shared public-belief CFV learning is
  now the most promising belief-value direction, but promotion should use the
  stratified evaluation surface and still require gameplay/search integration.
- The positive hand-CFV representation now has a saved checkpoint path via
  `scripts/train_public_belief_hand_cfv.py`. On the stratified `128/64`
  surface, the saved CUDA-trained checkpoint reached held-out MAE `0.3930` and
  RMSE `0.5233` against searched CFV labels. Read: this is the reusable learned
  value component to attach to depth-limited search experiments, not a direct
  Slumbot-play promotion.
- A river-only stratified CFV surface gives an even cleaner signal for
  DeepStack-style turn lookahead. Filtering the cached mixed pool to river
  states produced a balanced `64/32` split, and the shared hand-CFV belief probe
  passed `3/3` seeds with mean MAE delta `+0.1254` and mean RMSE delta
  `+0.1437`. The saved river-only checkpoint reached held-out MAE `0.4549` and
  RMSE `0.5588`. Batched CUDA inference over the 32-state holdout took
  `3.77 ms` mean (`0.118 ms/state`, `9.17M` hand labels/sec), compared with
  `1127.5 ms/state` average solver-label latency in the cache. Read: use the
  river-only learned CFV component first when testing learned leaf values for
  turn search.
- A first range-robustness diagnostic did not falsify the river CFV component.
  On 8 held-out river states, re-solving 50% uniform-mixed ranges gave MAE
  `0.4478`; fully uniform legal ranges gave MAE `0.4790`. These are close to
  the original river holdout MAE `0.4549`, although the sample is still small.
  Read: proceed to a bounded turn-search leaf experiment, but keep range
  robustness in the promotion gate.
- Direct search integration still needs dual-player CFVs. The current saved
  hand-CFV model predicts hero values only; DeepStack-style depth-limited
  re-solving needs value vectors for both players. The code now has a tested
  `compute_villain_cfv_vector` primitive so the next probe can train a
  dual-player public-belief value model instead of forcing a hero-only model
  into solver leaves.
- The first dual-player river probe is not stable enough for integration. On a
  tiny `16/8` river split it failed `0/3` seeds. On the full river `64/32`
  split it passed only `1/3` seeds: MAE deltas were `-0.0171`, `-0.0193`, and
  `+0.1536`; RMSE deltas were `-0.0381`, `-0.0223`, and `+0.1934`. Read:
  dual-player CFV learning is the correct target shape, but raw belief with the
  current small MLP/player-indicator architecture is not promotable as a search
  leaf yet.
- A mechanism-specific separate-player-head variant improves the dual-player
  result but still needs confirmation. On the same full river `64/32` split,
  MAE improved in all three seeds (`+0.0397`, `+0.1393`, `+0.0273`) and the
  strict criterion passed `2/3`; RMSE deltas were `+0.0394`, `+0.1674`, and
  `-0.0411`. Read: separate hero/villain value heads are the next architecture
  to harden; do not use the older shared scalar head for leaf-value work.
- A learned belief bottleneck removes the immediate dual-player instability on
  the same cached river split. With `--head-mode separate
  --belief-bottleneck-dim 32`, all three seeds passed; MAE deltas were
  `+0.0989`, `+0.1042`, and `+0.0622`, while RMSE deltas were `+0.1586`,
  `+0.1659`, and `+0.0839`. Read: the next leaf-value architecture should use
  separate player heads plus a learned belief compression stage, but it still
  needs a larger river surface before solver integration.
- The larger balanced river surface preserves the feature-baseline improvement
  but fails a harder zero-CFV baseline. On a new gameplay-reachable river pool
  split into `128/64` public states, separate-head plus 32-dim belief bottleneck
  improved over the feature-only model in all three seeds, but every seed was
  worse than predicting zero CFV: zero MAE/RMSE were `0.4257/0.6297`, while the
  belief model MAEs were `0.5051`, `0.5010`, and `0.5485`. It also loses to
  train-constant baselines: train-median MAE is `0.3984`, and train-mean RMSE is
  `0.5250`. Read: the previous dual-player probe gate was too weak;
  learned-leaf work is blocked until the model beats trivial constant
  baselines.
- A saved dual-player checkpoint path now exists for that architecture via
  `scripts/train_public_belief_dual_hand_cfv.py`. The first larger-surface
  checkpoint trained from cached dual labels on CUDA and reproduced the seed
  `20260530` holdout metrics (`MAE=0.5051`, `RMSE=0.6826`), but it fails the
  constant-baseline gate. Inference is fast (`0.242 ms/state`, about `7,583x` faster
  than cached solver labels), so the bottleneck is target/model quality rather
  than deployment latency.
- Scaling the river surface from `128/64` to `384/128` public states helps but
  does not solve the constant-baseline failure. The belief model still improves
  over the feature-only baseline on average (`MAE delta +0.0758`, `RMSE delta
  +0.0933`) and nearly matches zero MAE in the best seed, but pass count remains
  `0/3` because train-constant baselines are better (`train-median MAE=0.4098`,
  `train-mean RMSE=0.5292`). Read: naive data scaling alone is not enough at
  this size; the next method needs a better value target/objective or stronger
  generalization regularizer, judged against the constant gate.
- Centering CFVs by public-state/player means does not rescue the current MLP.
  On the `384/128` split, a state-centered residual probe failed `0/3` seeds
  against a zero-residual baseline (`MAE=0.3569`, `RMSE=0.4483`). Increasing the
  hidden size from `64` to `256` improved the best seed (`MAE=0.3818`,
  `RMSE=0.5001`) but still failed. Read: the failure is not only an absolute
  value offset; the current additive public+hand+belief MLP is not learning
  holdout hand-residual structure well enough.
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
  duplicate-swapped incumbent comparison with a required positive lower 95%
  bound, and fixed-state resolver diagnostics.
- Persistent knobs are registered through `add-knob`, require a mechanism and
  removal criterion, and reject broad sweep-shaped defaults. This keeps the
  workflow focused on falsifying mechanisms instead of benchmark tuning.
- New research-log entries record a metrics artifact path plus key metrics
  instead of pasting full raw command JSON into `RESEARCH_LOG.md`.

## Learned River Leaf Status

- Hero-only river hand-CFV inference is fast enough for search leaves, but a
  learned leaf cannot be promoted from hero-only values. Resolver leaves need
  both players' counterfactual value vectors.
- The first dual-player public-belief hand-CFV path with separate heads and a
  32-dim belief bottleneck looked positive on `128/64`, but the stricter
  zero/train-constant baseline invalidated it. Its checkpoint was fast
  (`0.242 ms/state`, about `7,583x` faster than solver labels), so the blocker
  is value quality, not inference latency.
- Scaling the river public-state surface to `384/128` did not rescue the flat
  additive public+hand+belief MLP. It failed train-constant baselines in all
  three seeds, and centered-residual checks showed the model was not just
  missing public-state/player offsets.
- The first useful representation change is learned card-set interaction:
  `--card-encoder deepset` projects private hand, board cards, public misc
  features, player, and belief separately, then adds a learned hand-board
  interaction. Hidden-64 Deepset beats zero in all three seeds and is near the
  train-constant gate; hidden-128 regresses, so blind width scaling is not the
  mechanism.
- A three-seed Deepset-64 prediction ensemble passes the hardened constant gate
  on the `384/128` river split: model MAE/RMSE `0.3405/0.4400`, best
  train-constant `0.4098/0.5292`, zero `0.4480/0.6383`.
- The saved-checkpoint ensemble path also passes: three persisted Deepset-64
  checkpoints evaluated together reproduce the same holdout MAE/RMSE, improve
  range-weighted value-sum residual over zero prediction
  (`0.3220` residual error vs `0.8756` zero-prediction residual), and run at
  `0.793 ms/state`, about `2,254x` faster than cached solver labels. Treat
  this as a promising learned-leaf direction, not a search-ready promotion.
- Dual-player range robustness now has a diagnostic script. On eight held-out
  river states, the saved Deepset-64 ensemble passed both a 50% mix toward
  uniform legal ranges and fully uniform legal ranges after re-solving labels:
  mix `0.5` MAE/RMSE `0.3087/0.3725`, mix `1.0` MAE/RMSE `0.3030/0.3669`.
  Both beat zero and train-constant baselines, so exact-blueprint-range
  overfitting is not the current learned-leaf blocker.
- CPU CFR now has a `showdown_leaf_fn` diagnostic hook. A passthrough hook
  exactly reproduces default solver regrets/strategies, and torch backends
  reject the hook explicitly. This is the first infrastructure piece for fixed
  resolver-state learned-leaf A/B; it does not yet make the learned leaf part
  of gameplay.
- The dual-CFV target builder now supports turn and river states. A mixed
  turn+river `64/32` Deepset-64 probe is promising but unstable: `1/3` seeds
  passed, all seeds beat zero, and the mean best-constant deltas were
  `-0.0026` MAE and `+0.0012` RMSE. Label generation is the immediate practical
  bottleneck for this route, averaging `3.35s` per train state and `3.84s` per
  holdout state with the current `torch-cuda` solver backend.
- Related-work anchor: Deep Sets supports permutation-aware learned set
  encoders for unordered card inputs, while Deep CFR/ReBeL/Supremus support
  learned value approximators paired with search rather than manual card
  abstraction. Sources: https://arxiv.org/abs/1703.06114,
  https://arxiv.org/abs/1811.00164, https://arxiv.org/abs/2007.13544, and
  https://arxiv.org/abs/2007.10442.

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

Current learned-leaf failure class: `range_belief`.

Flat public-belief hand-CFV models mostly learn broad value offsets and fail
trivial constant baselines on held-out river states. The saved Deepset-64
ensemble is the first reusable local result to beat those baselines, and it
survives small perturbed-range checks. The next research cycle should test the
learned card/range interaction inside fixed-state resolver diagnostics before
any Slumbot spend.

Tertiary learned-value bottleneck: `label_throughput`.

Street-specific dual-CFV learning is more principled than trying to call a
river model inside every turn CFR terminal, but mixed turn labels are slow at
the current solver speed. The next mixed-street step should target label
throughput or a turn-only split, not a hyperparameter sweep.

Label-throughput update:

- For turn labels, the CPU solver backend beat the current Python-driven
  `torch-cuda` backend on a four-case check: `1.10s` versus `1.45s` mean per
  case. This is consistent with the solver note that torch-CUDA is not a fused
  solver.
- `--label-jobs` now enables parallel CPU dual-CFV label generation. Naive
  multiprocessing was slower from BLAS oversubscription; the implementation
  caps threadpools per worker when `label_jobs > 1`.
- On an uncached mixed `8/4` CPU benchmark, controlled four-worker generation
  reduced wall time from `27.26s` to `18.38s` (`1.48x`). Use
  `--solver-backend cpu --label-jobs 4` for CPU label builds; keep
  `label_jobs=1` for default single-worker runs and do not combine label jobs
  with `torch-cuda`.
- A filtered turn-only `64/32` probe is stronger than mixed turn+river:
  Deepset-64 passed `2/3` seeds, all seeds beat zero, and mean best-constant
  deltas were `-0.0009` MAE and `-0.0006` RMSE. This supports street-specific
  dual-CFV networks, but it is still seed-sensitive on the small surface. The
  next gate should be a larger turn-only surface or a saved turn ensemble.
- The saved turn-only Deepset-64 ensemble passes the hardened checkpoint gate:
  model MAE/RMSE `0.2125/0.2835`, best train-constant `0.2376/0.3120`, zero
  `0.2505/0.3534`, and value-sum residual error `0.1855` versus zero
  residual `0.3457`. Inference is `0.781 ms/state`, about `9,180x` faster than
  the CPU label solver. The next gate is a larger independent turn surface.
- Turn range robustness also passes on eight held-out turn states. With a 50%
  mix toward uniform legal ranges, MAE/RMSE is `0.1917/0.2531`; with fully
  uniform legal ranges, MAE/RMSE is `0.1995/0.2597`. Both beat zero and
  train-constant baselines after re-solving perturbed labels.
- A larger independent turn-only `128/64` restored-history surface confirms the
  ensemble story. Single seeds all missed strict constant MAE, but the saved
  three-checkpoint Deepset-64 ensemble passed decisively: MAE/RMSE
  `0.2220/0.2886`, best constants `0.2500/0.3346`, zero `0.2634/0.3745`, and
  about `8,662x` solver-label speedup per state. The remaining blocker is now
  resolver A/B integration, not held-out turn value quality.
- The same larger independent turn ensemble also passed a full-uniform-range
  perturbation check on eight held-out turn states: MAE/RMSE `0.1984/0.2472`
  versus best train-constant `0.2439/0.3076` and zero `0.2483/0.3437`.
  Value-sum residual error was `0.2837`, below the zero residual target mean
  `0.3069`. This closes the immediate value-quality/range-robustness gate.
- Fixed resolver learned-leaf A/B is now implemented as
  `scripts/eval_learned_river_leaf_resolver_ab.py`. The diagnostic replaces
  only the measured turn decision subtree's normal equity terminal leaves with
  a saved river dual-CFV ensemble averaged over legal river cards. A two-case,
  five-iteration smoke is mechanically green but strategically mixed: only one
  case had an applicable learned leaf, and it flipped from all-in to call with
  leaf-only action L1 drift `1.3017`. This is evidence to investigate, not a
  promotion signal.
- A guarded applicable-case scan now uses `--target-leaf-applied` to avoid
  diluting the A/B with all-in-only cases where no learned leaf is used. The
  first `target_leaf_applied=2` scan blocks integration: applicable-case action
  agreement was `0/2`, mean leaf-only L1 drift was `0.7685`, and a root-turn
  case took `84.9s` for three iterations because it generated `47,664` leaf
  prediction states. The current learned river continuation path is a useful
  diagnostic, but it is not a gameplay-ready search component.
- Leaf inference now uses a vectorized all-hands dual-CFV predictor that
  computes public/belief embeddings once per state and broadcasts over hand
  chunks. It matches the older pairwise predictor in unit tests. On the same
  root-turn case, neural prediction time dropped from `37.6s` to `7.4s` and
  total learned solve time dropped from `84.9s` to `54.0s`. The remaining
  runtime is mostly CPU-side leaf-state construction and denominator
  accumulation, not the MLP forward pass.
- Denominator accumulation now uses BLAS matrix-vector products instead of
  allocating `valid_m * reach` arrays per leaf state. The same root-turn case
  dropped again from `54.0s` to `11.4s` for three iterations, while neural
  prediction stayed near `7.4s`. The diagnostic is now fast enough for small
  samples, but action agreement remains `0/2`, so speed does not change the
  integration blocker.
- Leaf-range attribution now exports actual river leaf states from turn solver
  terminal reaches with `scripts/build_learned_river_leaf_cases.py` and
  re-solves them with the existing dual-CFV evaluator. On four exported river
  leaf states, the river ensemble failed badly: model MAE/RMSE
  `0.3553/0.4300` versus zero `0.2560/0.3112`, and value-sum residual error
  `0.7938` against target residual scale `0.0467`. The action drift is
  therefore best explained as terminal-reach distribution shift, not a
  trustworthy correction of equity-only leaves.
- Tiny resolver-leaf training did not rescue this. The exporter now supports
  `--start-index` and `--max-rivers-per-terminal` for disjoint, less
  single-terminal-heavy splits, but a `16/8` diverse leaf split still failed
  with MAE `9.83` versus zero `0.258`. Treat this as a data-scale/protocol
  blocker, not a prompt to tune learning rates.
- A wider source-capped leaf split improves the pathology but still fails. The
  exporter now also supports `--max-leaf-states-per-source`; a `32/16` split
  capped at four leaf states per turn source and one river per terminal reduced
  holdout MAE to `0.913`, but zero and train-constant baselines remain near
  `0.31`. This confirms that learned river continuations should stay disabled
  until the leaf-value dataset is materially larger and stratified by source and
  terminal structure.

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
10. For learned river leaves, do not integrate the current river-continuation
    callback into gameplay. Next work should train/evaluate value networks on
    a materially larger resolver-generated leaf dataset with source/terminal
    stratification, or use the turn-CFV ensemble as a street-level continuation
    target, before another broad A/B.
