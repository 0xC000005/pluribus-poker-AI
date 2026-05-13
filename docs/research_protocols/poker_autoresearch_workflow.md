# Poker Autoresearch Workflow

Status: approved and implemented for safe evaluation/logging automation, with
methodology-review, mechanism-review, synthesis, calibration-phase, and
knob-governance gates.

## Objective

Build a research loop for an elegant, novel, SOTA-oriented heads-up no-limit
hold'em engine that can train on a personal PC, use learned search knowledge
during play, and beat Slumbot plus stronger public baselines without
hand-crafted poker-strategy rules.

The active direction is **neural regret-field resolving**: learn reusable
public-belief regret/policy initializers, use them to warm-start CFR+/resolving,
and let search remain the correction operator. Search, abstraction, regret
matching, legality handling, and learned public-belief representations are
allowed. Ad hoc opponent-specific action hacks, street-specific human heuristics,
hard learned value-cut replacement, and benchmark-tuned policy argmax patches are
not promoted.

## Workflow Contract

Each cycle follows HEAD:

1. **Hypothesize:** state one falsifiable claim, the expected metric movement,
   and the failure class being targeted.
2. **Execute:** run one focused experiment, audit, implementation, or literature
   check.
3. **Analyze:** compare against the incumbent on the same evaluation ladder and
   compute budget.
4. **Decide:** update state, append the research log, and either commit a
   focused change or record why no code change was justified.

Do not run training merely because more training looks productive. Each run must
answer a specific question about strategy quality, evaluation hardness, compute
throughput, representation quality, or search. The main current question is
whether amortized neural initialization can make a low-budget resolver behave
more like a high-budget teacher on root-disjoint public states.

## Commit Policy

Do not commit after every small file edit or individual gate. Batch changes by
research objective and commit only at natural boundaries: workflow feature
complete, experiment batch complete, methodology review complete, or
documentation synchronized. Commit messages should include the objective, files
changed, tests or gates run, key result, and review or related-work status.

Continuous mode must not commit autonomously. It may produce run artifacts,
metrics, queue entries, and log entries; a human or supervising agent should
review the batch before committing.

## Methodology Review Gate

Run a methodology review before changing the learning method, evaluation
protocol, checkpoint-promotion rule, or any persistent research knob. The gate
creates a review bundle under `autoresearch-session/poker_reviews/` with:

- `review.md`: independent-verifier findings based on local files/artifacts.
- `related_work.md`: at least one primary source URL and a transfer analysis.
- `benchmark_audit.md`: objective-drift and benchmark-hacking checks.
- `mechanism_review.md`: learned object, search boundary, train distribution,
  eval distribution, falsifier, pass action, fail action, and related-work
  delta.
- `team_review.md`: routing for research lead, verifier, literature scout, and
  benchmark auditor roles.
- `decision.json`: one of `proceed`, `revise`, `abandon`, or
  `gather_more_evidence`.

The validator rejects pending `TODO`/`PENDING` review files and decisions
without sources. When the independent verifier is invoked, related-work review,
benchmark-hacking audit, and mechanism review are mandatory. Use separate
sub-agents for these roles when available; the review files are the durable
source of truth.

Review bundles are ignored because they may include local artifact paths and
large evidence notes. For any review used to justify a methodology decision,
write a tracked digest manifest:

```bash
python scripts/poker_autoresearch.py write-review-manifest \
  --review-dir autoresearch-session/poker_reviews/<review_id>
```

Tracked manifests live under
`docs/research_protocols/poker_review_manifests/`.

## Failure Synthesis Gate

Every five non-review experiments, stop expanding the experiment surface and
write a causal synthesis. The synthesis must name the current causal model,
retired hypotheses, live hypotheses, and one next falsifier. This prevents
long unattended runs from accumulating diagnostics without converting them into
a sharper research program.

```bash
python scripts/poker_autoresearch.py synthesis-status
python scripts/poker_autoresearch.py enqueue-synthesis \
  --subject "callback-state DCVN failures"
```

The synthesis validator rejects pending fields. A failed synthesis gate is a
methodology failure, not a strategy-quality result.

## Objective-Drift Guard

Autoresearch experiments may change candidate/training code and configuration,
but protected evaluation surfaces are immutable by default. Protected surfaces
include local evaluation scripts, Slumbot adapters, solver benchmarks,
promotion logic, parsers, seed lists, and parity tests. Changing those files
requires a completed methodology review and benchmark audit.

Run the audit when keeping a candidate or before any methodology commit:

```bash
python scripts/poker_objective_audit.py --base-ref HEAD
python scripts/poker_autoresearch.py objective-audit \
  --changed-path scripts/play_slumbot.py \
  --review-dir autoresearch-session/poker_reviews/<review_id>
```

The long-term objective remains the controlling policy: novel, compute-efficient
Texas hold'em methods that transfer to Slumbot and stronger bots on personal-PC
hardware. Visible smoke metrics are diagnostics, not promotion targets.

## Falsification Ladder

Before spending Slumbot confidence hands or treating a candidate as promotable,
queue a falsification ladder. This is the local POPPER-inspired counter-test
stage: it tries to falsify the candidate's mechanism with distinct blockers
before live evaluation.

```bash
python scripts/poker_autoresearch.py enqueue-falsification \
  --candidate models/candidate.pt \
  --mechanism "search-distilled policy targets reduce Slumbot transfer loss" \
  --n-games 500 \
  --max-resolver-cases 3 \
  --changed-path poker_ai/deep_cfr/networks.py
```

The ladder currently runs:

- objective-drift audit;
- duplicate-swapped candidate-vs-incumbent comparison requiring positive lower
  95% confidence bound;
- fixed-state resolver diagnostics.

Passing the ladder is still not promotion. It means the candidate survived the
cheap counter-tests and may justify sparse live Slumbot confirmation. A
candidate with `promotable=false` or non-positive comparison lower bound should
fail this ladder even if the evaluator itself ran successfully.

## Research Knob Governance

Persistent knobs are allowed only when they test one named mechanism. Each knob
must record a single default, failure class, mechanism, rationale, and removal
criterion in `poker_knobs.tsv`. Broad sweeps and list-shaped defaults are
rejected. Keep at most five active knobs unless the goal file is deliberately
changed after methodology review.

During `callback_state_calibration_debug`, new GPU training runs, live Slumbot
smokes, and model-size/search knobs are blocked. The allowed next actions are
calibration audit, methodology/mechanism review, failure synthesis, and
objective audit:

```bash
python scripts/poker_autoresearch.py set-phase \
  --phase callback_state_calibration_debug \
  --reason "callback-state DCVN scale-up failed supervised and leaf gates"
python scripts/poker_autoresearch.py enqueue-calibration-audit \
  --train-dual-cache <train_callback_cache.npz> \
  --holdout-dual-cache <holdout_callback_cache.npz> \
  --supervised-metrics <train_metrics.json> \
  --leaf-ab <leaf_ab_metrics.json>
```

Return to `open_research` only after the calibration audit explains the failure
well enough to select one falsifiable next test.

The current mainline phase is `neural_regret_field_resolving`. The prior
callback-state DCVN and successor-cut work remains useful as negative evidence,
but it is no longer the active method unless a completed methodology review
reopens it.

```bash
python scripts/poker_autoresearch.py set-phase \
  --phase neural_regret_field_resolving \
  --reason "pivot to learned CFR/resolving warm starts after hard value-cut failures"
```

Approved next actions in this phase:

- implement solver warm-start interfaces that preserve zero-initializer parity;
- export root-disjoint teacher targets from higher-budget resolving;
- train a public-belief regret/policy initializer, not a hard value oracle;
- evaluate low-budget vanilla CFR+ versus neural-warm-start CFR+ against the
  same higher-budget teacher.

Blocked anti-patterns:

- scaling hard learned leaf/successor value replacement as the mainline;
- sweeping final-distribution policy mixing weights;
- adding model-size knobs before a root-disjoint warm-start resolver gate;
- queueing generic GPU Deep CFR training or live Slumbot smokes before the
  local warm-start resolver gate passes.

## Research State

The approved implementation should create local resumability state under
`autoresearch-session/`:

- `poker_goal.json`: durable objective, constraints, hard stop conditions.
- `poker_state.json`: current phase, incumbent checkpoint, last verified
  metrics, active hypothesis, and next queue.
- `poker_knobs.tsv`: every new knob with status, default, failure class,
  mechanism, rationale, and removal criterion.
- `poker_reviews/`: methodology-review bundles with verifier notes, related
  work, benchmark audits, team routing, and decisions.
- `poker_runs/`: ignored run artifacts, configs, raw logs, metrics JSON, and
  Slumbot transcripts.

Tracked documentation should live under `docs/research_protocols/` and the repo
root research log if one is added. Model checkpoints and large run artifacts
must stay out of git.

## Evaluation Hardness Ladder

Results are not promotable unless they pass the appropriate ladder level:

- **Tier 0: integrity.** Unit/parity tests for legal masks, feature encoders,
  Slumbot action mapping, checkpoint loading, and solver compatibility. The
  feature-encoding parity script runs under `NUMBA_ENABLE_CUDASIM=1` so the
  input-contract check does not require a visible CUDA device. Legal-mask tests
  include minimum-raise parity so under-minimum fractional raise buckets are not
  exposed to Slumbot play.
- **Tier 1: local smoke.** Fast fixed-seed evaluation against random and simple
  baseline policies. The implemented `eval-local` gate evaluates
  `models/slumbot_2p_iter1000.pt` for a small fixed-seed sample against random
  opponents. This catches broken loading/evaluation and gives a noisy local
  proxy, but it does not prove strength.
- **Tier 2: incumbent comparison.** Fixed-seed local comparison against the
  current best checkpoint using the same seeds and action settings. The
  implemented `eval-incumbent-self-compare` gate verifies that comparison
  metrics and promotion blockers are emitted; future candidate checkpoints
  should use the same protocol before any live Slumbot confidence run.
  The stronger `eval-head-to-head-self-compare` gate validates duplicate-swapped
  local model-vs-model evaluation before candidate-vs-incumbent use.
- **Tier 3: Slumbot smoke.** Short live Slumbot run for integration,
  action-mapping, latency, and solver stability.
- **Tier 4: Slumbot confidence.** Longer Slumbot run with chips/hand, mbb/hand,
  confidence interval, all-in rate, fold/call/raise distribution, mapping error
  rate, and solver latency.
- **Tier 5: tournament track.** Multi-player or tournament-like evaluation only
  after the heads-up Slumbot track has a hard baseline and reproducible metric.

The primary metric for Slumbot work is lower 95% confidence bound of chips/hand
or mbb/hand versus the incumbent. Training changes must also report iters/hour,
samples/sec, and train seconds/iteration.

GPU training scripts call `scripts/cuda_env.py` before importing Numba so the
process can discover the pip NVVM package and force the local GPU compute
capability. This avoids shell-level `LD_LIBRARY_PATH` overrides, which can hide
the GPU from this environment.

## Failure Classes

Every failed or inconclusive cycle assigns one primary class:

- `eval_invalid`: metric is too noisy, leaky, slow, or non-mechanical.
- `rules_parity`: CPU, fast, CUDA, solver, or Slumbot state logic disagrees.
- `action_mapping`: continuous or Slumbot actions are mapped poorly.
- `train_fit`: the value or policy network fails to fit training targets.
- `strategy_quality`: local training improves but play quality does not.
- `history_representation`: the network cannot use action history effectively.
- `range_belief`: range tracker or public-belief state is wrong.
- `search_quality`: subgame search is unstable, too shallow, or too slow.
- `abstraction_limit`: card, action, stack, or raise abstraction is too coarse.
- `compute_bottleneck`: throughput prevents the needed training scale.
- `distribution_shift`: self-play policy does not transfer to Slumbot behavior.

Only diagnosed failure classes justify changing architecture, abstraction,
training objective, solver behavior, or evaluation protocol.

## Neural Architecture Policy

Modern neural networks are allowed and expected. ReBeL is a 2020 result, and
DeepStack is the 2017 poker result; both predate many now-standard architecture
and training improvements. The workflow should consider stronger set encoders,
attention/transformer blocks, learned action-sequence encoders, residual trunks,
uncertainty heads, and mixed precision when they serve the learned-search
mechanism.

Architecture changes are not progress by themselves. A larger or newer network
must name the learned object, the search boundary, and the local falsifier. For
the current phase, that means improving a public-belief regret/policy
initializer and passing the root-disjoint warm-start resolver gate. Offline
target fit, local random wins, or a better-looking Slumbot smoke cannot justify
mainline promotion without the search-behavior gate.

## Literature And Review Gate

Online research is required before adopting a new RL/search method, changing the
core Deep CFR family, adding public-belief search, or promoting an architecture
as a mainline direction. The literature note must state:

- diagnostic question;
- papers or primary sources checked;
- what transfers to this codebase;
- what does not transfer because of compute, action abstraction, or Slumbot
  constraints;
- the smallest experiment that tests the idea locally.

Method changes should be classified as `canonical`, `supported_adjacent`, or
`speculative_local_heuristic`.

Use `enqueue-review` for any change that needs independent verification or
related work. This keeps review artifacts in the workflow queue instead of
burying them in chat.

## Neural Regret-Field Targets

The approved next mechanism is not bounded policy imitation. It is a learned
regret/policy field that initializes CFR+/resolving at public-belief states.
The network may use stronger modern architecture, but its output must enter the
search loop as an initializer or calibration signal that search can correct.

The smallest useful target artifact should contain root-disjoint public states,
legal masks, public cards, private-hand set encodings, public action sequences,
reach/belief summaries, low-budget vanilla solver output, and higher-budget
teacher output. The first pass can train policy logits only if the evaluation
turns them into CFR/regret initializers rather than final played actions.

Required A/B:

- vanilla low-budget CFR+ versus neural-warm-start low-budget CFR+;
- both compared against the same higher-budget teacher;
- metrics: root action L1/KL, top-action agreement, all-in probability/top rate,
  illegal-action count, latency, and root-disjoint split identity;
- fail action: retire the learned object or change the target, not sweep
  architecture size on the same holdout.

Queue the implemented gate with:

```bash
python scripts/poker_autoresearch.py enqueue-warm-start-resolver \
  --checkpoint autoresearch-session/search_consistency_restored200_100x2k_20260513/search_consistency_allroots_allhand_policy_train128_iter5_seed20260643.pt \
  --cases autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --cfv-cache autoresearch-session/search_consistency_restored200_100x2k_20260513/public_belief_successor_pool256_seed20260627_cache.npz \
  --start-index 128 \
  --limit 64 \
  --min-evaluated 64
```

The first run of this gate failed for the old joint-PBS policy checkpoint even
though root-disjointness and legality passed. That result is evidence against
reusing a policy-imitation head as the regret-field initializer; the next
learned object should target regret/policy deltas that directly improve
low-budget resolving toward a higher-budget teacher.

Before training that learned object, run the oracle sanity check:

```bash
python scripts/eval_regret_oracle_warm_start.py \
  --cases autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --cfv-cache autoresearch-session/search_consistency_restored200_100x2k_20260513/public_belief_successor_pool256_seed20260627_cache.npz \
  --start-index 128 \
  --limit 64 \
  --low-iterations 5 \
  --reference-iterations 25 \
  --min-evaluated 64
```

The first 64-case oracle passed when it seeded both the teacher's selected-node
`regret_sum` and `strategy_sum`; regret-only seeding was not calibrated enough
on the smoke slice. This means the learned label should be the solver-state
field, not only a final policy distribution.

Legacy search-consistency, policy-head calibration, and CFV/DCVN scripts remain
diagnostic tools. Direct target fit is not promotion evidence. A run that only
learns a target file but worsens resolver drift or collapses into all-in
selection fails the search-quality criterion. The solver, training masks, CUDA
masks, Slumbot adapter, and target builder must share the same 9-action legality
contract; search must not clamp an illegal fractional bucket into a different
legal raise size.

Use `scripts/eval_public_belief_probe.py` before wiring range inputs into the
main trainer. The probe compares feature-only target prediction to raw
hero/villain range-vector prediction on held-out gameplay-distributed targets.
It is diagnostic only: passing means the belief representation is worth a
larger value-target experiment, not that a checkpoint is promotable.

Use `scripts/eval_public_belief_value_probe.py` for the next stricter
representation check. It computes searched scalar hero-EV targets from solved
turn/river public states, then compares feature-only value regression against
feature-plus-public-belief regression. Use `--train-value-cache` and
`--holdout-value-cache` for repeated probe seeds so the expensive CPU solver
labels are reused. This probe is also diagnostic only; promotion still requires
the falsification ladder and Slumbot confirmation.

Use `scripts/eval_public_belief_cfv_probe.py` for the literature-shaped value
probe. It zeroes private-card feature slots, computes a searched
counterfactual value vector over all 1326 hero private hands, and trains a
masked vector regressor. Use `--train-cfv-cache` and `--holdout-cfv-cache` for
repeated seeds. This is the preferred value-probe shape before any mainline
public-belief value-head work.

Use `scripts/eval_public_belief_hand_cfv_probe.py` when testing representation
learning over the same CFV labels. It trains on valid `(public state, hero
hand)` pairs with shared public-state and hand-card encoders, and optionally a
learned public-belief encoder. This is still diagnostic only; require stability
on a balanced target surface before promoting the architecture.

Use `scripts/train_public_belief_hand_cfv.py` after the diagnostic passes. It
trains the belief-conditioned shared hand-CFV model, saves normalization
metadata with the checkpoint, and exposes a load/predict path for later
depth-limited search integration. Treat this as a reusable learned component,
not as a playing-policy promotion by itself.

Use `scripts/benchmark_public_belief_hand_cfv.py` before wiring the learned
component into search. It measures batched checkpoint inference on a CFV cache
and compares it to the solver latencies stored with the labels. Learned leaf
evaluation must be much cheaper than solving the same public states, otherwise
it is not a useful PC-limited search primitive.

Use `scripts/eval_hand_cfv_range_robustness.py` before treating a saved CFV
checkpoint as a search leaf. It perturbs river public-belief ranges, re-solves
those perturbed states, and compares model CFVs against the new labels. Passing
the original holdout alone is not enough, because search iterations query value
functions at ranges different from the blueprint tracker distribution.

Before a CFV checkpoint is wired into a solver leaf, require dual-player value
coverage. A hero-only CFV model is a diagnostic component; a depth-limited
re-solver needs both hero and villain counterfactual value vectors at the leaf.
Use the tested `compute_hero_cfv_vector` and `compute_villain_cfv_vector`
helpers as the target source for the dual-player probe.

Use `scripts/eval_public_belief_dual_hand_cfv_probe.py` for that probe. It
trains on both players' hand-CFV labels with a player indicator and compares
public+hand+player against public+hand+player+belief. Do not integrate a
learned CFV checkpoint into turn search unless this dual-player probe is stable
across seeds on a balanced river surface. Prefer `--head-mode separate` when
testing leaf-value architectures; the shared scalar head is retained as a
baseline because it underfits the two payoff frames. For the current
dual-player river probe, also use `--belief-bottleneck-dim 32` as the default
candidate architecture before considering solver integration; it tests a
learned compression of the public-belief range vector instead of feeding the
raw range vector directly into the value body. The probe must also beat a
zero-CFV baseline and train-constant baselines on MAE/RMSE; improving over a
weak feature-only model is not enough.

Only after that harder larger-surface probe passes should a reusable checkpoint
from `scripts/train_public_belief_dual_hand_cfv.py` be considered for learned
leaf diagnostics. The checkpoint is still not a gameplay model. Its next
required gate is a fixed resolver-state learned-leaf A/B that checks
both-player CFV error, zero-sum residual, root-action drift versus the full
solver, and latency.

Use `scripts/stratify_search_targets.py` to build that balanced target surface
from generated gameplay-distributed artifacts. With CFV caches supplied, it
splits by target street and searched-CFV mean bins, writes matching target/case
files, and preserves split CFV caches. This prevents tiny independent
train/holdout files from turning ordinary value-distribution noise into a false
architecture pass or fail. Use `--streets 3` to isolate river CFV labels before
testing a learned value function intended for turn lookahead leaf evaluation.

For blueprint rollout targets, set `--blueprint-target-streets` deliberately.
The default `2,3` collects desired target streets round-robin so target files
are not accidentally all turn. Use `3` for river-only smoke tests and inspect
the street histogram in the metadata before using a target artifact.

## Implemented Automation

The executable runner is `scripts/poker_autoresearch.py`. It automates
evaluation and logging only; it does not autonomously rewrite learning code.

Use these commands from the repository root:

```bash
python scripts/poker_autoresearch.py init
python scripts/poker_autoresearch.py status
python scripts/poker_autoresearch.py set-incumbent \
  --checkpoint models/slumbot_2p_iter1000.pt \
  --reason "best available local heads-up Slumbot-track checkpoint"
python scripts/poker_autoresearch.py gate tier0
python scripts/poker_autoresearch.py gate eval-local
python scripts/poker_autoresearch.py gate eval-local-confidence
python scripts/poker_autoresearch.py gate eval-local-multiseed
python scripts/poker_autoresearch.py gate eval-incumbent-self-compare
python scripts/poker_autoresearch.py gate eval-head-to-head-self-compare
python scripts/poker_autoresearch.py gate slumbot-smoke
python scripts/poker_autoresearch.py new-cycle \
  --hypothesis "Tier 0 should pass before unattended work." \
  --cycle-type experiment \
  --failure-class eval_invalid \
  --gate tier0
python scripts/poker_autoresearch.py enqueue \
  --hypothesis "Tier 0 should pass before unattended work." \
  --cycle-type experiment \
  --failure-class eval_invalid \
  --gate tier0
python scripts/poker_autoresearch.py enqueue-review \
  --subject "New search objective" \
  --trigger method_change \
  --claim "The proposed objective should improve Slumbot transfer."
python scripts/poker_methodology_review.py \
  --review-dir autoresearch-session/poker_reviews/<review_id> \
  --require-complete
python scripts/poker_objective_audit.py --base-ref HEAD
python scripts/poker_autoresearch.py objective-audit \
  --changed-path scripts/play_slumbot.py \
  --review-dir autoresearch-session/poker_reviews/<review_id>
python scripts/poker_autoresearch.py enqueue-falsification \
  --candidate models/candidate.pt \
  --mechanism "search-distilled policy targets reduce Slumbot transfer loss" \
  --max-resolver-cases 3
python scripts/poker_autoresearch.py add-knob \
  --name search_target_weight \
  --default 0.05 \
  --failure-class search_quality \
  --mechanism "Test whether resolver-distilled policy targets reduce blueprint-vs-resolver drift and improve transfer evidence." \
  --rationale "Single auxiliary-loss weight isolates the reviewed search-consistency mechanism." \
  --removal-criterion "Retire if held-out resolver drift or falsification-ladder evidence fails to improve against the no-target control."
python scripts/build_search_targets.py \
  --output autoresearch-session/search_targets/sampled_turn_river_train.npz \
  --sampled-cases 64 \
  --seed 20260512 \
  --solver-iterations 25 \
  --solver-backend auto
python scripts/poker_autoresearch.py enqueue-compare \
  --candidate models/candidate.pt \
  --n-games 500 \
  --seeds 20260511,20260512,20260513 \
  --head-to-head \
  --strategy-source regret
python scripts/poker_autoresearch.py enqueue-slumbot \
  --model models/candidate.pt \
  --hands 10 \
  --greedy \
  --no-allin \
  --no-solver
python scripts/poker_autoresearch.py enqueue-train \
  --n-iterations 50 \
  --n-traversals 4000 \
  --n-training-steps 1500 \
  --search-targets autoresearch-session/search_targets/sampled_turn_river_train.npz \
  --search-target-weight 0.05 \
  --prefix candidate_gpu \
  --save-every 25 \
  --auto-compare \
  --compare-strategy-source regret
python scripts/eval_search_targets.py \
  --checkpoint models/candidate.pt \
  --targets autoresearch-session/search_targets/sampled_turn_river_holdout.npz \
  --strategy-source policy-head
python scripts/poker_autoresearch.py close-cycle \
  --run-id <run_id> \
  --outcome passed \
  --failure-class none \
  --metrics-path <run_dir>/metrics.json \
  --summary "Tier 0 integrity gate passed."
python scripts/poker_autoresearch.py continuous --max-cycles 1 --sleep-seconds 0
python scripts/poker_autoresearch.py continuous --max-idle-checks 1 --sleep-seconds 0
```

Local generated state is under `autoresearch-session/`:

- `poker_goal.json`: objective, constraints, gate commands, hard stops.
- `poker_state.json`: incumbent, active cycle, queue, history, last metrics.
- `poker_knobs.tsv`: knob ledger.
- `poker_reviews/`: methodology-review, related-work, benchmark-audit, and
  team-routing artifacts.
- `poker_runs/`: ignored cycle artifacts and `metrics.json` files.

The runner appends cycle summaries to `RESEARCH_LOG.md`. New entries include a
metrics file path and a short key-metrics JSON summary instead of embedding
full raw command logs.
When a gate command prints JSON, the runner stores it under
`commands[].stdout_json` in the cycle `metrics.json`.
`eval-local-confidence` uses 1,000 fixed-seed games to reduce noise relative to
the fast 32-game smoke gate while still staying PC-friendly.
`eval-local-multiseed` repeats that local benchmark over three seeds.
`eval-incumbent-self-compare` compares the incumbent checkpoint against itself
over matching seeds and emits candidate/baseline delta metrics plus promotion
blockers. Local random-opponent comparison remains a mechanical health check;
it is not sufficient to promote a new strategy.
Use `enqueue-compare` for a real candidate checkpoint. It materializes a
one-off comparison gate in `poker_goal.json`, queues it, and defaults the
baseline to the recorded incumbent checkpoint.
Pass `--head-to-head` to queue duplicate-swapped candidate-vs-incumbent play;
omit it only for the cheaper candidate-vs-random delta diagnostic.
Pass `--strategy-source policy-head` only when every evaluated checkpoint has
trained `policy_head` weights. Legacy checkpoints are rejected for policy-head
evaluation so the workflow cannot silently benchmark random initialized heads.
Use `enqueue-train --save-every N --auto-compare` for longer GPU runs. The
training gate writes periodic checkpoints, emits them in JSON, then continuous
mode queues head-to-head incumbent comparisons for every emitted checkpoint.
These comparisons require `--require-positive-lower95`, so a finite negative
lower confidence bound fails the gate instead of being logged as a pass. This
avoids judging a long run only by its final checkpoint when the learning curve
is non-monotonic.
Use `scripts/build_search_targets.py` to create resolver-target datasets and
pass them to `enqueue-train` or `scripts/run_gpu_deep_cfr.py` with
`--search-targets`. Prefer `--sampled-cases` plus a recorded seed for
experiments. The script also writes `<output>.cases.json`; use a separate seed
for held-out resolver benchmarks. The training path records
`search_target_weight` and target count in the emitted metrics JSON.
Use `--compare-strategy-source policy-head` for auto-queued comparisons only
after the incumbent itself is a policy-head-capable checkpoint.
Use `enqueue-slumbot` only for sparse live checks after local comparison says a
candidate is interesting. It creates a one-off live smoke gate and records
Slumbot chips/hand, CI, elapsed seconds, seconds/hand, action mix, increment
mix, policy/solver/fallback decision counts, parse/API errors, and
action-mapping drift. Solver-enabled runs also record solver latency, cache
hits, active hand count after learned range pruning, full hand count, and prune
ratio when the wrapper emits those diagnostics.
`slumbot-smoke` runs five live Slumbot hands with greedy, no-all-in, no-solver
diagnostic settings and parses the final chips/hand summary into JSON.
The Slumbot wrapper also emits `elapsed_seconds` and `seconds_per_hand` so
solver latency is visible in the research log.

## Hard Stop Conditions

Stop and ask for review if:

- the metric is not mechanical or cannot be parsed;
- Tier 0 integrity fails;
- Slumbot credentials, network access, or API limits block evaluation;
- a proposed change modifies multiple research axes in one cycle;
- a method, evaluation, promotion, or knob change has no completed methodology
  review;
- a knob addition looks like a broad sweep rather than a mechanism test;
- a run would overwrite the incumbent checkpoint without an explicit backup;
- the workflow wants to add hand-crafted opponent or street rules;
- compute cost or runtime exceeds the configured local budget.

## Continuous Mode

Continuous mode consumes `hypothesis_queue` entries from
`autoresearch-session/poker_state.json`. In unbounded mode it idles when the
queue is empty, rechecking for new work until a stop condition appears. Bounded
dry runs can use `--max-cycles` or `--max-idle-checks`.

It stops when:

- `autoresearch-session/STOP` exists;
- the queue is empty in a bounded `--max-cycles` run;
- `--max-idle-checks` is reached in a dry run;
- a gate fails;
- readiness checks fail;
- `--max-cycles` is reached.

For a safe unattended launch after readiness is verified:

```bash
python scripts/poker_autoresearch.py enqueue \
  --hypothesis "Run Tier 0 integrity before the next research action." \
  --cycle-type experiment \
  --failure-class eval_invalid \
  --gate tier0
python scripts/poker_autoresearch.py continuous --sleep-seconds 30
```

To stop it, create:

```bash
touch autoresearch-session/STOP
```

This workflow approval does not approve any specific model architecture change
or long Slumbot training run. Those require their own HEAD cycle, explicit gate,
and local budget.
