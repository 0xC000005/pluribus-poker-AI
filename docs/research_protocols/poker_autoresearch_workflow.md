# Poker Autoresearch Workflow

Status: approved and implemented for safe evaluation/logging automation, with
methodology-review and knob-governance gates.

## Objective

Build a research loop for this repository whose long-run target is a learned
poker engine that can train on a personal PC, use search efficiently during
play, and beat Slumbot and stronger public opponents without hand-crafted
poker-strategy rules.

The active direction is full-deck, heads-up Deep CFR-style self-play with GPU
traversal where possible, a learned blueprint policy, and online subgame search.
Search, abstraction, regret matching, and legality handling are allowed. Ad hoc
rules such as opponent-specific action hacks, street-specific human heuristics,
or manually curated poker features are not promoted unless they are temporary
diagnostics with a removal criterion.

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

Do not run training merely because more training looks productive. Each run
must answer a specific question about strategy quality, evaluation hardness,
compute throughput, representation quality, or search.

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
- `team_review.md`: routing for research lead, verifier, literature scout, and
  benchmark auditor roles.
- `decision.json`: one of `proceed`, `revise`, `abandon`, or
  `gather_more_evidence`.

The validator rejects pending `TODO`/`PENDING` review files and decisions
without sources. When the independent verifier is invoked, related-work review
and benchmark-hacking audit are mandatory. Use separate sub-agents for these
roles when available; the review files are the durable source of truth.

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

## Search-Consistency Targets

The approved next mechanism is bounded search-consistency training: use the
turn/river resolver to generate policy-head targets, then train the existing
policy head toward those search distributions under the 9-action legal mask.
This is not a hand-coded poker rule; the target comes from search. Keep the
search-target weight at `0.0` unless a completed methodology review and knob
entry justify enabling it.

Fixed public-state targets are useful for smoke tests and plumbing checks, but
they are not a training distribution. Any candidate meant for promotion should
use sampled train/held-out public states and must still pass the falsification
ladder before Slumbot confirmation.

Direct target fit is not promotion evidence. Evaluate held-out target fit with
`scripts/eval_search_targets.py`, then cross-check resolver drift and all-in
rate. A run that only learns the target file but worsens blueprint-vs-resolver
drift or collapses into all-in selection fails the search-quality criterion.
The solver, training masks, CUDA masks, Slumbot adapter, and target builder must
share the same 9-action legality contract; search must not clamp an illegal
fractional bucket into a different legal raise size.

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
This avoids judging a long run only by its final checkpoint when the learning
curve is non-monotonic.
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
