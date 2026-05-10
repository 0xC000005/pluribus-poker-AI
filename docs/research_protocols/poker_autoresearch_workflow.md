# Poker Autoresearch Workflow

Status: approved and implemented for safe evaluation/logging automation.

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

## Research State

The approved implementation should create local resumability state under
`autoresearch-session/`:

- `poker_goal.json`: durable objective, constraints, hard stop conditions.
- `poker_state.json`: current phase, incumbent checkpoint, last verified
  metrics, active hypothesis, and next queue.
- `poker_knobs.tsv`: every new knob with default, failure class, rationale, and
  removal criterion.
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
  input-contract check does not require a visible CUDA device.
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

## Literature Gate

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
python scripts/poker_autoresearch.py enqueue-compare \
  --candidate models/candidate.pt \
  --n-games 500 \
  --seeds 20260511,20260512,20260513 \
  --head-to-head
python scripts/poker_autoresearch.py enqueue-slumbot \
  --model models/candidate.pt \
  --hands 10 \
  --greedy \
  --no-allin \
  --no-solver
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
- `poker_runs/`: ignored cycle artifacts and `metrics.json` files.

The runner appends cycle summaries to `RESEARCH_LOG.md`.
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
Use `enqueue-slumbot` only for sparse live checks after local comparison says a
candidate is interesting. It creates a one-off live smoke gate and records
Slumbot chips/hand, CI, elapsed seconds, seconds/hand, action mix, increment
mix, policy/solver/fallback decision counts, parse/API errors, and
action-mapping drift.
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
