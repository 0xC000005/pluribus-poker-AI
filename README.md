# Poker AI

This repository is currently focused on building a full-deck poker AI that
learns from local self-play and is evaluated against heads-up no-limit Texas
hold'em baselines such as Slumbot. The active engine is Deep CFR with a 9-action
abstraction, CUDA traversal, and turn/river range-vs-range CFR+ solving. Older
short-deck tabular MCCFR and clustering code remains in the tree for reference,
but it is not the recommended path for new work.

## Current Engine

- `poker_ai/deep_cfr/` contains the active Deep CFR stack: neural networks,
  reservoir buffers, CPU traversal, fast numpy traversal, and CUDA traversal.
- `poker_ai/deep_cfr/cuda/` contains the Numba CUDA trainer and kernels used to
  move self-play traversal, legal masks, regret matching, action sampling, and
  propagation onto the GPU.
- `poker_ai/games/full_deck/` is the canonical full 52-card game state and
  feature encoder used for correctness and parity checks.
- `scripts/train_slumbot_2p.py` trains a local 2-player, 200BB self-play model;
  despite the name, it must not train from Slumbot hands or traces.
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

New Deep CFR checkpoints consume the 12 public action-history slots by default
and save `uses_betting_history=true`. Older checkpoints without that metadata
load with the legacy masked-history path for compatibility. Keep feature and
legal-mask behavior aligned across `full_deck/state.py`,
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
runs so the workflow evaluates intermediate checkpoints against the incumbent.
Auto-comparisons require a positive lower 95% bound; finite but negative deltas
are logged as rejected evidence:

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

The current objective is publication-grade local self-play policy improvement:
self-play generates experience, neural policy/value models learn from it, an
optional general search/resolving operator may improve decisions or targets, and
candidates are promoted through a checkpoint league before any external
confidence run. This is AlphaZero-like in spirit while respecting
imperfect-information constraints.

The executable AlphaZero-style smoke is the repeatable neural policy-iteration
loop: `python scripts/run_neural_policy_iteration_loop.py --n-generations 2
--output-dir autoresearch-session/neural_policy_iteration/az_loop --prefix
az_loop --self-play-hands 512 --parallel-self-play-hands 128
--self-play-state-backend fast --max-improvement-targets 256 --train-steps 32
--hidden-dim 128 --batch-size 512 --teacher-mode public_world_rollout
--rollout-worlds 8 --h2h-games 1000 --device auto --output-json
autoresearch-session/neural_policy_iteration/az_loop/summary.json`. Treat this
as an internal self-play loop and league gate, not Slumbot evidence.
When judging NPI against plug-in RL controls, add
`--mixed-control-checkpoint tianshou-rainbow:<rainbow.pt>` or
`--mixed-control-checkpoint tianshou-ppo:<ppo.pt>`. The mixed evaluator loads
Tianshou checkpoints as maintained-library policy adapters and applies the same
native duplicate-swapped lower95 gate. Add `--require-control-gate` for
unattended goal runs so the loop fails if any generation lacks a fixed-control
result. Do not reimplement Rainbow/PPO locally.

External RL libraries are controls unless wrapped by the native league. For
example, `uv run python scripts/run_rlcard_nfsp_pilot.py --train-episodes 100
--eval-games 100 --device auto` checks plug-and-play NFSP plumbing, but RLCard
no-limit Hold'em uses 5 actions and is not Slumbot-parity evidence. The native
9-action NFSP control already uses Double-DQN targets and can opt into a
dueling Q head with `--q-network-arch dueling`; promote neither path without
positive duplicate-swapped native-league confidence. A Tianshou Rainbow control
is available with `uv run --with tianshou python
scripts/run_tianshou_rainbow_native_control.py --device auto`; it is a
single-agent value-learning baseline against local random opponents, not an
equilibrium method. Add `--num-envs 8 --vector-env-backend subproc` when using
it as a serious throughput control; this uses Tianshou vector environments
instead of reimplementing Rainbow locally. The vectorized 10k-step smoke was
too small, but two 65k-step repeats beat the weak native NFSP h512 10k
checkpoint in 1k-game local H2H. Treat this as a serious pure-RL baseline, not
Slumbot/SOTA evidence. Use `--baseline-kind npi` with
`--checkpoint-in <rainbow.pt> --baseline-checkpoint <npi.pt>` to compare
Rainbow directly against neural policy-iteration checkpoints; the current 65k
Rainbow repeats beat the two-generation NPI gen1 checkpoint, so NPI candidates
must clear this stronger baseline before Slumbot evaluation. A 270k random-
opponent scale-up did not improve confidence against native NFSP, so prefer
self-play or league opponent sampling over simply adding more random-opponent
Rainbow steps. The control also supports fixed learned-opponent training with
`--opponent-kind native-nfsp --opponent-checkpoint <pt>`. Learned-opponent
subproc now uses Tianshou `context="spawn"` and records setup/train/total
timing, but current h256 A/B evidence is slower than dummy, so keep it
diagnostic-only unless a fresh same-scale A/B proves a real speedup. The first
18k NFSP-opponent diagnostic was slower and weaker than the random-opponent
baseline, so treat it as infrastructure for future league/self-play work.
The same control supports previous-checkpoint Rainbow opponents with
`--opponent-kind rainbow --opponent-checkpoint <rainbow.pt>` and Rainbow-vs-
Rainbow league checks with `--baseline-kind rainbow`. The first 18k child beat
several Rainbow checkpoints in local H2H but not native NFSP with confidence,
so it was not promotion evidence. It also supports repeated
`--opponent-checkpoint` flags to sample from a mixed Rainbow checkpoint league.
The first two-parent 18k mixed-league child was only mixed/neutral versus
Rainbow parents and native NFSP while still beating NPI gen1, so the next useful
pure-RL step is cheaper batched learned-opponent inference or a maintained
multi-agent self-play backend before larger scaling.
There is also a Tianshou PPO control at
`scripts/run_tianshou_ppo_native_control.py`; use it as a stochastic
policy-gradient falsifier, not as the mainline unless it clears native league
gates. Its H2H evaluator supports `--stochastic-eval`; the first shared-policy
MARL PPO pilot through PettingZoo trained at `842.96` env steps/sec but failed
native NFSP under both deterministic and stochastic evaluation, so PPO remains
a control rather than the current clean RL incumbent.
For true plug-and-play multi-agent RL, use
`poker_ai.research.native_pettingzoo.NativeNoLimitHoldemAECEnv`: it exposes the
native 126-feature, 9-action, legal-mask game as a PettingZoo AEC environment
and wraps with Tianshou's `PettingZooEnv`. This is the preferred bridge for
maintained MARL libraries before adding any local PPO/Rainbow/NFSP algorithm
code. The first maintained-library MARL Rainbow self-play pilot is available at
`scripts/run_tianshou_marl_rainbow_native_control.py`; its 18k-step checkpoint
was evaluable but not stronger than the current Rainbow/NFSP baselines, so
treat it as a workflow bridge rather than promotion evidence. Add
`--shared-policy` to train one shared actor across both seats; this is the
cleaner AlphaZero-style option, but the first 18k run was still only neutral
versus Rainbow/NFSP controls and should not be promoted. Add
`--vector-env-backend subproc` for the faster shared-policy path: the first
18k subproc run reached `871.0` env steps/sec and passed the current local
smoke league across Rainbow/NFSP/NPI controls. Treat this as the strongest
local plug-in RL evidence so far, still not Slumbot or SOTA evidence. The
script also supports `--checkpoint-in <pt>` for N -> N+1 continuation. The
first 18k gen2 continuation from the passed gen1 checkpoint failed the local
league gate, so do not assume continued Rainbow self-play is monotonic; every
new generation must beat its parent and controls before promotion. A fresh
66k-step shared-policy MARL Rainbow run beat native NFSP but failed to beat the
18k shared-MARL incumbent, so parent H2H remains mandatory before broader
league or Slumbot evaluation. A second 66k seed also failed that parent gate;
the next clean RL step is population/league self-play or another maintained
imperfect-information RL backend, not more single-checkpoint scaling. A first
sampled-opponent Rainbow response oracle against four frozen checkpoints was
slow and lost to the incumbent, so future population work should use explicit
PSRO/meta-strategy machinery or batched opponent inference rather than the
current sampled-opponent wrapper unchanged. Learned checkpoint opponents in
that wrapper now accept `--opponent-device auto|cpu|cuda`; `auto` uses CUDA for
in-process learned opponents when the learner is on CUDA, but keeps subproc
workers on CPU unless CUDA is requested. The first A/B showed only a small
throughput gain, so this is a resource-use fix rather than a full
population-training solution. `uv run --with open-spiel` can load
OpenSpiel's `universal_poker` and PSRO v2 modules, making it the preferred
maintained-library compatibility spike for explicit population methods. Run
`uv run --with open-spiel python scripts/probe_openspiel_poker_contract.py
--output-json autoresearch-session/openspiel/contract_probe.json` before using
OpenSpiel evidence: the current probe shows default `universal_poker` is a
4-action `fcpa` abstraction, `fchpa` is 5 actions, and tested full-game hold'em
is 20001 actions, so it is reference machinery until an adapter preserves the
native 9-action contract or transfers back through native gates.
AgileRL is a candidate for plug-in population/self-play only after the contract
probe passes: `uv run --with agilerl --with pettingzoo python
scripts/probe_agilerl_pettingzoo_contract.py --output-json
autoresearch-session/agilerl/agilerl_pettingzoo_contract.json`. The current
probe preserves 9 actions and 126 features through PettingZoo's turn-based
parallel wrapper, so it can be tested as a bounded library smoke without local
RL algorithm code. The repeatable IPPO smoke is `uv run --with agilerl --with
pettingzoo python scripts/run_agilerl_ippo_native_control.py --max-steps 64
--evo-steps 32 --learn-step 32 --eval-steps 16 --hidden-dim 32 --batch-size 32
--update-epochs 1 --device auto --output-json
autoresearch-session/agilerl/agilerl_ippo_native_smoke.json`. This delegates
IPPO to AgileRL and is not promotion or Slumbot evidence. Checkpoints can enter
native H2H with `uv run --with agilerl --with pettingzoo python
scripts/run_agilerl_ippo_native_control.py --checkpoint-in <agilerl.pt>
--baseline-checkpoint models/native_nfsp_dqn_reservoir_h512_10k_seed20260517.pt
--eval-steps 1000 --device cuda --output-json
autoresearch-session/agilerl/<h2h>.json`; a 64-step smoke checkpoint loaded
legally but lost badly to native NFSP, so this is an evaluator adapter first.
The h128/4k control also failed native NFSP with confidence, so do not scale
plain IPPO unchanged. AgileRL MADDPG/MATD3 can be checked with `uv run --with
agilerl --with pettingzoo python scripts/probe_agilerl_offpolicy_contract.py
--algorithms MADDPG,MATD3 --output-json
autoresearch-session/agilerl/agilerl_offpolicy_contract.json`; the current
probe constructs both algorithms but both fail a train smoke with a tensor-rank
mismatch.
For native population work, run `uv run --with nashpy python
scripts/analyze_poker_empirical_game.py <h2h.json>... --solve-meta-strategy
--output-json autoresearch-session/empirical_game/<name>.json` before training
a response oracle. `scripts/eval_mixed_policy_h2h.py` fills cross-format native
H2H gaps such as NPI versus native NFSP; it is only an evaluator adapter.
The direct maintained Tianshou Rainbow response oracle to that incumbent is now
the current local plug-in RL incumbent:
`autoresearch-session/tianshou_rainbow/psro_response_to_shared_marl18k_h256_18k_dummy8_seed20260678.pt`.
It entered the complete 7-policy empirical game and became the pure
meta-strategy leader after 5k confidence checks against the prior incumbent,
native NFSP, and saved Rainbow controls. This is local population evidence, not
Slumbot/SOTA evidence. The next identical response-oracle generation failed
its parent gate, so the active autoresearch objective is PSRO/XDO-style local
population improvement: train only after solving the empirical game, change the
population/meta-policy training distribution, and reject candidates that fail
parent/population lower-bound gates. A spawn-subproc version is mechanically
available but did not speed up the h256 learned-opponent workload in bounded
A/B tests.

The NPI branch is paused until target collapse is resolved. Use
`--teacher-mode public_world_value_prior` for preflop public-world targets or
`--teacher-mode sampled_state_value_prior` for sampled multi-street self-play
states. Both consume the learned value net while keeping the stochastic actor
policy as a prior. These are not Slumbot or SOTA evidence yet: the latest
multi-street run beat a weak fixed control in local H2H but still selected call
on every held-out root in public-world and exact-CFR gates. The two-generation
follow-up improved soft exact-CFR fit and stochastic public-world EV, but greedy
root action quality remained too weak. Treat the target-collapse gate as a
promotion blocker that triggers root-disjoint verification, not as a reason to
add action-specific patches. Treat plug-in Rainbow/PPO/NFSP frameworks as
maintained-library controls; do not implement generic RL learner internals
locally unless a reviewed native-contract exception is required. Native
promotion still requires root-disjoint exact-CFR and local league gates.

Slumbot is evaluation-only. Do not use Slumbot hands, traces, action
likelihoods, response ranges, revealed cards, or trace-start states as training
data, target labels, curricula, opponent priors, checkpoint selectors, or
hyperparameter signals. Slumbot artifacts may be used after evaluation to
diagnose transfer failures, but not to shape the next candidate unless a
completed methodology review reframes the mechanism as a general locally trained
method.

Trace-derived response ranges, hard-state policy heads, detached search-target
injection, and post-hoc calibration are diagnostic evidence tracks. They are not
the default live opponent model and should not receive Slumbot confidence spend
without positive local self-play league evidence.

```bash
python scripts/poker_autoresearch.py init
python scripts/poker_autoresearch.py status
python scripts/poker_autoresearch.py continuous --sleep-seconds 30
python scripts/poker_autoresearch.py set-phase \
  --phase self_play_policy_improvement \
  --reason "local self-play first; Slumbot remains evaluation-only"
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
criterion plus a completed methodology review; broad sweeps are intentionally
rejected.

The autoresearch CLI, evaluation harnesses, Slumbot adapters, solver
benchmarks, promotion logic, parsers, seed lists, and parity tests are
protected surfaces. Audit them before keeping a candidate or committing
methodology changes. The audit now discovers unstaged, staged, and untracked
git changes when no explicit path is supplied, and protected changes require a
completed review, tracked manifest, and review-scope file. Current hard
protected-surface defaults are enforced even if an older local goal file has not
been refreshed, and review manifest/scope requirements are hard defaults for
protected changes:

```bash
python scripts/poker_autoresearch.py objective-audit \
  --review-dir autoresearch-session/poker_reviews/<review_id>
python scripts/poker_autoresearch.py commit-ready \
  --review-dir autoresearch-session/poker_reviews/<review_id>
```

Before spending Slumbot confirmation hands on a candidate, queue the local
falsification ladder. It runs objective-drift audit, duplicate-swapped
incumbent comparison with a required positive lower 95% bound, and fixed-state
resolver diagnostics:

```bash
python scripts/poker_autoresearch.py enqueue-falsification \
  --candidate models/candidate.pt \
  --mechanism "local self-play policy improvement" \
  --max-resolver-cases 3
```

The current mainline phase is `self_play_policy_improvement`: static learned
warm starts and trace-derived policy probes failed transfer gates, so the
default branch is a simpler local self-play loop plus optional general search
improvement. Set the phase explicitly before continuing a long run:

```bash
python scripts/poker_autoresearch.py set-phase \
  --phase self_play_policy_improvement \
  --reason "local self-play first; no Slumbot-derived training data"
```

Use the CUDA budget frontier as the live-search baseline:

```bash
python scripts/eval_cfr_budget_frontier.py \
  --cases autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --cfv-cache autoresearch-session/search_consistency_restored200_100x2k_20260513/public_belief_successor_pool256_seed20260627_cache.npz \
  --start-index 128 \
  --limit 128 \
  --budgets 50,75,100,125 \
  --reference-iterations 150 \
  --solver-backend torch-levelsync-cuda \
  --min-evaluated 128
```

Queue the same branch through the autoresearch loop when running unattended:

```bash
python scripts/poker_autoresearch.py enqueue-cfr-budget-frontier \
  --cases autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --cfv-cache autoresearch-session/search_consistency_restored200_100x2k_20260513/public_belief_successor_pool256_seed20260627_cache.npz \
  --start-index 128 \
  --limit 128 \
  --budgets 50,75,100,125 \
  --reference-iterations 150 \
  --solver-backend torch-levelsync-cuda \
  --min-evaluated 128

python scripts/poker_autoresearch.py enqueue-cfr-matrix-footprint \
  --cases autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --start-index 128 \
  --max-cases 128 \
  --chunk-memory-cap-mib 4096
```

`StreetSolver` also caches two immutable terminal/equity matrix entries for
repeated public-board resolves, and `fast_cfr.py` reuses the matching torch
matrix tensor bundles plus per-tree workspace tensors. This is a compute
optimization only: active hand subsets have separate cache keys and are covered
by unit tests.

Only reopen neural warm-start work after a methodology review explains how it
will beat the exact CUDA budget frontier. Historical command:

```bash
python scripts/poker_autoresearch.py enqueue-warm-start-resolver \
  --checkpoint autoresearch-session/search_consistency_restored200_100x2k_20260513/search_consistency_allroots_allhand_policy_train128_iter5_seed20260643.pt \
  --cases autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --cfv-cache autoresearch-session/search_consistency_restored200_100x2k_20260513/public_belief_successor_pool256_seed20260627_cache.npz \
  --start-index 128 \
  --limit 64 \
  --min-evaluated 64
```

For regret-policy warm-start checkpoints, add `--baseline-iterations 10` so
the gate compares the learned initializer against simply spending more CFR+
iterations at similar latency.

Use the oracle warm-start diagnostic before training a new regret-field model;
it verifies that teacher solver-state labels can move low-budget CFR+ toward
the higher-budget resolver:

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

Solver-update variants must use the same fixed A/B before any gameplay use.
`dcfr_plus` is currently only an opt-in negative-control diagnostic; on the
64-root holdout it worsened L1/KL versus low-budget CFR+, so `cfr_plus` remains
the default:

```bash
python scripts/eval_solver_update_gate.py \
  --cases autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --cfv-cache autoresearch-session/search_consistency_restored200_100x2k_20260513/public_belief_successor_pool256_seed20260627_cache.npz \
  --start-index 128 \
  --limit 64 \
  --low-iterations 5 \
  --reference-iterations 25 \
  --candidate-update dcfr_plus \
  --min-evaluated 64
```

Build the corresponding supervised regret/policy field labels with:

```bash
python scripts/build_regret_policy_warm_start_targets.py \
  --cases autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --cfv-cache autoresearch-session/search_consistency_restored200_100x2k_20260513/public_belief_successor_pool256_seed20260627_cache.npz \
  --output autoresearch-session/search_consistency_restored200_100x2k_20260513/regret_policy_warm_start_targets_train128_seed20260657.npz \
  --start-index 0 \
  --limit 128 \
  --low-iterations 5 \
  --reference-iterations 25
```

Train the first fixed-artifact probe with:

```bash
python scripts/train_regret_policy_warm_start.py \
  --train autoresearch-session/search_consistency_restored200_100x2k_20260513/regret_policy_warm_start_targets_train128_seed20260657.npz \
  --holdout autoresearch-session/search_consistency_restored200_100x2k_20260513/regret_policy_warm_start_targets_holdout64_seed20260657.npz \
  --output-checkpoint autoresearch-session/search_consistency_restored200_100x2k_20260513/regret_policy_warm_start_train128_holdout64_seed20260658.pt \
  --output-json autoresearch-session/search_consistency_restored200_100x2k_20260513/regret_policy_warm_start_train128_holdout64_seed20260658.json \
  --device auto \
  --hidden-dim 256 \
  --n-layers 2 \
  --epochs 12 \
  --batch-size 8192
```

## Playing Slumbot

Run Slumbot only as held-out evaluation after local self-play gates pass, or as
a small integration smoke. Do not feed traces or outcomes back into training or
checkpoint selection.

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

For the current depth-limited resolving branch, use the aligned contract pilot
instead of training detached leaf/value targets:

```bash
python scripts/run_depth_limited_resolving_contract_pilot.py \
  --cases-json autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --cfv-cache autoresearch-session/search_consistency_restored200_100x2k_20260513/public_belief_successor_pool256_seed20260627_cache.npz \
  --output-dir autoresearch-session/depth_limited_contract_20260515 \
  --prefix callback_leaf_aligned_16x8 \
  --train-limit 16 \
  --holdout-start-index 128 \
  --holdout-limit 8 \
  --solver-iterations 5 \
  --device cuda
```

This collects labels from the exact CFR leaf-callback distribution, trains the
dual hand-CFV model, and evaluates by inserting the checkpoint back through the
same callback. It is diagnostic-only until root-disjoint resolver A/B gates
clear at larger scale.

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
