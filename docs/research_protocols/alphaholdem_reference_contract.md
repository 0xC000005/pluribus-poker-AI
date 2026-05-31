# AlphaNLHoldem Reference Contract

Date: 2026-05-28

## Purpose

Use the unofficial `bupticybee/AlphaNLHoldem` project as a public
AlphaHoldem-style reference, not as mainline source code. It gives us a concrete
RLCard-era baseline to beat in addition to the native Slumbot-facing engine.

## Local Checkout

- Path: `reference_code/AlphaNLHoldem`
- Upstream: `https://github.com/bupticybee/AlphaNLHoldem`
- Observed commit: `d847dbc`
- License: AGPL-3.0 with an explicit non-commercial warning in the README.

The checkout is ignored by git. Do not copy code, weights, or assets into the
tracked tree without a separate license review. Treat it as audit material and
benchmark scaffolding only.
The executable probe now verifies this, not just documents it: the checkout
must be git-ignored, `tracked_external_reference_files` must be empty, and
`reference_integrity_passed` must be true before the RLCard benchmark is
considered ready.

## What The Reference Contains

- RLCard `no-limit-holdem`, 50bb, 1v1, 5-action environment.
- Ray RLlib IMPALA/VTrace-style training.
- TensorFlow 1.x card/action convolutional policy-value model.
- Historical checkpoint league with k-best / PFSP-like opponent sampling.
- Bundled checkpoint `weights/c_1048.pkl` plus external archive links for a
  week-long training run.

## Dual Benchmark Requirement

The project now has two different public surfaces:

1. **RLCard AlphaNLHoldem surface:** beat the unofficial AlphaNLHoldem reference
   checkpoint or a source-controlled reproduction inside its own 50bb/5-action
   RLCard-style environment. This checks whether our neural self-play method can
   beat the closest available AlphaHoldem-like public baseline.
2. **Native Slumbot-facing surface:** beat local native 9-action full-deck
   incumbents, then run held-out Slumbot only after internal gates pass. This
   remains the real target game and must not use Slumbot data for training.

Passing the RLCard surface alone is not SOTA evidence for Slumbot. Passing the
native surface alone is not evidence that we beat the public AlphaHoldem-style
reference. A credible claim needs both.

## Next Autoresearch Step

Build an isolated reference audit/eval bridge:

- Parse the AlphaNLHoldem observation, action, network, and league contracts.
- Load the bundled checkpoint if dependency/runtime constraints allow it.
- Add a read-only compatibility evaluator or reproduction plan under this
  repo's audit tooling.
- Keep all downloaded weights and generated comparison artifacts under ignored
  directories.

Do not train from AlphaNLHoldem hands or use its weights as initialization for
the native Slumbot-facing agent unless the workflow explicitly opens a separate
transfer-learning review.

## Executable Probe

Run:

```bash
python scripts/probe_alphanlholdem_reference.py \
  --reference-dir reference_code/AlphaNLHoldem \
  --native-num-actions 9 \
  --require-checkpoint \
  --output-json autoresearch-session/alphanlholdem_reference/contract_probe.json
```

The current checkout passes this probe. Key result:

- `ready_for_rlcard_benchmark=true`
- `reference_integrity_passed=true`
- `reference_checkout_ignored=true`
- `tracked_external_reference_files=[]`
- `commit=d847dbc`
- `action_count=5`
- `direct_native_action_match=false`
- `bundled_checkpoint_present=true`
- `license_risk=agpl_reference_only`
- `legacy_runtime=true`

This authorizes an isolated RLCard compatibility evaluator as the next step. It
does not authorize native promotion, Slumbot training, or tracked imports of the
reference code.

## Isolated H2H Smoke

Run:

```bash
uv run --with rlcard python scripts/eval_alphanlholdem_rlcard_reference.py \
  --candidate alphanlholdem \
  --baseline random \
  --games-per-seat 2 \
  --seed 20260528 \
  --device cpu \
  --output-json autoresearch-session/alphanlholdem_reference/reference_vs_random_smoke.json
```

The current smoke passed mechanically with the bundled checkpoint loaded through
a clean PyTorch compatibility reader. This confirms the RLCard benchmark path
can play hands without importing the AGPL reference package. It is not strength
evidence because four games are too noisy; it is only an evaluator validation.

Future public-reference gates should use the same evaluator shape with our
candidate policy as `--candidate` and `--baseline alphanlholdem`, plus a
predeclared game count and lower-confidence threshold.
The H2H artifact must also carry the reference-integrity fields from the probe:
`reference_integrity_passed=true`, `reference_checkout_ignored=true`, and
`tracked_external_reference_files=[]`. The pre-Slumbot promotion gate now
rejects RLCard wins over random/non-reference baselines and rejects artifacts
without clean reference-integrity evidence.

The gate supports separate checkpoint paths:

```bash
uv run --with rlcard python scripts/eval_alphanlholdem_rlcard_reference.py \
  --candidate alphanlholdem \
  --candidate-weights models/<candidate_alphanl_format>.pkl \
  --baseline alphanlholdem \
  --baseline-weights reference_code/AlphaNLHoldem/weights/c_1048.pkl \
  --games-per-seat <N> \
  --min-lower95-candidate-payoff 0.0 \
  --output-json autoresearch-session/alphanlholdem_reference/<candidate>_vs_reference.json
```

The lower-confidence threshold is fail-closed: a 2-games-per-seat self-match
with threshold `0.0` correctly failed because its lower bound was negative,
while the same run with a permissive diagnostic threshold passed. This is a gate
contract check, not a strength result.

## Environment-Native Candidate Rule

Do not adapt a model trained in the native 9-action Slumbot-facing environment
into RLCard's 5-action environment for promotion. The principled comparison is
to train a fresh RLCard-native candidate under the same general self-play/
population-improvement schema, then compare that candidate against the
AlphaNLHoldem reference checkpoint in the same RLCard environment.

The native Slumbot-facing environment should get its own native 9-action model,
trained locally under the same high-level method. The invariant we transfer
across environments is the learning algorithm and research discipline, not a
checkpoint, action projection, Slumbot trace, or hand-built policy mapping.

If another card environment is introduced, repeat this rule: train a new model
inside that environment. The model is allowed to learn and adapt to each
environment's own action space and observations; the project should not hide an
action-space mismatch behind a projection layer and call that learning.

## RLCard-Native NFSP Smoke

The current minimal env-native candidate path is RLCard NFSP:

```bash
uv run --with rlcard python scripts/run_rlcard_nfsp_pilot.py \
  --train-episodes 20 \
  --eval-games 4 \
  --seed 20260532 \
  --hidden-dim 64 \
  --min-buffer-size-to-learn 8 \
  --device cuda \
  --checkpoint-out autoresearch-session/alphanlholdem_reference/rlcard_nfsp_env_native_smoke.pt \
  --output-json autoresearch-session/alphanlholdem_reference/rlcard_nfsp_env_native_smoke.json
```

Then evaluate the checkpoint against the reference:

```bash
uv run --with rlcard python scripts/eval_alphanlholdem_rlcard_reference.py \
  --candidate rlcard-nfsp \
  --candidate-weights autoresearch-session/alphanlholdem_reference/rlcard_nfsp_env_native_smoke.pt \
  --baseline alphanlholdem \
  --baseline-weights reference_code/AlphaNLHoldem/weights/c_1048.pkl \
  --games-per-seat 1 \
  --seed 20260533 \
  --device cuda \
  --output-json autoresearch-session/alphanlholdem_reference/rlcard_nfsp_vs_reference_env_native_smoke.json
```

The first run is only an end-to-end plumbing smoke. It proves that an RLCard-
trained checkpoint can enter the same-environment AlphaNLHoldem gate with
`trained_environment_native=true` and `native_action_projection=false`; it does
not establish strength.

## RLCard-Native PPO Smoke

The first maintained actor-critic control is Tianshou PPO over the same RLCard
state/action surface:

```bash
uv run --with rlcard --with tianshou \
  python scripts/run_tianshou_ppo_rlcard_reference_control.py \
  --rollout-steps 64 \
  --updates 2 \
  --repeat 1 \
  --batch-size 32 \
  --hidden-dim 64 \
  --replay-size 512 \
  --eval-games 4 \
  --seed 20260540 \
  --device cuda \
  --checkpoint-out autoresearch-session/alphanlholdem_reference/rlcard_ppo_smoke_seed20260540.pt \
  --output-json autoresearch-session/alphanlholdem_reference/rlcard_ppo_smoke_seed20260540.json
```

Then run the same-environment reference smoke:

```bash
uv run --with rlcard --with tianshou \
  python scripts/eval_alphanlholdem_rlcard_reference.py \
  --candidate rlcard-ppo \
  --candidate-weights autoresearch-session/alphanlholdem_reference/rlcard_ppo_smoke_seed20260540.pt \
  --baseline alphanlholdem \
  --baseline-weights reference_code/AlphaNLHoldem/weights/c_1048.pkl \
  --games-per-seat 2 \
  --seed 20260541 \
  --device cuda \
  --output-json autoresearch-session/alphanlholdem_reference/rlcard_ppo_vs_alphanlholdem_smoke_seed20260541.json
```

This validates the maintained-library PPO bridge and checkpoint loader. Four
games are not strength evidence, even if the smoke score is positive.

The same runner can train a fresh child against a frozen RLCard PPO parent:

```bash
uv run --with rlcard --with tianshou \
  python scripts/run_tianshou_ppo_rlcard_reference_control.py \
  --rollout-steps 64 \
  --updates 2 \
  --repeat 1 \
  --batch-size 32 \
  --hidden-dim 64 \
  --replay-size 512 \
  --eval-games 4 \
  --seed 20260542 \
  --device cuda \
  --opponent-checkpoint autoresearch-session/alphanlholdem_reference/rlcard_ppo_smoke_seed20260540.pt \
  --opponent-device cuda \
  --checkpoint-out autoresearch-session/alphanlholdem_reference/rlcard_ppo_child_vs_parent_smoke_seed20260542.pt \
  --output-json autoresearch-session/alphanlholdem_reference/rlcard_ppo_child_vs_parent_smoke_seed20260542.json
```

This is the smallest population-training hook for RLCard PPO. Larger runs must
predeclare the parent pool, game count, and lower-confidence gate before being
interpreted as public-reference strength evidence.

## RLCard-Native IMPALA Smoke

The closest maintained-library analogue to the AlphaHoldem reference direction
is Ray RLlib IMPALA/V-trace. The current bridge is intentionally conservative:

```bash
uv run --with ray[rllib] --with rlcard --with tianshou \
  python scripts/run_rllib_impala_rlcard_reference_control.py \
  --train-iterations 1 \
  --rollout-fragment-length 32 \
  --train-batch-size 64 \
  --hidden-dim 32 \
  --device cpu \
  --checkpoint-dir autoresearch-session/alphanlholdem_reference/rllib_impala_cpu_smoke_seed20260550 \
  --output-json autoresearch-session/alphanlholdem_reference/rllib_impala_cpu_smoke_seed20260550.json
```

This trains in the RLCard environment with an RLlib action-mask observation
contract. Ray 2.55's old-API masked IMPALA path trains on CPU, but the CUDA
learner path failed V-trace tensor-shape checks during smoke testing. Keep this
as a related-work bridge and CPU diagnostic until a new-API masked RLModule or
a local GPU V-trace learner is implemented.

## RLCard-Native APPO Guard

APPO is the maintained RLlib path closest to the reference's V-trace actor-
learner idea while using the new action-mask RLModule stack. The current runner
is deliberately guarded because earlier inline probes hung:

```bash
uv run --with ray[rllib] --with rlcard \
  python scripts/run_rllib_appo_rlcard_reference_control.py \
  --train-iterations 1 \
  --rollout-fragment-length 16 \
  --train-batch-size 32 \
  --hidden-dim 32 \
  --num-env-runners 1 \
  --num-cpus 3 \
  --device cpu \
  --timeout-seconds 20 \
  --output-json autoresearch-session/alphanlholdem_reference/rllib_appo_supervised_timeout_seed20260553.json
```

This command currently fails closed with `status=timeout`; the supervisor kills
the child Ray process group and writes metrics. Do not run APPO strength gates
until a tiny CPU/GPU smoke completes without the timeout guard firing. If this
maintained-library route remains unreliable, queue a methodology review before
moving to a local GPU V-trace learner on the compiled rollout substrate.
