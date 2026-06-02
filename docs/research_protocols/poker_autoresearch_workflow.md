# Poker Autoresearch Workflow

Status: approved and implemented for safe evaluation/logging automation, with
methodology-review, mechanism-review, synthesis, calibration-phase, and
knob-governance gates.

## Objective

Build a research loop for an elegant, novel, publication-grade heads-up
no-limit hold'em method that can train on a personal PC, use learned
game-theoretic knowledge during play, and beat both a public AlphaHoldem-style
RLCard reference baseline and the held-out Slumbot environment without
hand-crafted poker-strategy rules. The preferred philosophy is
pure/game-theoretic RL: AlphaZero-like in spirit, but valid for imperfect
information. The standard is a method that would be defensible as a
NeurIPS/AAAI/ICLR-style contribution: a clear mechanism, root-cause diagnosis,
related-work grounding, falsifiers, and transfer evidence.

The active direction is **tabula-rasa game-theoretic neural self-play for
poker**. The mainline should be describable as one loop: a stochastic neural
policy/value learner starts from fresh weights, plays full local self-play
hands in a rules simulator, receives only observations, legal actions, and
terminal rewards, updates through an equilibrium-oriented RL rule such as
R-NaD/NashPG/MMD-style regularized self-play, and returns the updated network
to self-play. This is the poker analogue of AlphaZero/Atari-style learning:
the simulator and reward function are allowed; human strategy, Slumbot traces,
hand-coded poker tactics, and detached solver labels are not the mainline.

Slumbot is evaluation-only. Do not use Slumbot hands, traces, action
likelihoods, response ranges, revealed cards, or trace-start states as training
data, target generation, curriculum, opponent model, checkpoint selector, or
hyperparameter signal. Slumbot artifacts may be used only after evaluation to
diagnose why a locally promoted self-play candidate failed transfer.

CFR/search is allowed when it acts as a general game-theoretic evaluator,
sanity-check, or optional policy-improvement operator across local self-play
states. It must not become the default training-label source, the whole
deterministic player, a Slumbot-specific patch, a street-specific hand rule, or
a way to fit visible benchmark quirks. Exact NashConv/CFR on small games is a
truth meter after training, not the training signal itself. Public-belief
representations, mixed policies, regret/policy initializers, and learned search
corrections are allowed only when they preserve the same local tabula-rasa
self-play contract.

The goal is not “fewer research knobs.” Knobs are acceptable when they isolate a
mechanism. The workflow rejects knob sweeps that substitute for understanding.
Every major cycle should ask whether it attacks the root problem: the current
system has not yet produced a full-game native self-play learner whose
successive neural policies reliably improve generation over generation and then
transfer to held-out Slumbot evaluation.

The workflow must deliberately allocate time to paradigm creation, not only
paradigm exploitation. Major AI advances often combine a simplifying mechanism
with a compute-scalable training loop: AlphaGo Zero removed human expert data
and learned from self-play; Transformers removed recurrence/convolution around
sequence modeling in favor of parallel attention; diffusion models borrowed a
physical noising/denoising process and turned it into a scalable generative
objective. Poker autoresearch should look for the same kind of mechanism-level
compression: a learned object or update rule that makes the current
belief/search problem simpler, more scalable, and easier to falsify.

The primary promotion signal is a self-play checkpoint league: candidates must
improve against previous checkpoints, the incumbent, and native self-play
controls before any Slumbot confidence claim. A change that only exploits
Slumbot quirks, action mapping edge cases, or visible evaluation thresholds is a
failure even if chips/hand improves.

AlphaNLHoldem is now a reference-only public baseline surface. Keep the
unofficial checkout under ignored `reference_code/AlphaNLHoldem`, do not copy
AGPL code into the main tree without explicit license review, and do not treat
RLCard's 50bb/5-action game as native Slumbot evidence. The correct use is to
audit its ideas and, if practical, build an isolated same-environment
benchmark. Train an RLCard-native candidate for RLCard and a separate native
9-action candidate for the Slumbot-facing game under the same general training
schema. Do not project or adapt checkpoints across action spaces for promotion.
Passing only one surface is not enough for a SOTA claim.

This is a train-per-environment rule, not a weak preference. If the card
environment, stack/action abstraction, observation contract, or benchmark
surface changes, train a fresh model inside that environment. The transferable
object is the big idea: neural self-play/population learning, legal masking,
league gates, and confidence evaluation. A checkpoint trained in one card/action
environment is not promotion evidence in another. Checkpoint continuation is
allowed only within the same environment contract; cross-environment work must
restart from fresh weights and reuse only the training schema.
Executable training and H2H reports should expose this boundary with fields
such as `environment`, `trained_environment_native`, `native_action_projection`,
and, for native checkpoints, `rlcard_candidate=false`.

Held-out Slumbot confidence runs now require an executable pre-Slumbot
candidate promotion gate. `scripts/eval_candidate_promotion_gate.py` must see
positive-lower-bound RLCard AlphaNLHoldem evidence from an RLCard-native
candidate against an AlphaNLHoldem reference/reproduction baseline, clean
reference-integrity evidence for the ignored AGPL checkout, positive-lower-
bound native 9-action H2H evidence, and complete native empirical-game support
before a run above the sparse Slumbot smoke cap can be queued. The gate rejects
missing empirical-game evidence, wins over non-reference RLCard baselines,
Slumbot or AlphaNLHoldem training-data leakage, tracked reference-code imports,
and any native-to-RLCard action projection.
Use `python scripts/poker_autoresearch.py enqueue-promotion-gate ...` for the
normal autoresearch path so the evidence check is queued, logged, and available
to Slumbot confidence gating.

The current frontier is **scaling the validated R-NaD-style tabula-rasa
self-play learner to the native full-deck 9-action game**. Exact CUDA CFR and
small-game NashConv remain evaluators/frontiers when applied to local states,
but the next learned object must improve the self-play learner itself:
successive checkpoint strength, empirical-game support, counterfactual action
values, or full-game league strength under matched controls. Detached
Slumbot-trace policy fitting, response-range replacement, post-hoc calibration,
and solver-label imitation are diagnostic evidence, not mainline methods.

The first native scaling bridge is the compiled R-NaD trajectory-contract
smoke: fresh R-NaD policy/value networks now consume compiled 9-action
full-deck self-play trajectories and take finite updates without Slumbot data
or solver labels. The second bridge exports the R-NaD target policy as a
native-policy-compatible checkpoint, so parent/child generations can enter the
existing duplicate-swapped H2H and league gates. Treat both bridges as plumbing
evidence only. The next promotion question is whether a scaled native R-NaD
run improves against parents/population controls in H2H and empirical-game
gates. The first scaled probe (`500x512`, seed `20260605`) cleared the parent
H2H lower95 gate but failed the required population gate: it lost to saved
native NFSP and Rainbow controls, and the child/parent/NFSP empirical game
solved to pure NFSP support. This makes the candidate non-promotable. The next
R-NaD step must change the population/self-play regime or train
multi-generation candidates that beat saved native controls before Slumbot or
RLCard evaluation.

A direct checkpoint-continuation version was then tested: gen2 initialized from
gen1, trained another `500x512` native compiled R-NaD update budget, and beat
gen1 in duplicate-swapped H2H. This validates generation-to-generation
continuation, but it still failed saved native NFSP and Rainbow controls, and
the gen2/gen1/NFSP empirical game again solved to pure NFSP support. Do not
repeat same-budget R-NaD continuation unchanged. The next R-NaD-family step
must train against empirical-game/meta-policy support, add a principled
historical/average population objective, or pivot to another tabula-rasa
game-theoretic learner that can beat saved native controls before any
Slumbot/RLCard gate.

The goal contract is specific but not brittle. Failed mechanisms are expected
research evidence, not completion of the outer objective:

- **North star:** build a locally trained self-play poker method whose
  stochastic neural policy/value networks improve through self-play,
  population learning, and equilibrium-oriented RL dynamics, then train
  environment-native candidates that beat both the RLCard AlphaNLHoldem
  reference surface and the native Slumbot-facing 9-action evaluation path.
- **Internal success:** same-budget checkpoint-league lower95 turns positive
  across generations, empirical-game support includes the candidate, and
  root-disjoint decision diagnostics do not contradict the league result.
- **Public-reference success:** an isolated RLCard/AlphaNLHoldem benchmark
  shows positive lower-bound H2H for an RLCard-native candidate against the
  unofficial reference checkpoint or source-controlled reproduction.
- **Held-out success:** Slumbot confidence runs happen only after internal
  native gates pass; Slumbot remains separate from the RLCard reference gate.
- **Blocked pivots:** no loss-weight, target-count, hidden-size, or selector
  tuning after a failed mechanism unless a positive root-decision gate already
  exists.
- **Soft pivot rules:** after a failed decision-impact mechanism gate,
  document the result, queue failure synthesis, run related-work/paradigm
  innovation review, select the next mechanism, and continue the outer goal.
- **Hard stop rules:** stop only for explicit user STOP, readiness failure,
  invalid evaluation, benchmark hacking, unsafe objective drift, or a causal
  control with unmatched compute or rejected traversal chunks.

## Slumbot Evaluation Quarantine

Allowed Slumbot use:

- Sparse integration smokes after local gates pass.
- Held-out confidence runs after a candidate clears self-play league gates.
- Post-failure diagnostics that explain transfer loss without feeding back into
  training or checkpoint selection.

Forbidden Slumbot use:

- Training rows, replay starts, target labels, curricula, opponent-response
  priors, revealed-card supervision, or hyperparameter selection.
- Promoting a candidate because it improves a Slumbot trace or likelihood metric
  without local self-play evidence.

Trace-derived response ranges, hard-state policies, and action-likelihood probes
are diagnostic-only. They may localize a failure but must not become the default
live opponent model, a street-specific range patch, or a promotion target.

## Mainline Scope Rule

Keep only one mainline mechanism active at a time: the local self-play
policy-improvement loop. Diagnostics and falsifiers may be added, but they do
not become candidate methods unless a methodology review shows they are general,
locally trained, and judged by matched decision or league gates. If a branch
needs many Slumbot-specific probes to look good, retire it as diagnostic
evidence.

Policy-only RL is a first-class falsifier, not an afterthought. Use maintained
libraries such as Tianshou, RLCard, or OpenSpiel for generic PPO/Rainbow/NFSP
algorithms whenever the native 9-action wrapper can support them; do not
reimplement generic RL algorithms locally. A frozen neural policy from PPO,
Rainbow, NFSP, R-NaD-style dynamics, or another self-play method may be enough
if it beats the same local native controls without explicit beliefs or runtime
search. If a plug-in or native policy-only self-play checkpoint clears
duplicate-swapped H2H gates, the workflow should simplify toward that direction;
if it fails, learned latent belief/search remains justified by evidence rather
than assumption. Current gate consequence: scaled Tianshou Rainbow is the
strongest local pure-RL baseline, so new neural policy-iteration candidates
must beat it before any Slumbot evaluation. The first mixed Rainbow
checkpoint-league pilot is infrastructure only: it did not beat native NFSP
with confidence, and it exposed learned-opponent inference as the next pure-RL
scaling bottleneck. Prefer plug-in multi-agent RL backends or batched opponent
inference over local rewrites of Rainbow/PPO internals. The native PettingZoo
AEC adapter is the preferred bridge for this: it keeps the 9-action poker
contract while letting maintained MARL libraries own the learning algorithm.
The first Tianshou MARL Rainbow self-play smoke is a validated bridge but not a
strong candidate yet; require local H2H confidence before treating MARL
self-play as the mainline over the stronger single-agent Rainbow baseline. Use
shared-policy MARL as the cleaner AlphaZero-style variant because one network
learns from both seats; if it remains neutral after a larger maintained-library
run, pivot to a stronger plug-in self-play algorithm rather than adding local
anti-collapse heuristics. Current gate consequence: subproc shared-policy
Tianshou MARL Rainbow is the strongest clean RL path, with a passed smoke-scale
local H2H league summary. Scale this path with the same league summary before
using Slumbot confidence spend. A continuation from this checkpoint failed the
N+1 gate, so reject generations that do not beat their parent and controls.
Treat parent-vs-child failure as a signal to change the maintained self-play
algorithm or training distribution, not as permission to tune Slumbot-facing
rules.
The runner now has a research-log drift guard for this exact failure mode:
after repeated local target-consumer/search-label transfer failures, another
XDO/NPI target-consumer experiment is blocked until review. This is because
those branches can fit local search labels while failing whole-game
parent/incumbent H2H, which recreates the old misalignment problem. The guard
does not block maintained-library RL response-oracle work where the stochastic
neural policy is the actor being trained online.
Shared-policy MARL PPO is also available through the same PettingZoo bridge,
but its first h256/16k-step CUDA pilot failed versus native NFSP under both
deterministic and stochastic evaluation. Keep PPO as a plug-in policy/value
control, and prefer stronger maintained game-theoretic self-play algorithms or
larger fresh shared-policy MARL Rainbow gates over local PPO rewrites.
A larger fresh h256/66k shared-policy MARL Rainbow run beat native NFSP but
failed to beat the h256/18k shared-MARL incumbent. Treat this as a required
parent-gate failure: local strength can improve against one control while still
regressing against the current incumbent. Future plug-in RL runs must report
parent H2H before broad controls or Slumbot evaluation.
A second fresh h256/66k seed also failed the same parent gate, so stop simple
single-checkpoint scaling as the immediate loop. The next plug-in RL direction
should make prior policies part of the training distribution through a
maintained population/league self-play setup or an external imperfect-
information RL library, while preserving the native 9-action evaluation gate.
For the public RLCard reference surface, the same principle now applies:
fresh RLCard-native candidates may train against a local mixture of RLCard-
native checkpoints, but not against AlphaNLHoldem as training data and not via
native-to-RLCard checkpoint projection. Direct RLCard collection with Torch
checkpoint opponents is CPU-bound and should stay a smoke/plumbing path; scale
population learning through a compiled/vectorized collector before strength
claims.
The first cheap approximation to that idea, a slow sampled-opponent Rainbow
response oracle against four frozen checkpoints, also failed. Future
population work must be more principled than "sample frozen opponent and hope":
use explicit PSRO/OpenSpiel-style meta-strategies, XDO-like extensive-form
population structure, or batched native opponent inference with a clear parent
gate.
Tianshou `SubprocVectorEnv` with Torch checkpoint opponents is now available
only through `context="spawn"` and must report setup, train, and total seconds.
Do not treat inner `train_steps_per_second` alone as evidence: the current h256
learned-opponent A/B is slower than dummy once setup and train time are both
counted. Use spawn-subproc as a diagnostic path until a same-scale A/B proves a
real total-throughput gain.
For dummy/in-process learned-opponent runs, pass
`--opponent-device auto` to `scripts/run_tianshou_rainbow_native_control.py` so
fixed checkpoint opponents use the learner device when CUDA is available.
This is now the default in `run_control`; subproc workers still use CPU under
`auto` unless CUDA is requested explicitly. The first same-budget A/B showed
only a small gain (`173.67` to `180.07` train steps/sec), so this is a resource
correctness fix rather than a decisive speedup.
Implementation boundary: do not hand-roll Rainbow, PPO, NFSP, PSRO response
oracles, or other generic RL algorithms in this repo. Prefer plug-and-play
library implementations and spend local engineering effort only on the native
9-action environment bridge, observation/action masks, batched inference,
population evaluation, and promotion gates. Custom code is justified only when
it preserves the poker contract or measures whether a maintained algorithm is
actually improving local self-play decisions.
The durable `poker_goal.json` must keep
`neural_policy_iteration_policy.generic_rl_algorithm_policy` set to
`plug_in_maintained_libraries_only`; objective audit should treat changes to
this boundary as methodology drift.
NPI candidates must be checked against plug-in RL controls through the mixed
native evaluator, not by copying those algorithms. Use
`--mixed-control-checkpoint tianshou-rainbow:<pt>` or
`--mixed-control-checkpoint tianshou-ppo:<pt>` on
`scripts/run_neural_policy_iteration_loop.py` so every generation is judged
against maintained-library checkpoints with the same lower95 native H2H gate.
Use `--require-control-gate` for unattended/goal-mode NPI runs; a run with no
fixed-control result for every generated checkpoint must fail rather than pass
vacuously.
The durable `poker_goal.json` records this same rule under
`neural_policy_iteration_policy`, and the NPI loop plus mixed H2H evaluator are
protected objective-audit surfaces.
Before training another population response oracle, build the native empirical
game with `uv run --with nashpy python scripts/analyze_poker_empirical_game.py
<h2h.json>... --solve-meta-strategy --output-json
autoresearch-session/empirical_game/<name>.json`. If coverage is incomplete,
queue the missing duplicate-swapped native H2H pairs or move to a maintained
PSRO backend; do not train from an arbitrary uniform/sampled population and
call it PSRO evidence. The active objective is now PSRO/XDO-style local
population improvement: freeze the local incumbent recorded in
`autoresearch-session/poker_state.json`, solve the native empirical game,
train maintained-library response oracles against the empirical-game
meta-policy or a reviewed population approximation, and add a candidate only
after parent and population lower-bound gates pass. Slumbot remains held out
except tiny smoke checks for integration or catastrophic transfer.
Use `python scripts/poker_autoresearch.py drift-status` before resuming a stale
queue; `continuous` runs the same check before starting the next experiment.
OpenSpiel is installable and exposes `universal_poker` plus PSRO v2 modules, so
it is the preferred maintained-library compatibility spike for explicit
population methods. Treat it as a reference/adapter investigation first; it is
not promoted unless it can preserve the native 9-action abstraction or produce
evidence that transfers back through the native parent and local-league gates.
The repeatable probe is `uv run --with open-spiel python
scripts/probe_openspiel_poker_contract.py --output-json
autoresearch-session/openspiel/contract_probe.json`. The current detailed
artifact reports default `universal_poker` as `fcpa` with 4 actions, `fchpa`
as 5 actions, and full-game hold'em as `20001` chip actions for the tested
stack. Do not treat default OpenSpiel poker results as equivalent to the native
Slumbot-facing game.
AgileRL is an allowed plug-in candidate for population/self-play MARL because
it targets PettingZoo multi-agent training. First run
`uv run --with agilerl --with pettingzoo python scripts/probe_agilerl_pettingzoo_contract.py --output-json autoresearch-session/agilerl/agilerl_pettingzoo_contract.json`.
The current probe preserves the native 9-action, 126-feature contract through
PettingZoo `turn_based_aec_to_parallel`, with one active legal mask and one
inactive all-zero mask. The adapter strips observations to raw features and
moves legal masks into `infos["action_mask"]`, matching AgileRL's documented
mask path. Treat this only as permission for a bounded library training smoke;
if AgileRL requires hand-written PPO/DQN internals or Slumbot-specific
wrappers, reject that path. The first bounded AgileRL IPPO smoke now passes:
`uv run --with agilerl --with pettingzoo python scripts/run_agilerl_ippo_native_control.py --max-steps 64 --evo-steps 32 --learn-step 32 --eval-steps 16 --hidden-dim 32 --batch-size 32 --update-epochs 1 --device auto --output-json autoresearch-session/agilerl/agilerl_ippo_native_smoke.json`.
This proves the maintained-library training API can consume the adapter, but it
is not strength evidence or promotion-eligible. Use it next only as a matched
plug-in control or population/self-play experiment with parent and local-league
gates. AgileRL checkpoints can be evaluated through the same script:
`uv run --with agilerl --with pettingzoo python scripts/run_agilerl_ippo_native_control.py --checkpoint-in <agilerl.pt> --baseline-checkpoint models/native_nfsp_dqn_reservoir_h512_10k_seed20260517.pt --eval-steps 1000 --device cuda --output-json autoresearch-session/agilerl/<h2h>.json`.
This H2H path is an adapter only; a 64-step smoke checkpoint was legal but very
weak versus native NFSP, so require real parent/control gates before using it
to motivate larger AgileRL experiments. The first h128/4k AgileRL IPPO control
also failed native NFSP with confidence (`mean=-0.149962`, lower95
`-0.187153`), despite zero invalid actions. Do not scale plain IPPO unchanged;
use AgileRL next only if it supplies a stronger maintained population/self-play
mechanism, otherwise return to the current shared-MARL Rainbow incumbent and
explicit empirical-game gates. The off-policy AgileRL contract probe is
`uv run --with agilerl --with pettingzoo python scripts/probe_agilerl_offpolicy_contract.py --algorithms MADDPG,MATD3 --output-json autoresearch-session/agilerl/agilerl_offpolicy_contract.json`.
The current result is negative: MADDPG and MATD3 construct but both fail the
training smoke with a tensor-rank mismatch. Do not hand-write those algorithms;
only revisit this path if the adapter issue is resolved in an upstream-
compatible way.
The current local plug-in RL incumbent is the fast-state shared-policy Tianshou
MARL Rainbow continuation
`autoresearch-session/native_rollout_substrate/fast_state_shared_marl_continue_from_incumbent_h256_65k_dummy8_seed20260763.pt`.
It continued the prior Rainbow response incumbent through local two-seat
self-play, cleared the 20k duplicate-swapped parent H2H gate
(`mean=+0.003705`, `lower95=+0.000561`), and became pure support in the
complete 5-policy empirical game against the prior incumbent, PSRO response,
shared-MARL 18k, and native NFSP controls. This is local population evidence
only; Slumbot remains held out until further internal league progress is
repeatable. The next step is a same-path N+1 fast-state self-play continuation
that must beat this parent and remain in empirical-game support, not another
singleton fixed-opponent response recipe.

Algorithmic solver-update changes are allowed only as opt-in diagnostics until
they beat the fixed baseline gate. Use `scripts/eval_solver_update_gate.py` to
compare any low-budget update against low-budget CFR+ and a higher-budget CFR+
teacher on the same root-disjoint public states. A failed update should be
recorded as mechanism evidence, not promoted or tuned into another knob sweep.

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
throughput, representation quality, or search. The immediate question is
whether the clean local self-play loop can produce monotonic checkpoint-league
progress. If it cannot, add only one general improvement operator at a time,
preferably search-guided and consumed inside the same self-play loop.

For heterogeneous checkpoints, comparisons must name the candidate and baseline
strategy sources separately. Use split-source compare flags when a candidate has
an explicit `average-policy` head but the incumbent is a regret-source
checkpoint. Applying one `--strategy-source average-policy` to both sides is an
invalid gate, not a strategy result.

Every methodology-changing action must include a structured mechanism brief in
`review_scope.json`:

- `decision_object`: the object whose decisions should improve.
- `where_consumed`: where the object is used in training/search/play.
- `matched_control`: the same-budget control that makes the result causal.
- `primary_decision_gate`: the first gate that can show decision impact.
- `retirement_criterion`: the result that retires the mechanism.
- `flexibility_boundary`: what may change without changing the hypothesis.
- `anti_benchmark_hack`: why the change is not merely optimizing a visible
  metric.
- `neural_policy_role`: whether the neural policy/value network is the main
  actor, an auxiliary teacher/student, or unchanged by the reviewed change.
- `cfr_role`: whether CFR/resolving is acting as a mixed-strategy improvement
  teacher, evaluator, live fallback, or unchanged control.
- `stochastic_policy_contract`: how the change preserves mixed-strategy play
  and avoids default deterministic argmax deployment.

This follows current agent-design guidance: keep agent goals concrete and
measurable, prefer simple workflow guardrails before open-ended autonomy, and
evaluate the trajectory of tool use and decisions rather than trusting final
outputs alone. Related references that informed this contract include OpenAI's
agent workflow/evals guidance, Anthropic's recommendation to prefer simple
composable workflows before more autonomous agents, Voyager's automatic
curriculum plus self-verification loop, AI Scientist-style idea-experiment-
review cycles, and recent coding-agent evidence that negative guardrails are
safer than many positive style directives.

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
- `review_scope.json`: changed paths, protected hits, mechanism, expected gate,
  structured mechanism brief, decision impact, decision-impact gate, fallback
  if there is no decision impact, pass action, and fail action for the reviewed
  batch.
- `team_review.md`: routing for research lead, verifier, literature scout, and
  benchmark auditor roles.
- `decision.json`: one of `proceed`, `revise`, `abandon`, or
  `gather_more_evidence`.

The validator rejects pending `TODO`/`PENDING` review files, missing
`review_scope.json`, decisions without sources, and related-work files that do
not name what transfers, what does not transfer, and the smallest local test.
It also rejects reviewed CUDA/search primitives unless `review_scope.json`
states how the change gets the project closer to stronger root decisions per
millisecond or better self-play checkpoint-league strength, names the next
decision-impact gate, and names the fallback if that impact is absent. This is
the anti-infrastructure-drift rule: a primitive can be useful only if its path
to root decisions or league strength is explicit and falsifiable.
The validator also rejects missing or pending `mechanism_brief` fields, so the
workflow cannot approve a flexible research action without first naming its
decision object, matched control, and retirement criterion.
When the independent verifier is invoked, related-work review, benchmark-
hacking audit, and mechanism review are mandatory. Use separate sub-agents for
these roles when available; the review files are the durable source of truth.

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

## Paradigm Innovation Gate

Run this gate after two consecutive failed cycles in the same failure class, or
whenever the next proposed action is just a larger model, longer run, extra
loss weight, or selector threshold on the same target. The purpose is to decide
whether to continue normal science inside the current paradigm or start a
bounded novelty sprint.

The executable gate is:

```bash
python scripts/poker_autoresearch.py enqueue-innovation-review \
  --subject "learned search update operator" \
  --anomaly "oracle update signal exists but shallow selectors fail holdout"
python scripts/poker_innovation_review.py \
  --innovation-dir autoresearch-session/poker_reviews/<review_id> \
  --require-complete
```

The gate writes an ignored review bundle under
`autoresearch-session/poker_reviews/` with `innovation.md`,
`thought_experiments.md`, `related_work.md`, and `decision.json`. If the review
justifies protected workflow or evaluation changes, complete the usual
methodology review and tracked manifest as well. The innovation review must
answer:

- **Anomaly ledger:** What repeated failure is the current paradigm not
  explaining? Name the exact artifacts and metrics.
- **Current-practice limit:** How is this done today, and what are the limits?
  Use the Heilmeier question set: objective, current practice, novelty, impact,
  risks, cost, time, and exams.
- **Integrity check:** What result would make the idea look bad? Follow the
  Feynman rule: report the strongest reason the idea may be wrong before
  running it.
- **First-principles reduction:** What is the simplest underlying object being
  learned? For this repo, valid candidates include public-belief state,
  game-theoretic update, amortized search operator, opponent-response latent, or
  compute-scaled exact solver primitive.
- **Cross-paradigm analogy:** Borrow one mechanism from another field or AI
  paradigm, but translate it mechanically. Examples: diffusion as iterative
  denoising of policy/search state, control theory as stabilizing feedback on
  regret updates, statistical physics as energy/min-free-energy belief updates,
  or compiler/runtime ideas as fused sparse search execution.
- **Smallest decisive test:** Define one root-disjoint or Slumbot-held-out test
  that could falsify the mechanism. Do not use Slumbot chips alone as the first
  novelty gate.

The required thought experiments are: mechanism stress test, failure thought
experiment, transfer thought experiment, and compute thought experiment. The
related-work file must cite at least one source, explain what transfers, what
does not, the novelty delta, and the smallest local test.

The 2026 novelty-workflow review adds three operating rules. Karpathy-style
autoresearch is valuable for tight fixed-budget loops, but poker cannot use a
single visible metric as the whole objective; use those loops only after the
mechanism is already defensible. AI Scientist-style systems support explicit
idea generation, review, and tree-style exploration, so major direction changes
should first produce a reviewable mechanism object. AutoDiscovery-style
Bayesian surprise is the selection heuristic for novelty sprints: prioritize
anomalies that change the causal model, not those that are easiest to tune.

Decision rule:

- Stay in the current paradigm when a known positive control exists, the next
  test attacks a named failure mode, and the result can pass a decision-impact
  gate.
- Start a novelty sprint when failures repeat across target fits, selector
  variants, and architectures, or when an oracle shows signal that the current
  learned interface cannot extract.
- Abandon a novelty sprint after the smallest decisive test fails and no
  stronger positive control remains. Record the failed mechanism as reusable
  negative evidence, not as a reason to tune around the gate.

This gate is intentionally not a brainstorming license for unlimited knobs. It
is a structured way to generate one new mechanism-level hypothesis, grounded in
related work and judged by the same evidence ladder.

## Objective-Drift Guard

Autoresearch experiments may change candidate/training code and configuration,
but protected evaluation surfaces are immutable by default. Protected surfaces
include the autoresearch CLI, local evaluation scripts, Slumbot adapters,
solver benchmarks, promotion logic, parsers, seed lists, and parity tests. Pattern-matched files
such as `scripts/eval_*.py`, `scripts/*slumbot*.py`,
`scripts/*resolver*.py`, and `test/unit/test_*autoresearch*.py` are protected
even when they are newly added. Changing those files requires a completed
methodology review, a tracked review manifest, and a `review_scope.json` that
covers every changed path and every protected hit.

Run the audit when keeping a candidate or before any methodology commit:

```bash
python scripts/poker_objective_audit.py --base-ref HEAD
python scripts/poker_autoresearch.py objective-audit \
  --review-dir autoresearch-session/poker_reviews/<review_id>
python scripts/poker_autoresearch.py commit-ready \
  --review-dir autoresearch-session/poker_reviews/<review_id>
```

If no `--changed-path` is supplied, both audit commands read unstaged, staged,
untracked, and optional `--base-ref` changes from git. An empty audit now fails
unless `--allow-empty` is explicitly passed, so a no-op validation cannot look
like a reviewed methodology change. The audit also merges the current hard
protected-surface defaults at runtime, so older `poker_goal.json` files cannot
silently omit newly protected workflow or evaluator files. Review manifest and
review-scope requirements are also enforced as hard runtime defaults.

The long-term objective remains the controlling policy: novel, compute-efficient
Texas hold'em methods whose self-play league strength improves over time and
then transfers to Slumbot and stronger bots on personal-PC hardware. Visible
smoke metrics are diagnostics, not promotion targets. A paper-acceptable cycle
must name the learned object, the search or RL boundary, why related work
predicts the mechanism, what simpler explanation it falsifies, and how it will
fail safely.

## Self-Play League Gate

Before Slumbot confidence spend or promotion, run an internal checkpoint league.
The required evidence is a duplicate-swapped candidate-vs-incumbent comparison
with positive lower 95% confidence, plus comparisons against the previous
checkpoint and relevant self-play controls when available. Solver-coupled
changes must also pass fixed-state resolver diagnostics. Use the built-in smoke
gate to verify the evaluator path before relying on it:

```bash
python scripts/poker_autoresearch.py gate eval-self-play-league-smoke
```

Valid progress plots should track checkpoint iteration, opponent/baseline,
strategy source, number of games, seeds, mean chips/hand, lower/upper 95%
chips/hand, and promotion blockers. Slumbot chip rate, sparse API smokes, and
Slumbot-specific response patches are blocked as primary evidence until the
league ladder passes.

The CLI now enforces that distinction for queued Slumbot runs. Before an
internal self-play league has a passed record with positive lower95 evidence,
`enqueue-slumbot` may only create sparse integration smokes up to 50 hands. A
larger Slumbot run is treated as confidence spend and is rejected until that
internal league evidence exists. This guard does not make Slumbot a training
objective; it keeps Slumbot as held-out external validation after self-play
progress is already visible.

## Falsification Ladder

Before spending Slumbot confidence hands or treating a candidate as promotable,
queue a falsification ladder. This is the local POPPER-inspired counter-test
stage: it tries to falsify the candidate's mechanism with distinct blockers
before live evaluation. The ladder complements, but does not replace, the
self-play league gate.

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
  95% confidence bound in the self-play league;
- fixed-state resolver diagnostics.

Passing the ladder is still not promotion. It means the candidate survived the
cheap counter-tests and may justify sparse live Slumbot confirmation. A
candidate with `promotable=false` or non-positive comparison lower bound should
fail this ladder even if the evaluator itself ran successfully.

## Research Knob Governance

Persistent knobs are allowed only when they test one named mechanism. Each knob
must record a single default, failure class, mechanism, rationale, and removal
criterion in `poker_knobs.tsv`. Broad sweeps and list-shaped defaults are
rejected. A completed methodology review is required before a persistent knob
can be registered; the knob ledger records the review id. Keep at most five
active knobs unless the goal file is deliberately changed after methodology
review.

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

The current mainline phase is `self_play_policy_improvement`. Exact CUDA CFR
remains the search teacher and frontier, but the player should be a stochastic
neural policy/value network trained through local self-play. Matched-latency
static warm starts, shallow trace controllers, detached policy labels, and
post-hoc calibration remain negative evidence unless a new methodology review
explains how they fit the neural policy-iteration loop and improve root
decisions or self-play league strength.
The current single-EV callback-leaf MSE variant is not a promotion candidate:
the CFR10 full holdout is marginal, learned terminal leaves are slower than
exact terminal evaluation, and both linear public-belief drift prediction and
cheap structural abstention are too weak. A small neural drift predictor is
only partial internal evidence and fails external transfer, so it is not a
safety gate. A small diverse-opponent-range smoke on high-margin flips showed
signal but not enough: plain averaging improved mean L1 but did not restore
most decisions, while an oracle over the same variants was much stronger.

```bash
python scripts/poker_autoresearch.py set-phase \
  --phase self_play_policy_improvement \
  --reason "AlphaZero-style neural self-play policy iteration is the active mainline"
```

Queue exact-search gates through the autoresearch loop when they provide
teacher targets or controls for the neural policy-iteration loop, rather than
running them only as side commands:

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

python scripts/eval_same_topology_batched_cfr.py \
  --cases-json autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --topology-plan-json autoresearch-session/search_consistency_restored200_100x2k_20260513/cfr_topology_compatible_chunk_plan_holdout128_cap4096_seed20260522.json \
  --group-indices 0,1,2,3,4 \
  --iterations 25 \
  --device auto \
  --max-root-l1 0.001 \
  --min-speedup 1.0

python scripts/eval_same_topology_batched_cfr.py \
  --cases-json autoresearch-session/search_targets/restored200_turn_successor_pool256_policy_seed20260627.cases.json \
  --topology-plan-json autoresearch-session/search_consistency_restored200_100x2k_20260513/cfr_topology_compatible_chunk_plan_holdout128_cap4096_seed20260522.json \
  --group-indices 0,1,2,3,4 \
  --iterations 125 \
  --teacher-iterations 150 \
  --device auto \
  --terminal-eval-mode batched \
  --min-speedup 2.0 \
  --min-teacher-top-match-delta 0
```

Approved next actions in this phase:

- use the repeatable end-to-end neural self-play policy-iteration loop:
  generation `N` stochastic neural policy/value self-play, sampled public
  states, search-improved mixed-policy/value targets, policy/value update,
  generation `N+1` checkpoint initialization from `N`, root gate, and league
  smoke;
- run root-disjoint CUDA CFR budget frontiers when they define teacher targets
  or matched exact-search controls;
- profile exact GPU resolving latency and illegal-mass parity as teacher-cost
  diagnostics, not as a replacement for the neural player;
- preserve and inspect immutable terminal/equity matrix reuse before larger
  fused-solver rewrites; same public-board resolves may use the bounded
  `StreetSolver` cache and the bounded torch terminal tensor-bundle cache,
  but active-hand-pruned subsets must stay isolated;
- for fused/matrix CFR work, first run a memory-capped chunk plan and an
  exact-topology-compatible chunk plan; only batch roots with matching topology
  hashes unless the implementation explicitly supports padding/ragged trees;
  start with the largest exact-topology group or a 4 GiB-capped parity/
  throughput smoke, and promote same-topology batching only on root-decision
  parity plus latency until full internal-state equivalence is separately
  proven;
- at high CFR budgets, same-topology batching must pass a teacher-relative
  gate before use as a teacher or main accelerator. The current CFR125 versus
  CFR150 top-five screen preserved top-action agreement (`43/50`, equal to
  serial CFR125) but failed the `2x` speed bar (`1.676x` worst group), so it
  remains a diagnostic/candidate-screening tool unless a ragged/fused
  formulation materially changes throughput;
- do not continue same-topology terminal-mode tweaks as the main path. Hybrid
  modes that loop only showdown or only fold terminals both failed the
  worst-group teacher-relative speed/accuracy gate. A future exact-search
  sprint should change the execution formulation, such as ragged/segmented
  batching across heterogeneous roots, rather than toggling terminal classes;
- before implementing a ragged/segmented executor, run
  `scripts/analyze_cfr_matrix_footprint.py --include-ragged-potential`. The
  current restored200 holdout has enough heterogeneous work to justify one
  bounded smoke: all 128 roots active through depth 3, 120 active at depth 4,
  `42,652` nodes/edges in the largest level, and `64,332` terminal nodes;
- validate ragged terminal execution separately before full CFR integration.
  The current showdown-only smoke passes at tolerance `0.002` and shows the
  best speed at 32- to 64-root chunks (`3.15x` and `2.02x`); a full 128-root
  padded batch is slower (`1.76x`). Fold terminal products also pass on the
  64-root slice (`3.49x` hero-fold, `3.44x` villain-fold). Treat terminal-only
  speedups as preflight evidence until full root-decision parity passes;
- for ragged terminal chunks, sort roots by terminal-row count and scatter
  outputs back to original indices. On the restored200 holdout, sorting reduced
  showdown chunks from `24` to `6` under a `40%` padding cap; the largest
  planned 64-root chunk had `7.78%` padding and a `1.995x` terminal-only
  speedup. Use this sorted shape for the first full-CFR integration smoke;
- after the ragged terminal full-CFR bridge, do not treat terminal-only GEMM
  speed as a sufficient proxy. The bridge preserved actual-hand root decisions
  on the sorted 64-root chunk (`64/64` top matches, max L1 `5.57e-5`) but was
  slower than serial (`0.939x`). Future exact-search accelerator work should
  fuse the level recurrence and terminal products into one on-device segmented
  executor; otherwise CPU recurrence and host-device copies dominate;
- use `scripts/analyze_segmented_cfr_layout.py` as the packing contract before
  writing kernels. The sorted 64-root restored200 chunk has `74,190` nodes,
  `74,126` edges, five level segments, and max level size `33,585` edges. The
  next executor must keep this layout's offsets, edge arrays, and terminal
  segments resident on device across CFR iterations;
- the segmented executor should progress by recurrence stages: first
  uniform reach-forward, then regret-matched reach-forward, then terminal
  value segments, then backward segmented reductions, then regret/strategy
  updates. The current CUDA uniform forward smoke is only the first stage
  (`13.9 ms` per repeat on the sorted 64-root chunk);
- do not store the segmented executor's main regrets as dense
  `(nodes, actions, hands)` tensors. The first regret-matched CUDA forward
  pass was correct but slow (`318 ms` per pass) because active edge regrets
  were gathered from dense node-action storage. Use edge-aligned active-action
  regret/strategy tensors as the main device layout, with compatibility
  scatter only at evaluation/report boundaries;
- edge-aligned regret forward is the current positive executor primitive:
  sorted 64-root CUDA dropped from dense-gather `318 ms` to edge-native
  `67.8 ms` per pass. Continue from the edge-native layout when adding
  terminal values, backward reductions, and regret/strategy updates;
- backward value reduction is now the second positive edge-native recurrence
  primitive: sorted 64-root CUDA reduced random per-node values through the
  segmented edge strategy layout in about `12.8 ms` per pass. This does not
  yet validate poker utility semantics; the next exact-search step must add
  terminal value construction and regret/average-policy updates before any
  root-decision parity claim;
- segmented terminal value construction is now implemented, but terminal matrix
  expansion must stay per-root/chunked. Per-terminal matrix gathering attempted
  a `117 GiB` CUDA allocation on the sorted 64-root chunk. The fixed per-root
  terminal pass checked `49,396` terminal nodes against loop formulas and ran
  at about `206.5 ms` per pass. Treat this as a correctness primitive, not a
  promoted speed path, until a complete CFR iteration passes root-decision
  parity and quality-per-millisecond gates;
- edge-native CFR+ regret and average-strategy update is implemented on the
  active-action layout. On the 64-root restored200 slice, CUDA update took
  about `60.4 ms` per pass versus CPU `221.6 ms`. This completes the
  standalone recurrence pieces. The next exact-search sprint must compose one
  full segmented CFR iteration and compare root policy/regret outputs against
  the reference solver before any more primitive-only optimization;
- one complete segmented CFR iteration now passes the first root-decision
  parity gate. Use `scripts/analyze_segmented_cfr_layout.py
  --run-single-iteration-parity` to compare against `solve_cfr(...,
  n_iterations=1)`. The 64-root CUDA run matched actual-hand root policies
  exactly and reached `6.41x` speedup, with only float32-scale regret
  differences. The next segmented executor gate is multi-iteration parity and
  stability, because first-iteration root policies are still dominated by
  uniform fallback strategy;
- multi-iteration segmented CFR diagnostics should track `decision_passed` and
  `internal_state_passed` separately. Current evidence is decision-positive but
  internally drifting: iter2 over 64 roots and iter25 over four roots preserved
  all actual-hand root top actions with tiny root L1, while internal
  regret/strategy tensors exceeded strict tolerances. Do not use the segmented
  executor as a promoted solver until longer root-decision stability passes or
  the PyTorch/NumPy float32 recurrence drift is reduced;
- `segmented-cpu` and `segmented-cuda` are explicit experimental solver
  backends for controlled decision-latency A/B gates. They may be used in
  resolver benchmarks, Slumbot smokes, and CFR budget frontiers only when the
  result reports root decision quality per millisecond. Do not route `auto` to
  the segmented backend until the internal drift concern is either reduced or
  bounded by a longer root-disjoint decision-stability gate;
- after any new explicit solver backend passes a smoke, run a same-surface
  sequential A/B against the current exact CUDA backend. Discard concurrent GPU
  timing runs. The first segmented high-level smoke matched
  `torch-levelsync-cuda` actions `4/4` with max strategy L1 about `3e-6` and
  `1.66x` total-latency speedup on four fixed states; the next required gate is
  a larger root-disjoint fixed-state A/B;
- if the larger root-disjoint A/B reverses a smoke win, retire promotion and
  classify the backend as shape-dependent. The first 16-case turn-only
  segmented A/B preserved actions but was `1.38x` to `1.42x` slower in total
  latency than `torch-levelsync-cuda`, so future work must either predict where
  segmented is faster before dispatch or reduce turn/small-tree overhead;
- use self-play league gates as the primary promotion signal;
- only add learned search objects when the decisive test compares them against
  equal-latency exact CUDA CFR, not only against low-budget CFR;
- run aligned depth-limited resolving pilots where callback-leaf target
  collection, supervised training, and leaf-callback inference use the same
  public-belief/search boundary;
- when aligned leaf value fit does not predict root action drift, switch the
  next test to search-impact labels, safe/multi-valued depth-limit targets, or
  another decision-aware continuation objective instead of scaling the same
  MSE value target;
- require any learned safety/abstention model to beat constant, linear
  public-belief, and oracle-gap baselines on root-disjoint action-drift labels
  before it can influence live solving; if it fails external transfer, retire
  the abstention formulation and change the supervised search object;
- for diverse-range or multi-valued depth-limit work, first test an exact
  positive-control solver/value-set formulation and compare it against both
  plain range averaging and the single-EV learned leaf;
- review and prototype pure game-theoretic RL alternatives such as NFSP,
  RM-FSP, R-NaD/DeepNash-style dynamics, ReBeL-style public-belief self-play, or
  Student-of-Games-style guided search when they address the same root
  decision-transfer problem;
- evaluate any neural initializer, residual update, or policy prior against
  the same higher-budget teacher and a same-latency CUDA CFR baseline.

Blocked anti-patterns:

- scaling hard learned leaf/successor value replacement as the mainline;
- sweeping final-distribution policy mixing weights;
- sweeping discount/loss/model-size knobs without a mechanism review;
- adding Slumbot-specific exploit patches or action-mapping tricks as a method;
- reopening static warm-start mass sweeps without a methodology review and
  equal-latency CUDA budget comparison;
- spending Slumbot confidence hands before internal self-play league evidence.

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
- **Tier 0.5: self-play league smoke.** The implemented
  `eval-self-play-league-smoke` gate runs a duplicate-swapped same-checkpoint
  model-vs-model comparison to validate the league evaluator path before it is
  used for promotion claims.
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
- **Tier 2.5: counterfactual-EV replay.** For belief/range mechanisms, replay
  held-out Slumbot trace states with revealed-hand counterfactual action values.
  This gate asks whether the changed resolver strategy improves action EV, not
  merely whether it increases revealed-hand likelihood or changes top actions.
- **Tier 3: Slumbot smoke.** Short live Slumbot run for integration,
  action-mapping, latency, and solver stability.
- **Tier 4: Slumbot confidence.** Longer Slumbot run with chips/hand, mbb/hand,
  confidence interval, all-in rate, fold/call/raise distribution, mapping error
  rate, and solver latency.
- **Tier 5: tournament track.** Multi-player or tournament-like evaluation only
  after the heads-up Slumbot track has a hard baseline and reproducible metric.

The primary promotion metric is the self-play checkpoint-league lower 95%
confidence bound of chips/hand versus the incumbent or previous checkpoint.
For Slumbot-held-out validation, report lower 95% chips/hand or mbb/hand as
transfer evidence only after the internal league passes. Training changes must
also report iters/hour, samples/sec, and train seconds/iteration.

Checkpoint continuation claims require training-state continuity. Current
compact GPU Deep CFR checkpoints save model weights and buffer-size metadata,
but not replay reservoir contents. A `--resume` run from such a checkpoint is a
model warm start, not true Deep CFR continuation. Training metrics now emit
`resume_semantics` and `resume_restored_replay_buffers`; parent-child league
claims require restored replay buffers or must be reported as warm-start
diagnostics only.

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
- `belief_integration`: learned belief information changes search but has not
  improved counterfactual EV or downstream decision quality.
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

Online related-work research is required (a) whenever new candidate mechanisms or
ideas are GENERATED — e.g. in a paradigm-innovation review or a strategic decision
memo — and (b) before adopting a new RL/search method, changing the core Deep CFR
family, adding public-belief search, or promoting an architecture as a mainline
direction. It must be LIVE online literature search (the `deep-research` skill /
WebSearch / Hugging Face papers), NOT recited from model memory: internal-knowledge-only
novelty claims are an overclaim / benchmark-hacking risk and do not satisfy this gate.
The literature note must state:

- diagnostic question;
- papers or primary sources checked;
- what transfers to this codebase;
- what does not transfer because of compute, action abstraction, or Slumbot
  constraints;
- the explicit novelty delta vs the closest prior work, and whether that prior
  work has been demonstrated at personal-PC / imperfect-information-poker scale;
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
- for `regret_policy_warm_start_checkpoint` checkpoints, add
  `--baseline-iterations <n>` to compare against a uniform CFR+ budget before
  claiming the learned warm start is compute-efficient;
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

Export the fixed supervised labels with:

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

Then train the fixed-artifact probe:

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

The first direct and low-state residual probes both failed the root-disjoint
offline baseline. Treat this as evidence that the current supervised field
target is not enough by itself: the cheap solver's average policy is already
closer to the teacher than the learned probe. Do not wire this checkpoint into
resolver play until a field learner beats the low-solver policy baseline first.

Legacy search-consistency, policy-head calibration, and CFV/DCVN scripts remain
diagnostic tools. Direct target fit is not promotion evidence. A run that only
learns a target file but worsens resolver drift or collapses into all-in
selection fails the search-quality criterion. The solver, training masks, CUDA
masks, Slumbot adapter, and target builder must share the same 9-action legality
contract; search must not clamp an illegal fractional bucket into a different
legal raise size.

On CUDA-capable machines, `resolve_solver_backend("auto")` now selects the
exact `torch-levelsync-cuda` resolver and falls back to CPU only when CUDA is
unavailable. Treat this as an infrastructure default, not a strength claim:
benchmark solver changes with `scripts/eval_cfr_budget_frontier.py`, keep
`promotion=false` frontier outputs diagnostic, and use explicit
`--solver-backend cpu` when a CPU baseline is required.

Use `scripts/eval_public_belief_probe.py` before wiring range inputs into the
main trainer. The probe compares feature-only target prediction to raw
hero/villain range-vector prediction on held-out gameplay-distributed targets.
It is diagnostic only: passing means the belief representation is worth a
larger value-target experiment, not that a checkpoint is promotable.

Use `scripts/eval_slumbot_response_range_ev_gate.py` before any response-range
integration or live Slumbot spend. The gate compares baseline and learned
response-range resolver strategies under a revealed-hand counterfactual action
value evaluator. Passing revealed-hand likelihood is not enough; the response
strategy must show positive counterfactual-EV evidence. Even then, explicit
ranges remain diagnostic until a learned latent belief representation consumes
the signal and survives the usual resolver and Slumbot gates.

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
  --review-dir autoresearch-session/poker_reviews/<review_id>
python scripts/poker_autoresearch.py commit-ready \
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
  --removal-criterion "Retire if held-out resolver drift or falsification-ladder evidence fails to improve against the no-target control." \
  --review-dir autoresearch-session/poker_reviews/<review_id>
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
For fidelity-sensitive traversal comparisons, also set
`--max-rejected-traversal-chunks 0`. Rejected chunks are retried and do not
directly corrupt accepted samples, but they prove the pool budget is binding;
quality H2H under retry pressure cannot be interpreted as a semantic
frontier-indexing result.
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

## Failed Training Triage

When a fresh GPU Deep CFR training cycle fails against the restored-history
incumbent, do not immediately scale the same recipe. First run a within-run
ladder comparison between its later and earlier checkpoints. If the later
checkpoint does not beat the earlier checkpoint, classify the run as a
learning-direction failure. If it does, address any fidelity blockers such as
rejected traversal chunks before larger training or external Slumbot spend.
Rejected traversal chunks in the CUDA trainer are discarded retry attempts, not
accepted biased samples, when accepted requested-traversal counts still cover
the intended workload. Treat them as compute predictability and retry-pressure
signals. If a clean no-retry rerun worsens or fails the within-run ladder, do
not keep shrinking chunks; move to a method/sample-efficiency review.

## Continuous Mode

Continuous mode consumes `hypothesis_queue` entries from
`autoresearch-session/poker_state.json`. In unbounded mode it idles when the
queue is empty, rechecking for new work until a hard stop appears. Bounded dry
runs can use `--max-cycles` or `--max-idle-checks`.

It stops when:

- `autoresearch-session/STOP` exists;
- the queue is empty in a bounded `--max-cycles` run;
- `--max-idle-checks` is reached in a dry run;
- a gate fails and it is not an approved soft mechanism failure;
- readiness checks fail;
- `--max-cycles` is reached.

Use `--continue-on-mechanism-fail` for the outer autoresearch goal. In this
mode, failed mechanism gates with non-integrity failure classes are closed as
failed evidence, then the workflow automatically queues a failure synthesis and
paradigm innovation review. Integrity failures such as `eval_invalid`,
objective drift, benchmark hacking, action mapping, or legal-mask failures
remain hard stops. The queued review artifacts still need to be completed by
the agent with online related-work research and a decision before new
implementation work.

Continuation integrity is also a hard contract. Compact GPU Deep CFR
checkpoints are model-warm-start resumes unless training metrics say
`resume_restored_replay_buffers=true`. Any parent-child continuation hypothesis
must either queue training with `--save-replay-buffers` and resume with
`--require-replay-buffer-resume`, or explicitly label the run as a model
warm-start diagnostic.

For a safe unattended launch after readiness is verified:

```bash
python scripts/poker_autoresearch.py enqueue \
  --hypothesis "Run Tier 0 integrity before the next research action." \
  --cycle-type experiment \
  --failure-class eval_invalid \
  --gate tier0
python scripts/poker_autoresearch.py continuous \
  --sleep-seconds 30 \
  --continue-on-mechanism-fail
```

To stop it, create:

```bash
touch autoresearch-session/STOP
```

This workflow approval does not approve any specific model architecture change,
long local training run, or Slumbot confidence run. Those require their own
HEAD cycle, explicit gate, and local budget.
