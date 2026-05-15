# Next Continuation Target

Date: 2026-05-13

## Decision

Stop scaling hard learned leaf/successor value substitution and direct policy
argmax imitation as mainline methods. The latest root-disjoint successor-cut,
callback-state DCVN, policy mixing, and impact-gating checks all failed resolver
behavior gates. The next target is **publication-grade game-theoretic RL /
learned search**: prefer pure self-play/equilibrium-learning methods where
possible, and use CFR+/resolving as a principled imperfect-information
evaluator, teacher, or correction operator when necessary.

The required first gate is not Slumbot. It is a root-disjoint resolver A/B:
compare low-budget vanilla CFR+ against low-budget neural-warm-start CFR+, both
against the same higher-budget teacher, reporting root action L1/KL,
top-action agreement, illegal-action count, and latency.

Use modern neural architecture where it helps the learned game-theoretic
primitive: public-belief encoders, set/card encoders, action-sequence attention,
residual trunks, equilibrium/RL update heads, uncertainty heads, and mixed
precision are all valid candidates. They must still be evaluated by downstream
decision quality. A larger network that improves offline target fit but fails
the root-disjoint resolver A/B is not progress.

The first implementation of this gate is now in place. Reusing the old
joint-PBS policy checkpoint as the initializer failed the 64-root holdout gate:
root-disjointness passed, but warm-starting worsened L1/KL and action agreement
while adding substantial latency. The next target should therefore train a
native regret-field initializer from low-vs-high resolver deltas, not reuse a
policy-imitation head. The training target should include per-action regret or
advantage residuals needed to move the low-budget solver toward the teacher,
plus a latency budget that avoids per-hand inference inside every resolver node.

The oracle version of that target now passed. Copying the 25-iteration teacher
solver's selected-node `regret_sum` and `strategy_sum` into a 5-iteration solve
made the low-budget solver much closer to the teacher on 64 held-out roots
(`0.0697` L1 vs `0.5254` vanilla, `0.0050` KL vs `0.2602`, `0.9688` top-action
agreement vs `0.7656`, `1.04x` latency). This validates the warm-start
interface and narrows the next task: export root-disjoint teacher solver-state
labels and train a network to predict both regret and average-strategy mass at
the public decision node.

The first target exporter smoke is also in place. Four train roots produced
`4,512` all-hand rows with public features, private policy features, belief
rows, legal masks, low-solver fields, and teacher `regret_sum`/`strategy_sum`
fields. The next implementation step is a learner for these labels, followed by
the same root-disjoint warm-start resolver gate.

The first learner did not clear the offline baseline. A direct model and a
low-state residual diagnostic both failed to beat the 5-iteration solver's
average policy on 64 unseen roots. This suggests the target must become more
search-aware than per-hand supervised field regression, or the evaluation
should learn only selective corrections where the low solver is demonstrably
wrong rather than distilling all rows uniformly.

A fixed solver-update probe is now also falsified. `dcfr_plus` was added as an
opt-in diagnostic update and compared with 5-iteration CFR+ against the same
25-iteration CFR+ teacher on the 64 held-out roots. It failed the gate: L1
worsened from `0.5254` to `0.5404`, KL worsened from `0.2602` to `0.3533`, and
top-action agreement fell from `0.7656` to `0.7500`. Keep `cfr_plus` as the
default. The useful next step is not to tune discount exponents; it is to
collect or learn a more search-aware correction signal, or test a better
researched predictive/learned update with the same fixed A/B gate.

The source-backed PDCFR+ update has now been tested under the same rule. Its
methodology review passed because it is a published predictive CFR+ diagnostic,
not a Slumbot-specific patch, but the 64-root root-disjoint gate failed badly:
mean L1 worsened to `0.7062`, KL worsened to `0.9894`, and top-action agreement
fell to `0.6250` versus low CFR+ `0.5254`/`0.2602`/`0.7656`. This rules out
fixed published solver-update swaps as the immediate path. Keep them opt-in
negative controls and return to learned counterfactual/search-derived
corrections evaluated inside the resolver.

The objective is now sharpened further: do not make Slumbot the thing to hack.
Slumbot is a transfer benchmark. The preferred research path is an elegant
general method, closer in spirit to AlphaZero/Student-of-Games/DeepNash than to
benchmark-specific poker patches: self-play learning, public-belief
representations, equilibrium-oriented RL dynamics, and search or resolving only
where imperfect information makes pure lookahead invalid. The next local work
should therefore compare two root-cause families before another engineering
sweep:

- **Decision-focused learned-search correction:** generate interventions that
  actually reduce root action L1/KL after low-budget resolving, then train a
  model to predict those search-effective corrections rather than raw teacher
  fields.
- **Pure game-theoretic RL alternative:** review and prototype the smallest
  NFSP/RM-FSP, R-NaD/DeepNash-style, ReBeL, or Student-of-Games-inspired module
  that can run locally and be judged by the same root-disjoint decision gate.

The first pure-RL alternative has now been built as a native full-deck NFSP/DQN
control branch: episode-level NFSP mode sampling, same-player next-decision
DQN targets, reservoir average-policy memory, target-network bootstrapping,
checkpoint save/load, and duplicate-swapped native H2H are all in place. This
branch is useful infrastructure, but the results do not justify making it the
mainline: 10k CUDA episodes failed to beat the 2k checkpoint with positive
confidence, and the Double-DQN target-network variant was negative/inconclusive.
Do not keep scaling native NFSP alone unless it clears duplicate-swapped H2H;
the next principled work should return to learned-search/public-belief methods
or a substantially stronger game-theoretic RL mechanism such as R-NaD-style
regularized policy dynamics coupled to an evaluation gate.

The first R-NaD-style local primitive also has a falsified control result. A
native full-deck regularized-policy pilot trained one policy net from sampled
terminal payoff targets and passed legal-mask/unit checks, but the 2k CUDA
checkpoint lost badly to the 10k native NFSP reservoir control under 1k paired
duplicate-swapped H2H (`mean=-0.1122`, lower95 `-0.1402`). This does not reject
regularized Nash dynamics; it rejects using sparse terminal sampled payoff as
the policy-update signal. The next version must use a stronger counterfactual
or search-derived advantage target before any scale-up.

The latest search-diagnostic result points to dynamic CFR traces rather than
more static cut labels. A denser static cut-impact predictor failed root-disjoint
correlation (`r=-0.0441`), so the resolver now exposes per-iteration
`regret_sum` and `strategy_sum` snapshots at traced public nodes. A 4-case
smoke showed the trace is mechanically useful: 100 records, zero skips, and
mean L1 to the final average strategy falling to `1e-08` at the final
iteration. The next experiment should use these traces to build a fixed,
root-disjoint predictor of low-budget solver error or convergence uncertainty,
then test whether that predictor improves low-budget resolving against the
higher-budget teacher. Do not treat the trace script itself as a gameplay
method.

The first version of that predictor found usable signal. On a fixed 32-root
train / 32-root holdout split, iteration `5` trace features improved holdout
L1 prediction over the train-mean baseline (`0.2214` MAE vs `0.2802`) with
Pearson `0.4283`; predicted-high roots had actual L1 `0.7981` versus `0.4424`
for predicted-low roots. This is exactly the kind of mechanism-level evidence
that was missing from static cut metadata. The next step should be a selective
search budget gate: use early trace features to decide where to spend extra
CFR iterations, and compare equal or explicitly reported compute budgets
against the 25-iteration teacher on root-disjoint states.

That first selective budget gate is now negative. The binary top-quintile rule
selected `7/32` roots and improved low trace L1 (`0.3456` adaptive vs `0.5202`
low), but it did not beat a uniform comparable-iteration budget (`0.3119`).
This is a useful falsifier: trace features identify some hard roots, but a hard
switch is too lossy. The next continuation target should predict a residual
correction to the low solver's strategy/regret field, or learn a smooth compute
allocation, and continue to report uniform-budget baselines.

The first shallow residual did not work either. A fixed linear trace-to-final
policy map was worse than both low and uniform traces (`0.7235` L1 vs `0.5202`
low and `0.3119` uniform). This rules out the simplest residual shortcut. The
next principled path is not another linear probe; it should either train a
nonlinear public-belief trace model with root-disjoint data and uniform-budget
baselines, or step back to a counterfactual advantage target that matches how
CFR updates actually move the average strategy.

The current synthesis chooses the counterfactual delta version as the next
single test: predict the low-to-later regret/strategy change, not the final
policy. This keeps the target closer to the update operator and avoids treating
the 25-iteration strategy as a one-step imitation label.

The shallow low-to-later delta implementation failed, so the target family
survives only in a stronger form. Do not keep adding linear residual probes.
The next version must use either a nonlinear public-belief trace model with
enough root-disjoint data or a target closer to counterfactual regret/advantage
updates, and it must continue reporting low and uniform baselines.

A methodology review now permits exactly that bounded next step. The reviewed
decision is `proceed`, but diagnostic-only: a nonlinear public-belief
trace-delta model may be tested if it predicts update-aligned regret/strategy
deltas, keeps CFR+/resolving as the correction operator, reports low and
uniform trace baselines, and leaves Slumbot/promotion surfaces untouched. The
tracked manifest is
`docs/research_protocols/poker_review_manifests/20260514T221524Z-nonlinear-public-belief-trace-delta-model.json`.

The first nonlinear implementation on the existing compact trace rows is also
negative. A CUDA MLP slightly beat the low trace on target fit
(`0.3216` vs `0.3266` L1 to iteration-10 target) but failed decision quality
(`0.5253` final L1 vs `0.5202` low and `0.3119` uniform). This rules out
capacity tuning on the current trace-row features as the next move. The next
trace attempt must add richer public-belief/counterfactual inputs or switch to
a stronger regret/advantage target.

Adding public-belief cache features and range-shape summaries improved the
same diagnostic but did not pass it. The enriched MLP reached `0.4983` final
L1, better than low `0.5202`, but still much worse than uniform `0.3119`.
This narrows the blocker further: context helps, but the low-to-uniform
policy-delta target is still not the right learned object. The next step should
target counterfactual advantages/regret fields or test predicted fields inside
an actual resolver update.

The first direct resolver-in-the-loop test of the trained regret/policy-field
checkpoint is also non-promotable. On the same 64 root-disjoint holdout roots,
the low-state warm-start checkpoint improved mean L1 slightly (`0.5028` vs
`0.5254`) and reduced all-in probability gap (`0.0117` vs `0.0799`), but KL
worsened (`0.2688` vs `0.2602`), top-action agreement fell (`0.7031` vs
`0.7656`), and warm-start latency was `7.66x` vanilla low-budget CFR. This
confirms that predicted fields can sometimes move the root policy in the right
direction, but the current per-hand supervised field target is not aligned
enough with the action decision and is too slow. The next step should not be a
bigger MLP; it should learn a selective or counterfactual correction that is
evaluated by action-quality improvement per unit of extra compute.

The raw counterfactual-advantage trace diagnostic is now also negative. The
resolver can expose per-action child values at traced public nodes, and the
diagnostic derives an `advantage_policy` for the acting player. On the same
32-root holdout split, iteration-5 instantaneous advantage policy was much
worse than both the low average strategy and a uniform iteration-10 baseline
(`0.9789` L1 vs `0.5202` low and `0.3119` uniform; top-match `0.3750` vs
`0.7813`/`0.8438`). Cumulative regret policy was mildly better than low in L1
(`0.4745`) but still lost to uniform. Do not train a raw one-step advantage
imitator; use the trace as input to a smoother regret-update or budget policy.

The strongest cheap baseline is now plain extra CFR, not a learned shortcut.
Running uniform 10-iteration CFR+ against the same 25-iteration teacher on the
64 held-out roots improved mean L1 to `0.3630`, KL to `0.1251`, and top-action
agreement to `0.8125`, with mean latency about `350 ms` versus `182 ms` for
5-iteration CFR+ and `859 ms` for the teacher. Any learned warm start, trace
model, or update-rule candidate must beat this 10-iteration baseline on both
decision quality and latency-adjusted value before it is worth scaling.

The budget baseline now has a dedicated evaluator. Use
`scripts/eval_cfr_budget_frontier.py` for future CFR budget claims because it
builds one solver per root, solves the teacher once and each requested budget
once, then reports quality and latency without pretending this is a promotion
gate. The first 64-root frontier confirms the current target: CFR10 reaches
`0.3630` L1 and `0.1251` KL at about `1.94x` CFR5 latency. Solver reuse
improved the 8-root cProfile smoke from `32.711s` to `18.787s`; CPU CFR
strategy reuse improved it to `16.687s`; exact seven-card evaluator reuse
improved it to `13.913s`, all while preserving 64-root quality metrics and
zero illegal mass. The next acceleration target is a coarser batched/fused CFR
boundary or learned public-belief correction, not another surface-level
evaluator cache. A learned or fused method must improve this
decision-quality-per-compute frontier, not just beat CFR5.

The current `torch-cuda` resolver backend is not the answer to the GPU-use
concern. A fresh CFR10 fixed-state smoke over the four built-in public states
averaged `1219 ms` with `torch-cuda` versus `743 ms` on CPU. This means the
next compute improvement should not be "force torch-CUDA"; it should either
batch/fuse the CFR recurrence so the GPU receives coarse work, or move the
learned-search boundary to a compact public-belief correction that reduces
solver calls. Any such change must clear the CPU CFR10 baseline above.

The first matrix/fused CFR footprint diagnostic makes the compute target more
concrete. On the 64-root turn holdout used by the budget frontier, public-state
trees are graph-small but hand-state-heavy: up to `1,257` nodes with `1,128`
private hands, `93.4 MiB` mean solver state per root, and `5.98 GiB` if all 64
roots are held concurrently. The transition graphs are tiny. Therefore, the
next GPU step should be a level-synchronous batched/chunked CFR recurrence over
node-by-hand tensors, not another per-operation torch port or adjacency-only
matrix experiment. Keep CPU CFR10 as the quality/latency baseline.

The first local level-synchronous recurrence is implemented as `cpu-levelsync`
and should remain diagnostic-only. It preserves average-strategy behavior and
legality, but the 8-root budget smoke was slower than the current CPU solver
(`158/294 ms` for CFR5/CFR10 versus `120/243 ms`). This falsifies a NumPy
level-synchronous rewrite as a CPU optimization. The remaining principled
compute path is either a real fused/chunked GPU recurrence or a learned
public-belief correction that reduces solver work while beating uniform CFR10.

The Torch version of that same level-synchronous recurrence now provides the
first real GPU resolver win. On the 64-root turn holdout,
`torch-levelsync-cuda` preserved budget-frontier quality and cut CFR5/CFR10
solver latency to `39.7/55.4 ms`, while CPU was previously `144.8/280.4 ms`.
CPU CFR25 versus Torch CUDA CFR25 parity over the same roots had perfect
top-action agreement and small strategy drift (`0.0281` mean L1). The next
single continuation target is still **validation before promotion**. The
Slumbot smoke path now accepts explicit `torch-levelsync-cuda`, and a 3-hand
live smoke passed with two solver calls and no parse/API errors, but mean live
reported solver latency was `1270 ms`. Before changing `auto`, isolate whether
the remaining live cost is range construction, solver-tree construction,
callback pruning, or CUDA recurrence overhead, then compare against CPU on the
same fixed mixed turn/river states.

That attribution is now partially done. The mixed four-state benchmark shows
that CUDA has solved only the recurrence part: `torch-levelsync-cuda` cut CFR10
recurrence time from `325.4 ms` to `94.0 ms`, but residual setup overhead stayed
about `235 ms`. The next single continuation target should therefore be setup
amortization: cache or precompute board/active-hand payoff components where it
does not change ranges or action semantics, then rerun the same CPU/CUDA
attribution benchmark and the live smoke. A valid optimization must reduce
`avg_solver_overhead_latency_ms` without changing legality, root actions, or
solver strategy beyond normal numeric tolerance.

The first setup optimization is complete but only partially solves the
problem. Incremental turn evaluation reduced fixed-state CUDA overhead to
`205.2 ms` and total latency to `298.4 ms`, but a post-change 3-hand Slumbot
smoke still had a `1338 ms` reported solver call. The next continuation target
should instrument or reduce the remaining live-only cost around range pruning,
active-hand selection, tree construction, and CPU-to-GPU tensor preparation.
Do not flip `auto` until same-state live-style attribution shows the full path
beats CPU with margin.

The live-budget check changes that interpretation: Slumbot uses at least
`150` CFR iterations, not the `10`-iteration smoke budget. At `150` iterations,
the fixed mixed-state benchmark shows a large CUDA advantage (`818.5 ms` total
vs CPU `4871.0 ms`). The next continuation target should therefore shift from
"make CUDA recurrence faster" to "make live resolver policy safer and cheaper":
evaluate whether lower or adaptive iteration budgets preserve the CFR150 action
quality on root-disjoint states, then expose a reviewed live-budget policy if it
beats the current fixed `150/250/350` heuristic on decision quality per compute.

The root-disjoint budget frontier now points to CFR100 as the first candidate
budget policy. CFR100 preserved `98.44%` of CFR150 top actions with `0.1247`
mean L1 and cut mean solver time to `354 ms` from the CFR150 teacher's roughly
`525 ms`; CFR75 is faster but less aligned, and CFR125 is more faithful but
smaller compute savings. The opt-in `fast-live` budget profile now implements
that single reviewed policy with `100/150/250` iterations and leaves the
default `live` profile unchanged at `150/250/350`. A sparse Slumbot smoke
exercised the profile with five CUDA solver calls and no API/parse errors, but
this is not promotion evidence. The same-state live-profile comparison is now
available and has run on the 64-root holdout: `fast-live` was materially
cheaper (`593.6 ms` vs `919.7 ms`, `0.6455x`) and close in distribution
(`0.1276` L1, `0.0275` KL), but two roots changed top action. The next
continuation target should inspect those disagreement states and decide whether
a bounded live confidence comparison is worth the variance. Do not add a second
budget knob before that falsifier.

The disagreement inspection now argues against immediate confidence spend.
One disagreement root becomes progressively closer to a CFR500 teacher as
budget rises, but the other is non-monotonic: CFR150 and CFR350 match the
teacher while CFR250 does not. The next target should therefore be a
teacher-aligned boundary detector or disagreement filter for live budget
selection, evaluated on root-disjoint states. Keep it as a diagnostic gate; do
not turn it into a Slumbot-specific budget sweep.

The CFR500 holdout frontier confirms the shape of that target. Uniform CFR350
is the cleanest tested approximation to the CFR500 teacher but costs about
`1.19 s` per solve. The current `fast-live` profile is cheaper and slightly
better on top-action agreement than `live`, but it is worse on L1/KL. The next
implementation should therefore estimate action-boundary uncertainty from
available profile/frontier signals and selectively escalate ambiguous states;
success means improving teacher L1/KL per millisecond against `live`, not
maximizing a Slumbot smoke score.

The first boundary-signal diagnostic is positive enough to formalize. Profile
L1 between `live` and `fast-live` identifies high-error roots, and a fixed
train-top-4 threshold improved held-out L1/KL in both half-splits while adding
only about `3-5%` latency. The next implementation should turn this into a
protected diagnostic gate that reports train-derived threshold, selected roots,
L1/KL/latency against CFR500, and low/live/uniform-CFR350 baselines. Keep it
off the Slumbot live path until that gate survives a root-disjoint split.

That formal gate now exists and passes both half-splits on the current 64-root
pool. The next continuation target is scale, not live deployment: generate or
reuse a larger root-disjoint public-state pool, rerun the CFR500 frontier only
where computationally feasible, and test whether profile-L1 selective
escalation still improves L1/KL per millisecond against `live`. If it does not
hold, return to trace-derived uncertainty rather than tuning the top-k.

The 128-root scale-up passed only as an offline oracle. After adding naive
online decision latency, direct profile-L1 deployment fails the compute premise:
it would cost about `1.57-1.67 s` per solve (`1.77-1.83x` live) because it must
run both `fast-live` and `live` before deciding whether to escalate to CFR350.
The next target should therefore learn a cheap boundary predictor from already
available state or early-trace signals, using profile-L1 as a teacher label.
Success means preserving most of the selective L1/KL gain without the extra
profile solve; failure means return to uniform CUDA budgets or a different
search-boundary objective.

The first cheap predictor is now a negative control. A feature-only ridge gate
failed to beat a mean baseline on profile-L1 prediction in both root-disjoint
split directions, even though downstream selective escalation improved by
chance. Do not tune ridge alpha or feature subsets. The next continuation
target should use richer predeclared uncertainty evidence, preferably an early
CFR trace or blueprint-policy entropy/margin signal, and must still beat the
same mean-baseline and downstream L1/KL gates before any integration work.

That richer trace attempt has now also failed, so the target should change
rather than the feature set. Trace iteration-5 plus public-belief context did
not predict profile-L1 better than a mean baseline and recovered none of the
oracle selected roots. The next principled target is a direct decision objective:
predict whether extra iterations improve the same root against CFR500, or learn
a single-solve early-stopping/continuation policy from trace convergence that
does not require running two budget profiles.

The direct-decision scalar target has also failed under the same gate. Predicting
`live_l1 - CFR350_l1` did not beat a mean baseline for either feature rows or
trace rows. The next continuation target should stop using shallow scalar
selectors. Use a mechanism that observes the solver's own sequence while it is
already running: a learned early-stopping or continuation policy based on trace
convergence, or a sequence model that predicts the future policy trajectory.

The existing top-predicted-error trace budget gate remains a negative control:
it improves over low iteration 5, but loses to uniform iteration 10 on both
matched split directions. Therefore the next target should not be another
top-k escalation rule. It should ask whether the whole trace sequence can
predict a stopping/continuation decision that beats a fixed uniform budget.

The existing nonlinear trace-delta MLP has now been rechecked on those matched
traces and failed more strongly than the older compact split result. It fit the
tiny train roots to zero loss on CUDA, but holdout predicted-policy L1 was worse
than the low iteration-5 trace and far worse than uniform iteration 10 in both
directions. This retires single-row trace-delta MLP tuning as a continuation
target. The next bounded experiment should either model the trace sequence
(`0..5` or `0..10`) as a solver trajectory, or learn a direct stopping/
continuation action that is evaluated against the fixed uniform iteration-10
baseline.

The first shallow sequence-level check also failed. A ridge model over the full
iteration `0..5` trajectory did not beat train-mean target MAE in either
matched split and its adaptive continuation policy still lost to uniform
iteration 10. This retires the cheap trace-selector line for now: low-iteration
trace state is informative, but the information is not stable enough under the
current small root-disjoint splits and linear decision rule. The next
continuation target should change scale or mechanism: either make uniform CFR
cheap through a fused/batched GPU solver path, or learn a recurrent
solver-dynamics/update model trained on a larger root-disjoint trace corpus and
judged by the same uniform-budget baseline.

The immediate evidence now favors the fused GPU search path. On 128 fixed
roots, `torch-levelsync-cuda` CFR100/125 remained close to a CFR150 teacher,
with CFR125 reaching `99.21875%` action agreement at lower latency than CFR150.
The next continuation target should therefore be a same-state live-budget
validation, not another trace-selector probe: keep default `live` unchanged,
compare `fast-live`/CFR125-style budgets against the live teacher on fixed
states or bounded Slumbot confidence hands, and only then consider changing
runtime budget policy.

The first bounded Slumbot latency smoke is consistent with that direction:
fast-live and live both passed 50 hands with zero API/parse errors, and
fast-live reduced mean solver latency from `1436.9 ms` to `858.0 ms` in this
sample. The next continuation target should be a statistically cleaner strength
check, not another latency smoke: use larger Slumbot confidence hands only when
API time is acceptable, or prefer local duplicate-swapped gates where variance
is controlled.

The 1,000-hand follow-up says speed is not enough. Fast-live and live were both
operationally clean, but both were negative against Slumbot, with fast-live
`-143 +/- 202` chips/hand and live `-264 +/- 233`. Since the slower live profile
did not rescue the result, the next continuation target should return to
strategy quality: train or evaluate a stronger blueprint/search-calibration
candidate, then reuse the faster CUDA profile for confidence checks only after
local gates show a real improvement.

The historical-checkpoint follow-up also says checkpoint picking is not the
answer. `slumbot_2p_iter900.pt` had looked slightly better than `iter1000` in a
larger local duplicate-swapped H2H, but its 1,000-hand fast-live Slumbot check
was worse (`-533 +/- 296` chips/hand) and mostly all-in (`836/999` selected
actions). Do not spend more live hands on old checkpoints unless a new gate
predicts live transfer for a mechanism reason. The next task should diagnose
why the blueprint overbets/all-ins under live Slumbot distributions, or build a
root-disjoint calibration/evaluation gate that catches this before API spend.

The new street-level diagnostic narrows that task: on a 300-hand `iter1000`
fast-live attribution run, all `82` all-ins came from the early blueprint path
(`60` preflop and `22` flop), not from turn/river solving. The next continuation
target should build a local early-street blueprint calibration gate that
measures regret-policy action distribution against search/stronger-policy
targets on reachable preflop/flop states. Do not patch this with a manual
no-all-in rule; the goal is to make the learned policy/search interface assign
reasonable probability mass before live Slumbot spend.

The first local version of that gate is already informative: the existing
policy-calibration target sampler now emits `per_street_target_diagnostics`.
It flags `iter900` as pathological before API play (`58.09%` preflop top-action
all-in), consistent with the later `836/999` live all-in mix. It also flags the
incumbent `iter1000` as still too aggressive (`17.33%` preflop and `30.80%`
flop top-action all-in). The next research task should turn this diagnostic
into a falsifiable candidate gate: train or select a checkpoint whose
early-street top-action all-in and mean all-in probability fall materially while
preserving duplicate-swapped H2H and fixed-state resolver behavior.

Two bounded deployment diagnostics rule out easy fixes. Non-greedy mixed
regret-matching did not reduce live all-ins or chips lost, and a hard
`--no-allin` diagnostic was still negative while increasing solver load. The
next step should not be a deployment flag change. It should change the learning
objective or self-play/evaluation signal so the early blueprint learns a less
degenerate betting distribution under the existing 9-action contract.

The restored-history 200x2k candidate closes the all-in gap but still fails live
confidence (`-172 +/- 225` over 1,000 hands) and pays a latency cost from many
more solver entries. This falsifies a too-simple early all-in story. The next
research unit should evaluate early-street value/action quality directly:
compare candidate and incumbent decisions on sampled reachable preflop/flop
states using duplicate-swapped rollouts or a stronger local teacher, then only
spend Slumbot hands on candidates that pass both distribution sanity and value
transfer.

The first-action outcome diagnostic reinforces that direction. In a fresh
300-hand restored-history attribution run, the largest first-action bucket was
`call/chk`, and it was strongly negative (`-402` chips over `186` hands). This
does not justify a manual "raise more" rule; it just says the current local
self-play gate is not measuring early passive-action value against a strong
opponent. The next implementation target should be a local approximate
best-response or adversarial early-street evaluator that punishes weak passive
blueprint behavior before Slumbot spend.

The first approximate-BR attempt is not that evaluator. A DQN-style frozen
checkpoint BR with Monte-Carlo returns failed its passive-call positive control,
so low exploitability from that script is non-evidence. The next attempt should
start from a positive-control-first design: exact/brute-force response in a
small public-state abstraction, stronger offline fitted Q with balanced action
coverage, or a restricted preflop/flop action-value evaluator that can exploit
known passive and over-all-in controls before touching real candidates.

The first replacement is a restricted showdown action-value diagnostic. It
does not train a model and is not exploitability: it estimates current-hand
showdown equity against a random opponent range, then scores each legal first
action by pot odds under an "opponent calls, then checkdown" abstraction. The
positive controls pass: premium aces prefer all-in pressure, while seven-deuce
facing a large legal raise prefers fold. A 64-root smoke produced
`oracle_best_mean_payoff=97.02` chips versus `call_mean_payoff=3.75`, with
positive controls passing. Use this only as an evaluator-health and
early-action value sanity check; candidate promotion still requires
duplicate-swapped H2H, fixed-state resolver evidence, and Slumbot confidence.
Checkpoint attribution on the same seeded 128-root preflop distribution shows
why the next method must improve early action quality rather than only reduce
all-in frequency: `iter1000` selected all-in on `105/128` roots and had
`checkpoint_mean_selected_payoff=-17.36` with an `89.12` chip oracle gap, while
the restored-history checkpoint selected call on `93/128` roots and was also
negative (`-21.18`, oracle gap `92.94`). This is not promotion evidence; it is
a local falsifier showing both over-pressure and passive variants miss a
simple hand-conditioned value signal.
A wider 128-root historical sweep did not reveal a ready checkpoint: `iter300`
was least bad by restricted selected-action payoff (`+3.04`) but had only
`6.25%` oracle match, while later and restored/search-consistency checkpoints
all retained large oracle gaps (`80-122` chips). This rejects checkpoint
selection as the main fix and points toward changing the learned early policy
or advantage target so it captures hand-conditioned action value before
turn/river search is asked to repair the game.
Raw advantage alignment confirms this is not just a deployment argmax issue.
On the same 128 roots, incumbent advantage-vs-restricted-value correlation was
`-0.028`, and restored-history was `-0.128`. The advantage network is not
ranking root legal actions by even this simple value proxy, so the next
publishable mechanism should change the learning signal or architecture around
public belief/private hand-conditioned action values rather than only tuning
action sampling, checkpoint selection, or Slumbot flags.
A restricted-value capacity probe makes that interpretation more precise. The
same `ValueNetwork` class fit clean restricted root labels with
`holdout_top_action_match=0.8203` and `holdout_mean_action_corr=0.7949` after a
small CUDA run (`512` train roots, `128` holdout roots). This argues against
"the MLP is simply too small" as the main explanation. The stronger next method
should improve the generated game-theoretic target/data distribution for early
actions, not merely increase hidden size or layers.

Related work supports this boundary choice. Kim's 2024 GPU-CFR paper frames
CFR as dense/sparse matrix and vector operations and reports speedups that grow
with game size (`https://arxiv.org/abs/2408.14778`). DeepStack shows the other
principled option: maintain ranges and opponent counterfactual values, then use
neural values inside continual re-solving rather than replacing search outright
(`https://arxiv.org/abs/1701.01724`). DDCFR is relevant future work because it
learns dynamic discounting instead of hand-tuning fixed CFR update weights
(`https://openreview.net/forum?id=6PbvbLyqT6`), but prior local DCFR/PDCFR
negative controls mean learned discounting should not be added until the fused
or learned-search boundary is clear.

Legacy note: the original leaf-only value objective is retained below as
historical context and negative evidence.

## Mechanism Hypothesis

Train a joint continuation model over public belief states (PBS):

- Inputs: raw public features, raw card one-hots/set encodings, learned action
  sequence encoding, and learned public-belief/range encoding.
- Outputs: both-player hand CFVs plus a legal action policy head from the same
  trunk.
- Training data: search-generated turn/PBS states with exact or cached CFV
  labels and solver average-strategy policy targets from the same public state.
- Inference: use the model only inside fixed resolver diagnostics at first,
  either as a depth-limited value target or a warm-start/policy prior. Do not
  use it in live Slumbot play until fixed-state A/B passes.

This keeps the project bitter-lesson aligned: no hand-coded strategy rules,
no opponent-specific patches, and no threshold gates replacing learned search.

## Why This Replaces Leaf-Only CFV Scaling

The failed leaf probe suggests that terminal river-leaf values are too local or
too late for the observed turn action drift. DeepStack, Supremus, ReBeL, and
Student of Games all point to learned values coupled to search at public belief
states, not isolated one-state action distillation. Safe/nested subgame solving
also warns that imperfect-information subgame values depend on broader strategy
context, so per-hand leaf MAE is not enough.

Primary sources:

- NFSP: https://arxiv.org/abs/1603.01121
- DeepNash / R-NaD: https://arxiv.org/abs/2206.15378
- DeepStack: https://arxiv.org/abs/1701.01724
- Supremus / deep CFV networks: https://arxiv.org/abs/2007.10442
- ReBeL: https://arxiv.org/abs/2007.13544
- Student of Games: https://arxiv.org/abs/2112.03178
- Safe/nested subgame solving: https://arxiv.org/abs/1705.02955
- Depth-limited value functions: https://arxiv.org/abs/1906.06412
- Opponent-limited online search: https://proceedings.mlr.press/v202/liu23k.html

## Smallest Local Test

1. Build a fixed turn/PBS dataset with paired dual-CFV labels and solver policy
   targets from the same public states. Use
   `scripts/build_joint_pbs_continuation_targets.py` to merge only aligned
   policy/value rows; it rejects public feature mismatches before training and
   preserves private policy features separately from public value features.
2. Train a shared-trunk joint policy/value probe and compare against:
   feature-only value, value-only belief model, zero CFV, train constants, and
   existing search-target policy distillation.
3. Run fixed resolver A/B on the high-drift turn cases. Required diagnostics:
   root action agreement, action L1/KL drift, value-sum residual, leaf/continuation
   latency, and action legality.
4. Promote only if the joint target improves fixed resolver behavior without
   weakening objective-audit or parity gates.

## Stop Conditions

- Abandon if the joint target does not beat zero/train-constant value baselines
  and policy-target cross-entropy baselines on held-out PBS states.
- Abandon if fixed resolver A/B action drift worsens against the equity-leaf
  or full-solver baseline.
- Do not add a persistent hyperparameter knob unless it has a mechanism-level
  rationale and a documented removal criterion in `poker_knobs.tsv`.

## First Local Evidence

The first `128/64` turn/PBS split confirms the target is learnable only when the
model has a learned card-set interaction path. A flat joint probe failed value
baselines (`MAE 0.2717` vs best constant `0.2500`), and a flat value-only probe
failed harder (`MAE 0.3475`). The Deepset value-only sanity passed (`MAE 0.2295`,
`RMSE 0.3018`), and the Deepset joint probe passed both gates (`MAE 0.2109`,
`RMSE 0.2870`, policy KL `0.5719` vs legal-uniform `0.6881`).
A checkpointed rerun also passed (`joint_pbs_turn128x64_deepset_seed20260613.pt`),
so the next step can consume a durable artifact rather than an in-memory probe.
The first fresh repeat split also passed (`joint_pbs_turn_repeat128x64_deepset_seed20260617.pt`),
with value `MAE/RMSE 0.1855/0.2504` and policy KL `0.4350` vs legal-uniform
`0.5806`.

This is not gameplay-ready. Treat it as evidence to run an independent
methodology review and fixed resolver A/B with the joint continuation model
before any Slumbot spend.

The first diagnostic resolver A/B did execute, but it did not clear the
behavior bar: replacing turn terminal equity leaves with joint-PBS values on
8 holdout states produced action agreement `0.375` and mean action L1 drift
`0.6281`. Repeating the same 8-state A/B at 25 solver iterations kept action
agreement at `0.375` and worsened mean/max action L1 drift to `1.1618/1.5874`.
This falsifies direct terminal-leaf substitution for the current checkpoint.
The next mechanism should either train leaf-compatible continuation targets,
add a safer depth-limited cut interface, or compare against a stronger exact
reference before any live play.

A safer one-step successor cut interface was added next. It avoids terminal
leaf misuse by replacing only immediate non-terminal successor nodes under the
current turn decision. The interface executed, but the current checkpoint still
failed the fixed behavior check: on the same 8 holdout states, 5 iterations gave
agreement `0.375` and mean/max L1 drift `0.6949/1.0212`, while 25 iterations
gave agreement `0.25` and drift `1.0949/1.7699`. This suggests the next target
must train on the successor frontier distribution that the depth-limited solver
actually consumes, not only on current-decision root PBS states.

The first successor-frontier export/training check built value-only joint PBS
targets from real cut nodes without a second dual-CFV label pass. A small
`16`-root train / `8`-root holdout export produced `95/50` successor targets
with `0` client-policy targets, so the policy gate is inactive by design. The
Deepset value model failed held-out value baselines (`MAE/RMSE 0.2423/0.3233`
vs best constant `0.1994/0.2828`). Before scaling this path, inspect successor
target error by action topology, actor-to-act, reach entropy, and bet size.
The first attribution pass found the failure is concentrated in compounded
betting topologies: `bbc/bbc/kb` had MAE `0.5423`, `bbbc/bbc/b` had `0.4650`,
and `bbbc/bbc/k` had `0.4039`, versus `0.1310-0.2110` for simpler checked/bet
families. The next data step should stratify frontier exports by action shape
and high bet count rather than blindly increasing all targets.

The high-bet successor-frontier follow-up did not clear the gate. A targeted
`96`-cut train / `48`-cut holdout export with `min_bet_count=5` produced a CUDA
Deepset model that fit train (`MAE 0.0681`) but failed holdout (`MAE/RMSE
0.3786/0.4788`) against the best constant (`0.3173/0.3961`). That points away
from more epochs or bigger unstructured MLPs. The next continuation target
should encode the betting sequence and bet amounts in a reusable learned module,
or factor values by public action topology, before any resolver A/B retry.

The first learned action-sequence attempt also failed. A GRU over public
action tokens and log-scaled bet amounts fit train (`MAE 0.0677`) but worsened
the high-bet holdout (`MAE/RMSE 0.3944/0.4936`) relative to the same-seed
no-action baseline (`0.3705/0.4679`) and the best constant (`0.3173/0.3961`).
Keep the diagnostic encoder available for falsification, but do not promote it
into gameplay. The next hypothesis should change the target or factorization:
for example, train topology-conditioned continuation values, predict normalized
delta-to-constant residuals, or generate paired public-action augmentations so
the model must learn invariances rather than memorize action strings.

A simple topology factorization check also failed. Train-set constants grouped
by `action_shape` got `MAE/RMSE 0.3214/0.3986`, and constants grouped by
`actor_to_act,bet_count` got `0.3304/0.4108`, both worse than the global
train-mean constant `0.3173/0.3961`. Do not spend on topology buckets alone.
The next target should probe reach/range-conditioned value structure: support
entropy, top-mass, zero-sum residuals, and whether predicting residuals from a
range-aware baseline is easier than raw per-hand CFVs.

The first reach-shift diagnostic found root-disjoint high-bet train/holdout
coverage is still poor: `bbc/bbc/bb`, `cbbc/bbc/b`, and `cbc/bbbc/b` appear in
holdout but not in train, and holdout has materially more concentrated hero
reach (top-mass SMD about `0.75`, entropy SMD about `-0.64`). Completing the
full high-bet train pool increased train cuts from `96` to `119`, but the same
missing shapes remained and the model still failed (`MAE 0.3653` vs best
constant `0.3184`). The next data step should create a larger successor-frontier
pool first, then split with explicit action-shape and reach-coverage audits
before fitting any new continuation network.

That controlled split is now implemented. A `256`-case successor pool produced
`244` high-bet cut targets; the metadata-balanced cut-level split created
`182/62` train/holdout targets with no holdout-only action shapes and much
smaller reach shift (largest SMD `0.265`, hero top-10 reach SMD `0.156`). On
this cleaner split, the same CUDA Deepset continuation probe passed (`MAE/RMSE
0.1204/0.1911`) against the best constant baseline (`0.2815/0.3660`), while
action-shape constants remained weak (`MAE 0.2802`). This means successor
frontier values are learnable when coverage is controlled; the remaining
bottleneck is sparse frontier context, especially low legal-action-count states
where residual error is highest. The next principled step should improve the
continuation interface or target factorization for those sparse legal contexts,
then rerun fixed successor-cut resolver A/B before any Slumbot evaluation.

The fixed successor-cut resolver A/B was rerun with a stricter behavior gate and
a matching high-bet frontier filter. This corrected an earlier diagnostic flaw:
the high-bet-only checkpoint had been applied to every immediate successor cut.
With `min_bet_count=5`, nonmatching successors remain exact CFR and only matching
high-bet cuts use the learned continuation model. Even under that matched setup,
the checkpoint failed the resolver behavior gate. On 16 evaluated cases from a
64-case scan, 5 iterations gave action agreement `0.50` and mean L1 drift
`0.8220`; 25 iterations gave agreement `0.4375` and drift `1.0912`. Hard
successor-value replacement is therefore not ready. The next mechanism should
make the search integration safer, for example by learning calibrated residuals,
mixing learned and exact values, or using the network as a warm-start/policy
prior instead of a hard cut-value oracle.

The simplest calibration variant was tested and failed. A learned affine
calibration over successor CFV predictions was nearly identity and slightly
worsened holdout value MAE (`0.1204` raw to `0.1212` calibrated). Plugging the
calibrated values into the matched 5-iteration resolver A/B still failed
(`0.50` agreement, `0.8275` mean L1 drift). Simple scalar value calibration is
not the bottleneck. The next target should be policy-aware or search-aware:
train a boundary policy/prior from resolver targets, learn residuals conditioned
on local legal context, or train a value-mixing controller against the fixed
resolver behavior gate.

The first policy-boundary calibration is promising but not directly usable.
Training only the policy head on a `192/64` resolver-target split improved
held-out policy fit (`L1 0.2115 -> 0.0382`, `KL 0.0572 -> 0.0054`) and reduced
policy-head solver drift (`0.8880 -> 0.8097`). But its argmax selected all-in on
`29.69%` of holdout cases while the target top all-in rate was `0`. Treat this
as evidence for soft policy priors inside search, not direct policy-head action
selection. The next gate should evaluate entropy/top-action calibration and a
solver warm-start/prior path that cannot force unsafe top actions by itself.
The fixed resolver benchmark now has an optional policy-head behavior gate; the
calibrated head fails it under max all-in rate `0.05` and max mean L1 drift
`0.75`. The probability-level diagnostics explain why: calibrated mean all-in
probability matches the target (`0.1469` vs `0.1454`) and entropy is close, but
argmax behavior is unstable around near-tied soft actions.

A naive soft-prior integration was tested and failed. Mixing the calibrated
policy-head distribution into the final `5`-iteration solver strategy at weight
`0.25` did not move the solver closer to a `25`-iteration reference: mean L1
was `0.5950` for the mix versus `0.5778` for the low-budget solver alone, and
action agreement stayed flat at `0.5469`. The all-in top-action rate also stayed
at `0.6563`. Do not spend on more final-distribution mixing weights. A useful
policy prior must enter the CFR update/warm-start path, or be trained as a
residual correction against low-vs-high solver behavior.

The first node-local CFR+ warm-start attempt also failed. The solver now accepts
optional initial regret and strategy tensors, and zero initializers preserve the
old behavior. A diagnostic seeded the current public node with policy-head
regret mass for every private hand, then reran the same `5` vs `25` iteration
budget comparison. This worsened mean L1 (`0.5969` vs `0.5778` for the plain
low-budget solver), lowered action agreement (`0.5312` vs `0.5469`), and left
all-in rate unchanged (`0.6563`). Related work still supports regret-aware
warm-starting, but this simplified one-node CFR+ seeding is not enough. The next
mechanism should be either theorem-closer strategy-based warm starting or a
learned residual correction target, not a sweep over warm-start mass.

A first learned residual combiner also failed the promotion gate. It trained on
the existing `192` resolver policy targets using public features, the
`5`-iteration solver distribution, and the calibrated policy-head distribution,
then evaluated on the fixed `64` holdout targets. It beat the cheap solver
distribution by a wide margin (`0.0617` vs `0.6929` mean L1), but it did not
beat the calibrated policy head (`0.0382` mean L1) and still selected all-in as
the top action too often (`0.28125`). The useful signal is narrower: top-action
agreement improved over the policy head (`0.28125` vs `0.15625`), so ranking
information is present but not captured by plain cross-entropy. The next
mechanism should train decision-aware policy calibration from solver targets,
for example top-action margin or pairwise ranking loss, with a fixed holdout
gate.

Decision-aware policy calibration is now the first local pass in this line. A
legal-masked top-action margin loss trained from the solver target's own top
action fixed the policy-head argmax failure on the original `64` holdout:
top-1 match improved from `0.15625` to `1.0`, top all-in rate fell from
`0.296875` to `0.0`, and the fixed resolver policy-head behavior gate passed
with mean action L1 drift `0.6973` under the `0.75` gate. The tradeoff is worse
soft distribution fit (`L1 0.0382 -> 0.1110`), so this is not promotion-ready.
Next continuation target: validate the rank-aware checkpoint on fresh fixed
resolver-policy cases without tuning `rank_loss_weight` or `rank_margin`; only
then run a small Slumbot smoke.

Fresh validation failed and corrected the gate. A new `128`-case sampled
resolver-policy target set has target top-all-in rate `0.59375`, while the old
holdout had `0.0`; the rank-aware checkpoint had learned the old split's action
distribution and selected all-in `0.0` on fresh cases. A fixed absolute all-in
cap is therefore not a valid general gate. The resolver benchmark now supports a
solver-relative all-in gap gate. On the fresh set, both the calibrated and
rank-aware checkpoints fail, and rank-aware is worse (`top1_match 0.0391`,
solver all-in gap `0.8516`). Next target: build a larger policy-target pool and
stratify train/holdout by target top action and public-state pressure before
rerunning rank-aware calibration.

The coherent-label rerun partially supports that direction but fails the
behavior gate. A fresh sampled-only `384`-case target pool with uniform
`25`-iteration CPU solver labels was split `288/96` by street and target top
action. Fixed-margin rank-aware calibration improved holdout L1 (`0.8622 ->
0.6315`) and top-1 match (`0.0833 -> 0.6458`), but overselected all-in
(`0.9583` vs solver/target `0.6667`) and failed the solver-relative all-in gap.
The next target is confidence-weighted ranking: use the solver distribution's
top-vs-runner-up probability gap to decide how strongly to enforce rank
margins, rather than forcing every top label equally.

Confidence-weighted ranking improves the stochastic policy path, not greedy
argmax. On the coherent `96` holdout, it improved soft fit over fixed rank
(`L1 0.5818`, `KL 0.2578`) and matched solver mean all-in probability well
(`0.3907` vs `0.4358`, gap `0.0450`), passing the new resolver probability
gate with mean action L1 drift `0.5880`. But argmax all-in is still `0.9896`,
so do not evaluate this checkpoint with `--greedy`. Next target: run stochastic
fixed/Slumbot smoke for the confidence-weighted rank checkpoint, while reporting
both probability-level and top-action diagnostics.

The stochastic full-game local smoke did not transfer. In duplicate-swapped
head-to-head against the base restored checkpoint, both using sampled
`policy-head`, the confidence-weighted checkpoint averaged only `+1.85`
chips/hand with lower95 `-50.57` over `12288` games. The likely bottleneck is
coverage mismatch: the policy head was calibrated from turn/river resolver
targets, but full-game policy-head evaluation uses it preflop and flop too.
Next target: either generate all-street search/teacher targets, or add an
explicit evaluation/play path that uses blueprint regret before turn and the
calibrated stochastic policy head only on streets covered by resolver targets.

That coverage-aware path was implemented and falsified. `policy-head-covered`
now reads checkpoint calibration support, falls back to regret outside covered
streets, and is available in local eval, Slumbot play, and range tracking. A
metadata-correct copy of the confidence-rank checkpoint declared turn/river
coverage from its actual train targets. In duplicate-swapped H2H against the
same checkpoint using regret, the covered policy route averaged `-16.93`
chips/hand with lower95 `-55.99` over `12288` games. The simple support-mismatch
explanation is therefore not enough. Next target: stop tuning policy-head
imitation losses and move to value/search mechanisms or reached-state policy
targets that optimize full-game value directly.

Historical checkpoint selection is now a triage tool, not the answer. A restored
history round-robin over iterations `50/100/150/200` initially ranked iter100
highest by mean field delta, but focused confirmation against iter200/final
failed hard: `-165.33` chips/hand with lower95 `-228.62` over `12288` games.
Keep iter200/final as incumbent. Next target: return to search/value integration
and make successor-frontier value use safer than hard replacement, ideally by
learning uncertainty or residual structure that is checked against resolver
action drift rather than policy imitation loss.

That uncertainty direction now has a positive diagnostic. A learned metadata
risk model trained on the metadata-balanced high-bet successor split predicts
per-cut value MAE on holdout with moderate signal: full metadata Pearson
`0.4895`, top-quintile recall `0.5385`; structural-only Pearson `0.5093`,
top-quintile recall `0.4615`. The structural-only result matters because it can
be applied before deciding which successor nodes to cut, without relying on
inside-callback reach features. The next continuation mechanism should implement
selective successor cuts: only use the learned continuation value on predicted
low-risk structural cuts and leave predicted high-risk cuts exact. The required
gate remains the fixed successor-cut resolver A/B with action agreement and mean
action L1 drift, not offline value MAE alone.

The first selective-cut A/B failed that gate. The resolver now supports a
pre-solve structural risk predictor so fallbacks remain exact instead of
returning zeros from inside the cut callback. With the fixed train-median
abstention rule, selected coverage was exactly `0.5` (`26/52` matching cuts),
but behavior got worse: only `7` cases evaluated, action agreement was
`0.4286`, and mean action L1 drift was `0.8906` versus `0.8220` for prior hard
replacement. This means offline successor CFV error ranking is not the missing
piece by itself. The next continuation target should train on the quantity the
solver gate actually cares about: root action drift, cut regret residuals, or a
search-consistency objective that couples successor values to root policy
stability.

The one-cut search-impact diagnostic confirms this target mismatch. Replacing
one high-bet successor cut at a time produced mean/max root action L1 drift
`0.6291/1.6000` over `24` cut evaluations. The structural predicted-MAE score
was effectively uncorrelated with this drift (`r=0.0189`), and the median
abstention rule did not separate safer cuts (`0.6412` selected mean drift vs
`0.6219` rejected mean drift). The next continuation target should therefore
build a search-impact dataset: features for the public node plus candidate cut
metadata/reach/value predictions, labels from one-cut or leave-one-out root
action drift, and a fixed gate that tests whether predicted low-impact cuts
actually reduce resolver action drift.

A context-enriched `64`-cut audit shows that static root context is still not
enough. The diagnostic now records root action shape, root bet count, pot,
stacks, player position, and first-to-act flag. The larger audit again showed
no relationship between value-risk score and search impact (`r=-0.0186`), and a
quick group-split ridge probe over cut metadata plus root context also failed
(`r=-0.0249`, top-quintile recall `0.2`). The next target should therefore move
inside the search loop: collect dynamic reach/regret traces or train a
continuation model with an auxiliary search-consistency loss that penalizes root
strategy drift directly.

A static-belief callback diagnostic narrows this further. The target exporter
trained the continuation model on average-strategy frontier beliefs, while the
cut callback normally supplies current CFR-iteration reaches. Feeding the
callback saved target-pool beliefs for matching cuts improved the high-bet A/B
but did not come close to passing: mean action L1 drift fell from `0.8220` to
`0.7160`, and agreement rose from `0.50` to `0.5625`. Reach distribution
mismatch is therefore a contributor, not the full cause. The next target should
train or distill on dynamic callback beliefs gathered during CFR iterations,
then validate with the same fixed resolver A/B.

Callback-state DCVN training tested that dynamic-belief hypothesis directly.
A 4-root smoke using exact callback states passed the small leaf A/B, but the
scaled `32/16` root-disjoint cache failed value baselines and resolver drift
gates. The first calibration falsifier, opponent-reach-weighted Smooth L1,
worsened supervised MAE/RMSE and failed both projected and unprojected leaf
A/B. This keeps the search-boundary value-network path alive, but rejects
simple reach weighting as the missing mechanism.

A learned state/player-offset plus hand-residual head then tested the simplest
target-factorization idea without leaking label means into inference. It also
failed: supervised MAE/RMSE worsened versus the direct MSE checkpoint and
unprojected leaf A/B drift rose to `0.6627`. The next continuation target
should therefore change the training target or search coupling, not another
direct CFV head variant: predict search-consistency residuals against the exact
resolver, train uncertainty/mixing against root action drift on root-disjoint
callback states, or collect paired exact-vs-learned callback interventions so
the model learns which value errors actually matter to the root policy.

The dynamic high-bet successor-cut line was rechecked with public-root-disjoint
target slices. The old passing probe used a shape-balanced row split inside the
same public-root pool. With disjoint slices, a small `110/115` target split and
a broader `512/256` target split both failed value baselines; the broader
GRU/deepset probe reached `0.2713` MAE versus a `0.1962` train-median constant.
This makes root extrapolation, not just callback feature plumbing, a recurring
failure mode. Do not run successor-cut resolver A/B from row-level internal
passes unless the checkpoint first clears root-disjoint value calibration.

A slice-aware one-cut impact diagnostic confirms the same issue behaviorally.
On unseen roots `128..143`, the root-disjoint GRU high-bet checkpoint produced
only `0.6087` action agreement with mean/max single-cut L1 drift
`0.5592/1.3602`; the worst root changed a baseline call into multiple bet
actions. The useful next target is therefore action-impact-aware: collect
exact-vs-learned intervention labels and train either a conservative fallback
gate or a value correction against root action drift.

The simplest learned fallback gate is not enough with the current sparse labels.
A ridge/log structural predictor trained on train-root one-cut records had
negative held-out correlation (`r=-0.1752`) and selected higher-drift cuts than
it rejected. This argues against threshold hacking. The next continuation work
should collect denser paired interventions across public roots, or optimize a
search-consistency target that directly penalizes root action drift.

The denser-intervention check now also failed. Expanding the same root-disjoint
one-cut audit to `127` train cuts and `117` holdout cuts still left the
structural predictor with negative holdout correlation (`r=-0.0441`) and only a
weak abstention separation (`0.4514` selected mean L1 versus `0.4902` rejected).
More static cut/root metadata is therefore not the root fix. The next target
should move inside the search loop: dynamic CFR reach/regret traces,
root-drift residual targets, or an update that learns which continuation
errors change the root policy while search is running.

## 2026-05-15 Update: All-In Semantics Repair

Full-deck all-in response semantics are now corrected across CPU, fast, and
CUDA states. An unmatched all-in must leave active non-all-in opponents to
call, fold, or raise; only matched/no-further-betting all-in states should deal
to showdown. This is not a new strategy method, but it changes the traversal
distribution that future learning sees.

Treat older checkpoint evidence as potentially stale because prior traversal
could skip opponent response states after an all-in. The next principled step
after this repair is to retrain or run a controlled short corrected-semantics
baseline before drawing more conclusions from Slumbot or restricted
early-action diagnostics.

That controlled short baseline is now complete and is not promotable. A
20-iteration corrected-semantics CUDA run improved the restricted early-action
value proxy versus the stale incumbent (`+0.12` selected payoff and `0.410`
advantage/value correlation versus `-14.50` and `-0.025`), but local H2H against
the old incumbent was inconclusive (`+164` chips/hand with lower95 `-280`).
More importantly, corrected all-in semantics expanded traversal enough that the
fixed GPU traversal pool now visibly truncates many traversals. The fast
default reached `419` traversals/sec but often overfilled the pool; 1000 slots
fell to `199` traversals/sec and still overflowed later; 2500 slots fell to
`87` traversals/sec. The next target is therefore not a larger network or a
longer run. Add hard traversal-fidelity telemetry and test an adaptive or
state-aware traversal allocation that preserves all-in response states without
global slot inflation.

The immediate CUDA OOM from trying a 2M-slot pool is fixed by adaptive
value-net forward chunking, and the exact failed command now completes. This is
an engineering reliability improvement, not research evidence for scaling the
pool. The 2M/1000 probe still runs at only `199.6` traversals/sec and still
shows several-times-over pool demand, so the active blocker remains traversal
fidelity and tree-shape control under corrected all-in semantics.

Traversal-fidelity telemetry is now part of autoresearch training JSON. Future
training runs must report overflow chunk fraction and pool demand ratios before
their checkpoint evidence is interpreted. The next real research step can now
be framed cleanly: reduce traversal overflow by a principled state-aware
allocation or sampling design, then compare against the same fast default on
both throughput and downstream decision quality.

The first telemetry A/B rejects slot scaling as that design. Default traversal
is faster but severely truncated (`3.98x` mean pool demand, `8.38x` max);
1000-slot traversal reduces demand but still overflows every chunk and loses
substantial throughput. Since Deep CFR's external-sampling contract explores
all traverser actions, the current pool-exhaustion demotion is not an
algorithmically clean substitute. The next target should be a source-backed
MCCFR sampling redesign: streaming external-sampling traversal, outcome
sampling, average-strategy/action-subset sampling, or another estimator with
explicit correction and variance accounting.
