# Next Continuation Target

Date: 2026-05-13

## Decision

Stop scaling hard learned leaf/successor value substitution and direct policy
argmax imitation as mainline methods. The latest root-disjoint successor-cut,
callback-state DCVN, policy mixing, and impact-gating checks all failed resolver
behavior gates. The next target is **neural regret-field resolving**: learn a
public-belief regret/policy initializer that warm-starts CFR+/resolving, then
let search refine the decision during play.

The required first gate is not Slumbot. It is a root-disjoint resolver A/B:
compare low-budget vanilla CFR+ against low-budget neural-warm-start CFR+, both
against the same higher-budget teacher, reporting root action L1/KL,
top-action agreement, illegal-action count, and latency.

Use modern neural architecture where it helps the learned search primitive:
set/card encoders, attention over public action tokens, residual trunks,
uncertainty heads, and mixed precision are all valid candidates. They must still
be evaluated as search initializers. A larger network that improves offline
target fit but fails the root-disjoint resolver A/B is not progress.

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
