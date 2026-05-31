# Poker Autoresearch Self-Review

Date: 2026-05-21

## Objective Check

The durable objective remains: develop an elegant, publication-grade,
compute-efficient heads-up no-limit Hold'em method that trains on a personal PC,
uses learned public-belief/game-theoretic RL plus search, and avoids
Slumbot-specific strategy rules. Slumbot remains a transfer benchmark, not the
training objective.

## Related Work Read

ReBeL and Student of Games support the main philosophical direction: learn
public-belief/search-compatible objects and use search as the improvement
operator. Depth-limited solving supports values at coherent public-belief
boundaries, not isolated terminal callback hacks. AutoCFR and Deep Predictive
Discounted CFR support learning or predicting the regret/update process rather
than hand-tuning CFR variants or cloning final policies.

The pure-RL literature should remain active. Recent policy-gradient work argues
that tuned PPO-style methods can be competitive in imperfect-information games
(`https://arxiv.org/abs/2502.08938`), while the 2026 VRPO/Q-boosting result
points to Expected-SARSA(lambda) as a better estimator than sampled GAE-style
future-action backups (`https://arxiv.org/abs/2605.19235`).

## Recent Evidence

Exact public-belief successor-cut replay is now a positive control. Per-iteration
replay matched the full resolver exactly on the high-margin cut pool. Fixed
iteration 5 also preserved all six eligible root actions with `0.1968` mean L1.

Direct dense value learning is the current failure mode. The all-iteration
joint-PBS CFV probe missed the constant baseline. Iteration-5 absolute CFV
regression failed badly, and the iteration-5-to-final residual probe failed the
zero-delta baseline. This points to target/representation mismatch, not simply
insufficient model size.

Two additional checks sharpen the blocker. The old learned regret/policy
warm-start fails once CFR10 is included as a cheap uniform baseline. Batched
high-entropy PPO and a minimal `q_expected_mc` action-value baseline both remain
negative against the 10k NFSP reservoir control. Pure policy-gradient is still
philosophically attractive, but the local implementation needs a stronger
variance-reduced or counterfactual advantage signal before scale.

The first stronger variance-reduced version, `q_expected_lambda`, improved the
PPO-family result but did not pass: `mean=-0.008408`, `lower95=-0.022051`
against NFSP10k. This is enough to justify one matched-budget scale check if
compute is idle, but not enough to call policy-only solved.

The matched-budget scale check reached near parity (`mean=0.000085`,
`lower95=-0.010230` over 5k games), which is a meaningful improvement over the
earlier PPO controls but still not a pass. This makes policy-gradient RL a live
candidate again, provided the next change is estimator-level rather than a
hyperparameter sweep.

A fresh related-work pass strengthened the motivation for a centralized
training-only critic: Rudolph et al. argue PPO/PPG can be competitive in
imperfect-information games (`https://arxiv.org/abs/2502.08938`), Fan/Farina
identify sampled future-action variance and propose VRPO/Q-boosting for HUNL
(`https://arxiv.org/abs/2605.19235`), and AutoCFR frames update-rule design as
meta-learning rather than manual CFR tinkering
(`https://www.sciencedirect.com/science/article/pii/S0004370224001681`). The
local CTDE q-lambda test used that idea narrowly: the actor stayed
observation-only while the Q critic received opponent private cards and stack
state during training. The 2k result was mean-positive but inconclusive; the
10k result failed NFSP10k (`mean=-0.003581`, lower95 `-0.013593` over 5k).
This retires the simple centralized-critic feature fix.

The two immediate estimator/game-dynamics follow-ups were negative. A simple
target-network Expected-SARSA(lambda) variant trained on CUDA but regressed
against NFSP10k. A reviewed average-policy/fictitious-play PPO control also
trained cleanly, but the exported average policy failed much worse than the
same checkpoint's actor export. This means the naive supervised average-policy
layer is not yet the right equilibrium object; the current pure-RL live branch
needs either a more faithful RM-FSP/VRPO-style update or search-derived
counterfactual advantages.

## Candidate Next Steps

1. **Search-state warm-start/update field**: learn compact regret, strategy,
   reach, or update fields consumed inside CFR and evaluate by root-disjoint
   resolver behavior. This is the recommended branch because it matches the
   exact replay evidence and related work.
2. **Decision-projected continuation target**: compress CFVs into
   decision-relevant action-value or policy-delta projections before learning.
   This is lower risk than full CFV tensors but closer to policy imitation.
3. **Pure game-theoretic RL control**: continue NFSP/R-NaD/PPO-style controls
   only when the estimator or equilibrium mechanism changes materially. The
   simple target-Q and supervised-average FSP/PPO variants are now retired; the
   next credible pure-RL variant would need a more faithful RM-FSP/VRPO update
   or an explicit average-policy fit diagnostic before scaling.

## Decision

Proceed with a stronger update-aligned branch. The immediate negative controls
retire the current learned warm-start and plain PPO pilots. The next test should
either implement a true variance-reduced policy-gradient estimator
Expected-SARSA(lambda)/VRPO-style, or learn a search-derived counterfactual
advantage/update field that beats CFR10 inside resolving.

After the first `q_expected_lambda` result, the concrete next action is a
bounded matched-budget scale check of that estimator or a return to
search-derived counterfactual advantages if the scale check fails.

After the 10k scale check, the next action should improve the estimator itself:
fuller Expected-SARSA(lambda)/Q-boosting, population/average-policy self-play,
or search-derived counterfactual advantages. Do not interpret near-parity as a
Slumbot-ready agent.

After the target-Q and FSP/PPO failures, prefer search-derived counterfactual
advantages or a dedicated average-policy fit diagnostic over another
policy-only scale run.

After the counterfactual-advantage trust-region and recurrent trace-sequence
diagnostics, the small-data trace-control branch should be retired. Both a
direct one-step advantage update and an ordered GRU update failed the uniform
CFR10 baseline on root-disjoint holdout. The next credible learned-search
attempt needs either a larger root-disjoint trace corpus with the same gate or
a different target that directly fixes early-street blueprint calibration.

The early-street calibration diagnostic now confirms the target is local and
actionable: the current `iter1000` blueprint still fails preflop/flop all-in
readiness on fixed shared states. The next research step should improve the
learned blueprint distribution through calibration or training targets, then
rerun this screen before any further Slumbot confidence run.

That local calibration step now has a bounded positive result. A learned
reference distilled into only the `iter1000` policy head passed the fixed
early-street distribution screen and removed the previous top-action all-in
failure on the diagnostic sample. This should be treated as a gate-opening
diagnostic, not a solved poker agent: because the reference checkpoint still
failed Slumbot, the workflow should next test early-action value transfer before
any live confidence run.

The first such value-transfer check is encouraging but limited. The calibrated
policy head improves the restricted first-action payoff proxy over both the
source checkpoint and learned reference, yet that proxy is a showdown
abstraction. The workflow should now escalate to duplicate-swapped local
self-play or another opponent-response-aware evaluator before live API spend.

The escalation caught the failure. Duplicate-swapped local play shows the
calibrated policy head loses to both the source and reference policies, so the
workflow correctly prevented a premature Slumbot run. This is a useful example
of why single-metric calibration is not enough: distribution repair must be
coupled to full-hand strategic interaction.

The follow-up attribution was especially useful methodologically: it turned a
generic "H2H failed" result into a mechanism call. The problem is not a simple
remaining all-in argmax bug; it is rollout distribution shift from deploying a
separately cloned policy head. Future workflow steps should privilege tests
where the learned object is evaluated on the distribution it induces.

The quick `policy-head-covered` sanity check was worth doing and failed. This
keeps the methodology honest: the issue is not merely using an uncalibrated head
on later streets, so the next direction must change the learning/evaluation
contract rather than the deployment switch.

The student-induced calibration failure is an important negative result. The
workflow avoided knob-tuning and directly tested the covariate-shift hypothesis.
Because that did not pass H2H, the next work should change paradigm toward
coupled regret/value rollout or self-play game-theoretic RL rather than another
reference-cloning variant.

The SD-CFR-style mixture check is the first positive sign after that pivot. A
fixed restored-history checkpoint mixture improved over `iter1000` while staying
near the restored final checkpoint, which supports the principle that the
deployed object should remain coupled to the regret/value network history.
The workflow should now protect this as a deployment/integration question:
review first, implement Slumbot-compatible mixture semantics second, then use a
small smoke to validate action mapping before any live confidence run.

That implementation step is now complete. The adapter keeps the mixture
diagnostic-only, samples one checkpoint per hand, records the chosen checkpoint
in every trace record, and reuses the existing action mapping, range tracker,
and solver path. Both tiny no-solver and solver-enabled Slumbot smokes passed
integration checks. This validates the workflow's review-then-smoke discipline;
it does not validate strategy strength. The next useful work is a fixed
falsification ladder or confidence gate for the same mixture, with no tuning
based on smoke outcomes.

The first queued falsification exposed a gate-hardness issue and the workflow
corrected it before advancing. The holdout mixture mean was positive, but
across-seed lower95 was negative; treating that as a pass would have been
benchmark optimism. The strict multi-seed lower95 requirement now blocks
Slumbot confidence runs until the seed instability is explained or reduced by a
principled training/deployment change.

The immediate decomposition kept the workflow aligned: it diagnosed the mixture
instead of selecting the best checkpoint. Individual checkpoints show that the
late `iter200` snapshot is weak and the sparse weighted mixture is not a robust
average. SD-CFR's related-work contract is to reconstruct average play from the
stored value-network sequence, so the next method should improve snapshot
density/averaging semantics or retrain with explicit SD-CFR deployment in mind.
