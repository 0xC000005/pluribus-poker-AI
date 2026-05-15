# Sampled-Action Control-Variate Traversal Plan

Date: 2026-05-15

## Objective

Build an opt-in Deep CFR traversal prototype that reduces corrected-semantics
pool overflow by sampling traverser actions with a learned action-value
baseline/control variate. This is a training estimator, not a gameplay policy
or Slumbot-specific patch.

## Method Contract

- Keep the default trainer unchanged until the prototype clears gates.
- At traverser infosets, sample actions from a legal distribution with positive
  mass on every legal action.
- Use inverse-probability correction for sampled actions.
- Use a learned/search-derived baseline for action-value control variates.
- Start with at least four sampled traverser actions per infoset; one- and
  two-action estimates failed the first different-seed Dirichlet robustness
  gate even with stronger restricted-value baselines.
- Write stochastic regret samples to replay; do not use sampled estimates as
  direct one-shot action decisions.

## Related Work Anchor

- VR-MCCFR formalizes baselines/control variates for MCCFR variance reduction:
  https://arxiv.org/abs/1809.03057
- Low-/zero-variance baseline work supports predictive baselines as an
  estimator-quality target: https://arxiv.org/abs/1907.09633
- Without-replacement estimators avoid duplicate samples and can reduce
  variance in discrete sampling: https://arxiv.org/abs/2002.06043 and
  https://arxiv.org/abs/2002.09067
- This project should treat the restricted value baseline as a local proxy for
  those predictive baselines, not as a final poker value model.

## Required Comparisons

- Compare averaged sampled regret targets against exhaustive traverser regrets
  on deterministic small states.
- Require explicit estimator-quality thresholds before trainer integration on
  the selected action-sample budget: `--max-mean-abs-bias 3.0
  --min-mean-estimate-top-match 0.95` on a root-disjoint holdout with a fixed
  baseline checkpoint.
- Report mean absolute regret error, L2 error, top-action agreement of averaged
  targets, per-sample top-action agreement, and estimator standard deviation.
- Report traversal-fidelity metrics if the prototype touches CUDA or batched
  traversal: overflow fraction, pool demand ratio, slots per traversal, and
  throughput.
- Use the same roots, seeds, action abstraction, legal masks, and baseline
  checkpoint across exhaustive and sampled probes.

## Falsifiers

- Legal action with zero sampling probability.
- Lower pool demand achieved by dropping uncorrected actions.
- Averaged sampled-regret top-action agreement below the current restricted
  root diagnostic range (`~96%`) on comparable states.
- Traversal-level sample budget that improves throughput only by breaking root
  action ordering. The first CPU probe found sample-4 faster but top-unstable,
  while sample-8 matched top action but gave little speedup after exact
  enumeration.
- Fixed higher sample budgets that recover action ordering by nearly enumerating
  the tree. A four-seed sample-6 grid matched top actions but lost throughput,
  so fixed-count sampling alone is not the target method.
- Worse downstream restricted-action or local H2H evidence at equal compute.
- Any change to Slumbot adapters, promotion logic, legal masks, or parser
  thresholds.

## Next Implementation Step

Investigate a value- or advantage-priority adaptive sampler before CUDA
integration. Strategy-probability priority did not fix the four-seed traversal
grid; the sampler must protect low-probability actions that can still change
regret ordering.
