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
- Write stochastic regret samples to replay; do not use sampled estimates as
  direct one-shot action decisions.

## Related Work Anchor

- VR-MCCFR formalizes baselines/control variates for MCCFR variance reduction:
  https://arxiv.org/abs/1809.03057
- Low-/zero-variance baseline work supports predictive baselines as an
  estimator-quality target: https://arxiv.org/abs/1907.09633
- This project should treat the restricted value baseline as a local proxy for
  those predictive baselines, not as a final poker value model.

## Required Comparisons

- Compare averaged sampled regret targets against exhaustive traverser regrets
  on deterministic small states.
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
- Worse downstream restricted-action or local H2H evidence at equal compute.
- Any change to Slumbot adapters, promotion logic, legal masks, or parser
  thresholds.

## Next Implementation Step

Create a research-only CPU probe that runs exhaustive and sampled traverser
regret collection on a tiny deterministic state set. Use the XL restricted
value baseline as the first control variate, then replace it with search-derived
action-value labels if the restricted baseline fails the exhaustive comparison.
