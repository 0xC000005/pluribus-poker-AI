# Search-Update Selector Innovation Review

## Anomaly Ledger

Repeated learned proxies fit local targets without improving root decisions:
trace-sequence prediction, trace-delta MLPs, EV-factor CFV normalization, and
the first learned mirror-update selector all failed decision gates. The latest
result is sharper: a per-root mirror-update oracle beat CFR10 on a 32-root
holdout (`0.2291` L1 vs `0.3288`), but the learned selector failed on the
96/64 split (`0.5257` L1 vs CFR10 `0.3251`). The unresolved anomaly is not
"can an update help"; it is "what state lets a model know when an update helps."

## Current-Practice Limit

Today the repo either runs exact CFR-style search or trains supervised proxies
to imitate values/policies. Exact search is reliable but costs latency.
Supervised proxies are fast but misaligned with the decision. The next mechanism
must preserve search coupling while learning the missing selector/update state.

Heilmeier check: the objective is a personal-PC poker agent that improves root
decisions with learned search-state updates. The novelty is to learn the update
operator or selector consumed inside search, not the final action. Success would
reduce CFR budget or improve a fixed-budget decision. Risk is selector
overfitting. The midterm exam is root-disjoint CFR10/CFR25 comparison; the final
exam is Slumbot transfer after local gates pass.

## Integrity Check

The strongest reason this idea may be wrong is that the oracle eta may exploit
label leakage from high-budget traces that cannot be inferred from low-budget
state. If richer features do not close the oracle gap, the branch should shift
to exact GPU search acceleration rather than tune the selector.

## First-Principles Reduction

The object to learn is not a poker action. It is a stabilizing feedback law over
search dynamics: given current public belief, legal actions, low-budget policy,
regret/strategy mass, and counterfactual advantage geometry, decide whether the
next update should be conservative, aggressive, or skipped.

## Cross-Paradigm Analogy

Treat search improvement like iterative denoising or control feedback. The
current low-budget policy is a noisy sample of the high-budget fixed point. The
model should learn a state-dependent denoising/control step, while exact search
keeps the update grounded and legal.

## Smallest Decisive Test

Build a richer selector target from dynamic trace state:

- features: the whole iteration `0..5` trajectory, advantage geometry,
  uncertainty/margin, and public-belief summaries;
- target: per-root oracle update family or direct improvement over CFR10;
- gate: root-disjoint learned selector must beat both CFR5 and CFR10 in L1 and
  top-action match against CFR24/CFR50 reference;
- fail action: stop learned selector work and prioritize fused exact GPU search.

## Decisive Test Result

The richer selector test failed. `train_cfr_trace_oracle_eta_selector.py`
computed finite-grid per-root oracle etas and trained a deterministic
per-eta-loss selector from the full early trajectory, public-belief summaries,
and advantage geometry. It recovered train roots exactly, but failed held-out
roots:

- 32/32 split: selected `0.5016` L1 versus oracle `0.2291`, CFR5 `0.5102`,
  and CFR10 `0.3288`; selected-oracle eta match `0.15625`.
- 96/64 split: selected `0.5827` L1 versus oracle `0.3041`, CFR5 `0.4817`,
  and CFR10 `0.3251`; selected-oracle eta match `0.1875`.

Interpretation: the mirror-update family remains a positive-control signal, but
the static selector interface is not the right learned object. The next
innovation step should not be a regularization or threshold sweep. It should
either fuse/accelerate exact GPU search, or learn a solver-dynamics state
transition consumed inside the resolver so the model is trained on the same
closed-loop object it will control.
