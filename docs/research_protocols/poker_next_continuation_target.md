# Next Continuation Target

Date: 2026-05-13

## Decision

Stop scaling the current learned river-leaf dual-CFV objective. The shape-covered
`256/128` resolver-leaf split failed zero/train-constant gates across seeds, and
a wider `hidden_dim=128` sanity check failed harder. The next target should be a
search-integrated public-belief continuation model rather than another
leaf-only value fit.

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
   policy/value rows; it rejects feature mismatches before training.
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
