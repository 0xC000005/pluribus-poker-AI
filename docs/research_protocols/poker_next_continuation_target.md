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
