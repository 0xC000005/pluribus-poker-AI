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
- External-sampling MCCFR samples chance/opponent actions while enumerating the
  updating player's actions, which is the safer contract for regret target
  ordering: https://papers.neurips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games
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
- Aggregate grid metrics that hide one unstable public state. Learned
  priority-model sampling reached the aggregate sample-4 thresholds, but the
  stricter gate exposed a seed-specific top-action flip and high per-case bias.
- Perfect branch-value priority as a selection signal. The oracle-action-value
  diagnostic is intentionally too expensive for training, but it still failed
  the four-seed sample-4 gate. That means the current estimator and stochastic
  rollout coupling are the bottleneck, not just model capacity.
- Fixed higher sample budgets that recover action ordering by nearly enumerating
  the tree. A four-seed sample-6 grid matched top actions but lost throughput,
  so fixed-count sampling alone is not the target method.
- Worse downstream restricted-action or local H2H evidence at equal compute.
- Any change to Slumbot adapters, promotion logic, legal masks, or parser
  thresholds.

## Next Implementation Step

The cleaner opponent/chance randomization contract is now implemented in the
research-only traversal probe. The rerun did not rescue fixed-count traverser
action dropping: sample-4 was faster (`1.32x`) and low-bias (`0.0219` mean
absolute regret bias), but matched the exhaustive top action on only `2/4`
deterministic roots. Sample-6 and sample-8 matched all top actions only because
they effectively enumerated legal root actions, with mean speedup below `1.0x`.

Do not train a standalone priority model or tune fixed action-sample counts.
That family has now failed under the cleaned-up estimator contract.

Practically, this means the next acceleration attempt should preserve updating
player action enumeration and seek speed from batching, GPU kernels, caching, or
external/chance-sampling variance reduction, not from dropping traverser actions
at the regret-target node.

## Follow-Up: Enumeration-Preserving Compaction

The first follow-up gate supports active-frontier compaction as the next compute
branch. `analyze_traversal_compaction_potential.py` found that the fidelity-clean
GPU traversal artifact has a `7.570839` mean allocated/live slot ratio, zero
accepted overflow, zero pool exhaustion, and an estimated `0.867914` slot
reduction if live nonterminal rows are packed before the next expansion.

The companion CUDA smoke, `run_frontier_compaction_smoke.py`, compacted
1,000,000 synthetic rows at `13.0354%` live density with prefix-sum/scatter,
preserved stable live-row order and row values, and measured a synthetic
`4.178382x` speedup versus full-frontier processing. This is not trainer
promotion evidence; it only authorizes an opt-in traversal-frontier compaction
prototype that keeps traverser-action enumeration unchanged.

The first real trainer integration now supports that conclusion narrowly. The
safe integration path is stable-slot indexing, not physical row movement:
`frontier_indices` compact the live work rows while `parent_idx`,
`child_values`, and dormant traverser nodes keep their original slot IDs. A
seeded 2k CUDA traversal A/B improved traversal time from `3.623748s` to
`1.386260s` with comparable regret-sample scale (`2,423,258` versus
`2,394,481`) and zero overflow. Keep `--use-frontier-indexing` opt-in until a
multi-seed/training-quality gate passes.

A second 2x256 smoke with 50 requested training steps strengthened the compute
claim: measured traversal time dropped from `1.271133s` to `0.426919s`
(`2.98x`) and total measured iteration time dropped from `1.541787s` to
`0.660462s`, again with zero overflow. This still does not prove stronger
poker play; it only justifies using the opt-in path in the next training-quality
gate.

The first normal autoresearch training path now works with the opt-in flag when
using the default `7000` traversal slots. A 3-iteration smoke saved
`frontier_indexed_smoke_7000_final.pt`, emitted machine-readable metrics,
and had zero accepted overflow, zero pool exhaustion, and zero rejected chunks.
Do not interpret the short random evaluation as strength evidence; the next
quality step is a root-disjoint or duplicate-swapped comparison against an
otherwise matched non-indexed training run.

That matched comparison now exists at tiny scale. A non-indexed 3-iteration
control passed the same training gate, and the frontier-indexed checkpoint beat
it in a 1,200-game duplicate-swapped H2H with `+214.0` chips/hand and
across-seed lower95 `+115.2`. This is enough to continue the branch; it is not
enough to promote defaults or spend Slumbot confidence hands.

A longer unseeded 10-iteration matched pair was inconclusive in the opposite
direction: frontier-indexed training was faster but lost the H2H and had two
rejected retry chunks. The correct next falsifier is not another unseeded
quality run. Use the new training `--seed` flag to control initialization and
traversal reset RNG, and increase slots or pool if retry chunks reappear.

A seeded 8-iteration matched pair now gives the first higher-powered quality
support for the opt-in path. With seed `20260520` and `9000` traversal slots,
baseline training reached `876.696` iters/hour and `487.022` traversals/second,
while frontier indexing reached `1355.881` iters/hour and `753.194`
traversals/second. Both runs had zero overflow, zero pool exhaustion, and zero
rejected chunks. The initial 2,000-game H2H was underpowered, but the
40,000-game duplicate-swapped rerun passed with `+108.8` chips/hand and
lower95 `+42.8`. Keep the feature opt-in; the next promotion-relevant evidence
is longer matched training or Slumbot-transfer calibration, not default use.

A larger clean test retired that promotion path for now. With 30 iterations,
4x512 networks, a 3M pool, and 12000 slots, non-indexed baseline training ran
at `86.5` and `91.6` iters/hour on two seeds, while frontier indexing ran at
`356.7` and `376.7` iters/hour. Both clean matched pairs had zero overflow,
zero pool exhaustion, and zero rejected chunks, and a small traversal-signal
parity check passed. Checkpoint quality still failed: seed `20260530` lost the
matched H2H with `-110.0` chips/hand and lower95 `-174.3`; seed `20260534`
was closer at `-13.5` but lower95 was still `-78.4`. Do not queue more
frontier-indexed promotion runs until a new methodology review gives a reason;
return the main research loop to strategy-quality mechanisms.

One semantic correction came out of the failed smoke: CUDA `action=-1` now
means "skip" and no longer advances betting state. This matches the traversal
contract and is unit-tested, but it is a protected behavior correction rather
than a strength result.
