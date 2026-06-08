# Contribution phase goal — make the scale lever DEFENSIBLE (A → B → C → D)

Date: 2026-06-08. Owner-committed sequence (post-port recap+decide, evidence in
`autoresearch-session/rebel/post_port_recap_decide.json`). The batched-subgame solver is BUILT + demonstrated
at 236k infosets on one 8GB GPU, but the contribution is NOT yet defensible: (1) the headline exploitability
is a confounded off-path UPPER bound (assembled NashConv froze a single continuation; rose 0.38→0.49 with
iters), and (2) the tabula-rasa SELF-PLAY pillar (a learned PBS value-net leaf) is absent at scale
(`trunk_solve_batched` does exact per-belief re-solve; `iig_selfplay` is decoupled + 2-level-only). This
phase makes the EFFICIENCY claim MEASURABLE (A) and delivers the SELF-PLAY essence (B), then scales (C) and
writes up (D). Sequence is COUPLED A→B→C→D (not parallel): per Kovařík 2019 (arXiv:1906.06412), depth-limited
exploitability is ~linear in value-net error, so the clean metric (A) is the instrument that certifies the
learned leaf (B). Slumbot stays HELD-OUT throughout (the minimal claim does not need it).

## Definition of Done (gates; the loop runs until EA–ED pass)

- **EA — honest exploitability evaluator.** A continual-resolving best-response evaluator (CDBR-style;
  Milec/Kubíček/Lisý AAMAS 2024, arXiv:2112.12594) that re-solves at each reached belief using the agent's
  own leaf, giving a bound that DECREASES with compute. VALIDATE it against an exact best-response-vs-
  resolver (OpenSpiel sequence-form) on Leduc + Goofspiel-4 (gold standard, <100s CPU), then apply at
  Goofspiel-5. LBR (Lisý & Bowling 2017, arXiv:1612.07547) as the cheap step-0 probe. Re-state every prior
  number through it. ACCEPT: evaluator matches exact BR on small games to <1e-3; produces a monotone-with-
  compute bound; the G5 number is re-stated honestly (and: does a SAFE-RESOLVING gadget lower it?).
- **EB — PBS value-net self-play.** Wire a game-agnostic PBS value net (public state + both ranges → per-
  infostate CF values) as the depth-limit LEAF in `trunk_solve_batched`, REPLACING exact re-solve, with
  `solve_all_keys_soa(return_cont=True)` as the fast target generator. Add the self-play outer loop +
  bootstrapping on a ≥3-level game. ACCEPT: exploitability-vs-training-steps curve (measured by EA's
  evaluator) DECREASES; net leaf reaches within a small factor of the exact-re-solve agent; works on a
  ≥3-level game. (Watch: may need a safe-resolving gadget the batched path lacks.)
- **EC — scale ladder, certified.** Extend toward standardized large games (the exp-a-spiel suite is 6M–27M
  infosets; Rudolph et al. ICML 2025, arXiv:2502.08938) ONLY to rungs EA can still certify near-Nash;
  report throughput + VRAM per rung; characterize where the 8GB GPU caps.
- **ED — write-up + baselines.** Position the cross-key population-GEMM as the orthogonal THROUGHPUT axis;
  cite + differentiate up front (named cross-tree-vs-intra-tree figure): Kim GPU-CFR (2408.14778 2024;
  2605.14277 2026 — within-one-tree), TurboReBeL (ICLR'26 — iteration-amortization), LAMIR (2510.05048 —
  abstraction-size). Required baselines: a tuned model-free PG/VRPO (Fan & Farina 2605.19235) + GPU-batched
  tabular CFR. One small-poker instance (Leduc / Turn-Endgame-Hold'em) as the realism cap — NOT full
  HUNL-vs-Slumbot.

## Engineering constraints (carried from the port gates)
- float64 regret accumulation mandatory (float32 drifts 0.75 L1 at imperfect-info repeated infosets).
- CUDA index_add_ atomics non-deterministic (~1e-3 cuda-vs-cpu drift at near-ties; CPU parity exact).
- ~1000x throughput is the small/medium-private-dim regime (Leduc/Goofspiel); HUNL large-P needs the
  belief-batch / shared-board-matrix variant → contribution = general small/medium-P crossing, NOT HUNL.

## Novelty positioning (verified OPEN but narrow)
Contribution = (cross-key population-of-depth-limited-PBS-subgames GEMM) + (sound clean metric) + (general
substrate) INSIDE a ReBeL-style learned-PBS self-play loop. The self-play loop itself is NOT novel
(ReBeL/DeepStack/PoG); state the contribution narrowly or it reads as re-derivation of Kim/TurboReBeL.

## Citation hygiene (from the verification pass)
CDBR 2112.12594 = AAMAS 2024 (+ follow-up 2501.10464); exp-a-spiel 2502.08938 = ICML 2025 (NOT ICLR'26);
VRPO 2605.19235 + LAMIR 2510.05048 ARE ICLR'26; keep the two Kim papers' speedups distinct; Kovařík
1906.06412 (value-net-error→exploitability); Modicum 1805.08195 (depth-limit ancestor); LBR 1612.07547.
2601.17131 (Equilibrium Refinements) NOT independently verified — check before relying.

## Scope (honest, unchanged)
Contribution = ~1e3 → ~1e6-1e7 (realistically 1e5-1e6) infosets SOUND + GENERAL crossing on one consumer
GPU + the batched-subgame mechanism. Full HUNL near-Nash on 8GB is a STRETCH, not the bar.
