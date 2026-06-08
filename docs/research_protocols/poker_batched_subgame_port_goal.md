# Port Goal — GPU-resident cross-key batched depth-limited PBS solver

Date: 2026-06-07. Owner-set goal: drive this port to completion (keep running until the Definition-of-Done
gates G1–G5 all pass). Premise validated across 4 gates (RESEARCH_LOG 2026-06-07 GATE 0/0b/0c/0d):
subgames batch (1 topology/game), batching gives ~1000× on small subgames, the kernel is exact on real
chance/variable-depth/imperfect-info structure (float64), and it integrates into `trunk_solve` preserving
the equilibrium. The throughput win is still LATENT because 0b/0c/0d batched only within one key's cut
nodes (small B) and 0c/0d kept the per-iteration Python tree walk. The port realizes the win.

## STATUS (2026-06-08): G1 ✅ G2 ✅ G3 ✅ G4 ✅ (scale headline) · G5 closing

G1 flat-SoA kernel exact parity (Leduc/G4/Liar's-Dice); G2 54.7× throughput on G4; G3 trunk integration
exact + 32.5× end-to-end; G4 Goofspiel-5 (236k infosets) solved depth-limited on the 8GB GPU (best scale
lever: 0.38–0.49 vs MCCFR 0.93 / Deep CFR 1.05; assembled-NashConv caveat — off-path upper bound, true
continual-resolving exploitability is the top open item). See poker_next_continuation_target.md (2026-06-08)
for the open-work list. Module: poker_ai/rebel/iig_batched.py; tests: test/unit/test_rebel_iig_batched.py.

## Objective

A generic, GPU-resident, **cross-key** batched depth-limited PBS subgame solver in
`poker_ai/rebel/iig_batched.py`, integrated into the self-play loop, that scales the validated general
2p0s-IIG method past the unabstracted CPU ceiling (~few-k infosets) on ONE consumer GPU (RTX 3070 Ti, 8GB)
— without sacrificing generality (game-agnostic, tabula-rasa) or soundness. Slumbot stays held-out.

## Definition of Done (the loop runs until ALL pass; each is a commit milestone)

- **G1 — Compiler + kernel parity.** A flat-SoA compiler turns the shared subgame topology (per Probe A,
  one topology per game) into level-ordered tensor arrays, and a level-synchronous batched CFR+ kernel
  runs with NO per-iteration Python recursion, batched over ALL cut nodes across ALL public keys. Its
  equilibria match numpy `solve_subgame_equilibrium` to max strategy L1 < 1e-4 (float64) on **Leduc,
  Goofspiel-4, and Liar's Dice** (poker + non-poker; chance + variable depth + imperfect-info). New unit
  test `test/unit/test_rebel_iig_batched.py` green.
- **G2 — Throughput.** On Goofspiel-4, the GPU cross-key batched solve of all public-state subgames is
  ≥50× faster (wall-clock) than the serial numpy per-key `solve_subgame_equilibrium` loop, and the
  speedup grows with the number of keys. Recorded in RESEARCH_LOG.
- **G3 — Integration.** `iig_selfplay` self-play using the GPU batched solver as the per-belief re-solve
  leaf reproduces the GATE-0d Goofspiel-4 assembled NashConv (|diff| < 1e-3) AND its end-to-end iteration
  is faster than the serial path. New self-play path tested.
- **G4 — Scale (headline).** Solve **Goofspiel-5** (236k infosets — past the dense enumeration/CPU ceiling
  that capped every prior lever) via depth-limited self-play to a NashConv within a small factor of the
  cfr_plus dense baseline, in feasible wall-clock on the 8GB GPU. This is the ~1e3 → ~1e5+ crossing, the
  contribution's headline. Gated on exact NashConv where computable; characterize the scaling curve
  honestly (it may land at 1e5 not 1e7 — report what it actually reaches).
- **G5 — Regression + docs.** All `test/unit/test_rebel_*` stay green; the new compiler/kernel are tested;
  RESEARCH_LOG + this doc + the continuation-target doc reflect the final state. Committed.

## Increments (build → verify → commit each)

1. **SoA compiler + flat cross-key kernel** → G1. The crux. Base it on the proven-correct GATE-0c walk
   (`run_rebel_batched_generic_subgame.solve_key_batched`): compile the shared topology once into
   level-ordered arrays recording, per topology-node per batch-element, the global-iid index, terminal
   payoffs, and per-element chance probs; then iterate CFR+ as gather/scatter tensor ops over levels.
   Parity-gate against numpy on Leduc/G4/Liar's-Dice.
2. **Throughput measurement** → G2. Cross-key batch all keys; GPU; float64 accumulators (fp16/bf16 matmul
   optional on top). Compare wall-clock vs serial numpy on G4.
3. **Self-play integration** → G3. Drop the GPU solver into `iig_selfplay` as the re-solve leaf; confirm
   0d-equivalent NashConv + speed.
4. **Scaling ladder** → G4. G4 → Goofspiel-5; gate on exact NashConv; characterize the curve.
5. **Tests + docs + cleanup** → G5.

## Engineering constraints (from the gates)
- **float64 regret accumulation is mandatory** (0c: float32 drifts 0.75 L1 at imperfect-info repeated
  infosets). fp16/bf16 matmul is fine only with a float64 accumulator.
- The generic substrate folds private into iids (iid = info_state incl. private); cut nodes within a key
  share iids, across keys differ → the kernel's regret space is the union of all keys' below-cut iids.
- Chance probs/outcomes can differ per batch element (Leduc flop deal) → store per-element chance probs.
- Imperfect-info repeated infosets (simultaneous-move Goofspiel) → an iid appears at multiple topology
  positions; accumulate (scatter-add) correctly (0c verified the logic).

## Honest scope
Contribution = the ~1e3 → ~1e6-1e7 (realistically maybe 1e5-1e6) infosets SOUND + GENERAL crossing on one
consumer GPU + the batched-subgame mechanism (novelty OPEN; cite TurboReBeL/LAMIR/AlphaHoldem/Modicum/
VRPO). Full HUNL near-Nash on 8GB is a STRETCH, not the bar. If G4 caps below expectations, that ceiling
IS a publishable characterization. Pause for the owner only on genuine blockers or scope decisions.
