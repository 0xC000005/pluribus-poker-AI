# ED (#40) v1 Write-up Evidence Pack
Assembled 2026-06-09 (post commit 8def574). Single source of truth for the v1 arXiv/workshop artifact.
Target: priority-securing arXiv preprint + workshop/AAMAS-short. v2 (main-track) upgrades AFTER the
timeboxed >=3-level demo (D). All numbers verified by independent audit (re-derived from raw JSONs).

## The claim (one sentence)
Cross-key GEMM-fusion of the POPULATION of same-topology depth-limited PBS subgames — the inner re-solve
of train-time iterated continual resolving — is an unclaimed, hardware-independent cost-per-solve lever
that trains a general tabula-rasa self-play resolving agent ~10x faster to the SAME exploitability band
than within-tree GPU-CFR, with exact equilibrium parity.

## Lever taxonomy (lead with this, not the raw speedup)
iterated-resolving cost = (#solves) x (iters/solve) x (cost-PER-solve)
- #solves: TurboReBeL (ICLR'26, OpenReview yMo7Z670f6; poker-specific isomorphisms, 4xA100)
- iters/solve: DCFR / PCFR+ / NPCFR (2504.18917; meta-predictor over a distribution of subgames, applied SEQUENTIALLY — orthogonal+stackable) / Deep-Predictive-DCFR (2511.08174, AAAI'26) / AlphaEvolve-discovered variants (2602.16928)
- samples: VR-MCCFR / ESCHER / DREAM
- subgame SIZE: LAMIR (2510.05048, ICLR'26, single A100 <=8GB)
- cost-PER-solve (UNCLAIMED 4th lever — this work): population-GEMM fusion

## Evidence chain (all in repo; JSONs under autoresearch-session/rebel/, gitignored)
1. MICROBENCHMARK (commit c47cc19; costpersolve_bakeoff.json): fused solve_all_keys_soa over all keys vs
   the SAME SoA kernel per-key (fair within-tree-GPU-CFR baseline; Python walk rejected as strawman).
   G3(B=27) 8.77x, G4(B=64) 11.69x, G5(B=125) 2.37x; exact parity 0-7e-6 (max observed 4e-6).
   CONTROLLED population sweep (fixed subgame size, vary #co-solved): G4 fused time ~CONSTANT
   (197.6->202.4ms) as N: 1->12 => speedup ~= N (1.0/2.0/3.96/7.9/11.7x near-LINEAR); G5 fused time GROWS
   (291->1870ms) => saturates ~2.4x. REGIME MODEL: speedup ~ min(N, GPU-capacity/subgame-occupancy) —
   launch-overhead amortization; near-linear in the launch-bound many-small-subgames regime (= what
   iterated resolving structurally generates), saturating when one subgame fills the GPU.
2. END-TO-END WIN (commit 8def574; b0_e2e_g4_{fused,sequential}_seed{0..4}.json + e2e_band_summary_g4.json):
   G4, 5 matched seeds, ONE config, median-of-per-seed-ratios:
   - inner-resolve 11.79x; TOTAL-WALL 9.99x; inner-resolve = 98.4% of sequential baseline loop;
     Amdahl ceiling 10.09x ~= measured 9.99x (the regime model PREDICTED the e2e outcome).
   - band equivalence: fused final nash_conv median 0.0402 [10/90: 0.0238, 0.0533]; sequential 0.0469.
   - gates: CPU exact-parity unit test (test_fused_vs_sequential_soa_parity, strategy+leaf <1e-9);
     CPU determinism run: identical trajectories to all recorded digits.
   - CUDA finding: index_add_ atomics COMPOUND through training -> per-seed fused-vs-seq |diff| median
     0.0144 max 0.0389 = atomic noise, NO systematic bias (2/5 fused>seq); equivalence proven on CPU.
   - time-to-band tau-sweep PERSISTED: 5.5-11.4x (threshold-sensitive; headline = total-wall 9.99x).
3. SEARCH-FREE BOUNDARY (commit 8def574; searchfree_nfsp_g4_seed{0,1}.json): NFSP average-policy (fair;
   RPG/QPG current-policy ~uniform 1.4334 vs uniform 1.4167 = strawman REJECTED — methodology point worth
   one paragraph): 600s/~640k eps -> 1.3968/1.3966 vs resolving band 0.04 in ~77s. DIRECTIONAL only
   (NFSP lit needs 1e6-1e7 eps); VRPO (2605.19235, ICLR'26, beats Slumbot search-free) = CITED boundary,
   not reimplemented.
4. EXPLOITABILITY BAND at scale (commit 1ff6a8e; b0_perbelief_g5_curve*.json): per-belief V* targets
   (cheap BECAUSE of the fused solver) robustly beat the 0.21 fixed-continuation floor at G5/236k ->
   0.07-0.13 band; NO reproducible improves-with-target-compute curve (init/opt-noise dominated below
   ~0.13). Report exploitability as BAND with >=5-seed error bars. [B+ adds seeds 2-4 + G5 e2e both arms
   — APPEND RESULTS HERE WHEN LANDED.]
5. SCALE LADDER context (commits 041fc57/6b20c98/7ff3042): G4 near-Nash (0.0145 fixed-cfr+ targets;
   ~0.04 V*-net); G5/236k = the band; G6 build >280s (full-enumeration substrate ceiling ~236k; the
   SOLVER scales — 12ms/iter at G5 — the BUILD does not; lazy/SoA build = future work).

## Figures (generate from JSONs)
F1: controlled keysweep — speedup vs #co-solved subgames, G4 + G5 (the regime model picture).
F2: e2e wall-clock bars per arm (inner/train/measure stacked), 5 seeds, with Amdahl-ceiling overlay.
F3: band distributions (fused vs sequential finals, G4) + per-seed paired lines; CUDA-noise inset.
F4: exploitability-vs-wallclock overlay: fused arm curve, sequential arm curve, NFSP curve (log-x).

## Honest-scope paragraph (MUST appear; pre-agreed wording basis)
Demonstrated loop is 2-LEVEL/single-cut (per-belief targets independent given the belief); >=3-level
self-play CORRECTNESS was certified earlier (iig_selfplay on G4: exploit decreases 0.0506->0.0145 with
training, EB entry 20260608T01/02) but multi-level fused-vs-sequential TIMING is deferred (v2, timeboxed).
One game family (Goofspiel ladder; Leduc/liars-dice parity-tested for the solver but not e2e-timed).
One device (RTX 3070 Ti 8GB; regime model is hardware-independent in form — A100 replication = cheap v2
add). Exploitability noise: bands not curves; speedup claims are time-based, robust to it (both arms
share the seed). 2.4x saturated-regime bound stated as a CONTRIBUTION (predictive cost model), not hidden.

## Reviewer-objection preempts
- "Just batching underutilized GPU work": conjunction defense — (a) identifying the population structure
  inside iterated resolving, (b) exact-float64-parity fused CFR+ over variable-depth/chance/imperfect-recall
  subgame populations (regret+range state per subgame — NOT a trivial vmap), (c) predictive regime model,
  (d) Amdahl-survived e2e 10x in a real training loop. Plus: ZERO citing papers on Kim 2408.14778 (checked
  2026-06-09) — nobody has built population batching on the GPU-CFR line.
- "Batched solving exists" (Marris NeurIPS'22 2210.09257): learned amortized solver on NORMAL-FORM payoff
  matrices vs exact-parity CFR+ on EXTENSIVE-FORM depth-limited PBS subgames inside a training loop.
- "DeepStack already batches": net-eval batching != solve batching (one sentence).
- "Gadget equilibria are non-unique" (Kubicek 2601.17131, must-cite): our band claims use the on-policy
  re-solve continuation; gadget numbers reported alongside; acknowledge refinement-choice sensitivity.
- "Why not VRPO comparison": cited boundary; reimplementation invites unfair-tuning attack; the load-bearing
  baseline is the sequential within-tree GPU-CFR arm INSIDE every experiment (same kernel, exact parity).

## Citations (verified live)
Kim GPU-CFR 2408.14778 (+ zero-citations audit point); Kim&Sandholm 2605.14277 + NoRegret/gpugt lib
(v0.0.0.dev15, 2026-06-04 — ACTIVE line, scoop risk, cite the lib); TurboReBeL (ICLR'26 OpenReview);
LAMIR 2510.05048; NPCFR 2504.18917; USPTO US11204803 (task-distribution not GEMM-fusion); VRPO 2605.19235;
ReBeL 2007.13544; DeepStack 1701.01724; Student-of-Games/PoG; Brown&Sandholm 1805.08195 (depth-limited
unsoundness); Burch/Moravcik safe re-solving; NFSP 1603.01121; RPG/QPG 1810.09026; Kovarik 1906.06412
(net-error->exploitability, coherent-targets caveat); Kubicek/Lisy/Sandholm 2601.17131 (refinements);
exp-a-spiel 2502.08938 (scale standard); Pasur-GPU 2508.06559 (single-tree GPU-CFR data point);
Marris 2210.09257 (batched normal-form); CDBR 2112.12594; Srinivasan 1810.09026; OpenSpiel 1908.09453.
Verified live 2026-06-09 (were missing IDs in drafts): Student of Games 2112.03178 (Science Advances;
Schmid et al.); ESCHER 2206.04122 (ICLR'23, McAleer et al.); VR-MCCFR 1809.03057 (AAAI'19, Schmid et al.);
DREAM 2006.10410 (Steinberger/Lerer/Brown).

## Writing constraints
- Aggregation convention stated explicitly: median of per-seed ratios.
- "Identical to all recorded digits" (NOT "byte-identical").
- Never present the killed G5 run's 0.1162 as a result row (superseded by B+).
- No claims of lower exploitability — EQUAL-band-FASTER only.
- Slumbot held-out everywhere; uses_slumbot_data=false.
- Hardware framing: consumer-GPU is table-stakes (LAMIR), 4xA100 near-Nash is non-novel (TurboReBeL);
  the lever is per-FLOP / hardware-independent; multi-GPU EXTENDS it (population shards trivially).


## Corrections and additions (r4 editorial audit, 2026-06-10)

1. **The "12 ms/iter at G5 in-loop" figure (item 5 above) is STALE and superseded.** It predates the
   in-loop instrumentation and contradicts the released artifacts. Artifact-derived value:
   ~6.2 ms per CFR+ iteration in-loop INCLUDING leaf reconstruction —
   `b0_e2e_g5q_fused_seed{0..2}.json` `inner_resolve_wall_s_per_round` 74.81–74.98 s per
   40-belief round at 300 iters/solve => 6.23–6.25 ms/iter; `b0_perbelief_g5_curve_seed2.json`
   rounds 74.71–75.05 s => 6.23–6.25 ms/iter; pure-solve microbenchmark
   `costpersolve_bakeoff.json` rows[2] 1866.7/300 = 6.22 ms/iter. The paper (r4) states
   ~6.2 ms in-loop, indistinguishable from the microbenchmark; the in-loop figure is therefore
   artifact-backed and no longer rests on this pack.
2. **Float32-ablation drift (~0.75 L1 in average strategy at repeated imperfect-information
   infosets), quoted in paper §3.2, is added to this notes file** as a documented-but-not-yet-
   artifact-backed number (now the fifth entry of the paper's §7 list). The originating run
   record (game, run count) is still to be attached; treat as a single-configuration ablation
   note until a standalone artifact is re-derived at camera-ready.
