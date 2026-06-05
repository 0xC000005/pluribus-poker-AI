# Net-only self-play hits an exploitability ceiling on single-GPU HU NLHE — and the case for search-in-learning

**Status:** consolidated finding, 2026-06-04 (updated with §5b ReBeL de-risk Stage 0/1 results).
Authoritative; supersedes scattered RESEARCH_LOG entries 20260603T133428Z → 20260604T023839Z. Single
machine (RTX 3070 Ti, 8 GB), tabula-rasa, Slumbot held-out.

## Headline

A method-sound, tabula-rasa **net-only** R-NaD self-play policy for heads-up no-limit Texas Hold'em
plateaus at **~12–15K mbb/g LBR exploitability across six independent levers** on a single consumer GPU
— roughly **3× more exploitable than Slumbot (~4K)** and far from near-Nash (~0). The exploitability does
**not** move with capacity, optimization, representation, or readout-policy changes. The proven, net-centric,
**efficient** route to near-Nash HU NLHE is **search integrated into the learning loop (ReBeL / Student of
Games)** — which this repo already contains ~80% of. Decision: pursue ReBeL-style search-in-learning,
small-game-first; the SOTA-efficiency contribution is making it fit one consumer GPU.

## 1. The trusted ruler (method)

Prior selection rulers were discarded as untrustworthy:
- **H2H gauntlet** (duplicate-swapped lower95 vs a frozen population) is **non-transitive** — a candidate beat
  NFSP/K-best, tied the incumbent, and lost to the router/worst-gap. Retired as a *selection* ruler.
- **BR-LB** (learned-Q best-response proxy) **inverted** — ranked an always-all-in shover as *less* exploitable
  than the incumbent (failed its positive control). Demoted.

Two trusted gates were established:
- **Exact last-iterate NashConv** on small games (Kuhn/Leduc/small-NLHE) via OpenSpiel — positive-control
  validated (random/fold/shover rank 0.67–4.75 ≫ CFR+ ~0.002), uninvertable. R-NaD is method-sound here
  (small-NLHE 0.062; Leduc descends to ~0.39 at 120k).
- **LBR (Local Best Response)** — `scripts/run_lbr.py`, a one-ply best-response *lower bound* on full-deck
  exploitability (mbb/g). Positive-control gate **passes at N=500**: shover +11.7K, always_fold **+750
  (the textbook cost of folding — confirms units/payout)**, calling_station +33.7K, all lower95 > 0; a
  tighter reference policy +1.3K ≪ the degenerates. **A decode fix was load-bearing**: the policy must be read
  out as **masked-softmax** (R-NaD's actual deployed policy), not `regret_match` — verified 0.0 vs the gauntlet
  adapter; the wrong decode had inflated R-NaD from a misread +28K to the correct ~12K. LBR is a *lower bound*:
  a HIGH value refutes near-Nash; a LOW value is consistent-with but not proof-of near-Nash.

## 2. The six-axis net-only ceiling (result)

R-NaD (DeepNash recipe: fixed η=0.2 + increasing reference-reset windows, cix=0), full-deck HU NLHE,
LBR full-9 action set. Each lever moved independently; exploitability is flat:

| Lever | setting moved | LBR (mbb/g) | N |
|---|---|---|---|
| training budget | 12k → 24k → 48k → 96k iters | 12,992 / 10,648 / 16,013 / 12,817 | 200 |
| batch size | 512 → 4096 (GPU collector) | 13,987 → 12,529 / 13,063 | 500 |
| gradient clip | 10,000 (≈off) → 1.0 | fixed the 96k NaN; LBR unchanged | 500 |
| reset density | coarse → denser increasing schedule | within ~13K | 500 |
| representation | flat MLP → AlphaHoldem-style CNN encoder | 14,971 / 14,862 | 500 |
| readout policy | EMA last-iterate → SL time-average | 12,528 → 11,476 (overlapping CIs) | 500 |

η-annealing (the QRE→Nash regularization lever) is **pre-falsified in-repo** (decaying α lets gradient noise
dominate; NashPG) and excluded. The capacity axes *and* the two floor-setting axes the literature flags
(readout, regularization) were both tested. **The ceiling is defensibly real**, not a one-config artifact.

## 3. Relative soundness + calibration

LBR is transitive (same instrument, same config), so the *ordering* is trustworthy even where absolute values
are loose:

| Agent | LBR (mbb/g) |
|---|---|
| greedy Rainbow **incumbent** | ~27,975 |
| policy **router** | ~15,886 |
| **R-NaD (net-only)** | ~9,600–13,700 |
| Slumbot (calibration; ACPC abstraction-CFR) | ~4,000 |
| ReBeL (search-in-learning) | ~881 |
| DeepStack / near-Nash | ~0 |

So stochastic R-NaD is the **least-exploitable agent in the local league** (sounder than the greedy incumbent
it *draws* in H2H — the gauntlet "draws" were relative soundness, not weakness) — but it is **not near-Nash**,
and the gap to Slumbot/ReBeL/Nash is the search gap, not a tuning gap.

## 4. Interpretation

Net-only model-free self-play reaching near-Nash HU NLHE is only known at **cluster scale** (DeepNash). On a
single consumer GPU it is capped (§2). The proven, net-centric path to near-Nash HU NLHE is **search
integrated into the learning loop** — ReBeL (NeurIPS 2020, ~881 mbb/g LBR) and Student of Games — which is the
AlphaZero philosophy for imperfect information (the network is the agent; search sharpens its training targets,
*not* a play-time bolt-on). This is **more efficient per unit of soundness** than net-only, because in-loop
search manufactures better targets. (Out of scope: Modicum / blueprint + play-time-only CFR, which demotes the
net to a range-prior.)

## 5. Decision + documented foundation for ReBeL

**Pursue ReBeL-style search-in-learning, small-game-first.** Not a pivot away from prior work — the repo
already contains ~80% of it:
- `scripts/fast_cfr.py::solve_cfr` (CPU) computes per-infostate counterfactual values, propagates the
  belief/range pair (= the public belief state), and exposes the learned-leaf hooks (`cut_node_fn` =
  ReBeL's `SetLeafValues`), per-iteration capture (`iteration_update_fn`), and warm-start args.
- `poker_ai/research/belief_value_probe.py::_HandCFVProbeNet` ≈ the PBS value net (belief→per-hand CFV).
- Open-loop CFV-target-harvest prototypes + exact Leduc NashConv harness + an exact leaf oracle exist.
- **New work = closing the loop** (~5 small files: `pbs_value_net`, `loop`, `leaf_eval`, `targets`, a Leduc gate runner).

**Plan: sound-with-fixes.** Must-fixes before trusting any gate:
1. Pin the CFV **normalization convention** (reach-weighted vs normalized) and verify with a **numerical oracle
   round-trip**, not a shape test.
2. Make value-average / policy-average / SampleLeaf weights **identical** (CFR+ here averages uniformly → use
   uniform t-sampling; defer linear-CFR + its protected-surface edit to scale-up).
3. Reframe the PASS criterion to the **oracle-leaf-control gap** (learned ≈ oracle → CFR+ ~0.002), not "beat R-NaD".

**Gating (small-game is a kill-switch, not a green light):** Leduc exact-NashConv → small-NLHE →
flop-truncated HUNL (a real >100-dim belief) before any multi-week scale-up. **First step (~½ day, no
training):** the constant-oracle round-trip on Leduc. **Full-deck data-gen is feasibility-marginal** on the
3070 Ti (CPU-only leaf hooks) — the GPU leaf-hook port is the headline **SOTA-efficiency** engineering, gated
behind the small-game proof.

## 5b. De-risk results — Stage 0 PASS, Stage 1 findings (2026-06-04)

Built small-game-first on the **trusted** Leduc tree (all game logic derived from OpenSpiel; new
code in `poker_ai/rebel/{leduc,leaf_eval,loop}.py`). The plan deliberately did **not** build on
`fast_cfr.solve_cfr` (HU-NLHE-specific, CPU-leaf-only) — it used the same exact belief-weighted-Q
recursion as the trusted small-game gates, so the value layer is verifiable against OpenSpiel
NashConv.

**Stage 0 — constant-oracle round-trip: PASS** (`scripts/run_rebel_leduc_roundtrip.py`,
`test/unit/test_rebel_leduc_roundtrip.py`, 4 tests green). Resolves both correctness must-fixes:
- *Must-fix #1 (CFV convention).* The leaf evaluator produces, and the trunk consumes, the **pinned
  normalized convention** (`leaf_eval.py` docstring): `v_i(c) = E[u_i | i holds c, opponent range]`,
  the opponent-range expectation normalized by opponent reach. The numerical round-trip reconstructs
  the exact round-1 q-values to **8.9e-16**; the un-normalized (counterfactual) convention consumed
  as normalized breaks it (error **2.24**) — a real negative control.
- *Must-fix #2 (averaging).* CFR+ with **uniform/own-reach** averaging (linear kept only as a probe)
  converges; linear CFR+ hits NashConv **0.0017** — proving tree + values + regrets are exact.

**Stage 1 — depth-limited solving findings** (`scripts/run_rebel_leduc_stage1.py`,
`autoresearch-session/rebel/leduc_stage1_findings.json`). Four measurements that **reshape the
scale-up design**:
1. **Exact-oracle control = full CFR+ / co-evolving CFR-D → NashConv 8.2e-4.** This is the
   depth-limited control ceiling; must-fix #3's "oracle-leaf-control gap" is the gap to *this*.
2. **Isolated subgame re-solving is unreliable** — both strategy and value. A frozen, isolated
   round-2 strategy is off-path exploitable (assembled NashConv **0.21** vs 8e-4). Its per-hand
   *values* match truth only at the most-reached entries and diverge as reach drops (≈**0.09** at
   >10% reach, ≈**0.77** at ~4%), **stable across 3k/12k/40k iters** → the thin-reach values are
   *under-determined* (equilibrium multiplicity), not slow convergence.
3. **Per-iteration re-solve biases the trunk.** Using a re-solved-equilibrium leaf each trunk
   iteration lets the opponent re-adapt inside the leaf value → biased regrets; the trunk fails to
   recover the equilibrium round-1 strategy (round-1 L1 **0.15**).
4. **Therefore (design constraint for Stage 2):** the value net must be a **fixed PBS-value
   function during each solve** (co-evolving CFR-D, *not* isolated equilibration); value **targets**
   must be the self-consistent CFVs read off the trunk solve (not isolated re-solves), averaged over
   the visited PBS distribution; **play must continually re-solve** for the actually-reached range.
   This is exactly the ReBeL/DeepStack design — now empirically motivated on our own substrate.

**Refined gating.** Stage 2 (Leduc, the next build): train a PBS value net on trunk-solve CFV
targets, plug it in as a fixed leaf in a CFR-D depth-limited solve, and require the learned-leaf
NashConv to stay within a small gap of the **8.2e-4** oracle control. Then small-NLHE →
flop-truncated HUNL before any multi-week scale-up. The under-determination of thin-reach leaf
values (#2) is a noted scale-up risk to watch (mitigated by averaging + net smoothing), not a
blocker — Leduc still solves to 8e-4 via proper CFR-D.

**Stage 2 results (2026-06-04) — two findings, no clean pass yet (a research juncture).** Built a
Leduc PBS value net (`poker_ai/rebel/pbs_value_net.py`; tiny MLP, reach-weighted MSE) used as a
fixed leaf in `loop.trunk_solve`. Metrics: A = held-out reach-weighted CFV MAE; B = round-1 strategy
L1 between the net-leaf trunk and the matching exact-leaf trunk.
- **(1) Learnability depends on target consistency** (Stage-1 #2, confirmed end-to-end): isolated
  per-range round-2 re-solve targets are under-determined across ranges → the net **underfits**
  (train MAE 0.16 ≈ val 0.16). A **consistent** blueprint-continuation target (CFV of continuing
  with one fixed near-equilibrium round-2 strategy) is **learned well** (train 0.048, val 0.055).
- **(2) Consistency↔quality tradeoff:** the per-range-resolve value is a *good* leaf (its exact
  trunk is 0.087 round-1 L1 from full CFR-D) but unlearnable-as-is; the blueprint value is learnable
  but a *crude* leaf (exact trunk 0.264 from full CFR-D — round-2 can't adapt to the trunk ranges).
- **Metric caveat:** round-1 strategy L1 is confounded (Leduc equilibria are non-unique/mixed;
  near-indifferent infosets inflate L1 — even exact leaves are 0.087–0.264 from full CFR-D, and a
  0.055 value MAE amplified to 0.45 strategy L1). A conclusive verdict needs an **exploitability**
  metric, which needs **safe re-solving** (the DeepStack/CFR-D gadget; finding #4).
- **Next (Stage 2c):** a **regularized (QRE/entropy) per-range round-2 solve** → unique (consistent)
  *and* near-optimal (good) targets, resolving the tradeoff; plus the safe-resolving gadget so the
  agent's exploitability (not strategy L1) is the metric. Artifacts:
  `autoresearch-session/rebel/leduc_stage2{,b_blueprint}.json`, `leduc_pbs_value_net.pt`.

**Stage 2c results (2026-06-04) — the gadget works; the depth-limited trunk is the wall.** Built the
DeepStack **safe-resolving gadget** (`loop.gadget_resolve`), the **QRE round-2 solve**
(`loop.solve_round2_qre`), and an exact-exploitability evaluator (`scripts/run_rebel_leduc_stage2c.py`;
the played agent is a *fixed* strategy σ1+σ2, so `nash_conv` is exact — the trustworthy metric).
- **Gadget validated (positive):** with a near-equilibrium σ1 and exact blueprint CFVs, gadget
  re-solving cuts exploitability **0.21 → 0.0079 (26.5×)**, within ~10× of the 8e-4 ceiling.
- **Blocker isolated (fundamental):** the **depth-limited trunk does not recover a good σ1.** Failure
  isolation — A perfect-σ1+exact-CFV **0.0098**; B trunk-σ1+exact-CFV **0.35**; C perfect-σ1+net-CFV
  **0.084**; D end-to-end **0.59**. Querying a per-range value function each trunk iteration is
  biased (the opponent re-optimizes inside a round-1 deviation; Stage-1 #3). Tested all leaves:
  blueprint σ1 → 0.35, QRE σ1 (τ=0.05/0.1) → 0.15–0.16, isolated → under-determined; **none** reach
  the perfect-σ1 0.0098. The faithful unbiased trunk is co-evolving CFR-D (= full CFR), which on a
  **2-round** game buys nothing from a net.
- **Verdict:** Leduc **validated the plumbing** (conventions, averaging, gadget, net learnability) but
  is **too shallow** to show the net's benefit or resolve the trunk-bias. Next: (a) reimplement the
  trunk with faithful ReBeL/CFR-D semantics (leaf value = value of the *co-evolving average*, not
  equilibrium-per-range), or (b) move to a **deeper small game (≥3 rounds)** where the net amortizes
  expensive subtrees and the bias is the genuine question. Artifacts:
  `autoresearch-session/rebel/leduc_stage2c_{findings,e2e}.json`.

## 5c. Gate diagnosis (2026-06-04) — the trunk bias is the known single-value-leaf unsoundness

The "faithful-loop gate" diagnostic phase **resolved the trunk-σ1 bias and is literature-grounded**:
- **Ruled out solver inaccuracy:** the isolated Nash round-2 re-solve is accurate at genuinely
  well-reached entries (CFV error vs the true oracle@star value **0.014** at reach-fraction >20%, 6000
  iters); the QRE leaf, by contrast, is badly inaccurate (**1.50** — entropy regularization distorts
  the values, so QRE was the wrong target choice).
- **Confirmed the bias is structural:** trunk solve with the *verified-accurate* exact per-range Nash
  leaf **still** gives σ1 at **0.18 L1 / 0.24 exploitable** (true gadget CFVs). Accurate values do not
  fix it.
- **Root cause + fix (Brown & Sandholm 2018, "Depth-Limited Solving", arXiv:1805.08195):** a **single**
  estimated leaf value is *unsound* in imperfect-information games — one value assumes the opponent
  plays one fixed continuation, so the trunk player isn't robust to the opponent's continuation
  choice → biased regrets. The fix is **multi-valued states** (the opponent chooses among several
  continuation strategies/value-vectors at the depth limit) or ReBeL's **PBS-CFR** (belief space).
  Both are known, sound, and reported feasible on **modest hardware (4-core CPU / 16 GB)** — directly
  relevant to the single-GPU goal. Our **gadget already implements opponent-choice for the round-2
  re-solve** (0.21→0.0079); the **trunk needs the same treatment**.
- **Net state:** every ReBeL mechanic is validated on Leduc *except* the trunk needs multi-valued
  states — solved, modest-hardware-feasible engineering, not a novel blocker. **The method is proven;
  the SOTA-efficiency contribution is running it on one GPU on a real game, not re-deriving trunk
  soundness on toy Leduc.** Next: implement the multi-valued-states / PBS-CFR trunk (faithful gate),
  then the efficiency main line on a real-belief game. Artifact:
  `autoresearch-session/rebel/leduc_gate_diagnosis.json`.

## 6. Reproducibility

- LBR gate + positive controls: `scripts/run_lbr.py`, `test/unit/test_lbr_positive_controls.py`,
  `autoresearch-session/lbr/positive_controls_verify_n500.json`.
- Six-axis data: `autoresearch-session/lbr/{budget_curve,hardened_compare,cnn_compare}/`,
  `autoresearch-session/20260604-lever-a-avgpolicy/` (readout axis).
- GPU R-NaD collector + CNN encoder: `poker_ai/rnad/{cuda_collector,encoder}.py`.
- Full timeline: RESEARCH_LOG.md entries 20260603T133428Z (anchor) → 20260604T023839Z (decision).
- Designs: workflows wp5fyfhw1 (review), wxe90uihq (ReBeL scoping).
- ReBeL de-risk (§5b): `poker_ai/rebel/{leduc,leaf_eval,loop,pbs_value_net}.py`,
  `scripts/run_rebel_leduc_roundtrip.py` + `test/unit/test_rebel_leduc_roundtrip.py` (Stage 0),
  `scripts/run_rebel_leduc_stage1.py` (Stage 1),
  `scripts/run_rebel_leduc_stage2.py` + `test/unit/test_rebel_stage2_plumbing.py` (Stage 2),
  `loop.{gadget_resolve,solve_round2_qre,qre_leaf_fn}` + `scripts/run_rebel_leduc_stage2c.py` (Stage 2c);
  artifacts `autoresearch-session/rebel/{leduc_roundtrip_stage0,leduc_stage1_findings,leduc_stage2,leduc_stage2b_blueprint,leduc_stage2c_findings,leduc_stage2c_e2e}.json`.
