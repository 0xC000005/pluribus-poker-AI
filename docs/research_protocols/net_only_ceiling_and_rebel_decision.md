# Net-only self-play hits an exploitability ceiling on single-GPU HU NLHE — and the case for search-in-learning

**Status:** consolidated finding, 2026-06-04. Authoritative; supersedes scattered RESEARCH_LOG entries
20260603T133428Z → 20260604T023839Z. Single machine (RTX 3070 Ti, 8 GB), tabula-rasa, Slumbot held-out.

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

## 6. Reproducibility

- LBR gate + positive controls: `scripts/run_lbr.py`, `test/unit/test_lbr_positive_controls.py`,
  `autoresearch-session/lbr/positive_controls_verify_n500.json`.
- Six-axis data: `autoresearch-session/lbr/{budget_curve,hardened_compare,cnn_compare}/`,
  `autoresearch-session/20260604-lever-a-avgpolicy/` (readout axis).
- GPU R-NaD collector + CNN encoder: `poker_ai/rnad/{cuda_collector,encoder}.py`.
- Full timeline: RESEARCH_LOG.md entries 20260603T133428Z (anchor) → 20260604T023839Z (decision).
- Designs: workflows wp5fyfhw1 (review), wxe90uihq (ReBeL scoping).
