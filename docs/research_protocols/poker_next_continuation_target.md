# Next Continuation Target

Date: 2026-05-13

## 2026-06-04 ReBeL de-risk: Stage 0 PASS, Stage 1 findings -> Stage 2 is next

CURRENT authoritative target (supersedes the branches below as the mainline; see
`docs/research_protocols/net_only_ceiling_and_rebel_decision.md` §5b for the consolidated record).
The net-only R-NaD ceiling is defensibly real across six axes; the decision is to pursue
ReBeL-style search-in-LEARNING on a single GPU, small-game-first. Built on the trusted Leduc tree
(`poker_ai/rebel/{leduc,leaf_eval,loop}.py`):

- **Stage 0 (constant-oracle round-trip) = PASS** (`scripts/run_rebel_leduc_roundtrip.py`,
  `test/unit/test_rebel_leduc_roundtrip.py`). Pinned the NORMALIZED CFV convention (round-trip q
  error 8.9e-16; wrong convention breaks at 2.24) and uniform/own-reach CFR+ averaging (linear CFR+
  NashConv 0.0017). Both correctness must-fixes resolved.
- **Stage 1 (depth-limited solving) findings** (`scripts/run_rebel_leduc_stage1.py`,
  `autoresearch-session/rebel/leduc_stage1_findings.json`): exact-oracle CONTROL = full CFR+/CFR-D
  -> NashConv 8.2e-4 (this is the oracle-leaf-control ceiling for the PASS metric, not "beat
  R-NaD"). Isolated subgame re-solving is the WRONG primitive: its frozen strategy is off-path
  exploitable (assembled NashConv 0.21), its thin-reach values are under-determined (err ~0.09 at
  >10% reach, ~0.77 at ~4%, stable across 3k/12k/40k iters), and per-iteration re-solve biases the
  trunk (round-1 L1 0.15). DESIGN CONSTRAINT: value net FIXED per solve (CFR-D), targets = the
  self-consistent CFVs read off the trunk solve averaged over the visited PBS distribution, and
  continual re-solving at play.
- **Stage 2 (PBS value net) done -- two findings, no clean pass yet**
  (`poker_ai/rebel/pbs_value_net.py`, `scripts/run_rebel_leduc_stage2.py`,
  `autoresearch-session/rebel/leduc_stage2{,b_blueprint}.json`). (1) Net learnability depends on
  TARGET CONSISTENCY (Stage-1 #2 end-to-end): isolated per-range re-solve targets -> underfit (val
  MAE 0.16); consistent blueprint-continuation targets -> learned (val MAE 0.055). (2)
  CONSISTENCY<->QUALITY tradeoff: per-range value is a good leaf (exact trunk 0.087 L1 from full
  CFR-D) but unlearnable-as-is; blueprint value is learnable but a crude leaf (exact trunk 0.264).
  Round-1 strategy L1 is a confounded metric (equilibrium multiplicity + near-indifference) -> a
  conclusive verdict needs an EXPLOITABILITY metric (-> safe-resolving gadget).
- **Stage 2c done -- gadget WORKS, depth-limited trunk sigma1 is the wall**
  (`loop.gadget_resolve`, `loop.solve_round2_qre`, `scripts/run_rebel_leduc_stage2c.py`,
  `autoresearch-session/rebel/leduc_stage2c_{findings,e2e}.json`). Safe-resolving gadget cuts
  exploitability 0.21 -> 0.0079 (26.5x) with a good sigma1 + exact CFVs (exact OpenSpiel NashConv;
  the played agent is a fixed strategy so the metric is exact). BUT failure isolation
  (A perfect-sigma1=0.0098; B trunk-sigma1=0.35; C net-CFV=0.084; D e2e=0.59) shows the
  depth-limited TRUNK does not recover a good sigma1 -- querying a per-range value function each
  iteration is biased (Stage-1 #3); blueprint/QRE/isolated leaves all fail (QRE sigma1 still 0.15).
  Leduc (2 rounds) validated the PLUMBING but is too shallow to show the net's benefit or resolve
  the trunk bias.
- **GATE DIAGNOSIS done (2026-06-04, literature-grounded)** -- the trunk-sigma1 bias is the KNOWN
  single-value-leaf unsoundness (Brown & Sandholm 2018, arXiv:1805.08195): a single leaf value is
  unsound in imperfect-info games (assumes one fixed opponent continuation). Ruled out solver
  accuracy (isolated Nash re-solve err 0.014 at well-reached; QRE badly off at 1.50) and confirmed
  the bias is structural (accurate exact leaf still gives sigma1 0.24 exploitable). FIX =
  MULTI-VALUED STATES (opponent chooses among several continuation strategies at the depth limit) or
  ReBeL PBS-CFR; both known + modest-hardware-feasible (4-core CPU/16GB per Brown-Sandholm). Our
  gadget already does opponent-choice for the round-2 re-solve; the TRUNK needs the same.
  `autoresearch-session/rebel/leduc_gate_diagnosis.json`.
- **DECISION (2026-06-05): build the faithful (multi-valued-states / PBS-CFR) ReBeL loop DIRECTLY on
  a real-belief game** (fold the gate into the efficiency main line; Leduc is too shallow to show the
  net's benefit). First target = **turn subgame + neural river leaf** (DeepStack-style): turn betting
  is the trunk, the river is the depth-limit leaf via `cut_node_fn`. Staged: 0 harness+exact-control,
  1 multi-valued-states (correctness gate), 2 PBS value net + throughput, 3 self-play loop, 4 scale.
- **STEP 0 STARTED** (`poker_ai/rebel/turn_river.py`, `scripts/run_rebel_turn_river.py`,
  `test/unit/test_rebel_turn_river.py`, `autoresearch-session/rebel/turn_river_step0.json`): harness
  built on the trusted `StreetSolver` + `solve_cfr` cut_node_fn. KEY FINDING (default 80bb spot,
  board Ah Kd 7c 2s): turn = 1128 hands / 459 nodes / 153 river-cut nodes; one EXACT river leaf =
  44 runouts x ~1.7s = 76s/cut; a full 100-iter turn solve with the exact oracle = ~323 HOURS ->
  the exact oracle is INFEASIBLE per-iteration in the trunk -> the learned PBS value net is
  LOAD-BEARING, not optional (the efficiency thesis, concrete on a real game).
- **REFRAME for the correctness gate (step 1):** because the exact oracle is per-iteration-infeasible
  at 80bb, run the multi-valued-states-vs-exact-control gate on a SMALL turn spot (short stacks ~10-20bb
  -> tiny turn tree, few cuts, cheap river solves -> feasible exact control). Keep the 80bb spot for
  the efficiency/throughput demonstration where the net is needed.
- **STEP 1 (#24) IN PROGRESS** -- per-hand river-CFV extractor designed (`river_subgame_cfv` in
  turn_river.py): reuse solve_cfr's terminal eval via avg-strategy-as-initial-regret + trace-root.
  - **SOLVER VALUE CONVENTION discovered** (fast_cfr.py L376-391): value = net chips from subgame
    start with the pre-existing pot awarded to the winner -> per valid pair
    `hero_val(a,b)+villain_val(b,a) = pot_start`. So the verification identity for the extractor is
    `sum(hr*hcfv)+sum(vr*vcfv) == pot * (hr @ valid @ vr)` (NOT naive zero-sum).
  - **CUT-NODE CONVENTION OFFSET** required: the river subgame value is net-from-river (awards the
    cut pot, which includes turn investments); to drop into the turn cut_node_fn it must be converted
    to net-from-turn-start by subtracting hi_cut/vi_cut (hero/villain turn investment at the cut), in
    counterfactual (opponent-reach-weighted) form. This offset differs per cut node so it does NOT
    cancel in regrets -> required for correctness.
  - **EXTRACTOR BUG FIXED + VERIFIED (2026-06-05):** the trace/hvals[0] approach was wrong (the
    opponent-node backward aggregation weights hero values by the villain's per-hand strategy -- an
    index mismatch that is fine for CFR regrets but wrong for the absolute counterfactual value).
    Replaced with `subgame_value_pass` (turn_river.py): a forward-reach-weighted sum over terminals
    (own range factored out -> counterfactual value). VERIFIED via the identity exactly across 3
    spots (e.g. 371.429==371.429); regression test in test_rebel_turn_river. `river_subgame_cfv`
    now uses it.
  - **EXACT-RIVER LEAF MACHINERY COMPLETE + VERIFIED + INTEGRATED (2026-06-05):** all layers built
    and numerically verified in `turn_river.py`: `subgame_value_pass` (extractor, identity exact) ->
    `turn_leaf_river_cfv` (44-runout averaging + turn<->river hand mapping, identity 550.53==550.53) ->
    convention offset (net-from-river -> net-from-turn-start, matches turn equity to ~1e-5 at all-in)
    -> `make_exact_river_showdown_fn` (drives solve_cfr via showdown_leaf_fn -- NOT cut_node_fn, which
    needs decision nodes; smoke runs the turn solve end-to-end). 7 turn_river unit tests green.
    Commits b478240 -> d5ecf49.
  - **PLAN A->B->C (goal-driven, 2026-06-05):**
    - **A DONE -- GPU-fast target gen:** `turn_leaf_river_cfv_batched` runs all 44 runouts in ONE
      same-topology batched GPU call (the board only changes matrices/hands). VERIFIED exact-match to
      the CPU path (identity 3670.21==3670.21, EV diff 0.156), **3.1x** (36.8s->11.8s). backend param
      threaded; per-solve GPU is only ~1.1x so batching was the win. Commits ff3e68e, bc2f03d.
      (Further win available: cache the 44 river matrices/trees across PBSs at a fixed cut state.)
    - **B (part 1 DONE) -- single-street best-response:** `street_br_value`/`street_nashconv`
      (turn_river.py). Forward fixed-player reach; backward MAX at BR nodes / SUM at fixed nodes.
      VERIFIED: agent-plays-avg BR == subgame_value_pass exactly (0.0) on 15/75-node trees; small-tree
      equilibrium NashConv <0.05*pot, degenerate clearly higher (8 tests green; commit d58f700).
      CAVEAT: on DEEP trees the BR exploits poorly-averaged low-reach infosets (NashConv grows with
      tree size, flat across iters) -> measure on SMALL/short-stack spots (also the gate's domain).
    - **B (part 2 NEXT) -- 2-street composition:** turn BR with river-BR leaf values (per cut: river
      BR vs the agent's river strategy, averaged over runouts + the net-from-turn offset). The
      exact-river-leaf agent's 2-street NashConv should be HIGH (single-value bias, like Leduc 0.21);
      the gadget-safe agent low. Build on small/short-stack spots.
    - **C (DONE -- efficiency thesis CONFIRMED):** RiverPBSNet (poker_ai/rebel/river_pbs_net.py)
      trained on exact-river-leaf targets at the check-check cut (1128 hands). LEARNABILITY: held-out
      reach-weighted MAE 3.9% of value scale (train~=val; plateaus by N~200). EFFICIENCY: net
      inference 0.059ms vs exact leaf 10632ms => **181,000x faster** -- the net turns the
      per-iteration-infeasible exact leaf (323 hrs/turn-solve) into a sub-ms forward pass, making
      single-GPU ReBeL feasible. 2 net tests green. Commit 28ca305 + this entry.
    - **NEXT (C extension -> the net-leaf gate):** a pot/stacks-GENERAL river net (features include
      the cut's pot/stacks; targets sampled across cut public states) so the net covers every river
      leaf the turn solve reaches; then use it as the live showdown_leaf_fn over a full turn solve
      and measure the net-leaf agent's 2-street exploitability (B) vs the exact-leaf control (the
      single-value-vs-multi-valued correctness gate, already proven on Leduc). Then flop-truncated
      HUNL. Slumbot held-out throughout.

## 2026-05-26 Simplification Reset

The active continuation target is reset to **local self-play first**. The
mainline is now a minimal policy-improvement loop: local self-play experience,
neural policy/value learning, optional general search improvement, and
promotion through a self-play checkpoint league. Slumbot remains a held-out
external benchmark and integration test only.

Demote the following branches to diagnostic-only evidence: Slumbot traces,
response-range replacement, action-likelihood fitting, trace-start hard-state
policy heads, post-hoc policy calibration, detached search-target injection,
and learned updates that are validated only on fixed trace artifacts. They
identified real failure modes, but they are too easy to turn into
benchmark-shaped mechanisms.

The next continuation should either:

- Consolidate the cleanest local self-play league baseline and measure whether
  it improves monotonically against previous checkpoints; or
- Add one general search-improvement operator inside that local loop, then
  falsify it by matched root-decision and league gates before any Slumbot run.

Do not spend Slumbot hands or optimize trace-derived targets until a locally
trained candidate clears the internal league gate with positive lower-bound
evidence.

2026-05-27 native rollout update: the queued native substrate spike now has an
executable gate. `scripts/eval_native_rollout_substrate.py` deterministically
replays canonical `full_deck/state.py` hands through `FastPokerState`, checks
legal masks, transitions, payouts, and feature vectors, then compares rollout
throughput. The first gate passed on 32 replay hands with `106` checked steps,
zero mismatches, and about `118x` hot-path speedup (`365` to `43325`
steps/sec). `scripts/run_tianshou_rainbow_native_control.py` now accepts
`--state-backend fast-state`; a tiny matched Rainbow smoke improved trainer
collection from about `183` to `848` train steps/sec. This is substrate
evidence only. The next continuation should run the maintained RL/population
loop on `fast-state` against population/history opponents and promote only by
duplicate-swapped H2H lower95 plus empirical-game support.

2026-05-28 batched learner update: the fast-state substrate is now strong
enough for real neural self-play candidates, but the first local policy update
families are falsified. A batched shared policy-gradient learner trained 8 x
2048 games at about `40309` rollout steps/sec and failed the incumbent H2H
gate (`mean=-0.014598`, lower95 `-0.032620`). Adding frozen history opponents
and a value head preserved clean parity and high throughput (`25738`
environment steps/sec) but failed harder (`mean=-0.0389705`, lower95
`-0.058533`). A first noncentralized Q-boosting/PPO-style variant with
Expected-SARSA(lambda) trajectory links also failed (`mean=-0.032145`,
lower95 `-0.051020`) despite clean speed/parity. The centralized-Q version now
exists too: it uses training-only opponent-card and stack/bet features for the
critic while preserving the actor observation, but the first smoke still failed
(`mean=-0.034328`, lower95 `-0.053347`). A 32 x 4096 scale check improved the
mean but still failed with high confidence (`mean=-0.024449`, lower95
`-0.035761`). Do not repeat this branch by changing only history capacity,
value weight, Q lambda, seed, or modest scale. The next continuation should
change the learning dynamics at a higher level: plug the fast batched substrate
into a maintained VRPO/PPO-style implementation if available, train a proper
empirical-game/meta-policy response oracle, or make the batched engine serve a
stronger population self-play objective. Promotion still requires
duplicate-swapped H2H lower95 and complete empirical-game support before any
Slumbot evaluation.

2026-05-28 maintained-response update: switching back to a maintained
Tianshou Rainbow response oracle on the fast-state backend produced a useful
parent-H2H pass but failed population robustness. The 131k-step UPC4 response
from the current local incumbent beat its parent (`mean=+0.008283`, lower95
`+0.000488`) but lost clearly to the prior 131k Rainbow response
(`mean=-0.021514`, lower95 `-0.033917`). This confirms the gate discipline:
parent improvement is not enough. The next continuation should train directly
against the empirical-game/meta-policy support or a broader population, then
require both parent H2H and empirical-game insertion before promotion.

2026-05-28 broader-population response update: the same maintained Rainbow
response-oracle shape trained against four Rainbow-family population members
did not clear the parent gate (`mean=+0.003301`, lower95 `-0.006336`) despite
`2254.88` train steps/sec. This rejects another same-budget population-response
retry as the immediate next move. The active bottleneck is now split: the
learning objective needs a stronger meta-policy/response target, and the
environment hot path still prevents much larger local neural population
training from keeping the GPU busy.

2026-05-28 compiled-transition update: the first compiled-array full-hand
transition primitive exists in `poker_ai/research/compiled_fast_rollout.py` and
is exposed via `scripts/eval_native_rollout_substrate.py --include-compiled-transition-benchmark`.
It uses Numba-compiled 2-player batch legal masks, action transitions, and
heads-up 7-card showdown/payout handling. The first strict smoke preserved the
existing canonical `full_deck/state.py` parity gate, passed random full-hand
terminal payout tests against `FastPokerState`, and measured `699618`
compiled transition steps/sec versus `102841` Python fast-state loop steps/sec
(`6.80x`) with zero showdown fallbacks. This clears the substrate integration
threshold for the transition/evaluator hot path. The next continuation should
feed this compiled batch into the maintained/local population learner path
while preserving the duplicate-swapped H2H and empirical-game promotion gates.

2026-05-28 compiled-collector update: the compiled batch now emits 126-feature
observations, legal masks, actions, player ids, rewards, and payoffs for a
learner-facing self-play collector. The first CUDA-policy smoke over 4096
games produced `23821` decisions at `152274` decisions/sec with zero truncated
games and zero Python showdown fallbacks. This finally attacks the low-GPU
utilization complaint at the correct layer: the neural actor can batch about
125 live decisions per forward while the environment transition stays in a
compiled batch. The next continuation is no longer another substrate-only
benchmark; route a maintained/local population RL update through this compiled
collector and judge policy quality with duplicate-swapped H2H and empirical-
game support.

2026-05-28 compiled-policy-gradient falsifier: the compiled collector now
feeds the existing history-population policy-gradient pilot through
`--pg-rollout-backend compiled`. This improved compute materially (`53684`
end-to-end env steps/sec, mean rollout `117260` env steps/sec, zero showdown
fallback), but the 8 x 2048 h256 CUDA checkpoint failed 2000-game H2H against
the local incumbent (`mean=-0.076120`, lower95 `-0.096993`). The interpretation
is important: low environment throughput was a real bottleneck, but it was not
the only bottleneck. Do not continue by tuning terminal-return PG, entropy,
history capacity, or seeds. The next continuation should put a stronger
off-policy/population response learner or empirical-game/meta-policy objective
on top of the compiled batch.

2026-05-28 AlphaNLHoldem reference update: the unofficial
`bupticybee/AlphaNLHoldem` repository is now the concrete public
AlphaHoldem-style reference surface. It is checked out locally under ignored
`reference_code/AlphaNLHoldem` at commit `d847dbc`. Its README states that it
is not the official AlphaHoldem implementation, uses RLCard no-limit hold'em
with 50bb stacks and 5 actions, and provides a TensorFlow/RLlib historical
league plus checkpoint data. The next continuation should build an isolated
reference audit/evaluation bridge, not copy AGPL code into the main tree. The
new benchmark contract is dual: our method should eventually beat this
AlphaNLHoldem/RLCard surface and also beat the native 9-action local league
before held-out Slumbot confidence evaluation. Do not treat RLCard success as
Slumbot evidence, and do not use Slumbot data to train for either surface.

2026-05-28 AlphaNLHoldem executable-probe update: added
`scripts/probe_alphanlholdem_reference.py` and
`poker_ai/research/alphanlholdem_reference.py`. The probe reads the ignored
checkout without importing AGPL code, verifies the RLCard wrapper, 5-action
contract, bundled `weights/c_1048.pkl`, TensorFlow/Ray legacy runtime, and
license boundary, then writes
`autoresearch-session/alphanlholdem_reference/contract_probe.json`. The current
checkout passes with `ready_for_rlcard_benchmark=true`, `action_count=5`,
`direct_native_action_match=false`, and `license_risk=agpl_reference_only`.
The next continuation is an isolated RLCard H2H evaluator or source-controlled
reproduction plan; do not use this probe as strength evidence.

2026-05-28 AlphaNLHoldem H2H-smoke update: added
`poker_ai/research/alphanlholdem_benchmark.py` and
`scripts/eval_alphanlholdem_rlcard_reference.py`. The evaluator implements the
AlphaNLHoldem observation contract, loads the bundled checkpoint into a clean
PyTorch inference path, and plays RLCard no-limit hold'em without importing the
AGPL reference package. Two 2-games-per-seat smokes succeeded:
`random` versus `alphanlholdem` and `alphanlholdem` versus `random`, writing
`autoresearch-session/alphanlholdem_reference/random_vs_reference_smoke.json`
and `reference_vs_random_smoke.json`. These are evaluator-validation smokes
only. The next real public-reference gate is our candidate policy versus
`--baseline alphanlholdem` with a predeclared sample size and confidence
threshold.

2026-05-28 AlphaNLHoldem source-controlled-candidate update: the RLCard
reference evaluator now accepts separate `--candidate-weights` and
`--baseline-weights` paths plus `--min-lower95-candidate-payoff`. This makes the
public-reference gate usable for a source-controlled AlphaNLHoldem-format
candidate or reproduction checkpoint. A self-match with the same bundled
checkpoint on both sides produced mean zero but failed a `0.0` lower95 gate at
2 games per seat, as it should with such a tiny confidence interval; the same
smoke passed with a permissive diagnostic threshold. The next missing bridge is
an exporter/adapter that lets this repo's locally trained candidate enter the
5-action RLCard reference surface without using AlphaNLHoldem traces or Slumbot
data.

2026-05-28 AlphaNLHoldem env-native correction: do not project a native
9-action Slumbot-facing checkpoint into RLCard's 5-action environment for
promotion. The transferable object is the high-level learning schema
(self-play/population improvement, legal masking, confidence-gated H2H), not a
checkpoint or action mapping. The next continuation is therefore an RLCard-
native training path that follows the same general algorithmic discipline and
compares its own RLCard-native checkpoint against the AlphaNLHoldem reference;
the native Slumbot-facing environment separately trains and gates a native
model. If another card environment is added, repeat this pattern: train a fresh
model in that environment and transfer only the algorithmic schema.

2026-05-28 RLCard-native PPO bridge update: added a maintained Tianshou PPO
control for the RLCard public-reference surface. It wraps RLCard no-limit
hold'em as a legal-masked single-agent Gymnasium environment, trains a PPO
actor/critic directly in the 5-action RLCard game, saves an env-native
checkpoint, and lets the AlphaNLHoldem evaluator load that checkpoint as
`--candidate rlcard-ppo`. A tiny CUDA smoke trained 128 rollout steps in
`0.43s` and then completed a 2-games-per-seat AlphaNLHoldem H2H smoke. This is
plumbing evidence only, not strength. The next continuation should scale the
RLCard-native actor-critic/league learner or move toward an AlphaHoldem/IMPALA-
style historical league, then require a predeclared positive lower95 gate
against the AlphaNLHoldem checkpoint.

2026-05-28 RLCard PPO population-hook update: the same maintained PPO runner
now accepts `--opponent-checkpoint`, allowing a fresh RLCard-native child to
train against a frozen RLCard PPO parent inside the RLCard environment. The
first tiny parent/child CUDA smoke completed and the child checkpoint loaded
into the AlphaNLHoldem evaluator. This remains a plumbing result. The next
meaningful public-reference experiment should train against a declared
historical parent pool or AlphaHoldem-style league for enough hands to make the
AlphaNLHoldem lower95 gate interpretable.

2026-05-28 RLCard PPO 8k falsifier: a first non-tiny maintained PPO
parent/child loop is falsified. The parent trained 8,192 rollout steps against
random, the child trained 8,192 steps against the frozen parent, and the child
failed the 50-games-per-seat AlphaNLHoldem lower95 gate (`mean=-2.81`,
`lower95=-6.171`, `upper95=+0.551`). Do not respond by tuning seeds or modest
step counts. The next continuation should change the learning structure toward
AlphaHoldem/IMPALA-style historical league or another stronger self-play
objective while preserving the train-per-environment rule.

2026-05-28 RLlib IMPALA bridge update: added an RLCard action-mask observation
format and `scripts/run_rllib_impala_rlcard_reference_control.py`. The dry-run
contract and CPU smoke pass with Ray 2.55 IMPALA/V-trace, using RLCard's own
5-action environment. The CUDA old-API action-mask path is explicitly blocked
after V-trace tensor-shape failures in the learner thread. The next GPU-capable
step should not keep patching Ray's old API; it should either implement a
new-API masked RLModule for IMPALA/APPO or a local GPU V-trace style learner on
the compiled rollout substrate.

2026-05-28 APPO probe update: Ray's new-API APPO plus the action-mask
RLModule example completed a one-iteration CPU probe on RLCard, but the GPU
probe hung and was killed. This keeps APPO as a plausible maintained-library
direction, but not yet a usable GPU path. The next continuation should first
make a controlled APPO runner with timeout/fail-closed CUDA validation or
switch to a local GPU V-trace learner only after documenting why maintained
APPO cannot be made reliable.

2026-05-28 controlled APPO runner update: the APPO runner now exists with
dry-run contract tests and a subprocess supervisor for unattended runs. The
supervised tiny RLCard smoke still timed out (`status=timeout`) before one Ray
APPO iteration completed, and the guard killed the child process group cleanly.
This converts APPO from an unsafe hang risk into a documented partial path, but
it is not a GPU-capable candidate. The next continuation should not tune APPO
strength. Either find a maintained APPO/IMPALA configuration that completes a
tiny masked RLCard smoke, or queue methodology review for a minimal local
GPU V-trace/actor-learner that consumes the compiled rollout substrate.

2026-05-28 local V-trace innovation review: the reviewed pivot now permits a
narrow local GPU V-trace actor-learner only because maintained Ray IMPALA/APPO
failed the integration smokes. The boundary is strict: local code may handle
trajectory packing, legal masks, and the standard V-trace policy/value update;
it may not add poker action heuristics, Slumbot/reference data, or broad custom
RL-library scope. The next decisive test is a tiny GPU learner smoke over
same-environment legal-mask trajectories, then AlphaNLHoldem/RLCard and native
9-action H2H gates only if reliability and parity pass.

2026-05-28 local V-trace primitive update: `poker_ai/research/local_vtrace.py`
now contains only the standard masked discrete-action V-trace recurrence and
policy/value loss helper, with hand-calculated unit tests. This is the first
local learner-side primitive after the maintained-library failure review. The
next continuation should connect it to generated RLCard/native trajectory
batches and require a tiny GPU smoke before any strength experiment.

2026-05-28 tiny V-trace trajectory smoke: `scripts/run_local_vtrace_rlcard_smoke.py`
now collects real RLCard legal-mask trajectories and runs one local V-trace
policy/value update. The 2-env x 4-step CUDA smoke passed with zero illegal
probability and finite loss. A larger direct-RLCard collection was CPU-bound,
so this validates the learner interface but not scale; the next step is to
feed the same loss from the compiled rollout collector rather than Python
RLCard stepping.

2026-05-28 compiled native V-trace smoke: `scripts/run_local_vtrace_compiled_native_smoke.py`
now consumes the compiled 9-action full-deck self-play collector, packs learner
decision chains into padded trajectory tensors, and applies the local V-trace
loss in one vectorized learner call. The 512-game CUDA scale smoke passed with
`3125` samples, zero illegal mass, zero Python showdown fallback, and about
`6790` samples/sec end-to-end. This is still plumbing, not strength. The next
continuation can turn this smoke into a checkpoint-producing learner and judge
it only by native duplicate-swapped H2H plus empirical-game gates.
9-action candidate.

2026-05-28 AlphaNLHoldem RLCard-native NFSP smoke: the public-reference gate
now accepts `--candidate rlcard-nfsp` checkpoints produced by
`scripts/run_rlcard_nfsp_pilot.py --checkpoint-out`. A tiny CUDA NFSP run in
RLCard's own 5-action environment wrote
`rlcard_nfsp_env_native_smoke.pt` with `trained_environment_native=true` and
`native_action_projection=false`. The same-environment AlphaNLHoldem H2H smoke
loaded that checkpoint as an RLCard agent and completed against the bundled
reference checkpoint. This validates the corrected plumbing path. It is not
strength evidence: the sample was only 1 game per seat and the NFSP run was
 only 20 episodes. The next continuation should scale an RLCard-native learner
or reproduction candidate under a predeclared budget, then gate it against
AlphaNLHoldem with confidence while preserving the separate native 9-action
league path.

2026-05-28 RLCard-native NFSP 1k gate result: a bounded 1,000-episode
RLCard-native NFSP candidate trained on CUDA and wrote
`rlcard_nfsp_env_native_1k_seed20260534.pt`. It failed the first
AlphaNLHoldem public-reference gate over 50 games per seat:
`mean=-4.63`, `lower95=-9.90`, `upper95=+0.64`, `passed=false`. This is not a
philosophy failure; it is an expected scale/objective failure for a tiny NFSP
baseline against a week-trained reference checkpoint. The next public-reference
continuation should either scale the RLCard-native learner by orders of
magnitude under a predeclared budget or build a stronger RLCard-native
AlphaNLHoldem-style league learner, not adapt native 9-action checkpoints.

2026-05-28 RLCard-native NFSP self-play 5k result: added
`--opponent-kind nfsp-self-play` to the RLCard NFSP pilot so both seats train
with maintained RLCard NFSP agents. The first bounded 5,000-episode CUDA
self-play checkpoint failed the same AlphaNLHoldem gate over 50 games per seat:
`mean=-4.04`, `lower95=-10.09`, `upper95=+2.01`, `passed=false`. This is more
philosophically aligned than the random-opponent baseline, but it is still not
competitive at this scale. The next public-reference move should be a stronger
RLCard-native league/training recipe, likely closer to AlphaNLHoldem's
historical-league IMPALA/VTrace setup or a maintained population learner, not
more small NFSP reruns.

2026-05-28 RLCard-native NFSP self-play 50k scale control: one predeclared
scale check trained the maintained RLCard NFSP self-play path for 50,000
episodes on CUDA. Training took `356.16s` at `140.46` episodes/sec and
`259.05` train steps/sec. The 200-game AlphaNLHoldem H2H gate improved the
mean versus 5k but still failed: `mean=-2.915`, `lower95=-8.249`,
`upper95=+2.419`, `passed=false`. This suggests scale helps but does not
justify continuing small NFSP unchanged as the main public-reference route.
The next continuation should implement or plug in a stronger RLCard-native
actor-critic/historical-league learner, using AlphaHoldem/IMPALA/VTrace as
related-work guidance, and keep NFSP as a control.

2026-05-29 stuck-learner innovation review (no GPU): resolved the three queued
reviews and corrected the reframing after a codebase check. Both methodology
reviews PROCEED (held-out AlphaNLHoldem benchmark contract; RLCard-native league
learner with the no-cross-projection boundary). The local GPU V-trace innovation
is now ABANDON: its own falsifier triggered (fresh `-1.36`, continue-from-ppo65k
`-4.37`, population-mix `-5.97`; all negative lower95). PREMISE CORRECTION: an
earlier draft claimed "we are not compute-bound, the GPU is idle, so build a
vectorized pipeline" — that is wrong on native. The native 9-action surface
ALREADY has a compiled self-play collector at `117-152K` decisions/sec, and a
compiled native V-trace/PG learner already consumed it and failed at TINY budgets
(compiled-PG `8x2048` -> mean `-0.076`; compiled native V-trace `4x512` ->
`+0.0066` vs parent / `-0.169` vs NFSP), with the log's own verdict "throughput
was a real bottleneck, but it was not the only bottleneck." The `~900-2600`
samples/sec figure is RLCARD-maintained-library only (direct rlcard is `31.7`/s,
CPU-bound) — a separate, harder surface (the native collector cannot serve its
5-action space). So the tiny prior native runs CANNOT decide scale vs objective
vs representation. The next experiment (NEW innovation review
`20260529T003000Z-gpu-throughput-vectorization-before-scaling-innovation`,
PROCEED, gated on explicit user GPU approval; directory slug predates the
correction): run the EXISTING fast native compiled self-play learner at a real
BUDGET x MODEL frontier — `>= ~10^6-10^7` steps at a GPU-saturating model/batch
(never done; all native runs were 2K-16K games) — as a scaling curve gated by
duplicate-swapped native 9-action H2H lower95 + empirical-game support, with
measured GPU utilization. NO new infra (the fast path exists), NO objective
change, NO c_1048 bar change. Falsifier: a flat/negative H2H curve at high
utilization and `>= ~10^6-10^7` steps isolates the limiter as objective/
representation and the next review MUST be a new-objective review, not more
budget. Honest tension: this is at core a "use more compute" move (vs the goal's
"not scale tweaks"), proposed because the user judges the compute unspent; veto
in favor of a new-objective review if preferred. Deep CFR is a
native-9-action-only contender for a later matched-budget bake-off (no
cross-projection). Slumbot stays held-out.

2026-05-29 (later) method-first reframe + decision memo (SUPERSEDES the
budget-frontier-only framing above): under the corrected METHOD-FIRST north star
(the contribution is a general bitter-lesson self-play-from-scratch mechanism
demonstrated on HU NLHE; AlphaNLHoldem/Slumbot are EVIDENCE, not the win
condition), a 27-agent mechanism-discovery workflow + live online related-work +
advisor produced `autoresearch-session/DECISION_MEMO.md`. Survivors (all PUBLISHED,
not novel -- which is fine and bitter-lesson-aligned): MMD/regularized-PG (Sokota
ICLR'23), ESCHER (McAleer ICLR'23), ReBeL (Brown NeurIPS'20). DECISIVE new
evidence: arXiv:2502.08938 (2025) shows tuned PG (PPO/PPG/MMD) match or beat
NFSP/PSRO/R-NaD/ESCHER on imperfect-info games, and that prior PG failures are
explained by (a) head-to-head being the WRONG metric (use exact exploitability)
and (b) an entropy-regularization regime far higher than single-agent defaults.
Our promotion ladder is duplicate-swapped H2H lower95 -- the indicted metric.
RUN-FIRST (reframed): high-entropy regularized PG / MMD evaluated by EXACT
EXPLOITABILITY on Kuhn/Leduc (CPU-only); the untried levers are the regime + the
metric, NOT the "moving magnet". Verifies regime+metric+implementation, NOT NLHE
scale (tiny-budget confound unbroken). The native budget-frontier is RECLASSIFIED
from "novel mechanism" to the scale-frontier CONTROL/measurement (no full-loop
throughput is even measured yet, so no NLHE budget is trustworthy). HEADLINE FORK
for the user: CPU exploitability gate runs regardless; the GPU slot is (A)
scale-frontier control vs (B) mechanism NLHE run -- lean (A)/parallel, user's
call. METHODOLOGY FLAG: H2H-vs-exploitability is a protected eval-surface change
-> route to methodology review + objective-drift audit, do not swap unilaterally.
No GPU run, no commit; awaiting the user's GPU go/no-go.

2026-05-26 update: the first executable neural self-play policy-iteration
contract smoke is now in place. `scripts/run_neural_policy_iteration_pilot.py`
uses a stochastic neural actor to collect local full-deck self-play states,
trains policy/value networks on legal mixed policy-improvement targets, and
exports checkpoint/metrics metadata that explicitly marks Slumbot-free training
and non-promotion status. Its current teacher is a legal mixed-target test
double, not CFR. The immediate continuation is therefore to replace that test
double with public-belief CFR/resolving targets on the same self-play state
distribution, then judge only by root-disjoint exact-CFR gates and self-play
league lower bounds.

2026-05-26 follow-up: public-belief CFR mode now skips ineligible states rather
than training on fallback labels, exposes `dropped_ineligible_self_play_states`,
and supports `--min-resolver-targets` as a hard gate. CUDA teacher routing works
with `--cfr-backend torch-levelsync --cfr-device cuda`, but the first 64-hand
smoke produced only `2` eligible resolver targets from `190` self-play states.
The next continuation should improve native target coverage/batching, not
switch to PettingZoo or optimize a different 5-action game.

2026-05-26 compute update: native batched policy inference is now available via
`--parallel-self-play-hands`. A 256-hand same-seed smoke improved collection
time only modestly (`2.586s` serial to `2.401s` with 32 parallel hands). With
the real public-belief CFR teacher, the 32-parallel smoke spent `2.431s` in
self-play collection, `3.929s` in one-state-at-a-time teacher target
generation, and `0.449s` in neural training. This explains low GPU utilization:
there are too few eligible CFR states and target generation is not batched. The
next continuation should attack batched/denser teacher targets, not neural
model size or external environment wrappers.

2026-05-26 coverage update: public-belief CFR mode now supports
`--min-searchable-self-play-states` with `--max-self-play-hands`. The collector
keeps native stochastic self-play running until enough eligible turn/river
street-root states are found or the cap is reached, then trains only on real
resolver targets. The 32-target CUDA smoke passed without fallback labels but
spent `9.500s` collecting states and `11.181s` solving teacher targets versus
`0.379s` training the network. Next continuation: batch teacher generation or
add dense early-street search targets, not PettingZoo or bigger networks.

2026-05-26 batching diagnostic: the first 16-target coverage run had
`legal_mask_pattern_groups=4` but `public_shape_groups=15`, with largest exact
public-shape group only `2`. This weakens the near-term case for a narrow
same-public-shape CFR batcher. The next reviewed pivot should prefer dense
early-street policy-improvement targets from local public belief/search, because
those cover the states the neural actor actually visits and give the GPU more
training signal per generated hand.

2026-05-26 scaled NPI update: the aligned CFR5 neural policy-iteration branch
does learn versus weak controls, but simply scaling from 32 to 96 real CFR5
targets and h64 to h128 did not beat the prior 32-target CFR5 checkpoint. The
96-target candidate beat random and the older CFR1 checkpoint in 1000-game
duplicate-swapped internal H2H, but was neutral against the prior CFR5 candidate
(`mean=-0.000465`, `lower95=-0.008744`) and slightly worse on the matched
32-root exact-CFR L1/KL gate (`0.2159/0.0795` versus `0.2107/0.0680`). It also
selected all-in as the top action on every held-out root. The next continuation
should not scale model size or target count blindly. Improve dense/batched
policy-improvement target generation under the same local self-play contract,
or batch the exact-CFR teacher itself, then rerun the same matched gate and
prior-candidate H2H.

2026-05-26 GPU-utilization note: low instantaneous `nvidia-smi` usage is
expected for the current NPI teacher path. The 96-target run spent `19.84s`
collecting self-play states and `33.49s` generating one-state-at-a-time CFR
targets, but only `0.77s` training the neural network. The GPU is available and
used, but the workload arrives as many small solver/inference bursts separated
by CPU poker simulation and Python orchestration. Higher utilization requires
batched teacher/resolver generation or a denser training-target pipeline, not
just a larger policy network.

2026-05-26 batched-CFR-teacher update: opt-in same-topology batching is now
wired into the NPI public-belief CFR teacher and validated on a real 16-target
CUDA smoke, but it is not the next scaling solution by itself. The batch path
solved `11/16` roots in one CUDA batch with no batch failures, yet total teacher
time was essentially unchanged (`5.623s` batched versus `5.683s` instrumented
serial). The new timing counters show why: solver context and terminal-matrix
construction consumed about `5.41s`, while CFR solving consumed only
`0.21-0.28s`. The next continuation should attack solver-context construction,
terminal matrix generation, ragged/fused terminal evaluation, or denser targets
that avoid one full street solver per label.

2026-05-26 continuation note: compact GPU Deep CFR checkpoints are model-only
for resume purposes. They preserve `buffer_sizes` metadata but not replay
reservoir contents, so `--resume` is a warm start unless
`resume_restored_replay_buffers=true` appears in training metrics. Do not treat
bufferless resumed children as true parent-child continuation evidence.
`scripts/poker_autoresearch_train.py --save-replay-buffers` now creates an
opt-in full replay checkpoint, and `--require-replay-buffer-resume` hard-fails
if a continuation run accidentally uses a compact checkpoint. Use those flags
for any experiment whose hypothesis depends on true Deep CFR continuation.

## Decision

Stop scaling hard learned leaf/successor value substitution and direct policy
argmax imitation as mainline methods. The latest root-disjoint successor-cut,
callback-state DCVN, policy mixing, and impact-gating checks all failed resolver
behavior gates. The next target is **publication-grade game-theoretic RL /
learned search**: prefer pure self-play/equilibrium-learning methods where
possible, and use CFR+/resolving as a principled imperfect-information
evaluator, teacher, or correction operator when necessary.

2026-05-16 update: the aligned callback-leaf branch produced the best local
learned-search signal so far, but it does not change the decision. Matching
label collection and inference at the exact CFR leaf-callback distribution
fixed one important mismatch, and the no-projection learned leaf passed some
root-disjoint behavior gates. The matched CFR10 `128x32` checkpoint remains
marginal on the full successor-pool holdout (`0.59375` action agreement,
`0.75097` mean L1) and only loosely positive on an external repeat-64 holdout
(`0.625` agreement, `0.6767` mean L1). It is also slower than exact terminal
leaf evaluation and can flip high-margin decisions. Leaf MAE, structural
abstention, linear public-belief drift prediction, and a small neural drift
predictor all fail as deployable safety mechanisms; the neural predictor fails
external transfer. The next continuation target is therefore not another
single-EV MSE scale run. It should be a safe or multi-valued depth-limit target,
or a search-impact correction target, validated by root-disjoint decision
behavior at a useful earlier cut boundary.

The first range-diversity smoke supports that direction but does not solve it.
On the eight worst high-margin action-flip roots, generic opponent-range
perturbation averaging improved mean L1 from `1.3715` to `1.0787` and restored
`3/8` top actions; an oracle over the same perturbation set reached `0.5733`
mean L1 and `5/8` agreement. This means range diversity contains signal, but
plain averaging is too weak. The next exact positive control should let the
solver reason over a value set or opponent choice at the depth limit, rather
than averaging strategies after the fact.

2026-05-19 update: the range-diversity smoke is now a reusable
diagnostic-only script with unit coverage:
`scripts/eval_callback_leaf_diverse_range_probe.py`. The script-backed run
reproduced the same eight-root signal and emits explicit promotion blockers.
Use it as a positive-control probe for value-set mechanisms, not as a gameplay
policy or Slumbot-tuned action selector.

The first integrated value-set smoke is partial but not sufficient. Feeding
aggregated diverse-range leaf values into CFR on the four worst high-margin
flips showed that naive `mean` aggregation is worse than post-hoc averaging,
while `hero_pessimistic` improves L1 (`1.1951` versus original `1.6180` and
post-hoc range average `1.2934`) but still only matches the exact top action on
`1/4` roots and costs about `3.5s` per learned solve. The next step should not
be a wider perturbation sweep. It should formulate the opponent-choice/value-set
gadget more correctly, or move the value-set boundary earlier so prediction can
be batched and amortized.

The first naive opponent-choice variant is now falsified. Choosing a single
variant per terminal leaf by villain range-weighted value produced `0/4`
agreement and `1.5653` mean L1 on the same high-margin roots, worse than
post-hoc averaging. This means the real next target is not "pick a variant at
each callback leaf"; it is to implement a faithful depth-limit gadget or an
earlier public-belief boundary where opponent continuation choices are part of
the resolved subgame state.

The current philosophically aligned target is now explicit: **ReBeL /
Student-of-Games style public-belief learning plus search**, with CFR/resolving
used as the game-theoretic improvement operator and verifier. This is the most
deep-learning-native direction that still respects imperfect-information game
theory. The immediate step is not another terminal callback heuristic. Build an
exact public-belief depth-limit gadget teacher on fixed states first; only if
that teacher improves root-disjoint resolver decisions should we train a neural
public-belief model to approximate it.

The first exact-teacher smokes passed their positive-control contract. On the
high-margin callback-leaf failure cases, the diagnostic found six eligible
successor-cut roots in the larger run and per-iteration replay of exact cut CFVs
preserved the full resolver exactly (`1.0` action agreement, `0.0` mean L1).
Static final-CFV replay was weaker (`0.6667` agreement, `0.4594` mean L1), so
the next neural target must approximate a dynamic public-belief/search state,
not only a terminal final-value table.

The first direct neural continuation probe failed that stronger target. A
root-disjoint `512/512` dynamic successor-cut dataset was exported, but the
small joint-PBS continuation model with DeepSet cards and GRU action encoding
missed even the constant value baseline (`0.2804` holdout MAE versus `0.1609`
train-mean constant MAE). This does not invalidate the exact boundary; it
invalidates detached per-hand CFV regression as the next scale-up. The next
test should learn a paired low-to-final dynamic correction or a resolver
warm-start/search-state field and evaluate it by decision behavior.

The paired low-to-final dense-value correction is now falsified too. Exact
`fixed_5` replay preserved all six eligible high-margin actions with `0.1968`
mean L1, but training the same joint-PBS continuation model on iteration-5
absolute values and iteration-5-to-final residuals failed badly. The residual
model reached `0.9444` holdout MAE versus a `0.1082` zero-delta baseline. The
next target should therefore leave dense per-hand CFV regression and move to a
search-state warm-start or learned update field that is consumed inside CFR.

The first uniform-baseline check of that old learned warm-start is negative.
Adding a CFR10 baseline to `scripts/eval_regret_policy_warm_start.py` rejects
the current low-state regret/policy field even on a 16-root smoke: the warm
start improved over CFR5 in L1 (`0.4814` vs `0.5729`) but lost to uniform CFR10
(`0.3114` L1, `0.0945` KL, `0.875` top-action agreement) and cost `2.76x` the
CFR5 latency. Future learned search-state candidates must beat CFR10, not just
CFR5.

The pure policy-gradient escape hatch was re-tested with better local mechanics
and remains negative. Batched high-entropy PPO now supports 32-hand rollouts per
update and trained on CUDA, but the 2k checkpoint lost to the 10k NFSP reservoir
control (`mean=-0.011995`, `lower95=-0.029515` over 1k duplicate-swapped
games). A minimal VRPO-inspired `q_expected_mc` advantage mode trained faster
but also lost (`mean=-0.013249`, `lower95=-0.031672`). This does not reject the
policy-gradient literature; it rejects local terminal-return PPO and the simple
MC Q-baseline. The next philosophically clean RL branch should implement a real
Expected-SARSA(lambda)/Q-boosted PPO target or use search-derived
counterfactual advantages, not merely scale this PPO pilot.

A same-player Expected-SARSA(lambda)-style PPO target is now implemented and is
the best policy-gradient control so far, but it still fails the local gate. The
2k `q_expected_lambda` checkpoint scored `mean=-0.008408`, `lower95=-0.022051`
versus NFSP10k over 1k duplicate-swapped games. This is a weak positive
mechanism signal, not a strength result. A matched-budget 10k run is defensible
only as a bounded scale check of this specific estimator; broad PPO knob sweeps
remain blocked.

The matched-budget 10k scale check has now been run. It reached statistical
near-parity with NFSP10k (`mean=0.000085`, `lower95=-0.010230`, `upper95=0.010399`
over 5k duplicate-swapped games) but still failed the positive-confidence gate.
This changes the next pure-RL target from "does PPO work at all?" to "can a
better variance-reduced/game-theoretic advantage make the near-tie decisively
positive?" The next action should not be a blind episode-count scale-up; it
should implement a fuller Expected-SARSA(lambda)/Q-boosting target, add a
population/average-policy opponent control, or use search-derived
counterfactual advantages while retaining the same H2H gate.

The CTDE Q-boosting follow-up has also been run. Giving the Q critic
training-only full-state features while keeping the actor observation-only did
not improve the matched gate: the 10k checkpoint scored `mean=-0.003581`,
`lower95=-0.013593` versus NFSP10k over 5k games. This rules out "just add
opponent cards to the critic" as the next pure-RL fix. Further pure-RL work
needs a materially different estimator or opponent-population/equilibrium
mechanism, not another PPO feature or episode-count scale.

Two of those pure-RL follow-ups are now falsified in their simple form. A
target-network Expected-SARSA(lambda) smoke lost to NFSP10k and was worse than
the online-Q lambda smoke. The reviewed average-policy/fictitious-play PPO
control trained the actor against a lagged average-policy opponent and exported
the average policy, but the average export failed badly (`mean=-0.030775`,
`lower95=-0.051581` over 1k duplicate-swapped games). The same checkpoint's
actor export was only mildly negative (`mean=-0.008752`, `lower95=-0.025487`),
so the simple supervised average-policy layer is not yet a useful equilibrium
object. The next pure-RL continuation should not scale this implementation; it
should either debug average-policy fit directly, implement a more faithful
RM-FSP/VRPO-style update, or return to search-derived counterfactual advantages
with the CFR10 baseline intact.

The average-policy fit diagnostic is now complete and does not point to a
simple fit/export bug: actor-vs-average top-action agreement is around
`0.76-0.79` with mean KL around `0.028`. The policy is being copied moderately
well, but the averaged policy is weaker than the actor at this early budget.
This retires "increase average-policy supervised fit" as the immediate fix.
The next principled branch should use search-derived counterfactual advantages
or a materially different RM-FSP/VRPO update rule.

The latest Slumbot-facing diagnostics refine that branch. Exact CUDA resolving
is now fast enough for bounded live diagnostics, but a 300-hand fast-live run
still lost and the no-all-in control only reduced variance. The persistent
failure is calibration under Slumbot's response distribution: blueprint
opponent-action likelihood is far below uniform on both traces, while a small
learned response probe transfers across the all-in-enabled and no-all-in traces
with double-digit log-lift gains. The next continuation target should therefore
learn a latent opponent-response/public-belief conditioning signal and consume
it inside search or policy improvement, with a counterfactual-EV replay gate.
Do not convert this into a manual Slumbot range heuristic; the learned signal
must survive cross-trace validation and improve decision EV.

That direct EV requirement has now rejected explicit cross-trace range
replacement. The response signal improved true-hand likelihood across traces,
but response-conditioned strategies failed beat-rate and selected-action EV
gates in both directions. The next version should not pass a hand range straight
to the resolver. It should train a decision-aware latent conditioner or
auxiliary representation that is consumed by the policy/search network and
validated by the same EV replay gate.

A shallow learned trust gate is now rejected as well. Observable range-shape
and solver-drift features contain modest signal, but a ridge selector trained
on one trace and evaluated on the other did not improve selected-action EV or
beat rate. The next continuation target should therefore be a representation
learning task, for example an auxiliary opponent-response head or latent
conditioner trained jointly with policy/search targets, followed by the same
cross-trace EV replay gate. Avoid another post-hoc switch unless it is embedded
in the learned policy/search objective.

The dense SD-CFR window decomposition is negative too. A predeclared late
checkpoint band from the clean 200x2000 GPU run reached only near parity under
a higher-confidence local H2H, so the next self-play path should not search for
a lucky checkpoint band. It should fix the training/deployment contract:
explicit average-policy semantics, a stronger game-theoretic RL objective, or a
search-state target whose deployment matches training.

The smallest explicit average-policy Deep CFR deployment test has now failed
after the workflow compare contract was repaired. The GPU run itself was clean
(`1M` average-strategy targets, zero rejected traversal chunks, about `403`
iterations/hour), but corrected candidate-average-policy versus
incumbent-regret H2H was negative/inconclusive rather than confidence-positive.
Do not scale the same 50x1000 setup or treat the earlier auto-compare error as
the explanation. Future average-policy work needs a materially stronger
self-play objective, larger justified scale, or a different equilibrium-learning
update; otherwise prefer search-state/public-belief targets with root-disjoint
decision gates.

The direct search-derived counterfactual-advantage branch has now had its
smallest fair test and failed. A trust-region update selected eta on train
traces and evaluated the same eta on holdout. It improved over CFR5 but did not
beat CFR10, so it is not a useful learned-search primitive by the current
standard. The next branch should stop trying to reuse one-row policies or
one-step advantage directions. Prefer a richer dynamic search-state object:
predict where additional CFR iterations matter, learn a multi-iteration update
operator, or return to public-belief/search-state fields that can beat CFR10.

The post-PPO exact CUDA frontier improved local resolving fidelity but did not
solve Slumbot transfer. CFR125 is close to the CFR150 teacher on held-out fixed
roots, and the `frontier-live` profile is now available, but a 100-hand
no-all-in Slumbot smoke was negative with stack-scale risk losses. A follow-up
trace response/range EV replay improved revealed-hand likelihood but worsened
selected-action and strategy EV. This rejects "better explicit Slumbot range"
as the immediate deployable fix.

The next continuation target is therefore a **trace-start / hard-state
self-play contract**, not another response-range patch or Slumbot chip sample.
Use failed live trace states as data-augmented game starts or search-state
training contexts, with an explicit source/context flag so the learner can
distinguish normal self-play from hard states. Before training, build the
smallest verified contract that reconstructs observation features, legal masks,
street/action context, and provenance from Slumbot trace cases. Only after that
contract is tested should native RL or search-guided updates consume it. The
gate remains decision impact first: held-out root value/action improvement and
native league transfer before any larger Slumbot run.

That smallest contract now exists and passed on the failed frontier-live trace:
39 observation-only hard states with exact Slumbot feature and legal-mask
parity, plus one explicit source flag. It also shows why naive native
environment restarts are risky: 21 of 118 observed Slumbot bet actions are
soft-mapped below a 0.95 top-weight threshold into the local 9-action
abstraction. The next decision is architectural, not a knob choice. A full
DAGS-style environment restart requires reconstructing local `PokerState`
transitions from Slumbot continuous bet strings, which is nontrivial and may
introduce a new abstraction mismatch. A safer immediate step is a feature-level
hard-state update: consume the verified hard-state observations in a
variance-reduced search/RL objective, require held-out decision-impact
improvement, and keep the contract marked non-deployable until it proves
transfer.

The first feature-level hard-state policy probe failed this gate. It could fit
source records but did not generalize to held-out hard states, and the
multi-trace leave-frontier-out version was worse despite positive held-out
oracle EV. This means the issue is not absence of signal; it is a mismatch
between detached feature-policy fitting and the decision/search process. The
next continuation should not scale this policy head. Prefer either (1) a
continuous-bet trace-state reconstructor so DAGS-style self-play can truly
restart from those states, or (2) an in-resolver/variance-reduced update that
uses hard states as search queries while staying inside the solver's decision
loop.

The realized-trajectory-return search-guided actor bridge has now had its
smallest fair smoke and failed. It fixed the previous "local scorer only" issue
by assigning complete-hand returns to the behavior actions actually taken, but
both the matched base and search-improved policy-head actors regressed held-out
fixed-root selected-action value, policy EV, and oracle gap. This points to
variance and counterfactual credit assignment as the active bottleneck. The next
continuation should not tune return temperature. It should implement or test a
variance-reduced search-guided RL update, such as VRPO/Q-boosted
Expected-SARSA(lambda) under search-guided behavior, or a search-derived
counterfactual advantage update with the same matched decision-impact gate.

The first VRPO-style behavior-only Q-boosting smoke is also not enough. Adding
`--behavior-mode q_boosted` to raw-sequence q-lambda PPO changed the sampling
policy and used CUDA correctly. A review fixed the Q-mode PPO contract so
expected-Q baselines are frozen at rollout time. After that fix, the matched 1k
duplicate-swapped H2H versus policy behavior was weakly positive but still
negative-lower95 (`mean=+0.002858`, lower95 `-0.008387`). The native league
check against stronger controls failed (`best_worst_lower95=-0.037461`). Do not
run Slumbot from the 1k checkpoint. Because the correction changes the training
contract used by older q-lambda checkpoints, the next continuation should run
one matched corrected h512/2k policy-vs-Q-boost A/B. If that fails native
controls, move Q/search information into the policy-improvement target itself
or pivot back to learned public-belief/search updates consumed inside
resolving.

The matched corrected h512/2k A/B did not fail. Q-boosted behavior beat the
corrected policy-behavior twin with positive 5k lower95 and beat NFSP10k plus
flat q-lambda 10k, but it tied raw-sequence q-lambda 10k and lost to CTDE
q-lambda 10k in 20k confirmations. The next continuation should not be Slumbot
yet. Run exactly one corrected CTDE + Q-boost h512/2k test, because it combines
the two locally useful ingredients under the same self-play/RL philosophy. If
that still fails CTDE/native controls, stop PPO-family scaling and pivot back
to learned public-belief/search-state updates consumed inside resolving.

The CTDE + Q-boost test has now failed the strict native league. It improves
random-eval payoff and beats NFSP10k, but it does not beat the non-CTDE
Q-boost checkpoint or the CTDE/raw-sequence controls with positive confidence.
Therefore the next continuation is no longer another PPO variant. The workflow
should pivot back to a resolver-consumed learned object: public-belief/search-
state updates trained on root-disjoint traces, or exact GPU resolving frontier
work that directly improves root decisions under the same budget.

The refreshed exact CUDA frontier says the next concrete baseline is CFR125
versus CFR150 on held-out public roots: CFR125 reached `1.0` top-action
agreement, `0.060359` L1, and zero illegal mass at about `432 ms/root`. The
next continuation should use this as the teacher/baseline. Either integrate
this exact search budget into a controlled live/self-play path, or train a
learned resolver update that beats the CFR100/125 frontier by quality per
millisecond on root-disjoint states.

2026-05-22 update: the old low-state warm-start interface is rejected, but the
solver-consumed update interface has a valid positive control. A CPU diagnostic
`iteration_update_fn` now lets CFR inspect reach/value/regret/strategy state
between iterations and replace regret or strategy sums consumed by the next
iteration. On 8 root-disjoint turn states, an oracle that copied the CFR25
selected-node regret and strategy state into a CFR5 solve exactly matched the
CFR25 root policy (`0.0` L1/KL, `1.0` top-action agreement) with mean latency
`2180 ms` versus `9202 ms` for CFR25. This is not a strength result because it
uses teacher state at inference time. The next continuation target is a learned
in-resolver update diagnostic trained on disjoint traces and judged against
uniform CFR10/CFR25 by root policy L1/KL, top-action agreement, and latency.
Do not return to external eta selectors, final-policy imitation, or Slumbot
confidence runs until this learned update object either passes or is falsified.

The first learned update through that hook is falsified in the shallow
aggregate-policy form. A trace-delta MLP trained on root trace summaries and
consumed by the hook improved vanilla CFR5 only marginally (`0.5767` vs
`0.5951` mean L1) and lost to uniform CFR10 (`0.4148` mean L1, `0.875`
top-action agreement). This preserves the hook direction but rejects root
policy-vector injection. The next attempt must predict/update a richer
per-hand regret/strategy field or improve exact CFR runtime; it should not
add another scalar selector or policy-target fit metric.

The existing low-state per-hand regret/strategy checkpoint also failed when
reused as a closed-loop hook update. It had zero illegal mass and was
root-disjoint, but hook-CFR5 reached only `0.6961` mean L1 and `0.375`
top-action agreement versus CFR25, while uniform CFR10 reached `0.3847` and
`0.875`. This rejects both current learned hook surfaces: aggregate root-policy
injection and static per-hand field prediction. The next principled move is
not a small architecture or epoch tweak. Either train a model specifically on
closed-loop hook trajectories with a direct CFR10/CFR25 decision loss, or shift
to decision-preserving exact search acceleration so the runtime search budget
can increase without adding learned miscalibration.

The exact-search acceleration check keeps the existing CUDA path but rejects
the newer segmented path for the active shape. `torch-levelsync-cuda` matched
CPU top actions on 8/8 root-disjoint turn states and was about `55x` faster,
though dense policies differ slightly because of float32 recurrence drift
(`0.00836` mean L1). `segmented-cuda` matched `torch-levelsync-cuda` top
actions but was slower (`1.38x` candidate/reference latency ratio), so it is
not the next lever. The next experiment should use `torch-levelsync-cuda` as
the exact evaluator and either spend the saved budget on stronger CFR teachers
or train a genuinely closed-loop update learner against those teachers.

The first post-synthesis exact-budget frontier supports that choice. On 16
root-disjoint turn states, `torch-levelsync-cuda` improves smoothly against a
CFR100 teacher as budget rises: CFR5 `0.8671` mean L1 and `0.50` action
agreement, CFR25 `0.4323` and `0.6875`, CFR50 `0.2273` and `0.875`, with CFR50
mean latency `173.5 ms`. The next continuation should treat CFR50/CFR100 exact
search as the current teacher/runtime improvement source. Before another neural
update model, verify whether this stronger exact budget improves fixed-state
or live-profile decisions under the actual runtime budget; if learning is used,
the target must be the closed-loop CFR5-to-CFR50/CFR100 improvement.

The higher CFR200 frontier narrows the blocker further. On 8 roots, CFR50,
CFR100, and CFR150 all matched the CFR200 top action, while dense L1 improved
from `0.3118` to `0.1451` to `0.0657`. Live-scale exact search is therefore
not the most suspicious failure source for these turn/river shapes. The next
mainline should shift one level up: improve the self-play blueprint/range
object that feeds the resolver, preferably with search-improved self-play or a
public-belief policy/value training contract where the deployed resolver and
training target share the same distribution.

The first recurrent dynamic search-state version is also falsified at the
current data scale. A GRU over the ordered CFR trace from iterations `0..5`
fit the 32 train roots but failed the root-disjoint 32-root holdout:
predicted-policy L1 was `0.6631`, worse than low CFR5 `0.5102` and uniform
CFR10 `0.3288`, and top-action agreement was `0.5625` versus uniform `0.84375`.
This retires small-data recurrent trace tuning. A future update-model attempt
must first collect a substantially larger root-disjoint trace corpus or change
the supervision target; otherwise prefer fused uniform GPU search and early
street blueprint calibration.

The 96-root follow-up does not justify collecting a larger corpus for the same
sequence-to-policy target. The learning curve is weak and remains below low
CFR5 and uniform CFR10. The next target should now change the learned object or
the compute frontier: exact GPU search acceleration, a learned multi-iteration
update operator validated inside the resolver, or a decision-aware
public-belief value/search state. Do not run another plain GRU trace-policy
scale check without a methodology review explaining the new mechanism.

The required first gate is not Slumbot. It is a root-disjoint resolver A/B:
compare low-budget vanilla CFR+ against low-budget neural-warm-start CFR+, both
against the same higher-budget teacher, reporting root action L1/KL,
top-action agreement, illegal-action count, and latency.

Use modern neural architecture where it helps the learned game-theoretic
primitive: public-belief encoders, set/card encoders, action-sequence attention,
residual trunks, equilibrium/RL update heads, uncertainty heads, and mixed
precision are all valid candidates. They must still be evaluated by downstream
decision quality. A larger network that improves offline target fit but fails
the root-disjoint resolver A/B is not progress.

The first implementation of this gate is now in place. Reusing the old
joint-PBS policy checkpoint as the initializer failed the 64-root holdout gate:
root-disjointness passed, but warm-starting worsened L1/KL and action agreement
while adding substantial latency. The next target should therefore train a
native regret-field initializer from low-vs-high resolver deltas, not reuse a
policy-imitation head. The training target should include per-action regret or
advantage residuals needed to move the low-budget solver toward the teacher,
plus a latency budget that avoids per-hand inference inside every resolver node.

The oracle version of that target now passed. Copying the 25-iteration teacher
solver's selected-node `regret_sum` and `strategy_sum` into a 5-iteration solve
made the low-budget solver much closer to the teacher on 64 held-out roots
(`0.0697` L1 vs `0.5254` vanilla, `0.0050` KL vs `0.2602`, `0.9688` top-action
agreement vs `0.7656`, `1.04x` latency). This validates the warm-start
interface and narrows the next task: export root-disjoint teacher solver-state
labels and train a network to predict both regret and average-strategy mass at
the public decision node.

The first target exporter smoke is also in place. Four train roots produced
`4,512` all-hand rows with public features, private policy features, belief
rows, legal masks, low-solver fields, and teacher `regret_sum`/`strategy_sum`
fields. The next implementation step is a learner for these labels, followed by
the same root-disjoint warm-start resolver gate.

The first learner did not clear the offline baseline. A direct model and a
low-state residual diagnostic both failed to beat the 5-iteration solver's
average policy on 64 unseen roots. This suggests the target must become more
search-aware than per-hand supervised field regression, or the evaluation
should learn only selective corrections where the low solver is demonstrably
wrong rather than distilling all rows uniformly.

A fixed solver-update probe is now also falsified. `dcfr_plus` was added as an
opt-in diagnostic update and compared with 5-iteration CFR+ against the same
25-iteration CFR+ teacher on the 64 held-out roots. It failed the gate: L1
worsened from `0.5254` to `0.5404`, KL worsened from `0.2602` to `0.3533`, and
top-action agreement fell from `0.7656` to `0.7500`. Keep `cfr_plus` as the
default. The useful next step is not to tune discount exponents; it is to
collect or learn a more search-aware correction signal, or test a better
researched predictive/learned update with the same fixed A/B gate.

The source-backed PDCFR+ update has now been tested under the same rule. Its
methodology review passed because it is a published predictive CFR+ diagnostic,
not a Slumbot-specific patch, but the 64-root root-disjoint gate failed badly:
mean L1 worsened to `0.7062`, KL worsened to `0.9894`, and top-action agreement
fell to `0.6250` versus low CFR+ `0.5254`/`0.2602`/`0.7656`. This rules out
fixed published solver-update swaps as the immediate path. Keep them opt-in
negative controls and return to learned counterfactual/search-derived
corrections evaluated inside the resolver.

The objective is now sharpened further: do not make Slumbot the thing to hack.
Slumbot is a transfer benchmark. The preferred research path is an elegant
general method, closer in spirit to AlphaZero/Student-of-Games/DeepNash than to
benchmark-specific poker patches: self-play learning, public-belief
representations, equilibrium-oriented RL dynamics, and search or resolving only
where imperfect information makes pure lookahead invalid. The next local work
should therefore compare two root-cause families before another engineering
sweep:

- **Decision-focused learned-search correction:** generate interventions that
  actually reduce root action L1/KL after low-budget resolving, then train a
  model to predict those search-effective corrections rather than raw teacher
  fields.
- **Pure game-theoretic RL alternative:** review and prototype the smallest
  NFSP/RM-FSP, R-NaD/DeepNash-style, ReBeL, or Student-of-Games-inspired module
  that can run locally and be judged by the same root-disjoint decision gate.

The first pure-RL alternative has now been built as a native full-deck NFSP/DQN
control branch: episode-level NFSP mode sampling, same-player next-decision
DQN targets, reservoir average-policy memory, target-network bootstrapping,
checkpoint save/load, and duplicate-swapped native H2H are all in place. This
branch is useful infrastructure, but the results do not justify making it the
mainline: 10k CUDA episodes failed to beat the 2k checkpoint with positive
confidence, and the Double-DQN target-network variant was negative/inconclusive.
Do not keep scaling native NFSP alone unless it clears duplicate-swapped H2H;
the next principled work should return to learned-search/public-belief methods
or a substantially stronger game-theoretic RL mechanism such as R-NaD-style
regularized policy dynamics coupled to an evaluation gate.

The first R-NaD-style local primitive also has a falsified control result. A
native full-deck regularized-policy pilot trained one policy net from sampled
terminal payoff targets and passed legal-mask/unit checks, but the 2k CUDA
checkpoint lost badly to the 10k native NFSP reservoir control under 1k paired
duplicate-swapped H2H (`mean=-0.1122`, lower95 `-0.1402`). This does not reject
regularized Nash dynamics; it rejects using sparse terminal sampled payoff as
the policy-update signal. The next version must use a stronger counterfactual
or search-derived advantage target before any scale-up.

The latest search-diagnostic result points to dynamic CFR traces rather than
more static cut labels. A denser static cut-impact predictor failed root-disjoint
correlation (`r=-0.0441`), so the resolver now exposes per-iteration
`regret_sum` and `strategy_sum` snapshots at traced public nodes. A 4-case
smoke showed the trace is mechanically useful: 100 records, zero skips, and
mean L1 to the final average strategy falling to `1e-08` at the final
iteration. The next experiment should use these traces to build a fixed,
root-disjoint predictor of low-budget solver error or convergence uncertainty,
then test whether that predictor improves low-budget resolving against the
higher-budget teacher. Do not treat the trace script itself as a gameplay
method.

The first version of that predictor found usable signal. On a fixed 32-root
train / 32-root holdout split, iteration `5` trace features improved holdout
L1 prediction over the train-mean baseline (`0.2214` MAE vs `0.2802`) with
Pearson `0.4283`; predicted-high roots had actual L1 `0.7981` versus `0.4424`
for predicted-low roots. This is exactly the kind of mechanism-level evidence
that was missing from static cut metadata. The next step should be a selective
search budget gate: use early trace features to decide where to spend extra
CFR iterations, and compare equal or explicitly reported compute budgets
against the 25-iteration teacher on root-disjoint states.

That first selective budget gate is now negative. The binary top-quintile rule
selected `7/32` roots and improved low trace L1 (`0.3456` adaptive vs `0.5202`
low), but it did not beat a uniform comparable-iteration budget (`0.3119`).
This is a useful falsifier: trace features identify some hard roots, but a hard
switch is too lossy. The next continuation target should predict a residual
correction to the low solver's strategy/regret field, or learn a smooth compute
allocation, and continue to report uniform-budget baselines.

The first shallow residual did not work either. A fixed linear trace-to-final
policy map was worse than both low and uniform traces (`0.7235` L1 vs `0.5202`
low and `0.3119` uniform). This rules out the simplest residual shortcut. The
next principled path is not another linear probe; it should either train a
nonlinear public-belief trace model with root-disjoint data and uniform-budget
baselines, or step back to a counterfactual advantage target that matches how
CFR updates actually move the average strategy.

The current synthesis chooses the counterfactual delta version as the next
single test: predict the low-to-later regret/strategy change, not the final
policy. This keeps the target closer to the update operator and avoids treating
the 25-iteration strategy as a one-step imitation label.

The shallow low-to-later delta implementation failed, so the target family
survives only in a stronger form. Do not keep adding linear residual probes.
The next version must use either a nonlinear public-belief trace model with
enough root-disjoint data or a target closer to counterfactual regret/advantage
updates, and it must continue reporting low and uniform baselines.

A methodology review now permits exactly that bounded next step. The reviewed
decision is `proceed`, but diagnostic-only: a nonlinear public-belief
trace-delta model may be tested if it predicts update-aligned regret/strategy
deltas, keeps CFR+/resolving as the correction operator, reports low and
uniform trace baselines, and leaves Slumbot/promotion surfaces untouched. The
tracked manifest is
`docs/research_protocols/poker_review_manifests/20260514T221524Z-nonlinear-public-belief-trace-delta-model.json`.

The first nonlinear implementation on the existing compact trace rows is also
negative. A CUDA MLP slightly beat the low trace on target fit
(`0.3216` vs `0.3266` L1 to iteration-10 target) but failed decision quality
(`0.5253` final L1 vs `0.5202` low and `0.3119` uniform). This rules out
capacity tuning on the current trace-row features as the next move. The next
trace attempt must add richer public-belief/counterfactual inputs or switch to
a stronger regret/advantage target.

Adding public-belief cache features and range-shape summaries improved the
same diagnostic but did not pass it. The enriched MLP reached `0.4983` final
L1, better than low `0.5202`, but still much worse than uniform `0.3119`.
This narrows the blocker further: context helps, but the low-to-uniform
policy-delta target is still not the right learned object. The next step should
target counterfactual advantages/regret fields or test predicted fields inside
an actual resolver update.

The first direct resolver-in-the-loop test of the trained regret/policy-field
checkpoint is also non-promotable. On the same 64 root-disjoint holdout roots,
the low-state warm-start checkpoint improved mean L1 slightly (`0.5028` vs
`0.5254`) and reduced all-in probability gap (`0.0117` vs `0.0799`), but KL
worsened (`0.2688` vs `0.2602`), top-action agreement fell (`0.7031` vs
`0.7656`), and warm-start latency was `7.66x` vanilla low-budget CFR. This
confirms that predicted fields can sometimes move the root policy in the right
direction, but the current per-hand supervised field target is not aligned
enough with the action decision and is too slow. The next step should not be a
bigger MLP; it should learn a selective or counterfactual correction that is
evaluated by action-quality improvement per unit of extra compute.

The raw counterfactual-advantage trace diagnostic is now also negative. The
resolver can expose per-action child values at traced public nodes, and the
diagnostic derives an `advantage_policy` for the acting player. On the same
32-root holdout split, iteration-5 instantaneous advantage policy was much
worse than both the low average strategy and a uniform iteration-10 baseline
(`0.9789` L1 vs `0.5202` low and `0.3119` uniform; top-match `0.3750` vs
`0.7813`/`0.8438`). Cumulative regret policy was mildly better than low in L1
(`0.4745`) but still lost to uniform. Do not train a raw one-step advantage
imitator; use the trace as input to a smoother regret-update or budget policy.

The strongest cheap baseline is now plain extra CFR, not a learned shortcut.
Running uniform 10-iteration CFR+ against the same 25-iteration teacher on the
64 held-out roots improved mean L1 to `0.3630`, KL to `0.1251`, and top-action
agreement to `0.8125`, with mean latency about `350 ms` versus `182 ms` for
5-iteration CFR+ and `859 ms` for the teacher. Any learned warm start, trace
model, or update-rule candidate must beat this 10-iteration baseline on both
decision quality and latency-adjusted value before it is worth scaling.

The budget baseline now has a dedicated evaluator. Use
`scripts/eval_cfr_budget_frontier.py` for future CFR budget claims because it
builds one solver per root, solves the teacher once and each requested budget
once, then reports quality and latency without pretending this is a promotion
gate. The first 64-root frontier confirms the current target: CFR10 reaches
`0.3630` L1 and `0.1251` KL at about `1.94x` CFR5 latency. Solver reuse
improved the 8-root cProfile smoke from `32.711s` to `18.787s`; CPU CFR
strategy reuse improved it to `16.687s`; exact seven-card evaluator reuse
improved it to `13.913s`, all while preserving 64-root quality metrics and
zero illegal mass. The next acceleration target is a coarser batched/fused CFR
boundary or learned public-belief correction, not another surface-level
evaluator cache. A learned or fused method must improve this
decision-quality-per-compute frontier, not just beat CFR5.

The current `torch-cuda` resolver backend is not the answer to the GPU-use
concern. A fresh CFR10 fixed-state smoke over the four built-in public states
averaged `1219 ms` with `torch-cuda` versus `743 ms` on CPU. This means the
next compute improvement should not be "force torch-CUDA"; it should either
batch/fuse the CFR recurrence so the GPU receives coarse work, or move the
learned-search boundary to a compact public-belief correction that reduces
solver calls. Any such change must clear the CPU CFR10 baseline above.

The first matrix/fused CFR footprint diagnostic makes the compute target more
concrete. On the 64-root turn holdout used by the budget frontier, public-state
trees are graph-small but hand-state-heavy: up to `1,257` nodes with `1,128`
private hands, `93.4 MiB` mean solver state per root, and `5.98 GiB` if all 64
roots are held concurrently. The transition graphs are tiny. Therefore, the
next GPU step should be a level-synchronous batched/chunked CFR recurrence over
node-by-hand tensors, not another per-operation torch port or adjacency-only
matrix experiment. Keep CPU CFR10 as the quality/latency baseline.

The first local level-synchronous recurrence is implemented as `cpu-levelsync`
and should remain diagnostic-only. It preserves average-strategy behavior and
legality, but the 8-root budget smoke was slower than the current CPU solver
(`158/294 ms` for CFR5/CFR10 versus `120/243 ms`). This falsifies a NumPy
level-synchronous rewrite as a CPU optimization. The remaining principled
compute path is either a real fused/chunked GPU recurrence or a learned
public-belief correction that reduces solver work while beating uniform CFR10.

The Torch version of that same level-synchronous recurrence now provides the
first real GPU resolver win. On the 64-root turn holdout,
`torch-levelsync-cuda` preserved budget-frontier quality and cut CFR5/CFR10
solver latency to `39.7/55.4 ms`, while CPU was previously `144.8/280.4 ms`.
CPU CFR25 versus Torch CUDA CFR25 parity over the same roots had perfect
top-action agreement and small strategy drift (`0.0281` mean L1). The next
single continuation target is still **validation before promotion**. The
Slumbot smoke path now accepts explicit `torch-levelsync-cuda`, and a 3-hand
live smoke passed with two solver calls and no parse/API errors, but mean live
reported solver latency was `1270 ms`. Before changing `auto`, isolate whether
the remaining live cost is range construction, solver-tree construction,
callback pruning, or CUDA recurrence overhead, then compare against CPU on the
same fixed mixed turn/river states.

That attribution is now partially done. The mixed four-state benchmark shows
that CUDA has solved only the recurrence part: `torch-levelsync-cuda` cut CFR10
recurrence time from `325.4 ms` to `94.0 ms`, but residual setup overhead stayed
about `235 ms`. The next single continuation target should therefore be setup
amortization: cache or precompute board/active-hand payoff components where it
does not change ranges or action semantics, then rerun the same CPU/CUDA
attribution benchmark and the live smoke. A valid optimization must reduce
`avg_solver_overhead_latency_ms` without changing legality, root actions, or
solver strategy beyond normal numeric tolerance.

The first setup optimization is complete but only partially solves the
problem. Incremental turn evaluation reduced fixed-state CUDA overhead to
`205.2 ms` and total latency to `298.4 ms`, but a post-change 3-hand Slumbot
smoke still had a `1338 ms` reported solver call. The next continuation target
should instrument or reduce the remaining live-only cost around range pruning,
active-hand selection, tree construction, and CPU-to-GPU tensor preparation.
Do not flip `auto` until same-state live-style attribution shows the full path
beats CPU with margin.

The live-budget check changes that interpretation: Slumbot uses at least
`150` CFR iterations, not the `10`-iteration smoke budget. At `150` iterations,
the fixed mixed-state benchmark shows a large CUDA advantage (`818.5 ms` total
vs CPU `4871.0 ms`). The next continuation target should therefore shift from
"make CUDA recurrence faster" to "make live resolver policy safer and cheaper":
evaluate whether lower or adaptive iteration budgets preserve the CFR150 action
quality on root-disjoint states, then expose a reviewed live-budget policy if it
beats the current fixed `150/250/350` heuristic on decision quality per compute.

The root-disjoint budget frontier now points to CFR100 as the first candidate
budget policy. CFR100 preserved `98.44%` of CFR150 top actions with `0.1247`
mean L1 and cut mean solver time to `354 ms` from the CFR150 teacher's roughly
`525 ms`; CFR75 is faster but less aligned, and CFR125 is more faithful but
smaller compute savings. The opt-in `fast-live` budget profile now implements
that single reviewed policy with `100/150/250` iterations and leaves the
default `live` profile unchanged at `150/250/350`. A sparse Slumbot smoke
exercised the profile with five CUDA solver calls and no API/parse errors, but
this is not promotion evidence. The same-state live-profile comparison is now
available and has run on the 64-root holdout: `fast-live` was materially
cheaper (`593.6 ms` vs `919.7 ms`, `0.6455x`) and close in distribution
(`0.1276` L1, `0.0275` KL), but two roots changed top action. The next
continuation target should inspect those disagreement states and decide whether
a bounded live confidence comparison is worth the variance. Do not add a second
budget knob before that falsifier.

The disagreement inspection now argues against immediate confidence spend.
One disagreement root becomes progressively closer to a CFR500 teacher as
budget rises, but the other is non-monotonic: CFR150 and CFR350 match the
teacher while CFR250 does not. The next target should therefore be a
teacher-aligned boundary detector or disagreement filter for live budget
selection, evaluated on root-disjoint states. Keep it as a diagnostic gate; do
not turn it into a Slumbot-specific budget sweep.

The CFR500 holdout frontier confirms the shape of that target. Uniform CFR350
is the cleanest tested approximation to the CFR500 teacher but costs about
`1.19 s` per solve. The current `fast-live` profile is cheaper and slightly
better on top-action agreement than `live`, but it is worse on L1/KL. The next
implementation should therefore estimate action-boundary uncertainty from
available profile/frontier signals and selectively escalate ambiguous states;
success means improving teacher L1/KL per millisecond against `live`, not
maximizing a Slumbot smoke score.

The first boundary-signal diagnostic is positive enough to formalize. Profile
L1 between `live` and `fast-live` identifies high-error roots, and a fixed
train-top-4 threshold improved held-out L1/KL in both half-splits while adding
only about `3-5%` latency. The next implementation should turn this into a
protected diagnostic gate that reports train-derived threshold, selected roots,
L1/KL/latency against CFR500, and low/live/uniform-CFR350 baselines. Keep it
off the Slumbot live path until that gate survives a root-disjoint split.

That formal gate now exists and passes both half-splits on the current 64-root
pool. The next continuation target is scale, not live deployment: generate or
reuse a larger root-disjoint public-state pool, rerun the CFR500 frontier only
where computationally feasible, and test whether profile-L1 selective
escalation still improves L1/KL per millisecond against `live`. If it does not
hold, return to trace-derived uncertainty rather than tuning the top-k.

The 128-root scale-up passed only as an offline oracle. After adding naive
online decision latency, direct profile-L1 deployment fails the compute premise:
it would cost about `1.57-1.67 s` per solve (`1.77-1.83x` live) because it must
run both `fast-live` and `live` before deciding whether to escalate to CFR350.
The next target should therefore learn a cheap boundary predictor from already
available state or early-trace signals, using profile-L1 as a teacher label.
Success means preserving most of the selective L1/KL gain without the extra
profile solve; failure means return to uniform CUDA budgets or a different
search-boundary objective.

The first cheap predictor is now a negative control. A feature-only ridge gate
failed to beat a mean baseline on profile-L1 prediction in both root-disjoint
split directions, even though downstream selective escalation improved by
chance. Do not tune ridge alpha or feature subsets. The next continuation
target should use richer predeclared uncertainty evidence, preferably an early
CFR trace or blueprint-policy entropy/margin signal, and must still beat the
same mean-baseline and downstream L1/KL gates before any integration work.

That richer trace attempt has now also failed, so the target should change
rather than the feature set. Trace iteration-5 plus public-belief context did
not predict profile-L1 better than a mean baseline and recovered none of the
oracle selected roots. The next principled target is a direct decision objective:
predict whether extra iterations improve the same root against CFR500, or learn
a single-solve early-stopping/continuation policy from trace convergence that
does not require running two budget profiles.

The direct-decision scalar target has also failed under the same gate. Predicting
`live_l1 - CFR350_l1` did not beat a mean baseline for either feature rows or
trace rows. The next continuation target should stop using shallow scalar
selectors. Use a mechanism that observes the solver's own sequence while it is
already running: a learned early-stopping or continuation policy based on trace
convergence, or a sequence model that predicts the future policy trajectory.

The existing top-predicted-error trace budget gate remains a negative control:
it improves over low iteration 5, but loses to uniform iteration 10 on both
matched split directions. Therefore the next target should not be another
top-k escalation rule. It should ask whether the whole trace sequence can
predict a stopping/continuation decision that beats a fixed uniform budget.

The existing nonlinear trace-delta MLP has now been rechecked on those matched
traces and failed more strongly than the older compact split result. It fit the
tiny train roots to zero loss on CUDA, but holdout predicted-policy L1 was worse
than the low iteration-5 trace and far worse than uniform iteration 10 in both
directions. This retires single-row trace-delta MLP tuning as a continuation
target. The next bounded experiment should either model the trace sequence
(`0..5` or `0..10`) as a solver trajectory, or learn a direct stopping/
continuation action that is evaluated against the fixed uniform iteration-10
baseline.

The first shallow sequence-level check also failed. A ridge model over the full
iteration `0..5` trajectory did not beat train-mean target MAE in either
matched split and its adaptive continuation policy still lost to uniform
iteration 10. This retires the cheap trace-selector line for now: low-iteration
trace state is informative, but the information is not stable enough under the
current small root-disjoint splits and linear decision rule. The next
continuation target should change scale or mechanism: either make uniform CFR
cheap through a fused/batched GPU solver path, or learn a recurrent
solver-dynamics/update model trained on a larger root-disjoint trace corpus and
judged by the same uniform-budget baseline.

The recurrent model branch has now been tested before collecting more data and
failed for exactly that reason: train fit did not transfer across the 32/32
root split. Do not treat this as an architecture-only failure. It is evidence
that the current trace corpus is too small and heterogeneous for learned
solver-dynamics generalization. The immediate continuation should therefore
favor larger trace data collection only if it is paired with a clear gate, or
return to uniform CUDA CFR and blueprint/search calibration work that already
has stronger local evidence.

The next calibration work has started with the right kind of gate. The new
fixed-state early-street diagnostic compares candidate and reference policies on
identical preflop/flop states and failed the current `iter1000` checkpoint:
`22.06%` preflop and `33.33%` flop top-action all-in, with `1.1537` mean L1 to
the reference. The continuation target is now blueprint calibration that changes
the learned policy distribution itself, not a manual action filter and not more
Slumbot spend.

The immediate evidence now favors the fused GPU search path. On 128 fixed
roots, `torch-levelsync-cuda` CFR100/125 remained close to a CFR150 teacher,
with CFR125 reaching `99.21875%` action agreement at lower latency than CFR150.
The next continuation target should therefore be a same-state live-budget
validation, not another trace-selector probe: keep default `live` unchanged,
compare `fast-live`/CFR125-style budgets against the live teacher on fixed
states or bounded Slumbot confidence hands, and only then consider changing
runtime budget policy.

The first bounded Slumbot latency smoke is consistent with that direction:
fast-live and live both passed 50 hands with zero API/parse errors, and
fast-live reduced mean solver latency from `1436.9 ms` to `858.0 ms` in this
sample. The next continuation target should be a statistically cleaner strength
check, not another latency smoke: use larger Slumbot confidence hands only when
API time is acceptable, or prefer local duplicate-swapped gates where variance
is controlled.

The 1,000-hand follow-up says speed is not enough. Fast-live and live were both
operationally clean, but both were negative against Slumbot, with fast-live
`-143 +/- 202` chips/hand and live `-264 +/- 233`. Since the slower live profile
did not rescue the result, the next continuation target should return to
strategy quality: train or evaluate a stronger blueprint/search-calibration
candidate, then reuse the faster CUDA profile for confidence checks only after
local gates show a real improvement.

The historical-checkpoint follow-up also says checkpoint picking is not the
answer. `slumbot_2p_iter900.pt` had looked slightly better than `iter1000` in a
larger local duplicate-swapped H2H, but its 1,000-hand fast-live Slumbot check
was worse (`-533 +/- 296` chips/hand) and mostly all-in (`836/999` selected
actions). Do not spend more live hands on old checkpoints unless a new gate
predicts live transfer for a mechanism reason. The next task should diagnose
why the blueprint overbets/all-ins under live Slumbot distributions, or build a
root-disjoint calibration/evaluation gate that catches this before API spend.

The new street-level diagnostic narrows that task: on a 300-hand `iter1000`
fast-live attribution run, all `82` all-ins came from the early blueprint path
(`60` preflop and `22` flop), not from turn/river solving. The next continuation
target should build a local early-street blueprint calibration gate that
measures regret-policy action distribution against search/stronger-policy
targets on reachable preflop/flop states. Do not patch this with a manual
no-all-in rule; the goal is to make the learned policy/search interface assign
reasonable probability mass before live Slumbot spend.

The first local version of that gate is already informative: the existing
policy-calibration target sampler now emits `per_street_target_diagnostics`.
It flags `iter900` as pathological before API play (`58.09%` preflop top-action
all-in), consistent with the later `836/999` live all-in mix. It also flags the
incumbent `iter1000` as still too aggressive (`17.33%` preflop and `30.80%`
flop top-action all-in). The next research task should turn this diagnostic
into a falsifiable candidate gate: train or select a checkpoint whose
early-street top-action all-in and mean all-in probability fall materially while
preserving duplicate-swapped H2H and fixed-state resolver behavior.

Two bounded deployment diagnostics rule out easy fixes. Non-greedy mixed
regret-matching did not reduce live all-ins or chips lost, and a hard
`--no-allin` diagnostic was still negative while increasing solver load. The
next step should not be a deployment flag change. It should change the learning
objective or self-play/evaluation signal so the early blueprint learns a less
degenerate betting distribution under the existing 9-action contract.

The restored-history 200x2k candidate closes the all-in gap but still fails live
confidence (`-172 +/- 225` over 1,000 hands) and pays a latency cost from many
more solver entries. This falsifies a too-simple early all-in story. The next
research unit should evaluate early-street value/action quality directly:
compare candidate and incumbent decisions on sampled reachable preflop/flop
states using duplicate-swapped rollouts or a stronger local teacher, then only
spend Slumbot hands on candidates that pass both distribution sanity and value
transfer.

A learned-reference policy-head calibration has now solved the first half of
that gate locally. Fitting only the `iter1000` policy head to the
`restored_history_200x2k_4x512_final` learned reference on fixed early states
passed the local distribution screen: top-action all-in dropped to `0%` on
preflop/flop and mean L1 to the reference fell to `0.6106`. The next step should
not be Slumbot promotion yet. It should ask whether this calibrated head
improves early-street action value/transfer on duplicate-swapped local rollouts
or a validated adversarial evaluator, because the reference itself did not beat
Slumbot.

The lightweight value proxy now supports continuing this branch. On the existing
restricted showdown first-action diagnostic, the calibrated policy head beat
both source and reference on selected payoff, oracle gap, and oracle-match rate.
Because this proxy assumes called actions go to showdown, it should not be the
promotion gate. The next unit should be a duplicate-swapped local self-play
comparison using the calibrated `policy-head` strategy source, with action-mix
and early first-action attribution recorded.

That duplicate-swapped comparison is now negative. The calibrated policy head
lost to both its source regret policy and the learned reference despite passing
distribution and one-step value screens. The next continuation target should not
be stronger behavior cloning from a fixed reference. It should diagnose what the
policy-head calibration changes during full-hand play: action mix by street,
early first-action buckets, and whether the policy-head-only deployment is
breaking later-street regret/search coupling.

The attribution result points away from fixed-reference cloning. The calibrated
head repaired fixed-state argmax all-ins but, in stochastic full-hand rollout,
still samples risky actions and makes common call/raise buckets lose against the
reference. The next continuation target should be rollout-aligned: either train
the policy object on its own induced trajectory distribution, or avoid deploying
a separate cloned head and return to SD-CFR/ReBeL-style regret/value rollout
selection where the played policy and the learned object remain coupled.

The obvious covered-head variant has been falsified too. Falling back to regret
outside calibrated streets still loses locally, so the next step should not be
another deployment-source toggle. It should be a rollout-aligned learner or a
checkpoint/value-network deployment path whose objective is trained and
evaluated on the same induced full-hand distribution.

The student-induced version has now failed that bar. It moved the target
distribution in the intended direction but still lost local H2H. The next
continuation target should retire reference policy-head cloning and use an
objective where rollout return or regret/value updates are part of training:
for example SD-CFR-style sampled value-net deployment, NFSP/FSP average-policy
self-play with H2H gates, or a ReBeL-like search/value loop. Do not continue by
tuning target size, target temperature, or calibration learning rate.

The first SD-CFR-style deployment check is now the live continuation target.
The restored-history iteration-checkpoint mixture beat `iter1000` regret by
`+29.09 +/- 1.69` chips/hand over three 2k-game duplicate-swapped runs and was
near-tied with restored final (`-0.53 +/- 1.05`). This is the first result after
the policy-head failures that keeps value/regret deployment coupled and still
improves a local H2H gate. The next work should not tune mixture weights by
benchmark outcome. It should add a reviewed, Slumbot-compatible SD-CFR mixture
strategy source that samples a fixed iteration checkpoint per hand, preserves
existing action mapping and solver hooks, and then runs a tiny API smoke before
any confidence spend.

That adapter now exists and passed its integration smokes. The next continuation
target is therefore a bounded falsification ladder for the exact same fixed
mixture semantics: first local duplicate-swapped and trace/mapping gates, then
only a small predeclared Slumbot confidence check if the local gates remain
positive. Do not alter mixture weights, checkpoint glob, solver profile, or
Slumbot deployment flags in response to tiny smoke chip outcomes.

The queue can now represent that local mixture falsification directly via
`enqueue-sd-cfr-mixture-falsification`, which runs the objective audit and
`eval_sd_cfr_mixture.py` for fixed candidate globs/checkpoints. Use this before
any larger Slumbot check so the continuous workflow does not silently collapse a
checkpoint mixture into one selected model.

After the first queued holdout run, the ladder must be hardened: the mean was
positive but the across-seed lower95 was negative. The gate now treats that as
failed. The next continuation target is a mechanism diagnosis of mixture
instability, not live API spend: inspect per-seed action/outcome attribution,
single-checkpoint contribution, and whether the local duplicate-swapped
baseline is too noisy or the mixture is genuinely brittle.

The single-checkpoint contribution check narrows the next step further. The
late `iter200` snapshot is weak while `iter100` is strong, so the sparse
four-checkpoint linear mixture is not a faithful enough SD-CFR average. The next
reviewed objective should be "proper SD-CFR snapshot semantics": save or restore
a denser sequence of value networks, define the averaging weights before
evaluation from the training iteration contract, and rerun the strict lower95
gate. Do not promote `iter100` directly; that would be checkpoint selection on
the diagnostic set.

The first-action outcome diagnostic reinforces that direction. In a fresh
300-hand restored-history attribution run, the largest first-action bucket was
`call/chk`, and it was strongly negative (`-402` chips over `186` hands). This
does not justify a manual "raise more" rule; it just says the current local
self-play gate is not measuring early passive-action value against a strong
opponent. The next implementation target should be a local approximate
best-response or adversarial early-street evaluator that punishes weak passive
blueprint behavior before Slumbot spend.

The first approximate-BR attempt is not that evaluator. A DQN-style frozen
checkpoint BR with Monte-Carlo returns failed its passive-call positive control,
so low exploitability from that script is non-evidence. The next attempt should
start from a positive-control-first design: exact/brute-force response in a
small public-state abstraction, stronger offline fitted Q with balanced action
coverage, or a restricted preflop/flop action-value evaluator that can exploit
known passive and over-all-in controls before touching real candidates.

The first replacement is a restricted showdown action-value diagnostic. It
does not train a model and is not exploitability: it estimates current-hand
showdown equity against a random opponent range, then scores each legal first
action by pot odds under an "opponent calls, then checkdown" abstraction. The
positive controls pass: premium aces prefer all-in pressure, while seven-deuce
facing a large legal raise prefers fold. A 64-root smoke produced
`oracle_best_mean_payoff=97.02` chips versus `call_mean_payoff=3.75`, with
positive controls passing. Use this only as an evaluator-health and
early-action value sanity check; candidate promotion still requires
duplicate-swapped H2H, fixed-state resolver evidence, and Slumbot confidence.
Checkpoint attribution on the same seeded 128-root preflop distribution shows
why the next method must improve early action quality rather than only reduce
all-in frequency: `iter1000` selected all-in on `105/128` roots and had
`checkpoint_mean_selected_payoff=-17.36` with an `89.12` chip oracle gap, while
the restored-history checkpoint selected call on `93/128` roots and was also
negative (`-21.18`, oracle gap `92.94`). This is not promotion evidence; it is
a local falsifier showing both over-pressure and passive variants miss a
simple hand-conditioned value signal.
A wider 128-root historical sweep did not reveal a ready checkpoint: `iter300`
was least bad by restricted selected-action payoff (`+3.04`) but had only
`6.25%` oracle match, while later and restored/search-consistency checkpoints
all retained large oracle gaps (`80-122` chips). This rejects checkpoint
selection as the main fix and points toward changing the learned early policy
or advantage target so it captures hand-conditioned action value before
turn/river search is asked to repair the game.
Raw advantage alignment confirms this is not just a deployment argmax issue.
On the same 128 roots, incumbent advantage-vs-restricted-value correlation was
`-0.028`, and restored-history was `-0.128`. The advantage network is not
ranking root legal actions by even this simple value proxy, so the next
publishable mechanism should change the learning signal or architecture around
public belief/private hand-conditioned action values rather than only tuning
action sampling, checkpoint selection, or Slumbot flags.
A restricted-value capacity probe makes that interpretation more precise. The
same `ValueNetwork` class fit clean restricted root labels with
`holdout_top_action_match=0.8203` and `holdout_mean_action_corr=0.7949` after a
small CUDA run (`512` train roots, `128` holdout roots). This argues against
"the MLP is simply too small" as the main explanation. The stronger next method
should improve the generated game-theoretic target/data distribution for early
actions, not merely increase hidden size or layers.

Related work supports this boundary choice. Kim's 2024 GPU-CFR paper frames
CFR as dense/sparse matrix and vector operations and reports speedups that grow
with game size (`https://arxiv.org/abs/2408.14778`). DeepStack shows the other
principled option: maintain ranges and opponent counterfactual values, then use
neural values inside continual re-solving rather than replacing search outright
(`https://arxiv.org/abs/1701.01724`). DDCFR is relevant future work because it
learns dynamic discounting instead of hand-tuning fixed CFR update weights
(`https://openreview.net/forum?id=6PbvbLyqT6`), but prior local DCFR/PDCFR
negative controls mean learned discounting should not be added until the fused
or learned-search boundary is clear.

Legacy note: the original leaf-only value objective is retained below as
historical context and negative evidence.

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

- NFSP: https://arxiv.org/abs/1603.01121
- DeepNash / R-NaD: https://arxiv.org/abs/2206.15378
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

The first learned action-sequence attempt also failed. A GRU over public
action tokens and log-scaled bet amounts fit train (`MAE 0.0677`) but worsened
the high-bet holdout (`MAE/RMSE 0.3944/0.4936`) relative to the same-seed
no-action baseline (`0.3705/0.4679`) and the best constant (`0.3173/0.3961`).
Keep the diagnostic encoder available for falsification, but do not promote it
into gameplay. The next hypothesis should change the target or factorization:
for example, train topology-conditioned continuation values, predict normalized
delta-to-constant residuals, or generate paired public-action augmentations so
the model must learn invariances rather than memorize action strings.

A simple topology factorization check also failed. Train-set constants grouped
by `action_shape` got `MAE/RMSE 0.3214/0.3986`, and constants grouped by
`actor_to_act,bet_count` got `0.3304/0.4108`, both worse than the global
train-mean constant `0.3173/0.3961`. Do not spend on topology buckets alone.
The next target should probe reach/range-conditioned value structure: support
entropy, top-mass, zero-sum residuals, and whether predicting residuals from a
range-aware baseline is easier than raw per-hand CFVs.

The first reach-shift diagnostic found root-disjoint high-bet train/holdout
coverage is still poor: `bbc/bbc/bb`, `cbbc/bbc/b`, and `cbc/bbbc/b` appear in
holdout but not in train, and holdout has materially more concentrated hero
reach (top-mass SMD about `0.75`, entropy SMD about `-0.64`). Completing the
full high-bet train pool increased train cuts from `96` to `119`, but the same
missing shapes remained and the model still failed (`MAE 0.3653` vs best
constant `0.3184`). The next data step should create a larger successor-frontier
pool first, then split with explicit action-shape and reach-coverage audits
before fitting any new continuation network.

That controlled split is now implemented. A `256`-case successor pool produced
`244` high-bet cut targets; the metadata-balanced cut-level split created
`182/62` train/holdout targets with no holdout-only action shapes and much
smaller reach shift (largest SMD `0.265`, hero top-10 reach SMD `0.156`). On
this cleaner split, the same CUDA Deepset continuation probe passed (`MAE/RMSE
0.1204/0.1911`) against the best constant baseline (`0.2815/0.3660`), while
action-shape constants remained weak (`MAE 0.2802`). This means successor
frontier values are learnable when coverage is controlled; the remaining
bottleneck is sparse frontier context, especially low legal-action-count states
where residual error is highest. The next principled step should improve the
continuation interface or target factorization for those sparse legal contexts,
then rerun fixed successor-cut resolver A/B before any Slumbot evaluation.

The fixed successor-cut resolver A/B was rerun with a stricter behavior gate and
a matching high-bet frontier filter. This corrected an earlier diagnostic flaw:
the high-bet-only checkpoint had been applied to every immediate successor cut.
With `min_bet_count=5`, nonmatching successors remain exact CFR and only matching
high-bet cuts use the learned continuation model. Even under that matched setup,
the checkpoint failed the resolver behavior gate. On 16 evaluated cases from a
64-case scan, 5 iterations gave action agreement `0.50` and mean L1 drift
`0.8220`; 25 iterations gave agreement `0.4375` and drift `1.0912`. Hard
successor-value replacement is therefore not ready. The next mechanism should
make the search integration safer, for example by learning calibrated residuals,
mixing learned and exact values, or using the network as a warm-start/policy
prior instead of a hard cut-value oracle.

The simplest calibration variant was tested and failed. A learned affine
calibration over successor CFV predictions was nearly identity and slightly
worsened holdout value MAE (`0.1204` raw to `0.1212` calibrated). Plugging the
calibrated values into the matched 5-iteration resolver A/B still failed
(`0.50` agreement, `0.8275` mean L1 drift). Simple scalar value calibration is
not the bottleneck. The next target should be policy-aware or search-aware:
train a boundary policy/prior from resolver targets, learn residuals conditioned
on local legal context, or train a value-mixing controller against the fixed
resolver behavior gate.

The first policy-boundary calibration is promising but not directly usable.
Training only the policy head on a `192/64` resolver-target split improved
held-out policy fit (`L1 0.2115 -> 0.0382`, `KL 0.0572 -> 0.0054`) and reduced
policy-head solver drift (`0.8880 -> 0.8097`). But its argmax selected all-in on
`29.69%` of holdout cases while the target top all-in rate was `0`. Treat this
as evidence for soft policy priors inside search, not direct policy-head action
selection. The next gate should evaluate entropy/top-action calibration and a
solver warm-start/prior path that cannot force unsafe top actions by itself.
The fixed resolver benchmark now has an optional policy-head behavior gate; the
calibrated head fails it under max all-in rate `0.05` and max mean L1 drift
`0.75`. The probability-level diagnostics explain why: calibrated mean all-in
probability matches the target (`0.1469` vs `0.1454`) and entropy is close, but
argmax behavior is unstable around near-tied soft actions.

A naive soft-prior integration was tested and failed. Mixing the calibrated
policy-head distribution into the final `5`-iteration solver strategy at weight
`0.25` did not move the solver closer to a `25`-iteration reference: mean L1
was `0.5950` for the mix versus `0.5778` for the low-budget solver alone, and
action agreement stayed flat at `0.5469`. The all-in top-action rate also stayed
at `0.6563`. Do not spend on more final-distribution mixing weights. A useful
policy prior must enter the CFR update/warm-start path, or be trained as a
residual correction against low-vs-high solver behavior.

The first node-local CFR+ warm-start attempt also failed. The solver now accepts
optional initial regret and strategy tensors, and zero initializers preserve the
old behavior. A diagnostic seeded the current public node with policy-head
regret mass for every private hand, then reran the same `5` vs `25` iteration
budget comparison. This worsened mean L1 (`0.5969` vs `0.5778` for the plain
low-budget solver), lowered action agreement (`0.5312` vs `0.5469`), and left
all-in rate unchanged (`0.6563`). Related work still supports regret-aware
warm-starting, but this simplified one-node CFR+ seeding is not enough. The next
mechanism should be either theorem-closer strategy-based warm starting or a
learned residual correction target, not a sweep over warm-start mass.

A first learned residual combiner also failed the promotion gate. It trained on
the existing `192` resolver policy targets using public features, the
`5`-iteration solver distribution, and the calibrated policy-head distribution,
then evaluated on the fixed `64` holdout targets. It beat the cheap solver
distribution by a wide margin (`0.0617` vs `0.6929` mean L1), but it did not
beat the calibrated policy head (`0.0382` mean L1) and still selected all-in as
the top action too often (`0.28125`). The useful signal is narrower: top-action
agreement improved over the policy head (`0.28125` vs `0.15625`), so ranking
information is present but not captured by plain cross-entropy. The next
mechanism should train decision-aware policy calibration from solver targets,
for example top-action margin or pairwise ranking loss, with a fixed holdout
gate.

Decision-aware policy calibration is now the first local pass in this line. A
legal-masked top-action margin loss trained from the solver target's own top
action fixed the policy-head argmax failure on the original `64` holdout:
top-1 match improved from `0.15625` to `1.0`, top all-in rate fell from
`0.296875` to `0.0`, and the fixed resolver policy-head behavior gate passed
with mean action L1 drift `0.6973` under the `0.75` gate. The tradeoff is worse
soft distribution fit (`L1 0.0382 -> 0.1110`), so this is not promotion-ready.
Next continuation target: validate the rank-aware checkpoint on fresh fixed
resolver-policy cases without tuning `rank_loss_weight` or `rank_margin`; only
then run a small Slumbot smoke.

Fresh validation failed and corrected the gate. A new `128`-case sampled
resolver-policy target set has target top-all-in rate `0.59375`, while the old
holdout had `0.0`; the rank-aware checkpoint had learned the old split's action
distribution and selected all-in `0.0` on fresh cases. A fixed absolute all-in
cap is therefore not a valid general gate. The resolver benchmark now supports a
solver-relative all-in gap gate. On the fresh set, both the calibrated and
rank-aware checkpoints fail, and rank-aware is worse (`top1_match 0.0391`,
solver all-in gap `0.8516`). Next target: build a larger policy-target pool and
stratify train/holdout by target top action and public-state pressure before
rerunning rank-aware calibration.

The coherent-label rerun partially supports that direction but fails the
behavior gate. A fresh sampled-only `384`-case target pool with uniform
`25`-iteration CPU solver labels was split `288/96` by street and target top
action. Fixed-margin rank-aware calibration improved holdout L1 (`0.8622 ->
0.6315`) and top-1 match (`0.0833 -> 0.6458`), but overselected all-in
(`0.9583` vs solver/target `0.6667`) and failed the solver-relative all-in gap.
The next target is confidence-weighted ranking: use the solver distribution's
top-vs-runner-up probability gap to decide how strongly to enforce rank
margins, rather than forcing every top label equally.

Confidence-weighted ranking improves the stochastic policy path, not greedy
argmax. On the coherent `96` holdout, it improved soft fit over fixed rank
(`L1 0.5818`, `KL 0.2578`) and matched solver mean all-in probability well
(`0.3907` vs `0.4358`, gap `0.0450`), passing the new resolver probability
gate with mean action L1 drift `0.5880`. But argmax all-in is still `0.9896`,
so do not evaluate this checkpoint with `--greedy`. Next target: run stochastic
fixed/Slumbot smoke for the confidence-weighted rank checkpoint, while reporting
both probability-level and top-action diagnostics.

The stochastic full-game local smoke did not transfer. In duplicate-swapped
head-to-head against the base restored checkpoint, both using sampled
`policy-head`, the confidence-weighted checkpoint averaged only `+1.85`
chips/hand with lower95 `-50.57` over `12288` games. The likely bottleneck is
coverage mismatch: the policy head was calibrated from turn/river resolver
targets, but full-game policy-head evaluation uses it preflop and flop too.
Next target: either generate all-street search/teacher targets, or add an
explicit evaluation/play path that uses blueprint regret before turn and the
calibrated stochastic policy head only on streets covered by resolver targets.

That coverage-aware path was implemented and falsified. `policy-head-covered`
now reads checkpoint calibration support, falls back to regret outside covered
streets, and is available in local eval, Slumbot play, and range tracking. A
metadata-correct copy of the confidence-rank checkpoint declared turn/river
coverage from its actual train targets. In duplicate-swapped H2H against the
same checkpoint using regret, the covered policy route averaged `-16.93`
chips/hand with lower95 `-55.99` over `12288` games. The simple support-mismatch
explanation is therefore not enough. Next target: stop tuning policy-head
imitation losses and move to value/search mechanisms or reached-state policy
targets that optimize full-game value directly.

Historical checkpoint selection is now a triage tool, not the answer. A restored
history round-robin over iterations `50/100/150/200` initially ranked iter100
highest by mean field delta, but focused confirmation against iter200/final
failed hard: `-165.33` chips/hand with lower95 `-228.62` over `12288` games.
Keep iter200/final as incumbent. Next target: return to search/value integration
and make successor-frontier value use safer than hard replacement, ideally by
learning uncertainty or residual structure that is checked against resolver
action drift rather than policy imitation loss.

That uncertainty direction now has a positive diagnostic. A learned metadata
risk model trained on the metadata-balanced high-bet successor split predicts
per-cut value MAE on holdout with moderate signal: full metadata Pearson
`0.4895`, top-quintile recall `0.5385`; structural-only Pearson `0.5093`,
top-quintile recall `0.4615`. The structural-only result matters because it can
be applied before deciding which successor nodes to cut, without relying on
inside-callback reach features. The next continuation mechanism should implement
selective successor cuts: only use the learned continuation value on predicted
low-risk structural cuts and leave predicted high-risk cuts exact. The required
gate remains the fixed successor-cut resolver A/B with action agreement and mean
action L1 drift, not offline value MAE alone.

The first selective-cut A/B failed that gate. The resolver now supports a
pre-solve structural risk predictor so fallbacks remain exact instead of
returning zeros from inside the cut callback. With the fixed train-median
abstention rule, selected coverage was exactly `0.5` (`26/52` matching cuts),
but behavior got worse: only `7` cases evaluated, action agreement was
`0.4286`, and mean action L1 drift was `0.8906` versus `0.8220` for prior hard
replacement. This means offline successor CFV error ranking is not the missing
piece by itself. The next continuation target should train on the quantity the
solver gate actually cares about: root action drift, cut regret residuals, or a
search-consistency objective that couples successor values to root policy
stability.

The one-cut search-impact diagnostic confirms this target mismatch. Replacing
one high-bet successor cut at a time produced mean/max root action L1 drift
`0.6291/1.6000` over `24` cut evaluations. The structural predicted-MAE score
was effectively uncorrelated with this drift (`r=0.0189`), and the median
abstention rule did not separate safer cuts (`0.6412` selected mean drift vs
`0.6219` rejected mean drift). The next continuation target should therefore
build a search-impact dataset: features for the public node plus candidate cut
metadata/reach/value predictions, labels from one-cut or leave-one-out root
action drift, and a fixed gate that tests whether predicted low-impact cuts
actually reduce resolver action drift.

A context-enriched `64`-cut audit shows that static root context is still not
enough. The diagnostic now records root action shape, root bet count, pot,
stacks, player position, and first-to-act flag. The larger audit again showed
no relationship between value-risk score and search impact (`r=-0.0186`), and a
quick group-split ridge probe over cut metadata plus root context also failed
(`r=-0.0249`, top-quintile recall `0.2`). The next target should therefore move
inside the search loop: collect dynamic reach/regret traces or train a
continuation model with an auxiliary search-consistency loss that penalizes root
strategy drift directly.

A static-belief callback diagnostic narrows this further. The target exporter
trained the continuation model on average-strategy frontier beliefs, while the
cut callback normally supplies current CFR-iteration reaches. Feeding the
callback saved target-pool beliefs for matching cuts improved the high-bet A/B
but did not come close to passing: mean action L1 drift fell from `0.8220` to
`0.7160`, and agreement rose from `0.50` to `0.5625`. Reach distribution
mismatch is therefore a contributor, not the full cause. The next target should
train or distill on dynamic callback beliefs gathered during CFR iterations,
then validate with the same fixed resolver A/B.

Callback-state DCVN training tested that dynamic-belief hypothesis directly.
A 4-root smoke using exact callback states passed the small leaf A/B, but the
scaled `32/16` root-disjoint cache failed value baselines and resolver drift
gates. The first calibration falsifier, opponent-reach-weighted Smooth L1,
worsened supervised MAE/RMSE and failed both projected and unprojected leaf
A/B. This keeps the search-boundary value-network path alive, but rejects
simple reach weighting as the missing mechanism.

A learned state/player-offset plus hand-residual head then tested the simplest
target-factorization idea without leaking label means into inference. It also
failed: supervised MAE/RMSE worsened versus the direct MSE checkpoint and
unprojected leaf A/B drift rose to `0.6627`. The next continuation target
should therefore change the training target or search coupling, not another
direct CFV head variant: predict search-consistency residuals against the exact
resolver, train uncertainty/mixing against root action drift on root-disjoint
callback states, or collect paired exact-vs-learned callback interventions so
the model learns which value errors actually matter to the root policy.

The dynamic high-bet successor-cut line was rechecked with public-root-disjoint
target slices. The old passing probe used a shape-balanced row split inside the
same public-root pool. With disjoint slices, a small `110/115` target split and
a broader `512/256` target split both failed value baselines; the broader
GRU/deepset probe reached `0.2713` MAE versus a `0.1962` train-median constant.
This makes root extrapolation, not just callback feature plumbing, a recurring
failure mode. Do not run successor-cut resolver A/B from row-level internal
passes unless the checkpoint first clears root-disjoint value calibration.

A slice-aware one-cut impact diagnostic confirms the same issue behaviorally.
On unseen roots `128..143`, the root-disjoint GRU high-bet checkpoint produced
only `0.6087` action agreement with mean/max single-cut L1 drift
`0.5592/1.3602`; the worst root changed a baseline call into multiple bet
actions. The useful next target is therefore action-impact-aware: collect
exact-vs-learned intervention labels and train either a conservative fallback
gate or a value correction against root action drift.

The simplest learned fallback gate is not enough with the current sparse labels.
A ridge/log structural predictor trained on train-root one-cut records had
negative held-out correlation (`r=-0.1752`) and selected higher-drift cuts than
it rejected. This argues against threshold hacking. The next continuation work
should collect denser paired interventions across public roots, or optimize a
search-consistency target that directly penalizes root action drift.

The denser-intervention check now also failed. Expanding the same root-disjoint
one-cut audit to `127` train cuts and `117` holdout cuts still left the
structural predictor with negative holdout correlation (`r=-0.0441`) and only a
weak abstention separation (`0.4514` selected mean L1 versus `0.4902` rejected).
More static cut/root metadata is therefore not the root fix. The next target
should move inside the search loop: dynamic CFR reach/regret traces,
root-drift residual targets, or an update that learns which continuation
errors change the root policy while search is running.

## 2026-05-15 Update: All-In Semantics Repair

Full-deck all-in response semantics are now corrected across CPU, fast, and
CUDA states. An unmatched all-in must leave active non-all-in opponents to
call, fold, or raise; only matched/no-further-betting all-in states should deal
to showdown. This is not a new strategy method, but it changes the traversal
distribution that future learning sees.

Treat older checkpoint evidence as potentially stale because prior traversal
could skip opponent response states after an all-in. The next principled step
after this repair is to retrain or run a controlled short corrected-semantics
baseline before drawing more conclusions from Slumbot or restricted
early-action diagnostics.

That controlled short baseline is now complete and is not promotable. A
20-iteration corrected-semantics CUDA run improved the restricted early-action
value proxy versus the stale incumbent (`+0.12` selected payoff and `0.410`
advantage/value correlation versus `-14.50` and `-0.025`), but local H2H against
the old incumbent was inconclusive (`+164` chips/hand with lower95 `-280`).
More importantly, corrected all-in semantics expanded traversal enough that the
fixed GPU traversal pool now visibly truncates many traversals. The fast
default reached `419` traversals/sec but often overfilled the pool; 1000 slots
fell to `199` traversals/sec and still overflowed later; 2500 slots fell to
`87` traversals/sec. The next target is therefore not a larger network or a
longer run. Add hard traversal-fidelity telemetry and test an adaptive or
state-aware traversal allocation that preserves all-in response states without
global slot inflation.

The immediate CUDA OOM from trying a 2M-slot pool is fixed by adaptive
value-net forward chunking, and the exact failed command now completes. This is
an engineering reliability improvement, not research evidence for scaling the
pool. The 2M/1000 probe still runs at only `199.6` traversals/sec and still
shows several-times-over pool demand, so the active blocker remains traversal
fidelity and tree-shape control under corrected all-in semantics.

Traversal-fidelity telemetry is now part of autoresearch training JSON. Future
training runs must report overflow chunk fraction and pool demand ratios before
their checkpoint evidence is interpreted. The next real research step can now
be framed cleanly: reduce traversal overflow by a principled state-aware
allocation or sampling design, then compare against the same fast default on
both throughput and downstream decision quality.

The first telemetry A/B rejects slot scaling as that design. Default traversal
is faster but severely truncated (`3.98x` mean pool demand, `8.38x` max);
1000-slot traversal reduces demand but still overflows every chunk and loses
substantial throughput. Since Deep CFR's external-sampling contract explores
all traverser actions, the current pool-exhaustion demotion is not an
algorithmically clean substitute. The next target should be a source-backed
MCCFR sampling redesign: streaming external-sampling traversal, outcome
sampling, average-strategy/action-subset sampling, or another estimator with
explicit correction and variance accounting.

The sampled-action MCCFR redesign review is complete and allows only a
diagnostic prototype. The implementation target is not "sample fewer actions"
by fiat. It is to define an estimator, verify it against exhaustive traversal
on a small setting, report variance and pool demand, and only then run a
corrected-semantics Deep CFR probe.

The first estimator-level smoke has passed. A single-infoset
inverse-probability sampled regret estimator is unbiased in analytic tests, and
the toy external-sampling traversal with sampled opponent responses also
matches exhaustive expectation. Synthetic variance falls as sampled actions
increase. The next gate must move from toy traversal to a restricted full-deck
state: compare exhaustive external-sampling regret against sampled-action
regret on deterministic poker states before any CUDA trainer integration.

The restricted full-deck gate is now partial. The estimator works, but payoff
scale makes low sample counts too noisy: one sampled action has unacceptable
variance, while 16-32 sampled actions are tolerable in the restricted root
diagnostic. The next target should therefore be variance reduction at the
estimator level, not immediate CUDA integration.

The variance-reduction follow-up passed at the diagnostic level. Optional
action-value baselines preserve unbiasedness, and even crude baselines reduce
the full-deck estimator variance sharply. The next research objective is now
more specific: learn a baseline/control-variate action-value head that lets a
small sampled traverser-action set approximate exhaustive traversal regret.

The first learned baseline confirms the direction but does not fully solve the
problem. A larger restricted value baseline beats the crude mean baseline and
allows 2-8 sampled actions to pass the full-deck estimator diagnostic, while
one sampled action remains too noisy. The next step should be an
exhaustive-vs-sampled traversal comparison using two sampled traverser actions
plus a learned baseline, still outside the default CUDA trainer.

The 64-root check preserves that conclusion. Two sampled traverser actions plus
the learned baseline pass the restricted full-deck estimator diagnostic on a
broader deterministic slice, but variance is still large enough that this must
remain opt-in until compared against exhaustive traversal on small states.

The 256-root check keeps the same direction but clarifies risk: sample-2 can
pass with enough repetitions, yet it is visibly noisy; sample-4 and sample-8
are more stable. Use four sampled traverser actions as the first safer
diagnostic integration point, with sample-2 kept as a lower-bound ablation.

Scaling the learned baseline changed the conclusion: a stronger baseline makes
even one sampled action pass the 256-root restricted estimator gate. This
points toward a clean learned control-variate sampled traversal method. The
next gate is no longer "can sampling be unbiased"; it is whether a learned
baseline sampled traversal matches exhaustive traversal decisions on small
states before touching CUDA/default training.

The top-action diagnostic adds a caution: sampled estimates are useful as
stochastic regret targets, not as one-shot decisions. Even with the XL baseline,
individual sampled estimates match the exhaustive top-regret action only about
`0.70` at eight samples. The next probe should therefore compare averaged
sampled regret targets to exhaustive regret targets over repeated samples.

That averaged-target check now passes: the top action of the mean sampled
regret estimate matches exhaustive regret on roughly `96%` of 256 roots. This
is the clearest current path: learned control-variate sampled traversal should
write stochastic regret samples to the buffer, then rely on replay/training
averaging rather than using sampled estimates for immediate play decisions.

The integration contract is now captured in
`docs/research_protocols/sampled_action_control_variate_traversal_plan.md`.
Keep all implementation opt-in and research-only until exhaustive-vs-sampled
regret comparisons pass on deterministic small states.

## 2026-05-21 Dense SD-CFR Update

The dense SD-CFR average-strategy path is not the next scale target. A clean
GPU-bound every-iteration `200x2000` run had zero rejected traversal chunks and
used the GPU correctly, but the predeclared average lost the local falsification
gate and its individual checkpoints did not reveal a robust competent region.
Early-street diagnostics show the important distinction: the dense run can
reduce visible all-in pathology while still selecting poorly aligned early
actions. The next continuation target should therefore change the training or
search target itself: learn a decision-aware early-street action-value /
counterfactual correction, or return to the ReBeL / Student-of-Games shaped
public-belief search-state objective with a root-disjoint decision gate. Do not
treat more SD-CFR iterations, post-hoc snapshot weights, or policy-head cloning
as principled progress without a new methodology review.

The first public-information rollout diagnostic is now available as
`scripts/eval_public_action_rollout_values.py`. It samples opponent hole cards
and future deck order from the public root, scores every legal first action on
the same sampled worlds, and rolls out with a continuation policy. This rejects
the unsafe copied-hidden-state rollout shortcut and gives a stronger positive
control than the showdown-only restricted value diagnostic. It also exposed an
important evaluation-contract detail: live Slumbot greedy play for regret
sources selects by masked raw advantage, not by regret-matched policy
probability. The diagnostic now matches that deployment contract.

Corrected 64-root/32-world CUDA results still do not promote dense SD-CFR. The
incumbent had selected value `81.32`, oracle gap `21.00`, and top-action match
`0.50`; dense iter100 had selected value `75.17`, gap `35.17`, and match `0.25`;
dense final had selected value `53.17`, gap `57.26`, and match `0.03125`. This
is diagnostic-only evidence because the rollout continuation can still inherit
the model's own all-in/fold biases. The next principled step is to turn this
into a candidate teacher only after adding a fixed continuation/opponent
population or resolver-derived public-belief target, then validating on a
root-disjoint decision gate.

The fixed-continuation variant is now implemented. Using `iter1000` as the
predeclared continuation checkpoint makes the dense weakness clearer: dense
iter100 selected value falls to `31.73` with gap `70.58` and match `0.078`;
dense final selected value is `28.33` with gap `73.98` and match `0.125`. This
does not make `iter1000` a teacher by itself because it is all-in heavy. It
does show that the next useful target is an opponent population or
resolver-derived public-belief teacher, not more self-continuation rollout or
more dense SD-CFR scale.

The restored-history continuation falsifies the "incumbent continuation caused
the dense failure" explanation. Under `restored_history_200x2k_4x512_final` as
fixed continuation, dense iter100 scores `-9.20` selected value with gap
`132.25`, and dense final scores `-18.52` with gap `141.57`. The next step
should leave checkpoint-continuation diagnostics and build a resolver-derived
public-belief teacher or a predeclared checkpoint-population teacher with
root-disjoint gates.

The predeclared checkpoint-population continuation was also negative. Averaging
continuation strategies from `iter1000`, restored-history final, and the
corrected-all-in smoke produced incumbent selected value `75.76` with gap
`24.93`, while dense iter100 scored `4.75` with gap `95.94` and dense final
scored `5.92` with gap `94.77`. This retires checkpoint-continuation teachers
as the next mainline. The next continuation target is now specifically a
resolver-derived public-belief/action-value teacher.

The exact public-belief boundary was rechecked after the checkpoint-continuation
failures. Dynamic per-iteration cut replay still preserves the teacher exactly
on the restored200 successor pool (`1.0` action agreement, `0.0` mean L1 on 28
eligible roots), but static fixed-iteration-5 replay fails badly (`0.3929`
agreement, `0.9112` mean L1). This sharpens the target: the next learned object
must represent dynamic search state or update behavior consumed inside the
resolver. A static leaf-value table, fixed checkpoint continuation, or
single-iteration value distillation is not the mainline.

A larger root-disjoint CFR-trace scale check did not rescue the existing
dynamic predictor. The 96-root/64-root recurrent trace model trained on CUDA but
was worse than both CFR5 and uniform CFR10 on holdout root decisions. A linear
selective-budget predictor found weak signal, but selective CFR10 escalation on
the top 20% predicted roots still lost to simply running CFR10 everywhere. The
next branch should therefore emphasize exact search acceleration, better
amortized search-state interfaces, or richer game-theoretic update targets
rather than another small trace-policy predictor.

The CUDA level-synchronous exact-search frontier strengthens that conclusion.
On the same restored200 holdout, CFR25 against a CFR50 teacher reached `0.2715`
mean L1 with zero illegal mass at about `108 ms` per root, while CFR10 was
`0.5371` L1 at `56 ms`. The current practical baseline is therefore not a weak
learned replacement for search; it is GPU-backed exact resolving with a clear
quality/latency frontier. Learned components must either improve this frontier
or amortize a richer search-state update without degrading root decisions.

The opponent-reach EV-factor dual-CFV target is a useful reviewed control but
not the next mainline by itself. On the cached river 384x128 split it passed
the reconstructed-CFV constant-baseline gate, yet was worse than the direct raw
dual-CFV probe (`0.3970` vs `0.3844` MAE; `0.5252` vs `0.5093` RMSE). Do not
scale this factorization without a stronger decision-consumption argument.

The amortized mirror search-update branch sharpens the next target. A learned
eta selector failed the 96/64 root-disjoint gate (`0.5257` L1, worse than CFR5
and CFR10), but the per-root eta oracle beat CFR10 on a 32-root holdout
(`0.2291` L1, `0.9375` top match). The next update-operator attempt must learn
a richer selector from dynamic search-state features, or the project should
spend the next engineering cycle on exact GPU search acceleration instead.

The oracle-eta selector follow-up failed the same way after adding the full
early trace trajectory and direct per-eta decision-loss labels. It fit train
roots perfectly but selected damaging etas on root-disjoint holdout (`0.5827`
L1 on the 96/64 split versus CFR10 `0.3251`; `0.5016` on the 32/32 split
versus CFR10 `0.3288`). This rules out another shallow selector as the next
mainline. The next continuation target should either make exact
level-synchronous GPU resolving cheaper at the same quality, or learn a
stateful solver-dynamics object that is trained and consumed inside the
resolver rather than picked by an external static feature row.

The existing non-CFR+ update-rule gate does not change that conclusion.
`dcfr_plus` at five iterations improved mean L1 over CFR5 but worsened KL and
all-in probability calibration; `pdcfr_plus` failed outright. Both are CPU-only
in the current live-relevant path. The next exact-search branch should focus on
CFR+ budget/latency engineering, batch/caching, or an actually learned update
rule with a closed-loop gate, not manual selection among these update variants.

Before another variant of that branch is expanded, use the executable paradigm
innovation gate. The next mechanism must be chosen by first-principles anomaly
value, not by convenience: either an exact GPU CFR+ quality/latency primitive
that raises the search frontier, or a stateful learned solver-dynamics object
trained and consumed inside the resolver. The innovation review must name the
cross-paradigm analogy, thought experiments, related work, and one root-disjoint
decisive test before any new training or selector run.

The first reviewed closed-loop proxy failed. A one-step delta model over real
CFR trace states learned under teacher forcing, but its own held-out rollout
drifted worse than CFR5 and CFR10 even with a training-derived trust-region
cap. This retires semi-closed-loop policy-delta rollouts as the next mainline.
The next continuation should be one of two stronger branches: exact GPU CFR+
search acceleration/caching, or a true in-resolver learned update that modifies
regret/search state while exact CFR+ recomputes the next state. Do not add more
standalone trace-policy predictors before a fresh innovation review.

The first exact-GPU continuation step is now complete: `auto` uses
`torch-levelsync-cuda` when CUDA is available, with a reviewed CPU-vs-auto
fixed-state smoke showing about `5.36x` CFR50 latency improvement. Static
tensor caching was added for reused solver trees, but its measured gain is
minor. The next continuation should not be another backend-selection tweak; it
should measure whether the cheaper GPU default lets us raise live solver budget
or add a teacher-aligned boundary policy that improves root decision quality per
millisecond without weakening Slumbot or local promotion gates.

The live wrapper smoke passed after the default flip, so integration is no
longer the immediate blocker. The next continuation should be a decision-quality
experiment: either a same-state live-budget frontier judged by root distribution
quality per millisecond, or a reviewed boundary/disagreement policy that spends
extra exact GPU CFR only when the cheap solver is likely to cross an action
boundary. Do not interpret tiny Slumbot chip swings as progress.

The same-state budget-profile diagnostic now favors the second option. Fast-live
saved about 35% latency but still disagreed with live on four of 32 roots. The
next continuation target is a selective exact-search escalation gate: train or
derive a cheap boundary score from public state, low-budget strategy geometry,
or early trace features, then spend live/high budget only on the predicted
boundary roots. The pass condition should be root-disjoint improvement in
latency at fixed action/distribution agreement, not a new Slumbot score.

The first selective-escalation gate shows that public-state features alone are
too weak. Keep the selective-escalation framing, but move the next attempt to a
solver-native boundary signal: margin between top actions, entropy, fast/live
low-budget trace deltas, or other cheap quantities available from the exact
resolver before deciding whether to escalate. Do not train another CFV-only
ridge selector unless it is a negative control.

The first solver-native boundary signal is positive enough to continue, but
not enough to deploy. A ridge selector over only fast-live strategy geometry
passed both 64/64 root-disjoint directions for profile-L1 reduction while
staying below live latency, whereas action-disagreement classification remained
noisy. This is philosophically cleaner than a public-state heuristic: the
resolver spends more computation when its own cheap distribution indicates a
boundary. The next continuation should validate the same signal on a larger
fresh auto-CUDA profile and then wire it only as an opt-in budget policy with a
same-state live-profile gate. Do not promote it from the current 128-root
diagnostic or tune thresholds as the main research result.

The larger fresh holdout now supports a guarded opt-in path. Training on the
older 128 roots and holding out on fresh roots `64..127` passed with better L1
and top-action agreement below live latency, and `selective-fast-live` now runs
through the Slumbot wrapper with traceable score/threshold fields. The next
continuation target is a controlled same-state integration check: compare
`fast-live`, `selective-fast-live`, and `live` on fixed roots using the exported
policy. Only after that should we spend on a larger Slumbot run. If the fixed
root gate fails, retire the live wrapper path and return to exact GPU search
amortization or richer trace dynamics.

That controlled fixed-root integration check now passes under true online
latency accounting: escalated roots pay for both the cheap fast-live solve and
the live rerun. The next continuation target is therefore an opt-in Slumbot
confidence run for integration and transfer, not a default change. Keep `live`
as the default, collect traces with selector score/threshold/escalation, and
judge the run by API/parse health, solver latency, escalation rate, and chips
only with enough hands to avoid tiny-sample noise. If Slumbot transfer is bad
despite fixed-state gains, diagnose distribution shift in the selected roots
rather than tuning the selector threshold.

The first 100-hand no-all-in Slumbot triangle is only an integration/latency
smoke. It confirmed no API/parse regressions and showed selective latency close
to fast-live, but the chip intervals are much too wide and the runs are
unpaired. The next continuation should either run a larger all-in-enabled
confidence check or, more efficiently, analyze the selective trace roots:
compare selector scores, escalation decisions, action likelihood, and Slumbot
outcomes against the fixed-state profile distribution to see whether the
selector is firing on the right live states.

2026-05-21 objective clarification: do not let that Slumbot trace analysis
become the primary promotion path. The workflow now requires a self-play
checkpoint league as the first strength signal: candidate versus previous
checkpoint, candidate versus incumbent, and candidate versus native self-play
controls where available. Slumbot remains held-out external validation after
the internal league ladder passes. The immediate continuation target is to
validate the league evaluator, then use it to build a checkpoint-progress plot
for the current family before spending more Slumbot confidence hands.

The first self-play-first pass found a real internal checkpoint peak but not
Slumbot transfer. In the restored-history 200x2000 4x512 family, `iter100`
beats the old incumbent and neighboring checkpoints under duplicate-swapped
league gates, while the final checkpoint is worse than `iter100`. Promoting
the internal peak to local incumbent was therefore correct. Held-out Slumbot
validation then failed cleanly: `iter100` lost `-393 +/- 268` chips/hand over
1000 hands with no API/parse errors, no all-in actions, and acceptable
fast-live latency. Action/range diagnostics localize this to calibration:
Slumbot action log-lift is `-9.237` for the model but `+1.042` for a
leave-one-out street baseline, and true hand range log-lift worsens on
turn/river and large pots. The next continuation target should not be more
Slumbot chip trials. It should use the existing latent-response-conditioner
review: keep Slumbot traces as held-out diagnostics, train/validate a
decision-aware learned public-belief/response representation, and require
cross-trace selected-action EV improvement before any resolver integration.

The latest `iter100` cross-trace replay strengthens but does not solve that
branch. Calibrated response probes trained on one trace beat the model and
uniform on the other trace in both directions (`+0.630` and `+1.382` external
log-lift), confirming a real learned opponent-response signal. Direct
response-range replacement failed the executable gate on both a 64-case smoke
and a larger 192-case replay. The larger replay is decisive: mean strategy EV
was `-3.04`, only `32.3%` of states beat baseline, and promotion stayed
blocked even though true-hand likelihood improved by `+4.44` log-lift. A
shallow selector could rescue a small positive mean EV but had near-zero
cross-trace prediction correlation and weak beat rate. Continue with latent or
jointly trained conditioning inside policy/search; do not spend the next cycle
on threshold tuning or a manual range switch.

The first bounded "latent" selector over the same observable response-range
and solver features is also negative as a mainline. It passed the loose
diagnostic gate by improving over direct range replacement on mean EV, but it
overfit the small train artifact, had negative external prediction
correlations, and underperformed the simpler shallow selector on selected-action
EV. This retires post-hoc nonlinear selectors over the current summary
features. The next continuation should either integrate an auxiliary
response/public-belief head into the self-play/search network so the learned
state is consumed during policy improvement, or leave Slumbot-response modeling
as a diagnostic branch and return to self-play public-belief/search-state
targets.

Online related-work refresh after these failures favors that shift over more
Slumbot-response tooling. AlphaHoldem is the most relevant PC-oriented
precedent: end-to-end self-play RL, structured card/action tensors, historical
model opponents, multitask losses, and model selection, with reported wins
against SlumBot and DeepStack and millisecond inference. TurboReBeL and recent
policy-gradient theory support the other principled branch: accelerate
public-belief/search-state learning rather than hand-patching opponent ranges.
Therefore the next continuation target should be one of two reviewed tracks:
an AlphaHoldem-style self-play actor-critic league baseline inside this repo's
full-deck action contract, or a public-belief/search-state target that produces
root-disjoint decision improvement. Do not continue the direct SlumBot
opponent-response branch except as a diagnostic.

The AlphaHoldem-style branch now has its first bounded mechanism: native PPO
can opt into `--feature-mode raw_sequence`, which keeps raw card one-hots and
continuous game scalars while encoding betting history as ordered action tokens
and bet amounts instead of per-street action-count summaries. This is not a
strength claim and not a Deep CFR feature-contract change. The immediate
continuation target is a small GPU self-play pilot against the flat PPO control
and the native NFSP/q-lambda controls, judged by duplicate-swapped H2H league
gates before any SlumBot run.

That first league smoke found a real local signal but also a best-iterate
problem. Raw-sequence q-lambda PPO beats several native controls with positive
lower95 bounds, including NFSP10k and flat q-lambda PPO 10k, but it does not
yet confidence-beat the CTDE q-lambda 10k control and the 10k raw checkpoint
does not confidence-beat raw 2k. Continue with seed replication and checkpoint
league selection inside the self-play workflow. Do not move to SlumBot until
the raw-sequence branch clears a replicated local-control gate.

Seed replication supports the raw-sequence representation but does not close
the CTDE-control gap. The next continuation target is a larger native-policy
league run using `scripts/eval_native_policy_league.py` over the two raw 2k
seeds, raw 10k, and the native controls. Promote only if the best checkpoint
has positive worst-case lower95 against NFSP10k, flat q-lambda 10k, and CTDE
q-lambda 10k. Otherwise, keep raw-sequence q-lambda as a promising control and
return to a reviewed mechanism such as historical-opponent population training
or multitask actor/value/policy losses.

That larger summary gate failed promotion while selecting raw-sequence
q-lambda 10k as the current native best iterate. The bottleneck is now the
self-play training distribution: current-policy self-play and one-step
q-lambda improvement produce useful policies, but not robust monotonic league
progress. The next continuation target is a historical-opponent population
pilot with the same raw-sequence observation and q-lambda PPO objective, using
fixed snapshot scheduling and duplicate-swapped league gates. Keep SlumBot
held out.

The historical-opponent path is now implemented and smoke-tested. Continue
with a bounded 2k CUDA pilot using raw-sequence q-lambda PPO plus frozen
self-play snapshots, then gate it against raw q-lambda 2k/10k, NFSP10k, flat
q-lambda 10k, and CTDE q-lambda 10k. If it does not improve worst-case
lower95, retire this simple population schedule and review a more faithful
AlphaHoldem-style multitask/policy-selection objective.

The simple historical-opponent schedule failed that gate. Keep the code as an
opt-in control, but do not tune snapshot interval or capacity. The next
continuation should do an online related-work refresh focused on AlphaHoldem's
multitask/self-play objective and then queue a methodology review for the
smallest faithful next mechanism. The current best local checkpoint remains
raw-sequence q-lambda 10k, but it is not promoted because CTDE q-lambda 10k is
not cleared with positive worst-case lower95.

The K-best approximation also failed, so the workflow should stop spending
cycles on historical-pool schedules. The next continuation target is a reviewed
Trinal-Clip/multitask-loss pilot: keep raw-sequence observations and q-lambda
PPO, modify only the learning objective, and gate against the same native
control league. If that fails, the AlphaHoldem-lite path should pause and the
project should return to public-belief/search-state learning.

The Trinal-Clip pilot failed. Keep the raw-sequence representation as useful
native-policy evidence, but stop the AlphaHoldem-lite PPO line here unless a
more substantial architecture review is opened. The active next target should
return to public-belief/search-state learning: root-disjoint decision
improvement, learned search-state updates, or exact GPU search acceleration,
with native raw-sequence q-lambda 10k retained only as a local control.

2026-05-22 continuation update: the reviewed public-belief/search-state
warm-start gate has now rejected the current low-state regret/policy field
under the stricter uniform-budget baseline. On 64 root-disjoint holdout roots,
the learned warm start moved some roots in the right direction, but it lost to
uniform CFR10 on L1, KL, top-action agreement, and latency. Do not rerun this
same warm-start checkpoint, train another shallow selector, or scale a hidden
size/epoch sweep around it.

The next continuation target should be a mechanism-change review and gate:
learn a stateful search-update object that is consumed inside the resolver, or
raise the exact GPU CFR frontier. The pass condition must be root-disjoint
decision quality per millisecond against the same uniform CFR10/CFR25-style
baseline. A useful first hypothesis is that a learned update must modify
regret/search state inside the iterative solver with closed-loop loss, rather
than predict a final policy, choose an external eta, or seed one node with
per-hand supervised fields.

The first infrastructure step for that target is now in place. The CPU CFR
solver accepts an opt-in `iteration_update_fn` that can inspect the current
reach/value/regret/strategy state after each iteration and return replacement
regret or strategy sums for the next iteration. This is diagnostic-only:
`torch-levelsync` and segmented backends reject the hook until a separate GPU
contract is reviewed. The next research action should use this hook for a
small root-disjoint in-resolver update diagnostic, with uniform CFR10/CFR25 as
the baseline. Do not treat the hook itself as progress in model strength.

The in-resolver learned-hook diagnostics have now served their purpose. The
oracle hook proves the update surface is live, but both shallow aggregate-policy
injection and the old low-state per-hand regret/policy checkpoint lose to simply
spending a uniform CFR10 budget. Exact `torch-levelsync-cuda` search is the
current positive decision-impact lever: CFR50/CFR100/CFR150 move monotonically
toward CFR100/CFR200 teachers with strong root-action agreement at live-scale
latencies. Therefore the active continuation target is upstream
search-improved self-play/blueprint distillation, not another static hook.

The first search-improved blueprint target artifact is available at
`autoresearch-session/search_targets/search_improved_blueprint_turnriver_cfr100_incumbent_seed20260522.npz`.
It contains 64 belief-conditioned turn/river CFR100 targets sampled from the
current incumbent's own self-play distribution. Treat it as diagnostic until it
passes a target-sanity gate: the metadata shows high all-in pressure
(`0.5469` target top-action all-in rate, `0.4424` mean all-in probability).
The next action is to compare incumbent regret/policy behavior against these
targets and decide whether the target set is a useful teacher, needs a
train/holdout split, or is exposing a blueprint/range pathology.

That 64-target smoke has now failed as a deployable post-hoc policy patch.
Policy-head calibration improved held-out fit to the CFR100 targets, but it
over-generalized to all-in on covered turn/river streets and lost heavily to
the incumbent in duplicate-swapped self-play. Do not promote or Slumbot-test
this checkpoint. The next principled continuation is either larger
self-play-distribution target coverage or in-training search distillation, with
the pass/fail gate still based on self-play decision impact. A reasonable next
step is a larger CFR100 target set from the same incumbent distribution to test
whether the failure is small-sample oversteer or a deeper policy-interface
problem.

The larger 512-target version also failed as a post-hoc policy-head patch. It
reduced but did not remove all-in oversteer, and duplicate-swapped self-play
remained strongly negative. This narrows the next target: do not keep
calibrating tiny policy heads after training. Instead, run one in-training
search-distillation candidate using the 384-target train split, now that
search-target checkpoints record covered streets for `policy-head-covered`
evaluation. If that still loses to the incumbent, treat the current
action-policy interface as the bottleneck and open a review for richer
range/belief-conditioned policy inputs rather than increasing target count or
tuning loss weights.

That in-training search-distillation candidate is also not promotable. The
25x2k CUDA run was clean and fast, but the resulting checkpoint is far weaker
than the incumbent at this budget, and the covered policy head only gives a
noisy within-checkpoint gain versus its own regret source. Because this result
is confounded by the short training budget, the next continuation target is a
matched no-target 25x2k control with the same seed/settings. Compare
search-target versus no-target checkpoints directly and against the incumbent.
If the search-target loss does not improve same-budget self-play strength or
held-out root decisions, retire this simple action-policy distillation branch
and open a reviewed richer-interface mechanism instead of tuning loss weights.

The matched no-target control has now been run cleanly after increasing
traversal pool capacity to eliminate one rejected chunk. Against the control,
the search-target checkpoint is only weakly positive: `+65.2` chips/hand by
regret source and `+46.1` by `policy-head-covered`, with negative lower95
bounds in both cases. Treat simple action-policy search-target distillation as
non-promotable and stop tuning target counts or weights. The next continuation
should queue a methodology/innovation review for a richer mechanism: a
range/belief-conditioned policy or search-state update object consumed by the
resolver, with pass conditions based on root-disjoint decision quality and
same-budget self-play league strength.

2026-05-25 update: the first richer target-export bridge is implemented but
negative at smoke scale. The new per-iteration warm-start exporter aligns data
collection better with the resolver hook by exporting current selected-node CFR
state across iterations, but the first 8-root/4-root low-state-aware field
model made held-out in-resolver decisions worse than low CFR5 and uniform
CFR10. This rules out treating "more rows from the same field predictor" as the
next mainline. The continuation target remains learned search, but the next
step must be a substantial mechanism change: a bounded regret/advantage
residual consumed by the hook, a more structured public-belief/search-state
model with direct decision loss, or a deliberate pivot to exact CUDA CFR
budget/blueprint quality if no learned update beats the uniform search
baseline.

2026-05-25 second update: the bounded regret/advantage residual option has now
failed its smoke gate. The hook stayed legal and capped, but still made
root-disjoint decisions worse than low CFR5 and uniform CFR10. This retires the
current learned-hook family, but it must not complete the outer autoresearch
goal. Do not continue with residual-cap, loss-weight, or hidden-size tuning.
The next continuation requires an explicit pivot review. The two admissible
pivots are: exact CUDA search plus stronger blueprint/self-play quality as the
near-term mainline, or a materially different learned search object trained
with direct closed-loop root-decision loss rather than field
replacement/residual prediction. Continuous autoresearch should treat this as a
soft mechanism failure: document, synthesize, run related-work innovation
review, choose the next principled mechanism, and continue.

2026-05-25 third update: the soft-pivot workflow is implemented and the pivot
review has selected exact-search-first blueprint/self-play improvement. A
root-disjoint `torch-levelsync-cuda` budget frontier passed on 16 held-out
roots: CFR100 is substantially closer to CFR150 than CFR50 while staying legal.
The next continuation is therefore not another learned hook. It is to connect
the exact CUDA improvement operator to training: build or reuse a
self-play-distribution target set where exact search changes decisions, then
run a matched same-budget search-improved blueprint/control comparison before
any Slumbot confidence run.

2026-05-25 fourth update: the static disagreement-weighted version of that
bridge is now negative for promotion. It trained cleanly and used CUDA, but it
only produced a noisy `+43.4` chips/hand against the matched no-target 25x2k
control and remained far behind the restored-history incumbent. Do not tune the
same static target loss again. The next continuation is a reviewed
search-as-actor/expert-iteration pilot: exact CUDA CFR should generate improved
behavior on high-disagreement self-play roots, the deployed policy source must
consume that signal, and success must be judged first by held-out root decision
impact and then by same-budget self-play.

2026-05-25 fifth update: the first search-as-actor average-policy pilot passed
the root-fit side of that bridge but failed transfer. It learned the held-out
turn/river CFR100 target distribution better than the no-target control, yet it
lost badly in duplicate-swapped whole-game play against both the restored
history incumbent and the same-budget no-target control. The continuation is no
longer another static target replay. Diagnose the decision-transfer break:
measure where the average-policy actor is out of distribution by street,
coverage, legal action mass, and all-in/raise-frequency shift, then only queue a
new mechanism if it couples search improvement to the policy's own whole-game
state distribution.

2026-05-25 sixth update: the mixed external-plus-traversal search-as-actor
pilot completed that diagnostic repair but still failed transfer. Whole-game
coverage fixed the early all-in collapse, so the remaining blocker is not
primarily compute, legality, or target coverage. The candidate still chose
lower-value actions under fixed continuation and lost to the same-budget
no-target control. The next continuation must be decision-value-aware:
construct a reviewed policy-improvement test where exact search or public
rollout values produce an advantage/mirror-descent actor update on the policy's
own sampled states, then compare against the same no-target control. Do not
spend the next cycle on more static target mixtures, memory caps, or loss-weight
sweeps.

2026-05-25 seventh update: the first decision-value-aware actor pilot passed
the local and matched-control gates but not the stronger incumbent gate. This
validates the pivot from target imitation to value-aware policy improvement:
selected-action value improved on held-out public-rollout roots and H2H versus
the matched no-target 25x2k control became positive with lower95 above zero.
It is still weaker than the restored-history 200x2k incumbent. The next
continuation should not tune `eta` or root counts as knobs. It should integrate
the same value-aware actor update into a stronger self-play/average-policy
training path or produce a reviewed stronger-base variant, then require the
same held-out value gate plus restored-history H2H before any Slumbot run.

2026-05-25 eighth update: the stronger-base variant has now been tested as a
covered policy-head overlay on the restored-history checkpoint. It passed the
local public-rollout gate but failed promotion-quality H2H confidence and
became too all-in-heavy. Treat this as a useful positive signal, not a solved
method. The next continuation should move the value-aware actor update from
offline preflop rollout labels into the training distribution: collect
search/value-improved targets from the actor's own self-play states across
streets, or run an on-policy expert-iteration-style loop where the search actor
generates the next distribution. A new cycle must gate against restored-history
H2H and all-in/action-distribution drift before any Slumbot confidence run.

2026-05-25 ninth update: the on-policy collector path is now mechanically safer.
The first on-policy smoke exposed a workflow bug: it claimed multi-street
training while collecting only preflop targets. Required-street coverage is now
part of the collector/CLI contract, and the gated smoke produced actual
flop/turn/river rows. However, the resulting policy-head update is still weak
and degenerate: it chooses one raise size on every fixed-root eval state while
policy EV drops. The next continuation should add a decision/action-collapse
gate for value-aware actor updates, then rerun a small same-budget decision
gate. Do not launch a large H2H or Slumbot run from this smoke until the update
improves selected-action value without collapsing the action distribution.

2026-05-25 tenth update: the KL-anchored expected-Q version of the same
post-hoc actor path has now failed the stricter decision gate too. It is a
better objective than cross-entropy in-sample, but it still overgeneralizes
through the covered policy head and collapses fixed-root behavior. This points
away from more post-hoc policy-head calibration. The next continuation should
either make exact search the acting policy-improvement operator inside
self-play/training, or train an actor whose distribution is generated by the
same value-aware update loop it will later use. Treat one-off policy-head
patches as controls unless a new review explains why their train/eval
distribution mismatch is resolved.

2026-05-25 eleventh update: the first search-improved behavior collector has
now been tested. Greedy search-improved behavior improves top-action root
metrics but still fails policy-EV transfer; sampled search-improved behavior is
worse. This narrows the issue further: search must affect data generation, but
the current covered policy-head patch is still too weak a deployed actor
surface. The next continuation should stop adding post-hoc policy-head variants
and either integrate search-improved behavior into GPU Deep CFR average-policy
training, or run exact-search-as-actor self-play as the behavior policy and
train a real average actor from that distribution under matched no-search
controls.

2026-05-25 twelfth update: the real average-policy actor-surface smoke passed
the fixed-root decision gate and outperformed the no-search replay control on
that gate, but failed direct H2H transfer. This is the strongest signal so far,
but still not SOTA or Slumbot-ready. The next continuation should diagnose the
transfer gap: compare action/street distribution drift and run a slightly larger
same-mechanism fixed-root/H2H check only if it keeps the same matched no-search
control. If the gap persists, integrate search-improved behavior into GPU Deep
CFR training rather than standalone one-shot replay.

2026-05-25 thirteenth update: the bounded two-iteration replay diagnostic
failed for the search-improved branch while the no-search control passed. This
retire standalone replay as a mainline. The next continuation should move the
search-improved behavior signal into GPU Deep CFR average-policy training, where
the policy is updated during self-play/traversal rather than by isolated
calibration, or pivot to exact-search deployment/blueprint quality if that path
is too invasive for the next bounded sprint.

2026-05-25 fourteenth update: the first GPU Deep CFR average-policy injection
sprint fixed the missing artifact boundary and a control-contamination bug.
`scripts/build_decision_value_average_strategy_targets.py` now emits
on-policy decision-value target files that can be consumed by
`--average-strategy-targets`. `--average-strategy-memory-capacity 0` now
preserves an external-only average-policy run instead of silently collecting
additional traversal strategy memory. The tiny CUDA smoke proved mechanics
only: both matched branches trained with zero collected average-policy memory
and zero rejected traversal chunks, but the 200-pair H2H diagnostic for
search-improved external targets was negative. The next continuation should run
a non-tiny matched gate: build multi-street search-improved and base-behavior
target artifacts with required street coverage, train same-budget GPU Deep CFR
branches with external-only average-policy targets, then require a held-out
root decision gate and duplicate-swapped H2H before any Slumbot validation.

2026-05-25 fifteenth update: the non-tiny multi-street external-injection gate
failed the held-out root decision test. The failure is sharper than H2H noise:
both matched branches selected exactly `raise_1.0` on every held-out root, and
the search-improved branch had lower policy EV than the base-behavior control.
This retires detached external average-strategy target files as the mainline.
The next continuation should implement a closed-loop search-behavior
average-policy memory collector inside training: the search-improved actor must
generate trajectories and reservoir entries that train the deployed average
policy. Add a collapse gate on selected-action concentration before running
H2H. If that also fails, pivot away from average-policy target fitting toward
exact-search deployment or native policy-gradient self-play controls.

2026-05-25 sixteenth update: the action-collapse failure mode now has an
executable gate. Future fixed-root rollout checks should immediately run
`scripts/eval_policy_action_collapse_gate.py` and block H2H/Slumbot if the
candidate selects one action on most roots. This keeps the workflow focused on
decision surfaces, not noisy chip outcomes. The next mechanism remains
closed-loop search-behavior average-policy memory, but it must pass this gate
before any game-level comparison.

2026-05-25 seventeenth update: the mutable average-policy memory path now
exists and passed a mechanics smoke. `--seed-average-strategy-memory-from-targets`
routes decision-value targets into the real reservoir instead of using detached
external-only targets; the smoke ran on CUDA, seeded `34` search-improved rows,
kept `average_strategy_external_target_size=0`, and passed a small
selected-action collapse check. This is only infrastructure. The next
continuation should run the matched same-budget base-behavior versus
search-improved memory-seeded gate, then require fixed-root value improvement
and action-collapse pass before H2H.

2026-05-25 eighteenth update: the first matched memory-seeded root smoke failed
to beat the base-behavior control. Both branches passed the collapse guard, but
the search-improved branch had worse policy EV and oracle gap at 1x64. The next
continuation should either run a modest but still cheap same-budget gate
(`3x256` or similar, with both branches using memory seeding and zero rejected
chunks) or pivot if the same failure repeats. Do not launch H2H or Slumbot from
the 1x64 smoke.

2026-05-25 nineteenth update: the clean 3x128 memory-seeded gate repeated the
same collapse pattern. Both branches used real average-policy memory and zero
rejected chunks, but both selected `call` on `28/32` held-out roots. The
synthesis and innovation review now pivot the next continuation away from
supervised target fitting and toward search-as-behavior data generation: collect
matched trajectories where search-improved action selection actually acts, train
the average policy from that memory, and require root-decision plus collapse
gates before H2H.

2026-05-25 twentieth update: the bounded one-hot search-as-behavior memory gate
failed. `--target-mode behavior` now records the action actually chosen by the
behavior actor, and matched 3x128 CUDA runs were admissible, but both base and
search-improved average policies selected `fold` on all `32` fixed roots. This
means the failure is no longer just soft-target smoothing; small replay/memory
injection is not producing a robust deployed decision surface. The next
continuation should run a pivot review before more target variants. Prefer a
mechanism that makes search the acting policy in the live decision loop or a
native self-play RL control, with the same fixed-root and action-collapse gates.

2026-05-25 twenty-first update: the pivot review is complete and validated.
The next continuation should run the smallest forked test available in the
codebase: exact search-as-actor/root-decision diagnostics versus a native
self-play RL control. Use related-work framing from AlphaHoldem and Student of
Games. Treat a fixed-root collapse/value failure as decisive; do not spend
Slumbot or H2H time until a branch passes that gate.

2026-05-25 twenty-second update: exact search-as-actor passed the fixed-root
gate and action-collapse guard. This validates the decision-level mechanism
that target replay failed to preserve. The next continuation should wire this
actor into a local duplicate-swapped game path or Slumbot smoke with explicit
latency instrumentation. The gate for continuing is simple: same public-search
policy, no hidden-card leakage beyond sampled public worlds, bounded latency,
and positive local transfer versus the deployed average-policy baseline.

2026-05-25 twenty-third update: exact search-as-actor H2H transfer failed to
clear. The direct search actor improves isolated public roots, but sequential
play against the same checkpoint baseline is near-zero or negative within
confidence. The next continuation should diagnose transfer, not tune `eta` or
world count blindly: log search-vs-deployed action disagreements by street and
payoff delta, then decide whether the flaw is myopic per-decision search,
continuation-policy calibration, or lack of search-guided self-play training.
No Slumbot run from this branch until that diagnostic is positive.

2026-05-25 twenty-fourth update: the transfer attribution diagnostic is now in
place and points to sequential calibration, not simple action collapse. Search
disagreed with deployment on `28.64%` of recorded decisions and looked locally
better by `+146.81` chips/decision under the public-search scorer, yet the
duplicate-swapped H2H lower bound stayed negative. The next continuation should
test the scorer itself on disagreement states: compare the local search value
claim against realized duplicate-swapped continuation value, then either make
search a sequential self-play improvement operator or retire this exact
per-decision actor at the current budget. Avoid another scalar sweep until that
causal gap is resolved.

2026-05-25 twenty-fifth update: the local-scorer consistency check failed. The
sum of local search-minus-deployed values over a duplicate pair was essentially
uncorrelated with realized H2H delta (`pearson=-0.0439`, and `-0.1014` on
disagreements only). This turns the next target from "deploy exact search at
the root" into "make search sequentially calibrated." The next continuation
should either use search inside self-play trajectory generation as the policy
improvement operator, or learn a continuation-calibrated search/value update
whose predicted gain passes this pair-level transfer analyzer before any
Slumbot run.

2026-05-26 update: a self-consistent search-guided scorer diagnostic did not
close the calibration gap. The new `--scorer-continuation search-guided` mode
uses cheap inner one-step search for future rollout decisions and passed
mechanics, but its pair-level local-gain correlation with duplicate-swapped H2H
was still negative (`-0.3066` all decisions, `-0.3953` on disagreements). Treat
this as evidence that the blocker is trajectory-level coupling, not just the
rollout continuation policy inside the scorer. The next continuation should
make search generate sequential self-play data or learn from trajectory-level
outcomes before considering Slumbot.

2026-05-26 second update: the trajectory-level calibration gate and the
available on-policy KL-Q actor smoke both failed. A ridge calibrator over
recorded search decision traces could not beat a train-mean baseline on the
100-pair trace, and the on-policy search-improved KL-Q policy-head checkpoint
worsened fixed-root rollout values despite improving its local training
objective. The next continuation should stop trying to rescue the current
local-score actor target. Prefer either direct search-guided self-play
trajectory generation with realized returns, or a VRPO/RM-FSP-style
trajectory-level policy-gradient/fictitious-play branch.

2026-05-26 third update: after corrected native PPO controls failed strict
league promotion, the exact CUDA CFR frontier was refreshed and exposed as an
opt-in `frontier-live` resolver profile. This gives Slumbot/autoresearch
entrypoints a direct way to spend CFR125-quality search when intentionally
testing exact resolving. It remains non-default. The next principled research
step is still to beat the CFR100/125 quality-latency frontier with a learned
resolver-consumed update, or prove that exact resolving transfer justifies the
latency.

2026-05-26 fourth update: true replay-buffer continuation is implemented and
mechanically validated, but it did not produce a better local policy in the
first strategic gates. A small parent-child test lost to the parent, and the
follow-up checkpoint curve plus SD-CFR-style mixture also failed: the parent
beat iter9 with lower95 `+89.57` chips/hand and beat the iter6/iter9 mixture
with lower95 `+2242.51` chips/hand. This means the next continuation should not
be another short continuation, snapshot-density, or mixture-weight attempt. The
active target should be a reviewed self-play policy-improvement mechanism that
uses search or value estimates in the trajectory-generating loop and is judged
by held-out root decisions plus local checkpoint-league lower bounds before any
Slumbot evaluation.

2026-05-26 fifth update: the reviewed restored-checkpoint exact search-as-actor
gate passed fixed-root decision impact but again failed confidence-clean
whole-game transfer. The root result is useful (`+223.00` selected-action value
and no action collapse), but H2H was only `+1.625` chips/hand with lower95
`-60.30`, and local search gains weakly predicted realized duplicate-pair
deltas. The next continuation should be failure synthesis, not more
`eta/worlds` tuning. The synthesis should choose between two mechanism pivots:
a NeuRD/RM-FSP-style regularized self-play update with better game-theoretic
dynamics, or a resolver-consumed learned search object that improves decisions
inside the solver rather than acting myopically at each state.

2026-05-26 sixth update: the smallest NeuRD-style actor-update branch was
implemented and falsified as a sufficient next mechanism. The legal-logit
update passed tests and exported clean checkpoint metadata, but the matched
raw-sequence CTDE q-lambda h512/2k A/B tied PPO in local H2H (`+0.00005`
mean, lower95 `-0.001836`) despite better random eval. The next continuation
must not sweep NeuRD/PPO scalar knobs. Run a reviewed paradigm choice between
two larger mechanisms: a faithful population/regularized-Nash self-play update
that treats average strategy as part of the learning dynamics, or a
resolver-consumed learned search-state update judged inside resolving.

2026-05-26 seventh update: the one allowed population-coupled NeuRD+FSP test
also failed. NeuRD+FSP trained cleanly and had better random eval, but its
average-policy source did not beat matched PPO+FSP in 5000-game local H2H
(`-0.000258` mean, lower95 `-0.007451`). Treat this as the stop point for the
native PPO/NeuRD/FSP family as mainline. The next continuation should open a
reviewed public-belief guided-search learning sprint: learn from search queries
or a resolver-consumed search-state object on local self-play states, and judge
it inside held-out root decisions plus local league transfer before Slumbot.

2026-05-26 eighth update: the public-belief CUDA search frontier refresh
passed and should become the next baseline. On 64 root-disjoint local
successor-pool states, CFR25 was much closer to CFR50 than CFR5/10/15
(`0.2715` L1, `0.0622` KL, `0.78125` action agreement, `105.38 ms` mean
latency, zero illegal mass). The next continuation is not to train an arbitrary
policy head. Open a methodology review for a distribution-matched search-query
learner or resolver-consumed update object, and require it to beat simply
spending more exact CUDA CFR budget at matched latency.

2026-05-26 ninth update: the first search-query learner is not promoted but
points to the correct next interface. The summary-feature budget selector
failed the strict matched gate, but it did lower L1 versus uniform CFR10 and a
same-latency oracle selector was much stronger (`0.3811` L1 at `58.64 ms`).
This says selective search is real, while action/all-in/latency summaries are
too weak. The next continuation should export richer low-budget search-state
features from the frontier, especially the full legal strategy vector and
derived margins, then rerun the selector against the same exact-CFR controls.

2026-05-26 tenth update: the richer low-budget search-state interface passed
the local gate. Frontier records can now export strategy vectors, and the
selector uses a solver-native strategy/margin feature vector when available.
On the same 32/32 split, it beat uniform CFR10 in L1, action agreement to
CFR25, and latency while preserving zero illegal mass. The next continuation
should not celebrate this as strength; it should use this as the minimal proof
that the learned object must be search-state-native. Replace the ridge selector
with a small learned resolver-consumed search-state policy/update on the same
contract, then demand root-disjoint exact-CFR gates and local self-play league
evidence before any Slumbot confidence run.

2026-05-26 eleventh update: tighten the philosophy around neural self-play.
The preferred target is not "online CFR chooses every move" and not "generic
PPO/Rainbow benchmark tuning." It is AlphaZero-like policy iteration for poker:
a stochastic neural policy/value network plays full self-play games; public-
belief CFR improves selected states into stronger mixed strategies; the network
learns those improved strategies and returns to self-play. CFR is allowed as
the improvement operator because imperfect-information poker needs
equilibrium-aware pressure, but the neural network should become the main
player and should eventually need less live search. The next continuation
should design the smallest loop that proves this contract end to end: neural
self-play trajectories, CFR-improved mixed-policy targets, policy/value update,
root-disjoint exact-CFR gate, and self-play league lower-bound check.

2026-05-26 twelfth update: the workflow and goal now enforce that philosophy.
The active goal names AlphaZero-style neural self-play policy iteration, the
default goal template has the same north star, and methodology review
`mechanism_brief` fields now require `neural_policy_role`, `cfr_role`, and
`stochastic_policy_contract`. Future method changes must say whether the
neural net is the main stochastic actor, whether CFR is only the
policy-improvement teacher/control, and how mixed-strategy play is preserved.
The next continuation is implementation, not more prose: build the smallest
end-to-end loop that validates this contract locally.

2026-05-26 thirteenth update: dense preflop public-world rollout teacher is now
the next compute-aligned diagnostic. The public-belief CFR teacher is too sparse
for high GPU utilization because most local self-play states are not turn/river
street roots and solver calls are one state at a time. The new
`public_world_rollout` teacher stays inside local 9-action self-play and
produces legal mixed targets at every preflop root without using Slumbot data.
The first 512-hand CUDA smoke produced 256 rollout targets with no fallback,
but timings still show the blocker clearly: `self_play_collection_sec=4.7405`,
`teacher_target_sec=2.0527`, and `training_sec=0.4504`. Low `nvidia-smi`
utilization is therefore expected: the GPU is visible, but the current work is
dominated by CPU/Python environment stepping and rollout scoring. The next
principled target is vectorized/GPU-native self-play and batched rollout/search
scoring, not a larger network by itself.

2026-05-26 fourteenth update: a first CPU hot-path cleanup reduced public-world
teacher time without changing the algorithm. `score_first_actions_across_worlds`
now builds one root state per sampled public world and copies it for each legal
first action, instead of rebuilding the same world for every action-world pair.
The targeted red/green test showed the old 4-world case made 29 state builds;
the new contract permits only `n_worlds + 1`. On the same 512-hand CUDA smoke,
teacher generation improved from `2.0527s` to `1.5512s`, but the new internal
profile shows the deeper bottleneck is still self-play simulation:
`self_play_env_step_sec=3.4101`, `self_play_policy_inference_sec=0.0936`, and
`training_sec=0.4543`. The next material step is therefore not another small
state-construction cleanup; it is a vectorized/GPU-native environment stepper
or batched rollout/search evaluator that keeps the neural actor and search
teacher supplied with larger continuous batches.

2026-05-26 fifteenth update: the self-play half now has an opt-in fast-state
backend for `public_world_rollout`. This is not a new poker abstraction: it uses
the existing parity-tested `FastPokerState` 9-action full-deck simulator and
keeps the slow object-state backend as default for public-belief CFR, whose
solver input extraction still needs full-deck `PokerState` objects. On the same
512-hand CUDA dense-teacher smoke, fast state reduced
`self_play_collection_sec` from `4.7278` to `0.1896` and
`self_play_env_step_sec` from `3.4101` to `0.0170`; total elapsed dropped from
`6.9358` to `2.3976`. The remaining bottleneck is now teacher rollout/search
scoring (`teacher_target_sec=1.5407`) plus intentionally small neural training
(`training_sec=0.4570`). The next material target is therefore batched or
GPU-native public-world/search target scoring, not more self-play Python object
cleanup.

2026-05-26 sixteenth update: the remaining dense-teacher CPU path was reduced,
and a held-out root gate exposed the next learning blocker. A built-in
`call_policy` specialization removes repeated legal-mask checks during
public-world scoring; on the same 512-hand fast-state smoke,
`teacher_target_sec` fell from `1.5407` to `0.9537` and total elapsed from
`2.3976` to `1.8038`. The new
`eval_neural_policy_iteration_public_world_gate.py` diagnostic then compared an
untrained h128 policy to the 256-target trained checkpoint on the same 64
held-out roots. The trained checkpoint barely improved mixed policy EV
(`-14.7155` to `-14.0424`) and worsened selected-action EV (`-4.9688` to
`-7.0312`), selecting mostly call while oracle actions were split across fold,
call, and all-in. This is not a promotion failure for the main goal because the
gate is public-world, not exact CFR, but it is a warning: the dense preflop
teacher branch needs a stronger target/training interface before scaling. The
next principled step should be to connect this root-disjoint gate to exact-CFR
turn/river targets or add a non-collapsing mixed-policy training objective,
rather than only increasing hands or steps.

2026-05-26 seventeenth update: the neural policy-iteration branch now has a
root-disjoint exact-CFR gate with matched state-source control. The evaluator
can load one checkpoint as the candidate policy and a separate fixed checkpoint
as the self-play state source, so A/B comparisons see the same held-out public
roots instead of each policy generating its own easier or harder roots. A tiny
4-root CPU exact-CFR smoke is not promotion evidence, but it falsified one
methodology risk: training on public-belief CFR targets really can move the
neural policy toward resolver mixed targets on held-out matched roots. The
random h64 checkpoint measured `0.4218` mean L1 and `0.2444` target-to-policy
KL; the 4-target public-belief-CFR checkpoint measured `0.1508` L1 and
`0.0663` KL on the same roots. Top-action agreement remained tied at `0.5`,
with zero illegal mass and no fallback targets. The next continuation should
scale this exact-CFR gate modestly and improve the mixed-policy objective before
any Slumbot evaluation. If GPU utilization is low, treat that as a batching
problem in teacher/search generation, not as a reason to use Slumbot as data.

2026-05-26 eighteenth update: a modest 16-root CUDA exact-CFR gate preserved
the useful signal but exposed a sharper blocker. Against the same fixed
state-source roots and a CFR5 CUDA teacher, the random h64 checkpoint scored
`1.0571` L1, `0.7431` KL, and `0.0625` top-action agreement. The tiny 4-target
CFR checkpoint improved distribution fit (`0.8390` L1, `0.4715` KL) but still
had `0.0625` top-action agreement. The older 32-target checkpoint improved
top-action agreement to `0.875`, but only because its policy top action was
all-in on all 16 roots while the teacher's top action was all-in on 14/16.
That is not a clean strength signal. It means the gate must track mixed-policy
fit and action-collapse metrics together. The next mainline step is a
non-collapsing mixed-policy objective or state-balanced target set for the
neural actor, judged by exact-CFR L1/KL, top-action agreement, and maximum
policy top-action fraction before any league or Slumbot evaluation.

2026-05-26 nineteenth update: the apparent action-collapse blocker was partly
a teacher-budget mismatch. The older 32-target checkpoint had been trained on
CFR1 targets whose top action was all `1`, then judged against a CFR5 exact
gate whose top action was mostly `8`. This is precisely the kind of
train/eval search inconsistency the workflow is meant to catch. The training
metrics now record `cfr_iterations`, `cfr_backend`, and `cfr_device`, and the
pilot has an opt-in `policy_sample_weighting=inverse_target_top` diagnostic
for generic class imbalance. In this run, inverse-top weighting had no effect
because the CFR1 training targets all had the same top action. Matching the
teacher budget to CFR5 and giving the small network enough fit steps was much
more important: the CFR5/train128 h64 checkpoint reached `0.3133` mean L1,
`0.1300` KL, and `0.9375` top-action agreement on the same 16-root CFR5 CUDA
gate, versus `1.0571`/`0.7431`/`0.0625` for random and
`0.9637`/`0.6229`/`0.875` for the CFR1/train4 checkpoint. This is still a
small gate, but it is a coherent AlphaZero-style result: self-play state,
search-improved mixed target, neural fit, held-out exact-search agreement.

2026-05-26 twentieth update: the branch now has a minimal internal self-play
league gate for neural policy-iteration checkpoints. `eval_neural_policy_iteration_h2h.py`
loads two NPI checkpoints and runs duplicate-swapped policy-only games with no
resolver and no Slumbot data. The CFR5/train128 checkpoint cleared a first
1000-game lower-bound smoke against two local baselines: versus random h64,
mean payoff was `0.1102` with lower95 `0.0786`; versus the older
CFR1/train4 32-target checkpoint, mean payoff was `0.0898` with lower95
`0.0592`. This is internal lower-bound evidence, not SOTA. The next step is to
repeat the same pattern at a less toy scale: more root-disjoint CFR5/CFR10
teacher states, same-budget exact-CFR gates, and a small checkpoint league
before any Slumbot run.

2026-05-26 twenty-first update: the AlphaZero-style loop is now repeatable
across generations. `run_neural_policy_iteration_pilot.py` accepts
`--checkpoint-in`, initializes both policy and value nets from the parent
checkpoint, and records parent metadata. `run_neural_policy_iteration_loop.py`
now runs generation `N`, feeds it into generation `N+1`, and evaluates
`N+1` versus `N` with the duplicate-swapped internal league. The first tiny
dense public-world rollout smoke validated the contract but did not show
strength: gen1 used gen0 as its actor, generated 16 new targets, and got
`0.0` mean/lower95 in the 20-game H2H smoke. On the fixed 8-root public-world
diagnostic, mixed policy EV moved only from `115.4537` to `115.7927`, while
selected-action EV worsened from `131.25` to `37.5` because gen1 selected call
on all roots. The current bottleneck is therefore not whether the loop exists;
it is the non-collapsing policy/value training signal and a stronger
root-disjoint gate for repeated self-play improvement.

2026-05-27 native-substrate update: the `FastPokerState` rollout substrate now
has deterministic replay parity against `poker_ai/games/full_deck/state.py` and
cleared the 5x throughput gate by a wide margin. Integrated into the maintained
Tianshou Rainbow control, a matched tiny smoke improved end-to-end collection
from `182.53` to `847.89` train steps/sec. A real 131k-transition fast-state
population/history run collected at `2227.51` train steps/sec, but failed the
5000-game duplicate-swapped H2H gate versus the current local incumbent
(`mean=-0.00206`, `lower95=-0.01515`). This falsifies "rollout speed plus a
simple checkpoint-history mixture is enough" for the current Rainbow recipe.
The next continuation should either improve the population objective/learner
interface, or move more of the maintained RL loop's collection and opponent
inference into a truly batched/vectorized engine; it should not treat this
failed candidate as promotion evidence or use Slumbot for tuning.

2026-05-27 self-play continuation update: the fast-state backend is now wired
into the native PettingZoo AEC adapter and the maintained Tianshou MARL Rainbow
self-play script. A clean sequential smoke improved shared-policy MARL
collection from `241.86` to `1092.11` train steps/sec. Continuing the current
local incumbent as a shared two-seat policy for 65k fast-state self-play steps
ran at `4007.47` train steps/sec and cleared the local promotion ladder: 20k
parent H2H `mean=+0.003705`, `lower95=+0.000561`; complete 5-policy empirical
game solved to pure support on the new checkpoint. This is the first positive
fast-state self-play continuation result after the goal revision. The next
continuation should treat it as the local incumbent and test repeatability:
another self-play generation must beat this parent and stay in empirical-game
support before any sparse Slumbot evaluation.

2026-05-27 repeatability update: the first repeatability checks did not support
monotonic iteration. Two same-budget fast-state shared-MARL gen2 attempts from
the new local incumbent failed parent H2H (`mean=-0.006014`, `upper95=-0.002500`
and `mean=-0.000487`, `lower95=-0.004023`). A synthesis review passed with
`Verdict: revise`, identifying that the fast-state substrate is useful but the
current update operator is not reliably monotonic. Training-time
`--checkpoint-in` was then added to the single-agent Rainbow population wrapper
so population/history responses can initialize from the parent instead of
random weights. The first parent-initialized frozen-population continuation also
failed 20k parent H2H (`mean=-0.006490`, `upper95=-0.001783`). The next
continuation should not repeat these two recipes unchanged. The most concrete
engineering bottleneck is now the slow sequential H2H evaluator; the most
concrete research bottleneck is a stable population-improvement objective.

2026-05-27 fast-H2H update: checkpoint H2H evaluation now has an explicit
`--eval-state-backend fast-state` switch for Rainbow-compatible checkpoints.
The default remains canonical `full-deck`. On a matched 5k candidate-vs-parent
pair, canonical evaluation ran at `55.45` games/sec with `mean=+0.000746`,
while fast-state evaluation ran at `619.25` games/sec with `mean=+0.000529`.
The confidence intervals overlapped closely. Treat fast-state H2H as an
iteration accelerator for local research; keep canonical confirmation for
promotion-critical claims until more parity artifacts accumulate.

2026-05-27 n-step credit-assignment update: a standard Rainbow `n_step=5`
shared-MARL continuation from the local incumbent trained correctly on CUDA at
`3926.17` transitions/sec and passed a fast-state 5k H2H screen
(`mean=+0.014929`, `lower95=+0.006312`). Canonical full-deck confirmation did
not pass: 5k was inconclusive (`mean=+0.002613`, `lower95=-0.006009`) and 20k
remained inconclusive (`mean=+0.001050`, `lower95=-0.003229`). This keeps the
incumbent unchanged and falsifies "multi-step credit assignment alone fixes the
repeatability failure." The next research step should change the population or
objective structure, not keep sweeping small Rainbow knobs. The next engineering
step should make canonical H2H closer to the fast-state throughput without
weakening the promotion boundary.

2026-05-27 common-random H2H update: the engineering step is now partly done.
The native substrate gate includes policy-driven replay parity in addition to
random-action parity. Rainbow checkpoint H2H also has
`--eval-state-backend fast-state-canonical-deal`, which starts from canonical
full-deck deals, preserves the future random board-card sequence, and then uses
fast-state transitions. A root-cause bug was fixed in
`fast_state_from_full_deck_state`: it no longer constructs throwaway fast games
for player-order arrays, so conversion has no NumPy RNG side effect. On the
n-step candidate/parent pair, the common-random backend exactly matched the
full-deck 5k and 20k metrics while raising the strict 20k gate throughput from
`66.41` to `266.11` games/sec. Use this backend for local Rainbow H2H screening
and promotion gates; keep pure `fast-state` H2H as a speed diagnostic only.

2026-05-27 independent-MARL update: H2H evaluation now consumes seat-specific
Rainbow policies for independent multi-agent checkpoints instead of collapsing
all checkpoint formats to `player_0`. This enabled a clean test of non-shared
two-seat Rainbow continuation from the local incumbent. The candidate trained
quickly (`3893.28` transitions/sec) but failed hard against the shared-policy
incumbent over 20k common-random canonical games (`mean=-0.034453`,
`upper95=-0.028379`). This retires the simple "non-shared heads fix
repeatability" hypothesis at this budget. The next live hypothesis should alter
the population/objective update, such as explicit average-policy/fictitious-play
structure or response-oracle population support, rather than only changing the
head sharing topology.

2026-05-27 FSP-average-policy update: mixed-format H2H can now load native
PPO/FSP checkpoints and use either `full-deck` or `fast-state-canonical-deal`
where the checkpoint observation contract permits it. The CLI now exits
nonzero when an explicit lower95 gate fails. A direct test of the existing
raw-sequence CTDE/Q-lambda actor-PPO FSP checkpoint against the current Rainbow
incumbent failed decisively over 5k full-deck duplicate-swapped games
(`mean=-0.031976`, `upper95=-0.020008`). Because that checkpoint uses ordered
action-record observations, it cannot be evaluated through current fast-state
without changing the observation contract. Do not promote or spend Slumbot
hands on it. The next useful FSP/NFSP step is not to reuse this checkpoint; it
is to train an average-policy method directly on the native fast substrate or
use a maintained-library population/FSP implementation behind the native
9-action adapter.

2026-05-27 native-NFSP fast-state update: the native NFSP/FSP pilot now has
`--state-backend full-deck|fast-state`. A matched CUDA A/B at 1000 episodes
improved training environment-step throughput from `266.22` to `1359.54`
steps/sec (`5.11x`), although episode throughput was `4.19x`. This clears the
strict env-step substrate target for average-policy experiments, but not a
strength gate. The resulting 1000-episode fast-state NFSP checkpoint failed
1000 common-random canonical fast H2H games versus the local Rainbow incumbent
(`mean=-0.035504`, `lower95=-0.062882`). Keep this as infrastructure: average
policy/FSP is now cheap enough to test on the native substrate, but the current
small NFSP learner is not a promotable policy. The next step should change the
population/objective update, not scale this weak checkpoint blindly.

2026-05-27 population-NFSP update: a direct fast-state NFSP/FSP response smoke
against four frozen local Rainbow-family opponents also failed. The dueling
h256 learner trained mechanically on CUDA (`1223.08` train steps/sec), sampled
all four opponents equally, and exercised both learner seats, but lost to the
current local incumbent in 1000 common-random canonical fast H2H games
(`mean=-0.060930`, `lower95=-0.090022`). This falsifies the current small
native NFSP/DQN response operator, not average-policy game dynamics in general.
The next principled step should either use a maintained stronger average-policy
or regularized game-dynamics implementation through the native adapter, or move
to a batched/vectorized actor-critic learner whose promotion gate is complete
local league improvement. Do not run Slumbot on these NFSP checkpoints.

2026-05-27 batched-inference substrate update: the next concrete acceleration
target is validated. `scripts/eval_native_rollout_substrate.py` now supports
`--include-policy-inference-benchmark`, which compares one-state-at-a-time
neural actor inference against batched inference over many live fast states.
On CUDA with a hidden-256 masked MLP actor and 2048 fast-state games, sequential
inference reached `6256.75` steps/sec with 4450 policy forward calls; batched
inference reached `46196.01` steps/sec with 41 forward calls (`7.38x`). This is
not strength evidence. It says the next engineering integration should batch
actor inference inside self-play/population collection before applying actions,
so larger neural policies can use the GPU without changing poker rules.

2026-05-27 maintained-vectorization check: existing Tianshou vector batching
does help, but not enough. A matched fast-state shared-MARL Rainbow smoke with
`num_envs=8` reached `2890.73` train steps/sec; `num_envs=64` reached
`4879.52` train steps/sec (`1.69x`) with similar update time. This is far below
the synthetic batched actor ceiling above. The next implementation should not
be another learner knob; it should be a batched fast-state collection path that
groups live observations, performs one policy forward, applies actions, and
hands transitions to a maintained or minimal learner.

2026-05-27 batched-live-collector update: that path now exists as a substrate
primitive. `collect_batched_fast_self_play` emits arrays for features, legal
masks, actions, players, game indices, step indices, rewards, and final payoffs.
The CLI exposes it with `--include-batched-collector-benchmark`. On CUDA with
4096 games, hidden 256, and batch size 128, batched collection exactly matched
the sequential action/payoff checksums while improving throughput from
`7739.34` to `57742.11` steps/sec (`7.46x`) and reducing policy forwards from
14795 to 117. A current Tianshou vector64 smoke reached `5001.43` train
steps/sec, so this custom collector shows about `11.55x` collection headroom.
The next continuation should connect this transition batch to a learner/replay
interface and evaluate policy quality; do not treat the collector benchmark as
strength evidence.

2026-05-27 learner-interface update: the first bridge from batched transition
arrays to GPU training is in place. `run_batched_fast_policy_fit_smoke` and
`--include-batched-policy-fit-smoke` collect fast-state transition arrays and
fit a fresh masked policy head to the collected synthetic actor actions. On CUDA
with 4096 games, hidden 256, train batch 2048, and 200 updates, the loss fell
from `1.7843` to `0.3273`, with the fit loop processing about `2.22M`
samples/sec. This is still not strength evidence because the target is behavior
imitation, not policy improvement. The next useful implementation is to replace
the imitation label with an RL/game-dynamics target over the same batched data,
then gate any resulting checkpoint with duplicate-swapped H2H and empirical-game
support.

2026-05-28 batched-policy-gradient update: the first real RL candidate on the
batched substrate is implemented and falsified. `run_batched_fast_policy_gradient_pilot`
collects batched stochastic self-play, trains a shared policy-gradient actor
from final local returns, and saves a flat native-PPO-compatible checkpoint for
mixed H2H. The h256 CUDA smoke trained 8 x 2048 games in `1.78s` and collected
55436 decisions at about `40309.11` rollout steps/sec, but failed the current
local incumbent over 2000 common-random canonical fast H2H games
(`mean=-0.014598`, `lower95=-0.032620`). This is the expected falsification
surface: the substrate works, but naive shared self-play policy gradient is not
a reliable improvement operator. The next continuation should use the same
batched substrate with population/history opponents or a stronger game-dynamics
target before any Slumbot evaluation.

2026-05-28 environment-native V-trace checkpoint bridge: the train-per-
environment rule is now executable on the native side too. `scripts/run_local_vtrace_compiled_native_learner.py`
uses the compiled 9-action full-deck collector and the local masked V-trace
loss to write a native-PPO-compatible checkpoint marked
`environment=poker_ai:full_deck_hu_nlhe`, `trained_environment_native=true`,
`native_action_projection=false`, and `rlcard_candidate=false`. A 4 x 512 CUDA
smoke produced 6183 learner samples in `0.50s` (`12271` samples/sec), zero
illegal action mass, and zero Python showdown fallback; native self-H2H loaded
the checkpoint cleanly. A 100-game diagnostic against
`models/native_nfsp_dqn_reservoir_h512_10k_seed20260517.pt` was weak/noisy
(`mean=-0.0324`, lower95 `-0.1308`, upper95 `+0.0660`), so this is bridge
evidence only. Do not adapt this native checkpoint into RLCard. The next
continuation should scale/improve the native V-trace/population objective in
the native environment, while any RLCard/AlphaNLHoldem candidate must be
trained fresh inside RLCard under the same high-level schema.

2026-05-28 RLCard-native V-trace checkpoint bridge: the same schema now has a
fresh RLCard-side instantiation. `scripts/run_local_vtrace_rlcard_learner.py`
writes a 5-action `local_vtrace_rlcard` checkpoint that the AlphaNLHoldem
public-reference evaluator accepts as `--candidate rlcard-vtrace`. The tiny
CUDA bridge run passed mechanically, but only processed 8 samples at about
`31.7` samples/sec because direct RLCard stepping is Python/CPU-bound. A
20-game diagnostic against the AlphaNLHoldem checkpoint was not strength
evidence (`mean=-6.35`, lower95 `-21.09`, upper95 `+8.39`). The next
continuation should not scale this exact direct-RLCard learner blindly. Either
find a maintained historical-league/V-trace path that can run RLCard faster, or
keep RLCard as the public reference evaluator while advancing the scalable
native compiled learner and only translating the high-level schema back to
RLCard for benchmark runs.

2026-05-28 pre-Slumbot promotion-gate update: the dual-surface rule is now
executable. `scripts/eval_candidate_promotion_gate.py` combines an RLCard
AlphaNLHoldem H2H artifact, a native 9-action H2H artifact, and a native
empirical-game artifact; it rejects candidates unless all required surfaces
show positive lower-bound evidence, no leakage, and no cross-environment action
projection. `enqueue_slumbot_smoke(... hands > smoke_cap)` now requires both
the older self-play league pass and this promotion gate before spending
held-out Slumbot confidence hands. Running the gate on the current V-trace
bridge artifacts failed as intended with blockers
`rlcard_reference_lower95_below_threshold`, `native_h2h_lower95_below_threshold`,
and `empirical_game_missing`. The next continuation should generate stronger
native/RLCard candidates and fill empirical-game support before any Slumbot
confidence attempt.

2026-05-28 queued promotion-gate update: the same check is now a first-class
autoresearch queue action via `python scripts/poker_autoresearch.py enqueue-promotion-gate`.
It records the dual-surface command under `poker_goal.json`, emits output under
`autoresearch-session/candidate_promotion/`, and can be run by the normal gate
runner. A live queued run on the current weak V-trace bridge artifacts failed
as intended. Future Slumbot confidence work should point to a passed queued
promotion gate, not an ad hoc standalone check.

2026-05-28 native V-trace parent-population update: the compiled native V-trace
learner now accepts `--opponent-checkpoint` for frozen native-PPO-compatible
parent policies. This is the minimal historical-league hook on the compiled
native substrate. The first child trained 4 x 512 games against the previous
native V-trace checkpoint, processed 3198 learner samples in `0.50s`, and kept
zero illegal action mass and zero showdown fallback. It did not clear quality
gates: 200-game H2H versus parent had `mean=+0.00656` but lower95 `-0.02280`,
and 200-game H2H versus native NFSP failed clearly (`mean=-0.16943`, lower95
`-0.25467`). Do not scale this exact recipe by seed/budget alone; the next
candidate-side continuation needs a stronger objective or broader population
support before filling empirical-game gates.

2026-05-31 truth-ledger reset: the latest valid exact-small-game result is
`20260531T160000Z-corrected-registered-go-no-go-ppo-fixed`, superseding the
invalid/fabricated `064621Z` and `142119Z` closes. The corrected registered
small-NLHE OpenSpiel GO/NO-GO measured R-NaD `0.639 +/- 0.0212` exact NashConv
versus PPO+FIFO `2.2547 +/- 0.6477`; PPO did not become lead. The immediate
continuation is not full-HUNL or Slumbot. First harden the exact-small-game
baseline by comparing R-NaD against a literature-faithful PPO control
(`K`-best or equivalent historical league plus conservative learning-rate
check) and/or an MMD/FoReL-style regularized policy-gradient control under the
same exact NashConv, matched-compute, multi-seed protocol. Only after this gate
identifies a learner family worth scaling should full-HUNL GPU work resume.

2026-06-02 housekeeping + direction reset (exploitability-gated learner-family
bake-off; user-approved). Housekeeping: RESEARCH_LOG.md reaffirmed as the single
chronological source of truth; poker_autoresearch_current_findings.md de-staled
with a Status banner; and the three incumbent fields in poker_state.json
reconciled to the re-anchored local incumbent
`fast_state_shared_marl_continue_from_incumbent_h256_65k_dummy8_seed20260763.pt`
(blessed `incumbent_checkpoint` via the tested set-incumbent mutator; vestigial
`current_incumbent_checkpoint` corrected via session.py I/O; 699-entry history
preserved; backup poker_state.json.bak-20260602-housekeeping).

Standing diagnosis (RESEARCH_LOG 20260602T1710..1809): the local PSRO empirical
game and the seed-robust state-conditioned policy router are decision-time
SELECTORS over a frozen archive; every standalone response learner trained
against them (Gen-3 Rainbow, "stronger" NeuRD/NashPG) ties the router yet loses
to incumbent + K-best2. This is the signature of churning on a non-transitive
head-to-head ruler against a self-generated archive (Balduzzi 2018;
arXiv:2206.12301; APSRO arXiv:2207.06541), compounded by a learner whose
regularization is FIXED (run_native_neural_nashpg_compiled_learner.py:
reference_kl_weight=0.05, entropy_weight=0.02, no anneal), which converges to a
regularized-but-still-exploitable point (DeepNash arXiv:2206.15378; NashPG
arXiv:2510.18183). The cached probe corroborates: MMD at fixed alpha converged
(regularized_gap -> 0) yet stayed exploitable (Leduc best last-iterate 0.4799 at
alpha=0.01), and the torch R-NaD run was merely undertrained (4.75 -> 2.19 at
2000 steps, still falling), whereas CFR+ reaches avg 0.0023 / current 0.0169.

Chosen plan (sharpened version of the 2026-05-31 reset, NOT a new direction):

  Stage 1 - small-game exploitability hardening (diagnostic-first; no Slumbot; no
  protected-surface gate change yet):
    (a) Confirm canonical ANNEALED R-NaD drives exact NashConv on Leduc well below
        the fixed-regularization MMD floor (0.4799) toward the CFR+ regime, at
        adequate budget, multi-seed (>=3). The faithful annealing config is a
        MODERATE entropy_schedule_size so the reference net resets several times
        within the budget (sizes=[s],repeats=[1] => reset every s steps forever);
        s = budget/10 (~10 anneal rounds). The default size=20000 is the trap (no
        reset in a short run -> behaves like fixed-alpha MMD). In flight: cycle
        20260602T190047Z-canonical-annealed-r-nad-on-leduc-drives-exact, steps
        20000, s=2000, eta=0.2, seeds 1/2/3.
    (b) Then a matched-compute, multi-seed bake-off on small-NLHE by EXACT
        NashConv: R-NaD/NashPG-annealed vs a literature-faithful K-best/historical
        -league PPO control vs an MMD control. ONE principled default per method;
        NO hyperparameter sweep. Select the single learner family with the lowest
        last-iterate NashConv at matched compute.

  Stage 2 - scale the winner to full HU NLHE (net-only at inference), switching the
  in-house metric to approximate exploitability via a learned best response
  (Timbers et al. IJCAI'22, arXiv:2004.09677) where exact NashConv is infeasible;
  stage sizes and gate each before scaling. Slumbot remains strictly held-out
  confirmation only.

Governance: promoting exact exploitability/NashConv from DIAGNOSTIC to the
promotion GATE is a protected eval-surface change and requires a completed
methodology-review bundle + objective-drift audit BEFORE it gates anything. The
Stage-1(a) confirm runs as a diagnostic and supplies that bundle's evidence; if
it confirms, draft the bundle next (per the approved "both, in sequence" plan).

None of the candidate algorithms is novel (MMD, R-NaD/NashPG, NeuRD, ESCHER,
Exploitability Descent, AlphaHoldem trinal-clip/K-best are all published). The
defensible publication-grade contribution is the controlled, single-PC,
exploitability-gated comparison of net-only tabula-rasa learner families with one
principled default each - not a new equilibrium algorithm, and not a
tuned-to-Slumbot number.

2026-06-02 Stage-1 RESULTS + schedule correction (cycles 20260602T190047Z,
193843Z, 201724Z). Stage-1a (Leduc): neural R-NaD with a FLAT entropy-reset-every-2000
schedule plateaued at NashConv best 1.341 +/- 0.042 (the stated sub-0.48 bar was an
invalid tabular-vs-neural comparison; CFR+/MMD references are exact tabular). Stage-1a
follow-up (Leduc, 3 seeds): a faithful INCREASING-size schedule (1000,2000,4000,8000,16000)
at FIXED eta=0.2 broke the plateau to 0.940 +/- 0.076, still descending at 40k steps.
Stage-1b BAKE-OFF (small-NLHE, exact NashConv, matched 16k steps, 3 seeds):
R-NaD fixed-eta+increasing-schedule WON decisively at 0.0617 +/- 0.0079, vs PPO-FIFO
1.134 +/- 0.311, NashPG 1.155 +/- 0.812 (best-iterate 0.194; high last-iterate variance),
PPO-Kbest 1.866 +/- 0.695; tabular anchors MMD-a0.05 0.458 and CFR+ 0.0005. R-NaD-neural
beat tabular fixed-alpha MMD and approached the CFR+ floor; the schedule correction took
small-NLHE R-NaD from the old GO/NO-GO 0.639 to 0.062.

CORRECTION to the plan entry above: the R-NaD lever is FIXED regularization (eta) + an
INCREASING-size entropy schedule (longer refinement windows per reference reset), NOT
annealing eta->0. NashPG shows decaying alpha lets stochastic-gradient noise dominate the
diminishing regularization signal; the increasing-window schedule at fixed eta is the
DeepNash recipe and is what actually lowered NashConv here. Disregard the earlier
"annealed regularization / decay to ~0" wording.

DECISION: R-NaD (fixed eta + increasing entropy schedule) is the selected net-only
tabula-rasa learner family. NashPG is a high-variance second (best-iterate competitive);
if revisited, add an ESCHER-style learned history-value baseline for variance reduction.
K-best PPO underperformed FIFO at this scale/budget (contra AlphaHoldem's ablation) - note
for the full-HUNL opponent-schedule choice, do not assume K-best transfers.

NEXT (Stage 2): scale the selected R-NaD recipe toward full HU NLHE on the cuda/ substrate,
staging sizes (e.g. river/turn -> full) and gating each size; switch the in-house metric to
approximate exploitability via a learned best response (Timbers et al., IJCAI'22,
arXiv:2004.09677) where exact NashConv becomes infeasible. Architecture (set/sequence
encoder, transformer) is a Stage-2 SAMPLE-EFFICIENCY lever only, not an exploitability-floor
fix (a 2048-hidden net was ~100x worse than a 64-dim net on Leduc). Slumbot remains strictly
held-out confirmation only; promoting exploitability/NashConv from diagnostic to the
promotion GATE needs the drafted methodology bundle
(20260602T190837Z-exploitability-nashconv-promotion-gate) to clear objective-drift audit
first.

2026-06-03 RESUME POINT — trusted metric framework adopted; both-axes problem defined;
inner-update lever exhausted (CIX falsified). This supersedes the 2026-06-02 "Stage 2 = scale
R-NaD" plan above (that plan was wrong: native R-NaD already existed and failed; see below).

WHAT IS NOW ESTABLISHED:
- The objective-drift audit on bundle 20260602T190837Z PASSED (passed=true, protected_hits=[]).
- TWO-AXIS EVALUATION (the trusted ruler; replaces non-transitive H2H):
  * AXIS 1 (method soundness) = exact last-iterate NashConv on small games (Kuhn/Leduc/
    small-NLHE). VALIDATED: positive-control battery (random/fold-chump/shover) ranks
    most-exploitable >> near-Nash on all three games (cycle 20260603T005908Z); uninvertable.
    This is the only signal allowed to GREEN-LIGHT promotion. Anchors: CFR+ ~0.0005, R-NaD ~0.062.
  * AXIS 2 (native strength) = positive-lower95 sweep of a diverse FROZEN gauntlet on full-deck
    HUNL (incumbent + NFSP + K-best + policy-router + worst-gap PSRO). Non-transitive H2H vs a
    single opponent is NOT trustworthy (proven: a candidate at +0.007 vs incumbent lost router
    + worst-gap). A clean sweep is required; any loss = exploitable/non-transitive.
  * The learned-Q BR-LB (run_exploitability_lb*.py) is RETIRED from gating (it inverts: shover
    ranked less exploitable than incumbent). LBR (one-ply, exact solver/equity leaf, asymmetric:
    POSITIVE blocks, near-zero defers) is the SECONDARY native-exploitability screen - DESIGNED
    but PARKED until a full-deck candidate passes AXIS 1 and needs pre-Slumbot vetting.
  * Held-out Slumbot stays final-confirmation-only.

THE OPEN PROBLEM (crisp + measurable): find a SINGLE net-only tabula-rasa learner that passes
BOTH axes. Neither current candidate does: R-NaD/NeuRD is method-SOUND (0.062) but native-WEAK;
PPO-inner is native-competitive but method-UNSOUND (1.238) and non-transitive.

INNER-UPDATE LEVER EXHAUSTED (this session, cheap falsifications):
- The native R-NaD "failures" were largely a SCHEDULE confound: a flat single-window entropy
  schedule. The corrected fixed-eta + INCREASING-window schedule made native R-NaD competitive
  (ties incumbent) but still loses the gauntlet (cycle 20260602T220810Z/222310Z).
- NeuRD-CIX (cap importance weight 1/mu -> 1/(mu+cix_eta)) was implemented on impl-1 RNaDSolver
  (functional.py; RNaDConfig.cix_eta, default 0 = bit-identical, R-NaD torch parity 7/7).
  cix=0.1 stayed AXIS-1 SOUND (0.094) but was AXIS-2 FALSIFIED: behaviorally identical to cix=0
  across all 5 gauntlet opponents (delta within noise, 0/5 wins; cycle 20260603T023816Z). So the
  variance-of-importance-correction is NOT the binding native lever.
- PATTERN: three inner-update variants are now native-weak (plain R-NaD, PPO-inner non-transitive,
  R-NaD-CIX flat) => strong evidence the binding AXIS-2 lever is REPRESENTATION / OPPONENT-SCHEDULE,
  NOT the inner update (brainstorm Lens 4).

NEXT DIRECTION (decision deferred to user; do NOT auto-launch a multi-day build):
- Leading hypothesis (evidence-backed): AXIS-2 strength needs a stronger REPRESENTATION (card/
  action set-or-sequence encoder, AlphaHoldem-style) and/or a K-best historical OPPONENT SCHEDULE,
  keeping the method-sound R-NaD inner update as the anchor and gating on AXIS-1 soundness first.
  This is a protected-surface build (encoder threaded through training/eval), days not minutes.
- Cheaper alternative (one more inner-update attempt, governance-awkward = multi-change):
  NashPG-done-right (fixed-large reg alpha=0.2 + per-outer-round inner convergence + GAE advantage)
  - targets the inner-loop/bootstrap the CIX falsification implicated; risks a 4th inner-update miss.
- Parked governance items: complete the methodology bundle to ACTIVATE exact-NashConv as the
  official promotion gate; build the LBR secondary screen when a full-deck candidate is ready.

STATE NOTES: cix_eta lives in poker_ai/rnad/solver.py (RNaDConfig, default 0.0) + functional.py
v_trace/compute_rnad_loss + native_rnad.run_compiled_native_rnad_learner + the small-NLHE baseline
gate (--cix-eta); default 0 makes it inert/parity-safe. Many uncommitted edits this session
(functional.py, solver.py, native_rnad.py, gate scripts, docs, the metric bundle); CONTINUOUS_ACTIVE
is present so commits are blocked until it is cleared. tianshou 2.0.1 was installed (no torch/numba
downgrade) to load Rainbow checkpoints for native H2H.

2026-06-03 UPDATE — anchor characterization done; fork resolved to SCALE R-NaD; GPU enabler built.
The "decide after anchor results" fork (above) is RESOLVED. R-NaD anchor characterization
(RESEARCH_LOG 20260603T133428Z): the anchor is BUDGET-LIMITED, not floored, on both axes —
small-game NashConv kept descending (Leduc 0.94@40k -> 0.39@120k, still descending; Kuhn solved),
and native gauntlet gaps ALL close monotonically with budget (incumbent -0.052/-0.042/-0.026,
router -0.051/-0.043/-0.034, worstgap -0.081/-0.069/-0.044 at 1500/3000/6000 iter; at 6000 it
already BEATS nfsp +0.033 and kbest +0.026). So native weakness was substantially UNDERTRAINING,
not (yet) a representation ceiling — this partially un-confounds the "inner-update exhausted"
conclusion for R-NaD. DECISION: scale R-NaD compute before any AlphaHoldem-encoder (LEAD A) build;
reserve LEAD A for if/when R-NaD plateaus while still failing AXIS-2.

ENABLER BUILT (RESEARCH_LOG 20260603T160000Z): CUDANativeRNaDCollector (poker_ai/rnad/cuda_collector.py)
runs R-NaD self-play on the cuda/ GPU kernels — ~1.1M steps/s @B=4096 (2.5M @B=32768) vs the CPU
single-core ceiling ~137K. Opt-in via run_compiled_native_rnad_learner(substrate="cuda"); CPU is
default; 6 CUDA-guarded parity tests pass; no methodology bundle required (substrate port of a
diagnostic learner); DDR at docs/research_protocols/gpu_rnad_collector_design_decision.md.

NEXT (resume here): run LARGE-budget R-NaD on substrate="cuda" (big batch + ~24k-50k+ iter, the
DeepNash increasing-window schedule, cix=0) and re-run the AXIS-2 frozen gauntlet (eval_mixed_policy_h2h,
candidate-kind native-ppo) to test whether the still-closing native gaps cross zero — i.e. whether a
method-SOUND learner can clear AXIS-2 by compute alone. If gaps cross 0 / clean sweep -> first
both-axes learner (then falsification ladder + Slumbot confirmation). If they plateau still-negative
-> the representation lever (LEAD A: AlphaHoldem encoder + K-best, positioned vs NashPG arXiv 2510.18183)
is the next build. Working tree has uncommitted GPU-collector work (cuda_collector.py, the parity test,
the native_rnad substrate kwarg, the DDR, RESEARCH_LOG, this file) pending a commit.

2026-06-04 RESUME POINT — net-only ceiling consolidated; ReBeL search-in-learning is the line.
THE LEAD-A / both-axes / "scale R-NaD" framing above is SUPERSEDED. Consolidated finding:
docs/research_protocols/net_only_ceiling_and_rebel_decision.md (authoritative). Summary: a trusted LBR
exploitability gate (scripts/run_lbr.py, positive-controls pass N=500; decode = masked-softmax, verified
0.0 vs the gauntlet adapter) shows net-only R-NaD is FLAT at ~12-15K mbb/g across SIX levers (budget,
batch, grad-clip, reset-density, CNN representation, last-iterate-vs-time-average readout); eta-annealing
pre-falsified. Slumbot ~4K, ReBeL ~881, Nash ~0 -> net-only is ~3x too exploitable and capped on this
hardware. R-NaD is the least-exploitable LOCAL agent (incumbent ~28K > router ~16K > R-NaD ~12K) but NOT
near-Nash. DECISION (user, goal = tabula-rasa self-play SOTA-EFFICIENCY model): pursue ReBeL-style SEARCH
INTEGRATED INTO LEARNING (net-centric mainline; Modicum/play-time-bolt-on OUT of scope). Repo already has
~80% (fast_cfr learned-leaf hooks + per-infostate CFVs + belief=PBS; _HandCFVProbeNet ~= PBS value net;
CFV-target prototypes; exact Leduc NashConv + leaf oracle). ReBeL plan: sound-with-fixes, SMALL-GAME-FIRST
(Leduc kill-switch -> small-NLHE -> flop-truncated HUNL). The SOTA-EFFICIENCY contribution = fit near-Nash
HUNL onto ONE consumer GPU (the GPU leaf-hook port, gated behind the small-game proof).
NEXT (resume here): the user chose CONSOLIDATE-FIRST; the consolidation doc + this update are it. Then the
ReBeL de-risk FIRST STEP (~0.5 day, no training): constant-oracle round-trip on Leduc -- solve one subgame
exactly (run_qfr_leduc_gate oracle CFVs) -> feed back through a cut_node_fn constant leaf -> assert parent
exact NashConv ~0 (isolates the CFV-normalization + uniform-averaging must-fixes). Then Stage 1 oracle-leaf
Leduc loop -> Stage 2 learned-leaf bootstrap (gate: Leduc NashConv <= ~0.02, below R-NaD 0.06-0.4).

2026-06-05 RESUME POINT — ReBeL moved off toy Leduc onto a REAL-BELIEF game (turn subgame + neural
river leaf); steps A/B/C DONE incl. the live net-leaf exploitability gate. WHY THE MOVE: the Leduc
de-risk validated every ReBeL mechanic (CFV normalization convention, uniform averaging, safe-resolving
gadget, net learnability) but is TOO SHALLOW (2 rounds) to show a value net's compute benefit or to test
trunk soundness; the diagnosis (RESEARCH_LOG 20260604T203000Z, Brown-Sandholm 2018 arXiv:1805.08195)
is that a single-value leaf is unsound in imperfect-info games and the FIX (multi-valued states / PBS-CFR)
is known + modest-hardware-feasible -> the contribution is running it on ONE GPU on a real game, not
re-deriving soundness on Leduc. SUBSTRATE: `poker_ai/rebel/turn_river.py` + `river_pbs_net.py` +
`scripts/run_rebel_net_gate.py`, built on the trusted StreetSolver + fast_cfr.solve_cfr (its showdown_leaf_fn
hook is the turn->river depth limit). EVIDENCE (all verified, Slumbot held-out):
 - STEP 0: exact-river leaf is per-iteration-INFEASIBLE in the turn trunk (~323 hr/turn-solve) -> a learned
   value net is LOAD-BEARING. (RESEARCH_LOG 20260605T120000Z)
 - A (GPU-fast targets): `turn_leaf_river_cfv_batched` runs all 44 river runouts in one same-topology
   batched GPU call, exact-match to CPU (3670.21==3670.21), 3.1x.
 - B (2-street exploitability): `street_br_value`/`two_street_nashconv` (forward fixed-player reach; backward
   MAX at BR nodes / SUM at fixed; agent-avg BR == subgame_value_pass exactly). Degenerate-uniform agent
   0.35 pot vs exact-leaf agent 0.023 pot. CAVEAT: BR over-reads poorly-averaged low-reach infosets on DEEP
   trees -> measure on SMALL/short-stack spots (the gate's domain).
 - C (river PBS net) + C-COMPLETION (live net-leaf gate, 20260605T143000Z): a cut-GENERAL `CtxRiverNet`
   (ctx = pot/stacks scalars + both normalized ranges -> per-hand net-from-river CFVs), trained on
   exact-river targets across ALL 7 cut public states, plugged in as the LIVE showdown_leaf_fn over a full
   turn solve. Spot AhKd7c2s pot10bb stacks3bb: net val MAE 3.5% of scale; exact-leaf control 2-street
   NashConv 0.041 pot (1587.7s) vs NET-leaf 0.076 pot (1.44s) = +0.034 pot exploitability at 1102x less
   compute. Net inference vs exact leaf = 181,000x (microbench). => the learned leaf is a faithful, vastly
   cheaper substitute END-TO-END in a real solve, not just in isolation. The single-value-leaf design point
   is validated on the LEAF; the multi-valued-states question is a TRUNK/self-play concern (Step 3).
NEXT (resume here): STEP 3 = the self-play ReBeL TRAINING LOOP on this turn+river substrate -- sample PBSs
from trunk solves -> re-solve (gadget-safe) -> harvest SELF-CONSISTENT CFV targets off the trunk (not
isolated re-solves) -> update the net -> iterate, watching 2-street exploitability (metric B) trend down.
This is where the trunk multi-valued-states soundness question becomes live (Leduc Stage-2c/diagnosis).
THEN Step 4: scale (flop-truncated HUNL / deeper truncation) + an explicit compute go/no-go before any
multi-week run. Tasks #26 (Step 3), #27 (Step 4) track this. Uncommitted at this point: only
`scripts/run_rebel_net_gate.py` + this doc + RESEARCH_LOG (the C-completion commit).

2026-06-05 RESUME POINT — Step 3 (turn+river SELF-PLAY loop) DONE; result NEGATIVE-BUT-INFORMATIVE;
contribution REPRICED to single-GPU EFFICIENCY; build paused on a user decision. See RESEARCH_LOG
20260605T180000Z (authoritative). Built `poker_ai/rebel/self_play.py` (on-policy + exploration PBS
sampling, EXACT-river targets, reservoir, abort-on-drift) + `scripts/run_rebel_self_play.py` + tests
(27/27 rebel green; `run_rebel_net_gate.py` refactored to import the shared helpers). 8-iter run on the
C-gate spot: ITER 0 (broad random coverage) is BEST at 0.072 pot (~= single-shot 0.076); on-policy iters
DRIFT UP to plateau 0.084; net MAE rises 3.8%->7.4%; exact-leaf control floor 0.041. DECOMP of 0.072 =
0.041 perfect-leaf residual (24 trunk iters + BR over-reads low-reach) + 0.031 net-approx (reproducible)
+ 0.012 drift. ROOT CAUSE: river is the LAST street -> exact targets -> NO bootstrapping; self-play only
shifts the PBS distribution broad->peaked, which HURTS. Same shallowness as Leduc Stage-2c.
LITERATURE-CORROBORATED (textbook, not a failure): DeepStack uses NO river net + trains flop-from-turn-net
(bootstrap BETWEEN streets) on STRUCTURED pseudo-random ranges; ReBeL bootstraps only at NON-terminal
leaves + eps=0.25 exploration (uniform-random "fails to learn"); Supremus per-round-except-final, win is
NET QUALITY. REPRICE: "self-play is the win" FALSIFIED; surviving defensible contribution = a SOUND
near-Nash HUNL resolver on ONE 8GB consumer GPU with a VERIFIED exploitability residual decomposition --
matches the standing thesis (efficiency of a proven method, not re-deriving the method). TWO SEPARATED
LEVERS: (a) close 0.072->0.041 = SUPERVISED net quality (broader/structured coverage + bigger net +
continual resolving), NOT self-play; (b) make self-play MATTER = bootstrapping DEPTH.
NEXT (resume here): the user chose RECORD + PAUSE; this entry + the log are it. The recommended build
sequence when resumed is OPTION S then OPTION 1: (S) push the river net toward the 0.041 floor the
DeepStack way (structured pseudo-random coverage + bigger net + continual resolving) -- locks in the
single-GPU efficiency deliverable AND sets the matched-compute baseline; then (1) FLOP trunk + a
bootstrapped net TURN leaf (turn-net targets harvested by solving the turn subgame with the net river
leaf) -- the ONLY honest, matched-compute A/B for whether self-play bootstrapping beats supervised
amortization, gated behind a compute go/no-go (flop PBS dim >100 vs ~6 verified on Leduc; thin-reach
under-determination risk). OPTION 3 (truncate river betting) = optional weak/confounded de-risk, not the
headline. Guardrail: do NOT HP-sweep the existing loop to cosmetically match iter-0. If Option 1 cannot
beat the Option-S floor at matched compute, the honest publication is the supervised single-GPU efficiency
result with self-play reported as a characterized negative -- still a sharp contribution.

2026-06-05 RESUME POINT — OWNER REFRAME to a GENERAL IIG method; the per-street poker road (Option
S/1) is SUPERSEDED/PARKED. See RESEARCH_LOG 20260605T210000Z (authoritative). NEW GOAL: a GENERAL,
simple, tabula-rasa self-play method for 2-player zero-sum imperfect-info games -- "the same
computation solves ANY IIG" -- that is single-GPU-efficient (RTX 3070 Ti, 8GB); HUNL-vs-Slumbot is ONE
INSTANCE, not the hand-engineered target. The contribution is single-GPU EFFICIENCY of a PROVEN general
method (ReBeL/PoG/SoG), not a new algorithm. The generality/compute tradeoff is minimized with
GENERALITY-PRESERVING levers ONLY: GPU-batched subgame solves; TurboReBeL-style fix-strategy
multi-iteration data harvest (~450x, OpenReview 2025); DCFR/PCFR+; small nets; scale-the-game-not-the-
algorithm. Single-GPU general precedent exists (LAMIR single-A100, arXiv:2510.05048). The Step-3
negative result is DISSOLVED -- it was an artifact of pinning the depth limit at the TERMINAL river;
the general method uses a NON-terminal cut that bootstraps.
BUILD 1+2 DONE (foundation): poker_ai/rebel/iig.py -- a game-agnostic PublicTree (enumerated entirely
from OpenSpiel: terminal/chance/decision; private deals + the public cut are both 'chance') + CFR+ +
exact NashConv. The SAME code across games (scripts/run_rebel_iig_suite.py): Kuhn 0.00034, Leduc 0.00461
(BIT-IDENTICAL parity vs the trusted LeducTree, |diff|=0.0 -- the correctness gate), Liar's Dice 3-sided
0.00021, Liar's Dice 6-sided (the ReBeL benchmark, non-poker) 0.00439. 4 tests green
(test/unit/test_rebel_iig_general.py). It does NOT touch the golden-tested leduc.py/loop.py (it
parity-validates against them). The specialized self_play.py structured-coverage sampler +
scripts/run_rebel_supervised_gate.py are PARKED (uncommitted).
NEXT (resume here): BUILD 3 = generic depth-limited PBS net self-play on the PublicTree -- a public-state
abstraction (public_state_key / is_depth_limit on the generic tree; OpenSpiel gives information_state
but not public state, so this is the real new piece -- see Factored-Observation Games / Kovarik et al.
arXiv:1906.11110), ONE PBS value net over generic [public-state | range0 | range1], the safe-resolving
gadget, and the self-play loop solving the trunk with the net leaf at a NON-terminal cut -> harvest
self-consistent CFV targets -> retrain. Validate on Leduc (public-card cut) vs the leaf-oracle control,
then small games. Decouple loop.py/pbs_value_net.py/leaf_eval.py from Leduc (drop NCARDS/LeducTree) with
the iig parity gate as protection. THEN Build 3b: single-GPU efficiency levers + HUNL scaling go/no-go
(tasks #29, #30). Honest ceiling: small games provably near-Nash = GO; depth-limited HUNL LBR-competitive
= conditional GO; provably-near-Nash full HUNL on one 8GB GPU = NO-GO.

2026-06-05 update — BUILD 3a DONE (the keystone, the genuinely-new piece): the generic public-state /
PBS abstraction. poker_ai/rebel/iig_pbs.py (PBSStructure): public state via OpenSpiel's public observer
(card-INDEPENDENT at Leduc's board-deal cut, confirmed -> groups identically to leduc's round-1-betting
keys; per-game public_key_fn fallback for games without an observer, e.g. Liar's Dice); the ONE
game-specific hook is is_cut_fn (Leduc = board-deal chance node); each player's private state =
information_state_string(player) at the cut (OpenSpiel gives it even at the chance node). BIT-IDENTICAL
parity vs LeducTree.cut_reaches (5 public states; player-0/1 reach multisets max|diff|=0.00e+00). Belief
dims across games (the scaling driver): leduc 5pub/6priv, kuhn 1/3, liars_dice 3-sided 1/3, 6-sided 1/6.
2 tests green (test/unit/test_rebel_iig_pbs.py); report scripts/run_rebel_pbs_report.py. RESEARCH_LOG
20260605T233000Z. NEXT = BUILD 3b: the generic depth-limited net LEAF CONSUMPTION on this PBS structure
-> a values round-trip vs the exact control on Leduc (the Stage-0 convention check, generalized), then
the safe-resolving gadget + the self-play loop at a NON-terminal cut (dissolves Step-3 no-bootstrapping).
Then efficiency levers + HUNL scaling go/no-go.

2026-06-06 update — BUILD 3b-i DONE: the generic depth-limited solving substrate + the convention
round-trip. poker_ai/rebel/iig_solve.py (DepthLimitedGame): enumerates the game into terminal/chance/
decision/CUT (CUT carries public key + per-player private index + the continuation subtree);
full_values (ground truth), oracle_leaf_v (exact normalized-PBS leaf = generic ExactLeafOracle),
trunk_q_with_leaf (depth-limited consumption), trunk_solve (generic loop.trunk_solve twin; net/oracle
plug in via leaf_fn), blueprint_leaf_fn. CONVENTION ROUND-TRIP gate (game-agnostic Stage-0, on Leduc:
36 trunk infosets / 150 cut instances / 6 private): normalized leaf reproduces exact full-game above-cut
values to 4.4e-16 (matches trusted Stage-0 8.9e-16); un-normalized control breaks at 2.5-3.7 (matches
2.24). trunk_solve smoke green. 4 tests (test/unit/test_rebel_iig_solve.py); 10/10 generic-method tests
pass together. RESEARCH_LOG 20260606T000000Z. NEXT = BUILD 3b-ii: the safe-resolving GADGET on the
generic tree + the SELF-PLAY LOOP (trunk_solve with the net leaf at a NON-terminal cut -> harvest
self-consistent CFV targets via oracle_leaf_v -> train a generic PBS net -> iterate), validated on Leduc
vs the leaf-oracle control, then a second small game. Then efficiency levers + HUNL scaling go/no-go.

2026-06-06 update — BUILD 3b-ii DONE: the game-agnostic depth-limited PBS-net SELF-PLAY LOOP.
poker_ai/rebel/iig_selfplay.py (ONE PBSNet, ONE solver, ONE loop, generic interface): each iter
trunk_solve(net leaf) -> harvest EXACT leaf targets [broad random-range coverage + on-policy
belief-consistent ranges] under a fixed near-eq continuation (added DepthLimitedGame.cfr_plus +
precompute_cont) -> train -> iterate; control = trunk_solve(exact blueprint leaf). RESULT on Leduc
(8 iters): net reach-weighted MAE 8.4% -> 3.4% (LEARNS, matches/beats Stage-2b 5.5%); net-leaf sigma1
L1 vs exact-leaf control 0.394 -> 0.211 (tracks it; residual ~0.21 = the confounded-L1 floor on Leduc's
mixed equilibria, NOT net error). 1 test (test/unit/test_rebel_iig_selfplay.py); 11/11 generic-method
tests green; driver scripts/run_rebel_selfplay.py. RESEARCH_LOG 20260606T013000Z.
=> The general method's TRAINING LOOP is built + validated end-to-end on the game-agnostic substrate
(Build 1+2 solver -> 3a public-state/PBS -> 3b-i leaf+round-trip -> 3b-ii self-play loop).
RECONFIRMS THE SHALLOWNESS BOUND: Leduc (like turn+river) is 2-LEVEL (one cut) -> leaf is the final
round -> EXACT targets, NO bootstrapping (Step-3 finding); this loop is leaf-amortization + coverage.
NEXT = THE DEPTH FORK (Build 4 decision, USER): to show self-play BOOTSTRAPPING (beat supervised
coverage / clear the single-value-leaf trunk wall) needs EITHER (a) a >=3-LEVEL game -- a non-terminal
cut whose continuation is itself net-evaluated (e.g. a 3-round small game / flop-truncated HUNL with a
turn-net leaf), OR (b) MULTI-VALUED STATES in the generic trunk (Brown-Sandholm 2018, the documented fix
for single-value-leaf unsoundness). NOT YET BUILT (gated on this fork): the generic safe-resolving gadget
(play-time) + end-to-end NashConv-assembled exploitability (on 2-level Leduc both hit the documented
trunk wall regardless). Then single-GPU efficiency levers + HUNL scaling go/no-go (tasks #29 done core,
#30).

2026-06-06 update — BUILD 4 chosen = Option D (soundness then bootstrapping). PHASE 0 DONE: added the
exploitability infra to DepthLimitedGame (to_tabular / nash_conv / assemble via OpenSpiel) and CONFIRMED
the single-value-leaf TRUNK WALL on the game-agnostic substrate -- trunk_solve(exact blueprint leaf) ->
assembled sigma1 exploitability 0.367 vs the full-CFR Nash floor 0.0046 (~80x), the generic reproduction
of the Leduc Stage-2c/diagnosis. FAST GATE: a NAIVE symmetric multi-valued leaf (per-(priv0,priv1) min
over K continuations) made it WORSE (0.367->0.724) -- per-pair min over-powers the opponent; the correct
mechanism is a CONSISTENT regret-matched opponent continuation choice per its own private state (asymmetric
Brown-Sandholm), awkward for a symmetric self-play trunk. DECISION (per the Build-4 brief): skip multi-valued
-> PBS-CFR. Removed the buggy naive method; kept the exploitability infra (5 tests in
test/unit/test_rebel_iig_solve.py). RESEARCH_LOG 20260606T030000Z.
NEXT = PHASE 1: PBS-CFR trunk (new poker_ai/rebel/iig_pbs_cfr.py: CFR over public-belief-state nodes
carrying (pub_key, r0n, r1n); the value net / leaf_fn interface is unchanged; iig_pbs.cut_reaches gives
the belief vectors) -> sigma1 near-Nash on Leduc (sound-by-construction, BOTH players). Gate: assembled
sigma1 exploitability drops from 0.367 toward the Nash floor. Biggest uncertainty (from the brief):
whether the belief-tree CFR walk is a clean extension of _trunk_cfv or needs range-vectors-as-node-state.
THEN Phase 2: multi-cut Goofspiel(5) bootstrapping (exact NashConv computable, 4 cuts).

2026-06-06 update — PHASE 1 DECISIVE RESULT: LEDUC (2-level) IS ILL-POSED for depth-limited soundness;
Phase 1+2 MERGE onto a >=3-level game. Before building a full PBS-CFR trunk, ran the decisive cheap test
(RESEARCH_LOG 20260606T050000Z): built the generic per-belief subgame solver
(DepthLimitedGame.solve_subgame_equilibrium + per_belief_equilibrium_leaf_fn) and ran trunk_solve with
the STRONGEST possible leaf -- the per-belief EQUILIBRIUM value V_i(beta), what a perfect PBS net learns.
RESULT on Leduc: Nash floor 0.0046; single-value fixed leaf sigma1 0.4435; per-belief EQUILIBRIUM leaf
sigma1 0.2628 -- the perfect leaf HELPS (0.44->0.26) but does NOT clear the wall (still ~57x Nash). This
INDEPENDENTLY CONFIRMS the in-repo diagnosis (fixed exact per-range Nash leaf -> 0.239): two leaf types,
same ~0.25 residual. CONCLUSION: on a 2-level game, a depth-limited trunk with ANY value-function leaf is
fundamentally biased; only co-evolving full CFR-D (no depth-limit benefit) is sound. The value-net
soundness benefit -- like bootstrapping -- needs a >=3-LEVEL game (a NON-terminal cut above a subtree
with its own depth for co-evolution). So PBS-CFR-on-Leduc (Phase 1) is ILL-POSED; soundness + bootstrapping
both move to GOOFSPIEL(5) (2124 infosets, exact NashConv, 4 cuts). 6 tests in test_rebel_iig_solve.py.
NEXT (Build 4, redirected) = the >=3-LEVEL GOOFSPIEL build: DepthLimitedGame nested-cut support +
goofspiel public_key/is_cut + the depth-limited self-play with the net leaf at a non-terminal cut ->
assembled NashConv vs the exact oracle, where soundness AND bootstrapping are both genuinely testable.
Then #30 efficiency/scaling. (The generic substrate -- solver, PBS abstraction, leaf oracle, self-play
loop, subgame solver, exploitability infra -- is all built + tested; Goofspiel needs nested cuts + the
2 game-specific hooks.)

2026-06-06 update — GOOFSPIEL FOUNDATION DONE (the >=3-level testbed is now in the generic substrate).
Goofspiel is natively SIMULTANEOUS -> use OpenSpiel load_game_as_turn_based -> SEQUENTIAL; points_order=
random gives prize-reveal chance cuts (num_cards C -> C-1 cuts). No public observer -> added a custom
public_key_fn (iig_pbs.goofspiel_public_key: parse Point card sequence + Win sequence + Points;
private = P{p} hand/action lines) + goofspiel_is_cut + load_goofspiel. VERIFIED on Goofspiel(4)
turn-based: the SAME generic CFR+ reaches NashConv 0.00278 (exact Nash on a >=3-level non-poker game),
n_iset ~3608; PBS cut structure = 12 public states, belief dim 4/player. 2 tests
(test/unit/test_rebel_iig_goofspiel.py). RESEARCH_LOG 20260606T070000Z. This de-risks the build:
the >=3-level testbed is representable, solvable, multi-level-cut.
NEXT (the hard CORE of Build 4) = NESTED-CUT depth-limited self-play: extend DepthLimitedGame._build
with depth_level so is_cut fires at EACH round boundary (not just the first); oracle/leaf recursion
across cut levels (the leaf at cut-d is a net query over cut-(d+1) values, themselves net/exact ->
genuine bootstrapping); then depth-limited self-play with the net leaf at NON-terminal cuts -> assembled
NashConv vs the exact oracle. On Goofspiel (>=3 levels) BOTH trunk soundness AND bootstrapping are
genuinely testable (unlike 2-level Leduc, which Phase 1 proved ill-posed). Then #30 efficiency/scaling.

2026-06-06 update — KILL-SWITCH PASSES (DECISIVE POSITIVE). Per the juncture decision (Option C: minimal
kill-switch then pivot to scale), ran the cheapest decisive test on Goofspiel (>=3-level), reusing
existing machinery (scripts/run_rebel_goofspiel_killswitch.py): the cut at the prize-2 reveal is
NON-TERMINAL (continuation = rounds 2-4). CONVERGED (cont 800/trunk 500/subgame 150): Nash floor 0.0011;
base uniform 1.4167; SINGLE-VALUE (fixed) leaf sigma1 0.0047 = 4.3x Nash (~300x below base); per-belief
LIVE-RESOLVE leaf 0.2178 = 195x. => On a >=3-level game, a depth-limited trunk with a FIXED value-function
leaf (what a net learns) is NEAR-NASH (4.3x) -- the depth-limited soundness mechanism WORKS in our code,
in SHARP contrast to 2-level Leduc (same fixed leaf ~96x, ill-posed). The live re-solve being far worse
(195x) re-confirms the ReBeL rule: use a FIXED net leaf, not a live re-solve. RESEARCH_LOG 20260606T093000Z.
=> Every ReBeL/depth-limited mechanic is now validated on the generic substrate (poker + non-poker),
INCLUDING depth-limited soundness at >=3 levels. The mechanism class is de-risked IN OUR CODE.
NEXT = GREENLIT Option-B PIVOT: the single-CONSUMER-GPU EFFICIENCY FRONTIER + scaling study (the actual
contribution -- learned abstraction + GPU-batched solving on a real-belief game; HUNL one instance,
Slumbot held-out), with this kill-switch as the soundness rigor-appendix. (Task #30.) The full
nested-cut Goofspiel(5) characterization is intentionally NOT pursued -- per the brief it would re-prove
known ReBeL/PoG/LAMIR theory; the fixed-leaf kill-switch is sufficient soundness evidence.

2026-06-06 update — SCALE-READINESS PROBE DONE (the cheapest go/no-go before the abstraction build).
scripts/run_rebel_scale_probe.py walks the Goofspiel size ladder + measures net-leaf self-play
exploitability. RESULTS: FIRST CURVE POINT goofspiel(4) (3608 infosets) net-leaf self-play exploitability
0.0446 = 16x Nash floor 0.0028 (~32x below base 1.42); net MAE 5.2%; 18s on the CPU host. SIZE LADDER:
goofspiel(4)=3608 but goofspiel(5)=236,450 infosets -- PAST the unabstracted Python-tree-walk ceiling
(~few-k). GO/NO-GO: the de-risked method works END-TO-END on a >=3-level game on the GPU host, but the
unabstracted substrate CAPS at ~few-k infosets; VRAM is NOT binding (net is tiny; CPU tree-walk is) ->
a game-agnostic LEARNED ABSTRACTION is the MANDATORY next lever (GPU kernels are net-negative at this
scale; abstraction makes 8GB viable, LAMIR-style). RESEARCH_LOG 20260606T120000Z.
NEXT (the LOAD-BEARING build) = poker_ai/rebel/iig_clustering.py: a game-agnostic reach-weighted
infoset-clustering hook at DepthLimitedGame._iset_id, gated on compressing an intractable game
(goofspiel(5), 236k) to feasible (~5-10k) WITH BOUNDED NashConv degradation. The abstraction x
depth-limited-soundness coupling is THE single biggest remaining risk (mechanism de-risked with a FULL
leaf, NOT yet under abstraction). Then scale the ladder (goofspiel -> limit-Hold'em-truncated -> HUNL)
with the explicit compute go/no-go gates (GO if curve extends >=1 order of magnitude before the 8GB wall;
NO-GO if VRAM caps below goofspiel-10 or tree-walk stays intractable even WITH abstraction).

2026-06-06 update — ABSTRACTION-COHERENCE GATE: the load-bearing risk FIRED. Before building an abstract
solver, ran the cheapest test of the abstraction x soundness coupling (poker_ai/rebel/iig_clustering.py:
collect_features=info_state_tensor, cluster_infosets=KMeans per (player,action) group, strategy_incoherence
=within-cluster equilibrium-strategy L1 from the cluster mean). On Goofspiel(4) (exact eq): FEATURE
clustering (cheap, game-agnostic) incoherence 0.41 FLAT across 5x-45x; VALUE clustering (eq q-vector) 0.22
->0.36. => the cheap game-agnostic feature (info-state tensor) is NOT a sound abstraction signal (merges
strategically-divergent infosets ~0.41 L1 even at 5x -> would break depth-limited soundness); the right
signal is VALUE-based but CIRCULAR on big games (must solve to get values) AND still lossy. So the "cheap
learned abstraction" the scale brief assumed does NOT work as-is. 1 test (test_rebel_iig_clustering.py);
probe scripts/run_rebel_abstraction_probe.py. RESEARCH_LOG 20260606T140000Z.
NEXT = SCALE-LEVER FORK (owner decision): (A) VALUE-guided abstraction bootstrapped from a CHEAP coarse
solve (LAMIR-style; mitigates circularity, still lossy); (B) MCCFR / external-sampling CFR -- the classic
scale lever that AVOIDS full-tree enumeration entirely (no abstraction, game-agnostic; repo has legacy
tabular MCCFR); (C) accept the unabstracted ceiling and frame the single-consumer-GPU contribution at the
games it reaches (Leduc/Goofspiel-class) honestly. The mechanism + general pipeline are fully built +
de-risked; this fork is purely about HOW to push the scale/efficiency frontier on 8GB.

2026-06-06 update — LEVER B (naive MCCFR) ALSO FAILED its premise gate. Owner chose lever B; ran the
premise gate with OpenSpiel's production external-sampling MCCFR (scripts/run_rebel_mccfr_gate.py).
RESULT NEGATIVE: Goofspiel(4) full-tree cfr_plus NashConv 0.0015 (600 it) vs MCCFR 0.21 (4000 it) -- ~140x
worse; Goofspiel(5) (236k) MCCFR 0.93 after 4000 it (base 1.42) -- barely moves. Naive external-sampling
MCCFR is too variance-dominated for near-Nash in feasible wall-clock. RESEARCH_LOG 20260606T160000Z.
=> BOTH cheap scale levers have now failed their cheap gates (abstraction = incoherent; naive MCCFR =
too slow). STRATEGIC RECKONING (owner): the general METHOD is fully built + de-risked (a general, sound,
single-consumer-GPU depth-limited self-play method, validated on poker + non-poker, soundness at >=3
levels). SCALING it past ~few-k infosets is the hard part and the cheap wins are exhausted. Remaining
paths, all substantial + uncertain: (B') DREAM / variance-reduced MCCFR (control-variate baselines,
order-of-magnitude speedup but a real build); (D) DEEP CFR -- neural function approximation as IMPLICIT
abstraction over sampled traversals (modern scalable CFR, eps-Nash, no lossy bucketing; repo has a
full-deck Deep CFR to generalize); (C) ACCEPT the ceiling and write up the complete de-risked
single-consumer-GPU contribution + the soundness/abstraction/sampling characterization at the
Leduc/Goofspiel-class scale it reaches exactly (strong methods + reproducibility contribution; forfeits
the HUNL/Slumbot scaling claim). NEXT = owner decision among B'/D/C.
