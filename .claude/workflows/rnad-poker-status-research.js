export const meta = {
  name: 'rnad-poker-status-research',
  description: 'Online related-work + novelty research and repo-artifact inventory for the tabula-rasa R-NaD poker direction',
  phases: [
    { title: 'Research', detail: 'parallel multi-angle online related-work + repo inventory' },
    { title: 'Verify', detail: 'adversarially check the riskiest novelty claims' },
    { title: 'Synthesize', detail: 'related-work landscape + novelty verdict + recommended directions' },
  ],
}

const CTX = `PROJECT CONTEXT (today is 2026-05-30; your knowledge + live web search both valid):
Goal: a poker AI that learns by TABULA-RASA NEURAL SELF-PLAY. From the rules only, a random-init
neural net self-improves to an approximate Nash equilibrium using R-NaD / DeepNash-style
"Regularized Nash Dynamics" (reward-transform regularization toward a reference policy + V-trace +
periodic reference reset). HARD CONSTRAINTS: (1) NO play-time search/solver/CFR at inference -- the
trained net plays directly (net-centric, "AlphaZero-of-imperfect-information" but WITHOUT search);
(2) must train on ONE consumer GPU: a single RTX 3070 Ti (8GB VRAM, 32GB RAM). Target: beat an
AlphaHoldem-style RLCard end-to-end-RL baseline AND held-out Slumbot at heads-up no-limit Texas
hold'em (HUNL). Slumbot is HELD-OUT evaluation only (never training/targets/selection).
The user's NOVELTY claim is the RESULT itself: "making tabula-rasa neural self-play WIN on one
consumer GPU" (DeepNash needed TPU/large-scale compute). Current state: the R-NaD MECHANISM passed a
cheap gate -- exact Nash on Kuhn poker, monotone NashConv descent on Leduc poker (vs vanilla
self-play which cycled). Next candidate: port R-NaD to full HUNL on a custom CUDA game-sim substrate.`

const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    angle: { type: 'string' },
    summary: { type: 'string', description: '4-8 sentence synthesis of what the literature says on this angle' },
    key_papers: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { title: {type:'string'}, year: {type:'string'}, venue: {type:'string'}, url: {type:'string'}, relevance: {type:'string'} },
      required: ['title','relevance'] } },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { claim: {type:'string'}, evidence: {type:'string'}, compute_or_numbers: {type:'string'}, confidence: {type:'string'} },
      required: ['claim','evidence'] } },
    novelty_implications: { type: 'string', description: 'What this means for the novelty/feasibility of the single-GPU tabula-rasa claim' },
    risks_or_caveats: { type: 'string' },
    claims_to_verify: { type: 'array', items: { type: 'string' }, description: 'The 1-3 riskiest/most-load-bearing claims worth adversarial verification' },
  },
  required: ['angle','summary','findings','novelty_implications'],
}

const INVENTORY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    artifacts: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { path: {type:'string'}, what: {type:'string'}, status: {type:'string'} }, required: ['path','what'] } },
    gate_results: { type: 'string', description: 'Concrete numbers from the Kuhn/Leduc R-NaD gate result JSONs if present' },
    research_log_findings: { type: 'string', description: 'One-line each of the RESEARCH_LOG.md Findings (esp. 6-12)' },
    open_threads: { type: 'array', items: { type: 'string' } },
  },
  required: ['artifacts'],
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    claim: { type: 'string' },
    verdict: { type: 'string', description: 'supported | partially-supported | refuted | uncertain' },
    reasoning: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['claim','verdict','reasoning'],
}

const ANGLES = [
  { key: 'rnad_deepnash', q: `R-NaD / DeepNash (Perolat et al. 2022, "Mastering Stratego with model-free multiagent RL", Science) and the underlying "From Poincare recurrence to convergence in imperfect-information games" / Regularized Nash Dynamics. What EXACTLY is the algorithm (reward transform, NeuRD, V-trace, reference reset)? What COMPUTE did DeepNash use (TPU count, env steps, wall-clock)? Has anyone applied R-NaD to POKER / HUNL specifically? Any follow-up or open-source R-NaD scaling work 2023-2026? Any consumer-GPU or down-scaled reproductions?` },
  { key: 'alphaholdem_baseline', q: `AlphaHoldem (Zhao et al., AAAI 2022, "AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning"). Architecture, training method (end-to-end deep RL, pseudo-Siamese, K-best self-play), COMPUTE used (GPUs, days), and reported results vs Slumbot/DeepStack. Is there an open RLCard reimplementation? How strong is it and on what hardware was it trained?` },
  { key: 'limited_compute_methods', q: `State-of-the-art imperfect-information / HUNL poker methods 2019-2026 and their COMPUTE footprints: ReBeL (Brown 2020), Student of Games (DeepMind 2023), ESCHER, Deep CFR / DREAM / outcome-sampling MCCFR, Pluribus. Which need play-time search? Which are net-only? What is the smallest compute anyone has used to reach near-Slumbot-level HUNL? Any "single GPU" or "consumer hardware" equilibrium-learning poker results?` },
  { key: 'mmd_neurd_regularization', q: `Magnetic Mirror Descent (MMD, Sokota et al. 2023 "A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games"), NeuRD (Hennes et al. 2020), Friction-FoReL, QFR, and other last-iterate-convergent regularized self-play methods. How do they compare to R-NaD in convergence guarantees, simplicity, and compute? Could any be a SIMPLER single-GPU path to the same goal than full R-NaD? Last-iterate vs average-iterate convergence for neural function approximation.` },
  { key: 'tabula_rasa_no_search', q: `Can a NET-ONLY (no play-time search) policy actually reach strong HUNL play, or is search at inference effectively required for top performance? Evidence on net-only vs search-augmented gaps in large imperfect-info games. AlphaZero-style tabula-rasa lessons transferred to imperfect-information games. Is "no search at inference" a help or a handicap for the win-the-baselines goal?` },
  { key: 'novelty_publishability', q: `Assess NOVELTY/PUBLISHABILITY of the claim "tabula-rasa neural self-play (R-NaD-style) that beats strong HUNL baselines while trained on ONE consumer GPU." What is the closest prior art on COMPUTE-EFFICIENT equilibrium learning for poker? Has "single consumer GPU" poker-equilibrium been claimed before? What would reviewers consider the contribution vs incremental? What concrete results/ablations would make it a credible paper?` },
]

phase('Research')
const tasks = ANGLES.map(a => () => agent(
  `${CTX}\n\nRESEARCH ANGLE: ${a.key}\n${a.q}\n\nDo LIVE web research: use WebSearch and WebFetch (load them via ToolSearch first: query "select:WebSearch,WebFetch"). Also try the Hugging Face paper search MCP if available. Prefer primary sources (arXiv, proceedings, official repos). Cite URLs in evidence. Report concrete COMPUTE NUMBERS wherever possible (GPUs/TPUs, days, env-steps). Be precise and skeptical; separate established fact from speculation. Fill claims_to_verify with the 1-3 most load-bearing claims for the projects novelty/feasibility.`,
  { schema: RESEARCH_SCHEMA, phase: 'Research', label: 'research:' + a.key }
))
tasks.push(() => agent(
  `${CTX}\n\nTASK: Inventory the repo's R-NaD / equilibrium-learning research artifacts at /home/max/Documents/pluribus-poker-AI.\n` +
  `Use Read/Grep/Glob/Bash. Look at: scripts/run_rnad_smallgame_gate.py, scripts/run_rnad_tabular_gate.py, scripts/vendor/rnad.py, scripts/run_qfr_leduc_gate.py, scripts/run_exploitability_lb_gpu.py, scripts/run_exploitability_lb_batched.py; the autoresearch-session/ directory (esp. B_PROPOSAL.md, DECISION_MEMO.md, and any rnad/leduc/kuhn result JSONs); RESEARCH_LOG.md (extract the Findings list, especially Findings 6-12 -- the file is large, grep for "Finding" and "R-NaD"/"Leduc"); docs/research_protocols/poker_autoresearch_current_findings.md and poker_next_continuation_target.md. ` +
  `Report concrete gate numbers (Kuhn/Leduc NashConv), what each script does, the state of the NLHE CUDA substrate (poker_ai/deep_cfr/cuda/), and open threads. Be concrete with paths.`,
  { schema: INVENTORY_SCHEMA, phase: 'Research', label: 'inventory:repo' }
))

const all = await parallel(tasks)
const research = all.slice(0, ANGLES.length).filter(Boolean)
const inventory = all[ANGLES.length]
log(`Research done: ${research.length}/${ANGLES.length} angles + inventory=${inventory ? 'ok' : 'MISSING'}`)

phase('Verify')
const claims = []
for (const r of research) {
  const cs = r.claims_to_verify || []
  for (const c of cs) claims.push({ angle: r.angle, claim: c })
}
const topClaims = claims.slice(0, 8)
log(`Verifying ${topClaims.length} load-bearing claims`)
const verifyThunks = topClaims.map(c => () => agent(
  `${CTX}\n\nADVERSARIALLY VERIFY this claim from angle "${c.angle}":\n"${c.claim}"\n\n` +
  `Use WebSearch/WebFetch (ToolSearch "select:WebSearch,WebFetch"). Try to REFUTE it or find prior art that undercuts its novelty. Default to "uncertain" or "partially-supported" unless evidence is strong. Cite sources. Be a skeptic, especially on novelty/compute claims.`,
  { schema: VERDICT_SCHEMA, phase: 'Verify', label: 'verify:' + c.angle }
))
const verifyRaw = await parallel(verifyThunks)
const verifications = verifyRaw.filter(Boolean)

phase('Synthesize')
const memo = await agent(
  `${CTX}\n\nYou are the synthesis lead. Below are JSON research findings (multi-angle online related-work), a repo inventory, and adversarial verifications. Produce a tight MEMO (markdown) with these sections:\n` +
  `1. RELATED-WORK LANDSCAPE -- the 6-8 most relevant methods, each with one line + compute footprint + whether it uses play-time search.\n` +
  `2. NOVELTY VERDICT -- is "tabula-rasa neural self-play beating strong HUNL baselines on ONE consumer GPU" genuinely novel/publishable? Closest prior art? What's the real contribution vs what's incremental? Be honest and specific.\n` +
  `3. FEASIBILITY RISKS -- the top 3-5 concrete risks to the plan (e.g., net-only vs search gap, R-NaD compute at NLHE scale, 8GB VRAM limits, abstraction needs), each with severity.\n` +
  `4. RECOMMENDED DIRECTIONS -- 3-4 candidate next moves (e.g., proceed to NLHE R-NaD port; try simpler MMD/NeuRD variant first; add a smaller-scale milestone like Leduc-NLHE or limit-holdem; reduce-risk experiments), each with rationale + rough compute + what it would prove. Rank them.\n` +
  `Be concrete, cite specific papers/numbers from the inputs, and do not pad.\n\n` +
  `=== RESEARCH (JSON) ===\n${JSON.stringify(research)}\n\n=== INVENTORY (JSON) ===\n${JSON.stringify(inventory)}\n\n=== VERIFICATIONS (JSON) ===\n${JSON.stringify(verifications)}`,
  { phase: 'Synthesize', label: 'synthesis' }
)

return { research, inventory, verifications, memo }
