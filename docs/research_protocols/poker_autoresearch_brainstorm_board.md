# Poker Autoresearch Brainstorm Board

Status: active planning board after search-target distillation produced only
weak, non-promotable same-budget evidence.

## Online Research Transfer

- OpenAI agent guidance frames an agent as a system that independently executes
  a workflow with tools and guardrails; this supports a concrete goal contract,
  tool/workflow guardrails, and trace-style evaluation rather than a vague
  "continue research" objective.
- Anthropic's agent-design guidance recommends simple, composable workflows
  first and added autonomy only when flexibility is needed. For this repo, that
  means explicit methodology gates, not free-running experiment expansion.
- Voyager's automatic curriculum, skill library, and execution-feedback loop
  suggest a useful analogy: keep reusable research skills, but select next
  tasks by novelty and verified capability gains.
- AI Scientist and AI Scientist-v2 support idea-generation, experiment,
  write-up, and review loops, but their safety lessons argue for bounded
  time/compute, review artifacts, and guardrails before unattended execution.
- Recent coding-agent rule evidence suggests negative constraints are safer
  than many positive directives. The poker workflow should therefore forbid
  benchmark hacking, unmatched controls, and scalar knob tuning after failed
  mechanisms instead of prescribing every research move.

## Current Causal Model

Exact CUDA CFR has real decision signal on tested turn/river roots. The current
failure is interface transfer: supervised target fit and simple policy-head
distillation do not reliably become stronger self-play decisions. The next
mechanism must put the learned object inside the decision loop.

## Candidate Mechanisms

1. **Belief/range-conditioned search-state updater**
   - Learned object: compact public-belief/search-state embedding plus update.
   - Consumed by: low-budget resolver iterations.
   - Control: same-budget vanilla CFR or no-update resolver.
   - First gate: root-disjoint CFR10/CFR25 teacher-distance and top-action
     agreement per millisecond.
   - Retire if: no root-decision improvement at matched latency.

2. **Range-conditioned action-value correction head**
   - Learned object: action-value residual conditioned on public state, reach
     ranges, legal mask, and low-budget solver outputs.
   - Consumed by: root action selection or resolver warm-start.
   - Control: low-budget resolver plus incumbent policy without correction.
   - First gate: root-disjoint action-value ranking and duplicate-swapped
     same-budget self-play.
   - Retire if: supervised residual fit improves but root action quality does
     not.

3. **Iterative policy-state denoising**
   - Learned object: denoiser over noisy/low-budget root policies or regrets,
     analogous to diffusion-style iterative refinement.
   - Consumed by: repeated update steps inside resolving.
   - Control: spending the same wall-clock budget on extra exact CFR
     iterations.
   - First gate: improvement per millisecond versus uniform CFR budget.
   - Retire if: extra exact CFR still dominates at matched latency.

## Recommended Next Sprint

Start with candidate 1. It directly attacks the current failure surface and is
least likely to become another action-label imitation problem. The mechanism
brief must name:

- decision object: root action distribution, counterfactual action values, and
  regret/search-state update;
- where consumed: resolver loop, not post-hoc policy deployment;
- matched control: same roots, same latency/budget, vanilla CFR;
- retirement criterion: no root-disjoint improvement over matched control;
- anti-benchmark rule: no Slumbot validation until internal root and self-play
  gates pass.
