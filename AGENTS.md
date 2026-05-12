# Repository Guidelines

## Project Structure & Module Organization
- `poker_ai/deep_cfr/` is the active full-deck Deep CFR stack. Key files: `deep_cfr.py` (reference loop), `networks.py` (advantage/policy heads), `policy_targets.py` (optional search-distilled policy targets), `buffer.py`, and `cuda/` (Numba kernels + `GPUDeepCFRTrainer`).
- `poker_ai/research/` contains the autoresearch workflow primitives, local evaluation helpers, search-target builders, policy-head calibration utilities, fixed-state resolver benchmarks, promotion blockers, and Slumbot smoke parsers.
- `poker_ai/games/full_deck/` is the canonical 52-card state and feature encoder used for parity checks.
- `scripts/` contains runnable entrypoints: `run_gpu_deep_cfr.py`, `train_slumbot_2p.py`, `play_slumbot.py`, `solver.py`, and diagnostics. Live turn/river solving uses learned range-pruning before CFR.
- `test/unit/` holds fast regression tests such as `test_network_mask.py`, `test_legal_mask_parity.py`, and `test_slumbot_mapping.py`.
- `models/` and `research/` are local artifact directories for checkpoints/LUT data; keep generated binaries out of git. Legacy short-deck/tabular code remains for reference only.

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- CLI help: `poker_ai --help`
- Fast Deep CFR (CLI): `poker_ai train-fast-deep-cfr --help`
- GPU Deep CFR (script): `python scripts/run_gpu_deep_cfr.py --n-iterations 50 --n-traversals 400 --save-path ./models`
- Slumbot training: `python scripts/train_slumbot_2p.py --n-iterations 1000 --n-traversals 10000 --hidden-dim 512 --n-layers 4`
- Slumbot play/eval: `python scripts/play_slumbot.py --model models/slumbot_2p_iter1000.pt --hands 300 --greedy --solver-backend auto`
- Policy-head Slumbot diagnostic: `python scripts/play_slumbot.py --model models/<policy_head_checkpoint>.pt --hands 300 --greedy --strategy-source policy-head`
- Autoresearch status: `python scripts/poker_autoresearch.py status`
- Autoresearch comparison gate: `python scripts/poker_autoresearch.py gate eval-incumbent-self-compare`
- Autoresearch head-to-head gate: `python scripts/poker_autoresearch.py gate eval-head-to-head-self-compare`
- Queue methodology review: `python scripts/poker_autoresearch.py enqueue-review --subject "New search objective" --trigger method_change --claim "The objective should improve Slumbot transfer."`
- Validate methodology review: `python scripts/poker_methodology_review.py --review-dir autoresearch-session/poker_reviews/<review_id> --require-complete`
- Objective-drift audit: `python scripts/poker_objective_audit.py --base-ref HEAD`
- Objective-drift audit with review: `python scripts/poker_autoresearch.py objective-audit --changed-path scripts/play_slumbot.py --review-dir autoresearch-session/poker_reviews/<review_id>`
- Build sampled resolver search targets: `python scripts/build_search_targets.py --output autoresearch-session/search_targets/sampled_turn_river_train.npz --sampled-cases 64 --seed 20260512 --solver-iterations 25 --solver-backend auto`
- Queue falsification ladder: `python scripts/poker_autoresearch.py enqueue-falsification --candidate models/candidate.pt --mechanism "search-distilled policy targets reduce Slumbot transfer loss"`
- Register research knob: `python scripts/poker_autoresearch.py add-knob --name search_target_weight --default 0.05 --failure-class search_quality --mechanism "Test whether resolver-distilled policy targets reduce blueprint-vs-resolver drift and improve transfer evidence." --rationale "Single auxiliary-loss weight isolates the reviewed search-consistency mechanism." --removal-criterion "Retire if held-out resolver drift or falsification-ladder evidence fails to improve against the no-target control."`
- Queue GPU candidate training: `python scripts/poker_autoresearch.py enqueue-train --n-iterations 50 --n-traversals 4000 --prefix candidate_gpu --save-every 25 --auto-compare`
- Queue search-consistency training: `python scripts/poker_autoresearch.py enqueue-train --n-iterations 50 --n-traversals 4000 --search-targets autoresearch-session/search_targets/sampled_turn_river_train.npz --search-target-weight 0.05 --prefix search_consistency --save-every 25 --auto-compare`
- Evaluate search-target fit: `python scripts/eval_search_targets.py --checkpoint models/candidate.pt --targets autoresearch-session/search_targets/sampled_turn_river_holdout.npz --strategy-source policy-head`
- Build policy-head calibration targets: `python scripts/build_policy_calibration_targets.py --checkpoint models/control.pt --output autoresearch-session/policy_calibration/calib_targets.npz --n-targets 4096 --strategy-source regret`
- Build turn/river hand-sweep calibration targets: `python scripts/build_policy_calibration_targets.py --checkpoint models/control.pt --output autoresearch-session/policy_calibration/hand_sweep.npz --hand-sweep --sampled-cases 16 --hands-per-case 128 --strategy-source regret --target-temperature 2.0`
- Train policy-head calibration: `python scripts/train_policy_head_calibration.py --checkpoint models/control.pt --targets autoresearch-session/policy_calibration/calib_targets.npz --output autoresearch-session/policy_calibration/calibrated.pt --n-steps 600`
- Diagnose policy-teacher collapse: `python scripts/diagnose_policy_teacher.py --checkpoint models/control.pt --sampled-cases 16 --hands-per-case 128 --strategy-source regret --output autoresearch-session/policy_calibration/teacher_diag.json`
- Evaluate SD-CFR checkpoint mixture: `python scripts/eval_sd_cfr_mixture.py --candidate-glob 'models/run/*iter_*.pt' --baseline-checkpoint models/run/final.pt --n-games 300 --seeds 20260512,20260513,20260514 --output autoresearch-session/sd_cfr_mixture/mixture_h2h.json`
- Diagnose range likelihood: `python scripts/diagnose_range_tracker.py --checkpoint autoresearch-session/policy_calibration/calibrated.pt --cases-json autoresearch-session/search_targets/reachable_policyhead_holdout_16x5.cases.json --strategy-source policy-head`
- Resolver benchmark gate: `python scripts/poker_autoresearch.py gate eval-resolver-fixed-states`
- Resolver benchmark CLI: `python scripts/poker_resolver_benchmark.py --checkpoint models/candidate.pt --solver-iterations 25 --solver-backend auto`
- Queue candidate comparison: `python scripts/poker_autoresearch.py enqueue-compare --candidate models/candidate.pt --head-to-head`
- Queue candidate resolver check: `python scripts/poker_autoresearch.py enqueue-resolver --model models/candidate.pt`
- Unit checks: `pytest -q test/unit/test_network_mask.py test/unit/test_slumbot_mapping.py test/unit/test_legal_mask_parity.py test/unit/test_policy_targets.py test/unit/test_sd_cfr_mixture.py`

## Coding Style & Naming Conventions
- Python with 4-space indentation, type hints, and PEP 8 names (`snake_case` functions/modules, `PascalCase` classes).
- Keep the 9-action contract and feature/legal-mask behavior aligned across CPU, fast, CUDA, and Slumbot integration code.
- New Deep CFR checkpoints use the 12 public betting-history features by default and save `uses_betting_history=true`; checkpoints without this metadata load with the legacy masked-history contract.
- Avoid ad hoc poker strategy rules; prefer learned policies, regret matching, and principled search.

## Testing Guidelines
- Place isolated logic tests in `test/unit/`; broader flows in `test/functional/`.
- For any action-space, feature, or mapping change, add/update parity tests and include regression coverage.
- For performance changes, report `iters/hour`, `samples/sec`, and `train sec/iter` with command/config used.
- For longer GPU runs, use `--save-every` plus `--auto-compare` so autoresearch evaluates intermediate checkpoints instead of only the final model.
- Use `--compare-strategy-source policy-head` only when both candidate and incumbent checkpoints include trained `policy_head` weights; legacy checkpoints must use regret matching.
- Search-consistency targets must be generated by solver/search code and consumed through `--search-targets`; do not replace them with hard-coded no-all-in or street-specific policy rules.
- Policy-head calibration targets are behavior-cloning diagnostics for range-likelihood calibration. Target temperature may soften legal teacher distributions, but it must not suppress legal all-in actions or encode street-specific rules. Treat improved target loss or range dispersion as mechanism evidence only; do not promote without head-to-head, resolver, and Slumbot confidence evidence.
- Deep CFR can collect legal-mask average-strategy targets during traversal with `--average-strategy-weight > 0` and save an explicit `average-policy` network, but this path is experimental and off by default. First 5-iteration A/Bs regressed H2H, so do not scale it without stronger regret-network or averaging evidence.
- Diagnose teacher collapse before scaling calibration: a dominant top action or all-in majority is a teacher-quality blocker, not a signal to sweep more calibration hyperparameters.
- Prefer SD-CFR-style checkpoint-mixture diagnostics before adding another average-policy trainer; checkpoint selection must be fixed by iteration/glob, not by post-hoc benchmark wins.
- Do not zero betting-history inputs unless a learned sequence encoder is actually wired through training, evaluation, and Slumbot play. A 5-iteration local H2H gate strongly favored restored history, but this remains pre-Slumbot-confirmation evidence.
- For method, evaluation-protocol, checkpoint-promotion, or persistent-knob changes, enqueue and complete a methodology review. The review must include independent-verifier findings, related work with source URLs, and a benchmark-hacking audit.
- Treat evaluation harnesses, Slumbot adapters, solver benchmarks, promotion logic, parsers, seed lists, and parity tests as protected surfaces. Changes to them require a completed review and objective-drift audit.
- Use sub-agents as a review team when available: verifier for `review.md`, literature scout for `related_work.md`, benchmark auditor for `benchmark_audit.md`, and research lead for `decision.json`.
- Before spending Slumbot confidence hands on a candidate, queue a falsification ladder. It runs objective-drift audit, duplicate-swapped incumbent comparison requiring positive lower95, and fixed-state resolver diagnostics.
- One-off and auto-queued candidate comparisons must include `--require-positive-lower95`; finite but negative lower bounds are diagnostic failures, not passed gates.
- Add persistent knobs only through `add-knob`; each needs one mechanism, one default, one failure class, and a removal criterion. Do not use broad hyperparameter sweeps as research progress.
- Keep `--traversal-slots-per-traversal` at the fast default unless explicitly running a high-fidelity pool experiment; the 2,500-slot mode was much slower and not better in the first local gate.
- Treat `--solver-backend torch-cuda` as experimental; benchmark it against `cpu` before using it in live Slumbot gates.
- Local random-opponent gates are mechanical health checks only. Do not mark a checkpoint as promotable without incumbent or Slumbot confidence evidence.

## Commit & Pull Request Guidelines
- Use concise imperative commit subjects and focused diffs. Batch commits by research objective; do not commit every small file edit or commit from continuous autoresearch mode.
- Commit bodies for research changes should mention objective, files changed, tests/gates run, key result, and review or related-work status.
- When a task changes documentation or research methodology, append a concise `RESEARCH_LOG.md` entry or close the workflow cycle through the autoresearch log path; do not paste raw command JSON into the log.
- In PRs, include: what changed, why, exact test commands run, and benchmark deltas for trainer/kernel changes.
- Call out environment flags when relevant (for example `POKER_AI_COMPILE_VALUE_NET=1`, `LUT_DIR`, `TESTING_SUITE`).
