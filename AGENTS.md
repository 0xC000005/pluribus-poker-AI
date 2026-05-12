# Repository Guidelines

## Project Structure & Module Organization
- `poker_ai/deep_cfr/` is the active full-deck Deep CFR stack. Key files: `deep_cfr.py` (reference loop), `networks.py` (advantage/policy heads), `buffer.py`, and `cuda/` (Numba kernels + `GPUDeepCFRTrainer`).
- `poker_ai/research/` contains the autoresearch workflow primitives, local evaluation helpers, fixed-state resolver benchmarks, promotion blockers, and Slumbot smoke parsers.
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
- Queue falsification ladder: `python scripts/poker_autoresearch.py enqueue-falsification --candidate models/candidate.pt --mechanism "search-distilled policy targets reduce Slumbot transfer loss"`
- Register research knob: `python scripts/poker_autoresearch.py add-knob --name search_target_mix --default 0.0 --failure-class search_quality --mechanism "Test whether search-distilled targets reduce live transfer loss." --rationale "One variable isolates the target mechanism." --removal-criterion "Retire if Slumbot transfer remains negative after confirmation."`
- Queue GPU candidate training: `python scripts/poker_autoresearch.py enqueue-train --n-iterations 50 --n-traversals 4000 --prefix candidate_gpu --save-every 25 --auto-compare`
- Resolver benchmark gate: `python scripts/poker_autoresearch.py gate eval-resolver-fixed-states`
- Resolver benchmark CLI: `python scripts/poker_resolver_benchmark.py --checkpoint models/candidate.pt --solver-iterations 25 --solver-backend auto`
- Queue candidate comparison: `python scripts/poker_autoresearch.py enqueue-compare --candidate models/candidate.pt --head-to-head`
- Queue candidate resolver check: `python scripts/poker_autoresearch.py enqueue-resolver --model models/candidate.pt`
- Unit checks: `pytest -q test/unit/test_network_mask.py test/unit/test_slumbot_mapping.py test/unit/test_legal_mask_parity.py`

## Coding Style & Naming Conventions
- Python with 4-space indentation, type hints, and PEP 8 names (`snake_case` functions/modules, `PascalCase` classes).
- Keep the 9-action contract and feature/legal-mask behavior aligned across CPU, fast, CUDA, and Slumbot integration code.
- Avoid ad hoc poker strategy rules; prefer learned policies, regret matching, and principled search.

## Testing Guidelines
- Place isolated logic tests in `test/unit/`; broader flows in `test/functional/`.
- For any action-space, feature, or mapping change, add/update parity tests and include regression coverage.
- For performance changes, report `iters/hour`, `samples/sec`, and `train sec/iter` with command/config used.
- For longer GPU runs, use `--save-every` plus `--auto-compare` so autoresearch evaluates intermediate checkpoints instead of only the final model.
- Use `--compare-strategy-source policy-head` only when both candidate and incumbent checkpoints include trained `policy_head` weights; legacy checkpoints must use regret matching.
- For method, evaluation-protocol, checkpoint-promotion, or persistent-knob changes, enqueue and complete a methodology review. The review must include independent-verifier findings, related work with source URLs, and a benchmark-hacking audit.
- Treat evaluation harnesses, Slumbot adapters, solver benchmarks, promotion logic, parsers, seed lists, and parity tests as protected surfaces. Changes to them require a completed review and objective-drift audit.
- Use sub-agents as a review team when available: verifier for `review.md`, literature scout for `related_work.md`, benchmark auditor for `benchmark_audit.md`, and research lead for `decision.json`.
- Before spending Slumbot confidence hands on a candidate, queue a falsification ladder. It runs objective-drift audit, duplicate-swapped incumbent comparison requiring positive lower95, and fixed-state resolver diagnostics.
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
