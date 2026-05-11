# Repository Guidelines

## Project Structure & Module Organization
- `poker_ai/deep_cfr/` is the active full-deck Deep CFR stack. Key files: `deep_cfr.py` (reference loop), `networks.py` (advantage/policy heads), `buffer.py`, and `cuda/` (Numba kernels + `GPUDeepCFRTrainer`).
- `poker_ai/research/` contains the autoresearch workflow primitives, local evaluation helpers, fixed-state resolver benchmarks, promotion blockers, and Slumbot smoke parsers.
- `poker_ai/games/full_deck/` is the canonical 52-card state and feature encoder used for parity checks.
- `scripts/` contains runnable entrypoints: `run_gpu_deep_cfr.py`, `train_slumbot_2p.py`, `play_slumbot.py`, `solver.py`, and diagnostics.
- `test/unit/` holds fast regression tests such as `test_network_mask.py`, `test_legal_mask_parity.py`, and `test_slumbot_mapping.py`.
- `models/` and `research/` are local artifact directories for checkpoints/LUT data; keep generated binaries out of git. Legacy short-deck/tabular code remains for reference only.

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- CLI help: `poker_ai --help`
- Fast Deep CFR (CLI): `poker_ai train-fast-deep-cfr --help`
- GPU Deep CFR (script): `python scripts/run_gpu_deep_cfr.py --n-iterations 50 --n-traversals 400 --save-path ./models`
- Slumbot training: `python scripts/train_slumbot_2p.py --n-iterations 1000 --n-traversals 10000 --hidden-dim 512 --n-layers 4`
- Slumbot play/eval: `python scripts/play_slumbot.py --model models/slumbot_2p_iter1000.pt --hands 300 --greedy`
- Autoresearch status: `python scripts/poker_autoresearch.py status`
- Autoresearch comparison gate: `python scripts/poker_autoresearch.py gate eval-incumbent-self-compare`
- Autoresearch head-to-head gate: `python scripts/poker_autoresearch.py gate eval-head-to-head-self-compare`
- Resolver benchmark gate: `python scripts/poker_autoresearch.py gate eval-resolver-fixed-states`
- Resolver benchmark CLI: `python scripts/poker_resolver_benchmark.py --checkpoint models/candidate.pt --solver-iterations 25`
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
- Local random-opponent gates are mechanical health checks only. Do not mark a checkpoint as promotable without incumbent or Slumbot confidence evidence.

## Commit & Pull Request Guidelines
- Use concise imperative commit subjects and focused diffs.
- In PRs, include: what changed, why, exact test commands run, and benchmark deltas for trainer/kernel changes.
- Call out environment flags when relevant (for example `POKER_AI_COMPILE_VALUE_NET=1`, `LUT_DIR`, `TESTING_SUITE`).
