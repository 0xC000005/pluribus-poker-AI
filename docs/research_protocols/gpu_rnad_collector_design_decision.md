# Design-decision record: GPU R-NaD collector (cuda/ substrate)

**Date:** 2026-06-03 · **Status:** implemented, parity-test-gated · **Class:** compute-substrate port (diagnostic), NOT a method/eval/promotion change.

## What

`poker_ai/rnad/cuda_collector.py` — `CUDANativeRNaDCollector`: a GPU-resident R-NaD self-play
trajectory collector over the existing `cuda/` `GameBatch` kernels. It ports the numba-CPU
`CompiledNativeRNaDCollector` (`poker_ai/research/native_rnad.py`) so R-NaD training budget can
scale past the single-core CPU substrate (which plateaus ~137K steps/s; the GPU reaches
~1.1M steps/s at B=4096 and ~2.5M at B=32768 on an RTX 3070 Ti).

Opt-in via `run_compiled_native_rnad_learner(..., substrate="cuda")`. **`substrate="cpu"` is the
default and the CPU path is structurally unchanged.**

## Why this needs no methodology-review bundle

Per the governance rule, bundles are required for *method, evaluation-protocol,
checkpoint-promotion, and persistent-knob* changes. This change is none of those:

- **Same algorithm** (R-NaD); only the trajectory-collection substrate differs (numba-CPU → GPU
  kernels). Directly analogous to how `GPUDeepCFRTrainer` is a faster substrate for Deep CFR.
- **No evaluation-protocol / promotion / Slumbot** touch. The collector writes nothing; opponents
  are training self-play only; `initial_chips=1000` is the training stack (not the 20000 Slumbot stack).
- **No persistent knob**: `substrate` defaults to `"cpu"` and preserves existing behavior exactly.
- **No protected surface modified**: only new files added (`cuda_collector.py`,
  `test/unit/test_cuda_rnad_collector.py`) plus the opt-in kwarg in `native_rnad.py`. The three
  protected parity suites (`test_fast_vs_slow.py`, `test_gpu_optimizations.py`,
  `test_legal_mask_parity.py`) and the three state files (`state.py`, `fast_state.py`,
  `cuda/game_state.py`) are untouched; player ordering is unchanged (reuses HU `[1,0]`).

## Parity / correctness

Reuses the already-parity-tested kernels unchanged (`get_features`, `get_legal_mask`,
`apply_action`, `compute_winners`, `init_games`). Only net-new device code is a trivial
`current_player_kernel` (exposes the acting seat). Actions are sampled in torch (inverse-CDF, the
`GPUTreeCollector` convention) and the full behavior-policy vector is stored — the R-NaD loss
consumes `policy[T,B,A]`, so no per-action log-prob kernel is needed.

Deck RNG (GPU xorshift128+ vs CPU numpy) differs by construction, so GPU↔CPU equivalence is
**distributional, not bitwise** (same policy as `GPUTreeCollector`). `test_cuda_rnad_collector.py`
(CUDA-guarded, 6 tests) asserts: structural invariants, seeded reproducibility, GPU-vs-CPU
distributional equivalence (acting-seat mean 0.441 vs 0.443, per-action freqs within 0.05, reward
magnitude within 0.1), first-step feature invariants, end-to-end RNaDSolver training, and the
population/opponent path. All pass; `illegal_records == 0`, `needs_python_showdown == 0`, rewards
exactly zero-sum.

## Verdict

Sound, parity-test-gated, governance-clean. Use the GPU substrate to scale R-NaD budget for the
"anchor is budget-limited" follow-up (see RESEARCH_LOG 20260603T133428Z); promotion of any
resulting checkpoint still requires the standard falsification ladder + (once activated) the
exact-NashConv gate.
