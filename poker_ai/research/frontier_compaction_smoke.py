"""Research smoke for active-frontier compaction primitives."""

from __future__ import annotations

import numpy as np
from numba import cuda
from statistics import fmean
import time

import torch


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return resolved


def compact_indices_prefix_sum(live_mask: torch.Tensor) -> torch.Tensor:
    """Return stable live indices using prefix-sum/scatter compaction."""

    if live_mask.ndim != 1:
        raise ValueError("live_mask must be one-dimensional")
    live_bool = live_mask.to(dtype=torch.bool)
    live_count = int(live_bool.sum().item())
    indices = torch.empty(live_count, dtype=torch.long, device=live_bool.device)
    if live_count == 0:
        return indices

    live_i64 = live_bool.to(dtype=torch.long)
    positions = torch.cumsum(live_i64, dim=0) - 1
    source_indices = torch.arange(live_bool.numel(), device=live_bool.device)
    indices[positions[live_bool]] = source_indices[live_bool]
    return indices


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(device: torch.device, fn) -> float:
    _sync_if_cuda(device)
    started = time.perf_counter()
    fn()
    _sync_if_cuda(device)
    return (time.perf_counter() - started) * 1000.0


def _process_rows(rows: torch.Tensor, work_repeats: int) -> torch.Tensor:
    values = rows
    for _ in range(max(1, int(work_repeats))):
        values = values * 1.0001 + 0.0003
    return values.sum(dim=1)


@cuda.jit
def _gather_stage_kernel(stages, frontier_indices, out_stages, n_frontier):
    gid = cuda.grid(1)
    if gid >= n_frontier:
        return
    slot = frontier_indices[gid]
    out_stages[gid] = stages[slot]


def run_frontier_compaction_smoke(
    *,
    n_rows: int = 1_000_000,
    feature_dim: int = 64,
    live_density: float = 0.13,
    work_repeats: int = 2,
    repeats: int = 5,
    seed: int = 20260515,
    device: str = "auto",
    min_slot_reduction_fraction: float = 0.5,
) -> dict:
    """Run a synthetic frontier compaction smoke on CPU or CUDA."""

    if not 0.0 < live_density <= 1.0:
        raise ValueError("live_density must be in (0, 1]")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    torch_device = resolve_torch_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    live_mask_cpu = torch.rand(n_rows, generator=generator) < float(live_density)
    if not bool(live_mask_cpu.any()):
        live_mask_cpu[0] = True
    live_mask = live_mask_cpu.to(torch_device)
    payload = torch.arange(
        n_rows * feature_dim,
        dtype=torch.float32,
        device=torch_device,
    ).reshape(n_rows, feature_dim)

    live_count = int(live_mask.sum().item())
    slot_reduction = 1.0 - (live_count / float(n_rows))
    indices = compact_indices_prefix_sum(live_mask)
    compact_payload = payload.index_select(0, indices)
    compact_values = _process_rows(compact_payload, work_repeats)
    full_values = _process_rows(payload, work_repeats)[live_mask]
    correct = bool(torch.allclose(compact_values, full_values))
    stable_order = bool(torch.equal(indices, torch.nonzero(live_mask).flatten()))

    # Warm up kernels and allocator paths outside the measured repeats.
    _process_rows(payload, work_repeats)[live_mask]
    _process_rows(payload.index_select(0, indices), work_repeats)
    _sync_if_cuda(torch_device)

    full_ms: list[float] = []
    compact_ms: list[float] = []
    index_ms: list[float] = []
    for _ in range(repeats):
        full_ms.append(
            _time_ms(
                torch_device,
                lambda: _process_rows(payload, work_repeats)[live_mask],
            )
        )
        index_ms.append(
            _time_ms(torch_device, lambda: compact_indices_prefix_sum(live_mask))
        )
        compact_ms.append(
            _time_ms(
                torch_device,
                lambda: _process_rows(
                    payload.index_select(0, indices),
                    work_repeats,
                ),
            )
        )

    mean_full_ms = fmean(full_ms)
    mean_compact_ms = fmean(compact_ms)
    mean_index_ms = fmean(index_ms)
    mean_compact_total_ms = mean_index_ms + mean_compact_ms
    speedup_vs_full = (
        mean_full_ms / mean_compact_total_ms
        if mean_compact_total_ms > 0.0
        else 0.0
    )
    failures: list[str] = []
    if not correct:
        failures.append("compacted row processing does not match full-frontier live rows")
    if not stable_order:
        failures.append("prefix-sum/scatter compaction order differs from torch.nonzero")
    if slot_reduction < min_slot_reduction_fraction:
        failures.append(
            "slot reduction is below threshold "
            f"({slot_reduction:.6f} < {min_slot_reduction_fraction:.6f})"
        )

    return {
        "mode": "frontier_compaction_smoke",
        "device": str(torch_device),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch_device)
            if torch_device.type == "cuda"
            else None
        ),
        "n_rows": int(n_rows),
        "feature_dim": int(feature_dim),
        "live_density_target": float(live_density),
        "live_count": int(live_count),
        "observed_live_density": round(live_count / float(n_rows), 6),
        "slot_reduction_fraction": round(slot_reduction, 6),
        "work_repeats": int(work_repeats),
        "repeats": int(repeats),
        "mean_full_ms": round(mean_full_ms, 6),
        "mean_index_ms": round(mean_index_ms, 6),
        "mean_compact_ms": round(mean_compact_ms, 6),
        "mean_compact_total_ms": round(mean_compact_total_ms, 6),
        "speedup_vs_full": round(speedup_vs_full, 6),
        "correct": correct,
        "stable_order": stable_order,
        "promotion": False,
        "passed": not failures,
        "failures": failures,
        "next_action": (
            "port compaction into the CUDA traversal frontier while carrying "
            "path/action keys for determinism"
            if not failures
            else "fix compaction correctness before touching traversal kernels"
        ),
    }


def run_frontier_index_interop_smoke(
    *,
    n_rows: int = 1_000_000,
    live_density: float = 0.13,
    repeats: int = 5,
    seed: int = 20260515,
    device: str = "auto",
    min_slot_reduction_fraction: float = 0.5,
) -> dict:
    """Smoke-test PyTorch prefix-sum indices feeding a Numba mapped kernel."""

    torch_device = resolve_torch_device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("frontier index interop smoke requires CUDA")
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available")
    if not 0.0 < live_density <= 1.0:
        raise ValueError("live_density must be in (0, 1]")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    rng = np.random.default_rng(seed)
    active_mask = rng.random(n_rows) < live_density
    if not bool(active_mask.any()):
        active_mask[0] = True
    inactive_mask = ~active_mask
    waiting_mask = inactive_mask & (rng.random(n_rows) < 0.5)

    stages = np.full(n_rows, 4, dtype=np.int8)
    stages[active_mask] = rng.integers(0, 4, size=int(active_mask.sum()), dtype=np.int8)
    stages[waiting_mask] = 1
    is_traverser_node = np.zeros(n_rows, dtype=np.int8)
    n_children_expected = np.zeros(n_rows, dtype=np.int32)
    is_traverser_node[waiting_mask] = 1
    n_children_expected[waiting_mask] = 2

    d_stages = cuda.to_device(stages)
    d_is_traverser_node = cuda.to_device(is_traverser_node)
    d_n_children_expected = cuda.to_device(n_children_expected)

    stage_t = torch.as_tensor(d_stages, device=torch_device)
    trav_t = torch.as_tensor(d_is_traverser_node, device=torch_device)
    expected_t = torch.as_tensor(d_n_children_expected, device=torch_device)
    live_mask_t = (stage_t < 4) & ~((trav_t == 1) & (expected_t > 0))

    indices = compact_indices_prefix_sum(live_mask_t)
    d_indices = cuda.as_cuda_array(indices.detach())
    d_out = cuda.device_array(indices.numel(), dtype=np.int8)
    threads = 256
    blocks = (max(1, indices.numel()) + threads - 1) // threads

    _gather_stage_kernel[blocks, threads](d_stages, d_indices, d_out, indices.numel())
    cuda.synchronize()
    gathered = d_out.copy_to_host()
    expected_indices = np.nonzero(active_mask)[0]
    observed_indices = indices.detach().cpu().numpy()
    correct_indices = bool(np.array_equal(observed_indices, expected_indices))
    correct_gather = bool(np.array_equal(gathered, stages[expected_indices]))
    slot_reduction = 1.0 - (int(active_mask.sum()) / float(n_rows))

    index_ms: list[float] = []
    gather_ms: list[float] = []
    for _ in range(repeats):
        index_ms.append(
            _time_ms(torch_device, lambda: compact_indices_prefix_sum(live_mask_t))
        )
        cuda.synchronize()
        started = time.perf_counter()
        _gather_stage_kernel[blocks, threads](d_stages, d_indices, d_out, indices.numel())
        cuda.synchronize()
        gather_ms.append((time.perf_counter() - started) * 1000.0)

    failures: list[str] = []
    if not correct_indices:
        failures.append("frontier indices do not match CPU active-mask reference")
    if not correct_gather:
        failures.append("Numba mapped gather does not match source stages")
    if slot_reduction < min_slot_reduction_fraction:
        failures.append(
            "slot reduction is below threshold "
            f"({slot_reduction:.6f} < {min_slot_reduction_fraction:.6f})"
        )

    return {
        "mode": "frontier_index_interop_smoke",
        "device": str(torch_device),
        "cuda_device_name": torch.cuda.get_device_name(torch_device),
        "n_rows": int(n_rows),
        "live_density_target": float(live_density),
        "frontier_count": int(active_mask.sum()),
        "observed_live_density": round(int(active_mask.sum()) / float(n_rows), 6),
        "slot_reduction_fraction": round(slot_reduction, 6),
        "repeats": int(repeats),
        "mean_index_ms": round(fmean(index_ms), 6),
        "mean_numba_gather_ms": round(fmean(gather_ms), 6),
        "correct_indices": correct_indices,
        "correct_gather": correct_gather,
        "promotion": False,
        "passed": not failures,
        "failures": failures,
        "next_action": (
            "prototype mapped feature/mask/classify/fork/apply kernels over "
            "frontier_indices while keeping tree slots stable"
            if not failures
            else "fix frontier-index interop before trainer integration"
        ),
    }
