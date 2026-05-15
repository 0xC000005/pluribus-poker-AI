"""GPU-accelerated Deep CFR trainer using Numba CUDA.

Replaces the CPU-bound traversal with GPU-parallel game simulation.

Architecture:
  1. Start N games on GPU in a large pre-allocated pool
  2. At each step: extract features (GPU) → NN forward pass (GPU, zero-copy) →
     regret matching (GPU kernel) → classify+sample (GPU kernel) →
     fork traverser nodes (CPU bookkeeping + GPU copy) → apply actions (GPU)
  3. Collect regrets from completed traverser nodes

Optimizations over the original version:
  - Zero-copy Numba↔PyTorch via __cuda_array_interface__ (no D→H→D transfers)
  - GPU regret matching kernel (eliminates 53% CPU bottleneck)
  - GPU action sampling with xoroshiro128p RNG
"""

from __future__ import annotations

import gc
import logging
import time
from typing import List

import numpy as np
import torch
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.deep_cfr import regret_match, train_value_network
from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.networks import ValueNetwork, PolicyNetwork
from poker_ai.deep_cfr.policy_targets import (
    PolicyTargetBuffer,
    PolicyReservoirBuffer,
    train_average_policy_network,
)

from poker_ai.deep_cfr.cuda.lookup_tables import get_gpu_tables, FLUSH_SIZE, UNSUITED_SIZE
from poker_ai.deep_cfr.cuda.game_state import (
    GameBatch, create_game_batch, _get_device_orders, init_games_kernel,
)
from poker_ai.deep_cfr.cuda.game_kernels import (
    apply_action_kernel,
    compute_winners_kernel,
    get_features_kernel,
    get_legal_mask_kernel,
)
from poker_ai.deep_cfr.cuda.action_kernels import (
    regret_match_kernel,
    sample_action_kernel,
    classify_and_sample_kernel,
    fork_kernel,
    copy_from_parent_kernel,
    propagate_kernel,
    collect_policy_targets_kernel,
    reset_traversal_state_kernel,
    count_nonterminal_kernel,
)

logger = logging.getLogger("poker_ai.deep_cfr.cuda.gpu_trainer")

_GPU_CACHE_FLOAT32_SAMPLE_BYTES = 4
_GPU_CACHE_COMPACT_SAMPLE_BYTES = 2
_GPU_CACHE_ITERATION_BYTES = 4
_GPU_CACHE_SAFETY_FRACTION = 0.75
_DEFAULT_TRAVERSAL_POOL_MAX_SLOTS = 1_000_000
_DEFAULT_TRAVERSAL_SLOTS_PER_TRAVERSAL = 500
_DEFAULT_POLICY_SLOTS_PER_TRAVERSAL = 64
_DEFAULT_NN_FORWARD_CHUNK = 500_000
_MIN_NN_FORWARD_CHUNK = 8_192
_NN_FORWARD_TIGHT_MEMORY_BYTES = 2 * 1024**3
_NN_FORWARD_MAX_WORK_BYTES = 384 * 1024**2
_NN_FORWARD_MIN_WORK_BYTES = 64 * 1024**2


def _traversal_batch_size(
    *,
    n_traversals: int,
    pool_max_slots: int = _DEFAULT_TRAVERSAL_POOL_MAX_SLOTS,
    slots_per_traversal: int = _DEFAULT_TRAVERSAL_SLOTS_PER_TRAVERSAL,
) -> int:
    """Choose traversal chunk size from a fixed slot budget.

    Increasing ``slots_per_traversal`` reduces pool exhaustion by shrinking
    traversal chunks under the same workspace cap. The default preserves the
    faster historical 2,000-traversal chunks.
    """
    n_traversals = max(1, int(n_traversals))
    pool_max_slots = max(1, int(pool_max_slots))
    slots_per_traversal = max(1, int(slots_per_traversal))
    return max(1, min(n_traversals, pool_max_slots // slots_per_traversal))


def _nn_forward_chunk_size(
    value_net: torch.nn.Module,
    device: torch.device,
    *,
    free_bytes: int | None = None,
    max_chunk: int = _DEFAULT_NN_FORWARD_CHUNK,
) -> int:
    """Choose a safe NN inference chunk for large CUDA traversal pools.

    The Numba traversal workspace can leave little free GPU memory on 8 GB
    cards. A fixed 500k forward batch is then too large for 4x512 MLP
    activations, even under ``no_grad``. Keep the historical chunk when memory
    is plentiful, but shrink it under tight CUDA memory.
    """
    max_chunk = max(1, int(max_chunk))
    if device.type != "cuda":
        return max_chunk

    if free_bytes is None:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:
            return max(_MIN_NN_FORWARD_CHUNK, min(max_chunk, 100_000))

    free_bytes = max(0, int(free_bytes))
    if free_bytes >= _NN_FORWARD_TIGHT_MEMORY_BYTES:
        return max_chunk

    hidden_dim = max(1, int(getattr(value_net, "hidden_dim", 256)))
    work_bytes = max(
        _NN_FORWARD_MIN_WORK_BYTES,
        min(_NN_FORWARD_MAX_WORK_BYTES, int(free_bytes * 0.35)),
    )
    rows = work_bytes // (hidden_dim * 4)
    return max(_MIN_NN_FORWARD_CHUNK, min(max_chunk, int(rows)))


def _summarize_traversal_pool_stats(
    records: List[dict[str, float | int]],
) -> dict[str, float | int]:
    if not records:
        return {
            "traversal_chunks": 0,
            "traversal_overflow_chunks": 0,
            "traversal_overflow_chunk_fraction": 0.0,
            "traversal_mean_pool_demand_ratio": 0.0,
            "traversal_max_pool_demand_ratio": 0.0,
            "traversal_mean_slots_per_traversal": 0.0,
            "traversal_max_slots_per_traversal": 0.0,
            "traversal_regret_sample_fill_ratio": 0.0,
        }

    chunks = len(records)
    demand_ratios = [
        float(record["requested_slots"]) / max(1.0, float(record["pool_max_slots"]))
        for record in records
    ]
    slots_per_traversal = [
        float(record["requested_slots"]) / max(1.0, float(record["n_traversals"]))
        for record in records
    ]
    total_capacity = sum(max(1, int(record["pool_max_slots"])) for record in records)
    total_regret_samples = sum(int(record.get("regret_samples", 0)) for record in records)
    overflow_chunks = sum(1 for ratio in demand_ratios if ratio > 1.0)

    return {
        "traversal_chunks": chunks,
        "traversal_overflow_chunks": overflow_chunks,
        "traversal_overflow_chunk_fraction": round(overflow_chunks / chunks, 6),
        "traversal_mean_pool_demand_ratio": round(sum(demand_ratios) / chunks, 6),
        "traversal_max_pool_demand_ratio": round(max(demand_ratios), 6),
        "traversal_mean_slots_per_traversal": round(
            sum(slots_per_traversal) / chunks,
            6,
        ),
        "traversal_max_slots_per_traversal": round(max(slots_per_traversal), 6),
        "traversal_regret_sample_fill_ratio": round(
            total_regret_samples / max(1, total_capacity),
            6,
        ),
    }


def _build_iteration_profile(
    *,
    iteration: int,
    n_players: int,
    n_traversals: int,
    traversal_stats: list[dict[str, float | int]],
    traverse_seconds: float,
    train_seconds: float,
    train_batch_size: int,
    train_steps: int,
) -> dict[str, float | int]:
    requested_traversals = int(n_players) * int(n_traversals)
    regret_samples = sum(int(record.get("regret_samples", 0)) for record in traversal_stats)
    policy_samples = sum(int(record.get("policy_samples", 0)) for record in traversal_stats)
    train_sample_budget = int(train_batch_size) * int(train_steps)
    iteration_seconds = float(traverse_seconds) + float(train_seconds)
    return {
        "iteration": int(iteration),
        "requested_traversals": requested_traversals,
        "regret_samples": regret_samples,
        "policy_samples": policy_samples,
        "traverse_seconds": round(float(traverse_seconds), 6),
        "train_seconds": round(float(train_seconds), 6),
        "iteration_seconds": round(iteration_seconds, 6),
        "traversals_per_second": round(
            requested_traversals / max(float(traverse_seconds), 1e-9),
            6,
        ),
        "regret_samples_per_second": round(
            regret_samples / max(float(traverse_seconds), 1e-9),
            6,
        ),
        "train_batch_size": int(train_batch_size),
        "train_steps": int(train_steps),
        "train_sample_budget": train_sample_budget,
        "train_samples_per_second": round(
            train_sample_budget / max(float(train_seconds), 1e-9),
            6,
        ) if train_sample_budget > 0 else 0.0,
        **_summarize_traversal_pool_stats(traversal_stats),
    }


def _gpu_cache_nbytes(
    n_samples: int,
    *,
    sample_dtype_bytes: int = _GPU_CACHE_FLOAT32_SAMPLE_BYTES,
    iteration_dtype_bytes: int = _GPU_CACHE_ITERATION_BYTES,
) -> int:
    sample_bytes = (N_FEATURES + N_ACTIONS) * int(sample_dtype_bytes)
    return int(n_samples) * (sample_bytes + int(iteration_dtype_bytes))


def _gpu_cache_budget_allows(
    *,
    n_samples: int,
    free_bytes: int,
    safety_fraction: float = _GPU_CACHE_SAFETY_FRACTION,
    sample_dtype_bytes: int = _GPU_CACHE_FLOAT32_SAMPLE_BYTES,
    iteration_dtype_bytes: int = _GPU_CACHE_ITERATION_BYTES,
) -> bool:
    return _gpu_cache_nbytes(
        n_samples,
        sample_dtype_bytes=sample_dtype_bytes,
        iteration_dtype_bytes=iteration_dtype_bytes,
    ) <= int(float(free_bytes) * safety_fraction)


def _remap_legacy_value_state_dict(state: dict) -> dict:
    """Map older sequential ValueNetwork checkpoints onto trunk/adv_head."""
    if not any(key.startswith("net.") for key in state):
        return state
    linear_indices = sorted({int(key.split(".")[1]) for key in state if key.startswith("net.")})
    *trunk_indices, head_index = linear_indices
    remapped = {}
    for key, value in state.items():
        if not key.startswith("net."):
            remapped[key] = value
            continue
        _, index, param = key.split(".", 2)
        index = int(index)
        if index == head_index:
            remapped[f"adv_head.{param}"] = value
        elif index in trunk_indices:
            remapped[f"trunk.{index}.{param}"] = value
    return remapped


def _torch_dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class _MultiBufferView:
    """Lightweight wrapper that samples from multiple ReservoirBuffers.

    Avoids the 10+GB allocation of combining buffers into one array.
    Implements the same sample_batch() interface as ReservoirBuffer.
    """

    def __init__(
        self,
        buffers: list,
        *,
        gpu_cache_sample_dtype: torch.dtype = torch.float16,
        gpu_cache_safety_fraction: float = _GPU_CACHE_SAFETY_FRACTION,
    ):
        self.buffers = [b for b in buffers if b.size > 0]
        self.size = sum(b.size for b in self.buffers)
        self._gpu_cache_sample_dtype = gpu_cache_sample_dtype
        self._gpu_cache_safety_fraction = gpu_cache_safety_fraction
        self._gpu_cache_device: torch.device | None = None
        self._gpu_cache_sig: tuple[tuple[int, int], ...] | None = None
        self._gpu_features: torch.Tensor | None = None
        self._gpu_iterations: torch.Tensor | None = None
        self._gpu_advantages: torch.Tensor | None = None
        self._gpu_cache_disabled = False
        self._gpu_cache_budget_rejected_sig: tuple | None = None
        self._expected_sample_budget: int | None = None

    def __len__(self):
        return self.size

    def _signature(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(buf.size), int(getattr(buf, "_n_seen", buf.size)))
            for buf in self.buffers
        )

    def set_expected_sample_budget(self, n_samples: int):
        self._expected_sample_budget = max(0, int(n_samples))

    def _try_build_gpu_cache(self, device: torch.device) -> bool:
        if self._gpu_cache_disabled:
            return False
        sig = self._signature()
        if (
            self._gpu_cache_device == device
            and self._gpu_cache_sig == sig
            and self._gpu_features is not None
            and self._gpu_iterations is not None
            and self._gpu_advantages is not None
        ):
            return True

        try:
            sample_dtype = (
                self._gpu_cache_sample_dtype
                if device.type == "cuda"
                else torch.float32
            )
            sample_dtype_bytes = _torch_dtype_nbytes(sample_dtype)
            if (
                self._expected_sample_budget is not None
                and self._expected_sample_budget < self.size
            ):
                return False
            budget_rejected_sig = (
                device.type,
                device.index,
                sig,
                sample_dtype_bytes,
            )
            if device.type == "cuda":
                if self._gpu_cache_budget_rejected_sig == budget_rejected_sig:
                    return False
                free_bytes, _ = torch.cuda.mem_get_info(device)
                if not _gpu_cache_budget_allows(
                    n_samples=self.size,
                    free_bytes=free_bytes,
                    safety_fraction=self._gpu_cache_safety_fraction,
                    sample_dtype_bytes=sample_dtype_bytes,
                ):
                    logger.warning(
                        "Skipping GPU replay cache: %d samples need %.2f GiB "
                        "with %s feature/advantage tensors and "
                        "current free memory is %.2f GiB.",
                        self.size,
                        _gpu_cache_nbytes(
                            self.size,
                            sample_dtype_bytes=sample_dtype_bytes,
                        ) / (1024**3),
                        str(sample_dtype).replace("torch.", ""),
                        free_bytes / (1024**3),
                    )
                    self._gpu_cache_budget_rejected_sig = budget_rejected_sig
                    return False

            feat = torch.empty(
                (self.size, N_FEATURES), dtype=sample_dtype, device=device
            )
            iters = torch.empty((self.size,), dtype=torch.float32, device=device)
            advs = torch.empty(
                (self.size, N_ACTIONS), dtype=sample_dtype, device=device
            )

            cursor = 0
            for buf in self.buffers:
                n = int(buf.size)
                if n <= 0:
                    continue
                end = cursor + n
                feat[cursor:end].copy_(torch.from_numpy(buf.features[:n]))
                iters[cursor:end].copy_(
                    torch.from_numpy(buf.iterations[:n]).to(torch.float32)
                )
                advs[cursor:end].copy_(torch.from_numpy(buf.advantages[:n]))
                cursor = end

            self._gpu_features = feat
            self._gpu_iterations = iters
            self._gpu_advantages = advs
            self._gpu_cache_device = device
            self._gpu_cache_sig = sig
            self._gpu_cache_budget_rejected_sig = None
            return True
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning(
                    "Disabling GPU replay cache due to CUDA OOM; falling back to host sampling."
                )
                self._gpu_cache_disabled = True
                self._gpu_features = None
                self._gpu_iterations = None
                self._gpu_advantages = None
                torch.cuda.empty_cache()
                return False
            raise

    def release_gpu_cache(self):
        """Release cached replay tensors before returning to Numba traversal."""
        cache_device = self._gpu_cache_device
        self._gpu_features = None
        self._gpu_iterations = None
        self._gpu_advantages = None
        self._gpu_cache_device = None
        self._gpu_cache_sig = None
        self._gpu_cache_budget_rejected_sig = None
        if cache_device is not None and cache_device.type == "cuda":
            torch.cuda.empty_cache()

    def sample_batch(self, batch_size, device=None):
        """Sample a batch and write directly into preallocated torch tensors."""
        if self.size == 0:
            raise ValueError("Empty buffer")
        n_total = min(batch_size, self.size)

        if device is not None and device.type == "cuda":
            if self._try_build_gpu_cache(device):
                idx = torch.randint(0, self.size, (n_total,), device=device)
                feat = self._gpu_features.index_select(0, idx)
                iters = self._gpu_iterations.index_select(0, idx)
                advs = self._gpu_advantages.index_select(0, idx)
                return feat, iters, advs

        sizes = np.array([b.size for b in self.buffers], dtype=np.int64)
        probs = sizes / sizes.sum()
        counts = np.random.multinomial(n_total, probs)

        out_device = device if device is not None else torch.device("cpu")
        feat = torch.empty((n_total, N_FEATURES), dtype=torch.float32, device=out_device)
        iters = torch.empty((n_total,), dtype=torch.float32, device=out_device)
        advs = torch.empty((n_total, N_ACTIONS), dtype=torch.float32, device=out_device)

        cursor = 0
        for buf, n in zip(self.buffers, counts):
            if n <= 0:
                continue
            idx = np.random.randint(0, buf.size, size=n)

            feat_cpu = torch.from_numpy(buf.features[idx])
            iter_cpu = torch.from_numpy(buf.iterations[idx].astype(np.float32))
            adv_cpu = torch.from_numpy(buf.advantages[idx])

            end = cursor + n
            if device is None:
                feat[cursor:end] = feat_cpu
                iters[cursor:end] = iter_cpu
                advs[cursor:end] = adv_cpu
            else:
                feat[cursor:end] = feat_cpu.to(device, non_blocking=True)
                iters[cursor:end] = iter_cpu.to(device, non_blocking=True)
                advs[cursor:end] = adv_cpu.to(device, non_blocking=True)
            cursor = end

        return feat, iters, advs


def _get_orders(n_players):
    """Get device order arrays for the given player count."""
    orders = _get_device_orders()
    if n_players in orders:
        return orders[n_players]
    d_preflop = cuda.to_device(
        np.array(list(range(2, n_players)) + [0, 1], dtype=np.int8))
    d_postflop = cuda.to_device(np.arange(n_players, dtype=np.int8))
    return d_preflop, d_postflop


class _GPUTraverseWorkspace:
    """Reusable GPU buffers for traversal to avoid per-call allocation overhead."""

    def __init__(
        self,
        max_traversals: int,
        n_players: int,
        initial_chips: int,
        slots_per_traversal: int = _DEFAULT_TRAVERSAL_SLOTS_PER_TRAVERSAL,
        policy_slots_per_traversal: int = _DEFAULT_POLICY_SLOTS_PER_TRAVERSAL,
    ):
        self.max_traversals = max_traversals
        self.slots_per_traversal = max(1, int(slots_per_traversal))
        self.max_pool = max_traversals * self.slots_per_traversal
        self.policy_slots_per_traversal = max(1, int(policy_slots_per_traversal))
        self.policy_capacity = min(
            self.max_pool,
            max_traversals * self.policy_slots_per_traversal,
        )
        self.n_players = n_players
        self.initial_chips = initial_chips

        # Shared tables/orders.
        tables = get_gpu_tables()
        (
            self.d_flush_keys,
            self.d_flush_vals,
            self.d_unsuited_keys,
            self.d_unsuited_vals,
            self.d_card_lookup,
            _,
        ) = tables
        self.d_preflop, self.d_postflop = _get_orders(n_players)
        self.d_raise_fractions = cuda.to_device(
            np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
        )

        # Game state + random seeds for reinit.
        self.batch = GameBatch(self.max_pool, n_players)
        self.d_seeds = cuda.device_array((max_traversals, 2), dtype=np.int64)

        # Inference / action tensors.
        self.d_features = cuda.device_array((self.max_pool, N_FEATURES), dtype=np.float32)
        self.d_masks = cuda.device_array((self.max_pool, N_ACTIONS), dtype=np.float32)
        self.d_strategies = cuda.device_array((self.max_pool, N_ACTIONS), dtype=np.float32)
        self.d_actions_gpu = cuda.device_array(self.max_pool, dtype=np.int8)
        self.d_is_traverser = cuda.device_array(self.max_pool, dtype=np.int8)

        self.rng_states = create_xoroshiro128p_states(
            self.max_pool, seed=np.random.randint(1, 2**31)
        )

        # Traversal bookkeeping.
        self.d_parent_idx = cuda.device_array(self.max_pool, dtype=np.int32)
        self.d_parent_action = cuda.device_array(self.max_pool, dtype=np.int8)
        self.d_is_traverser_node = cuda.device_array(self.max_pool, dtype=np.int8)
        self.d_traverser_features = cuda.device_array((self.max_pool, N_FEATURES), dtype=np.float32)
        self.d_slot_strategy = cuda.device_array((self.max_pool, N_ACTIONS), dtype=np.float32)
        self.d_child_values = cuda.device_array((self.max_pool, N_ACTIONS), dtype=np.float32)
        self.d_n_children_done = cuda.device_array(self.max_pool, dtype=np.int32)
        self.d_n_children_expected = cuda.device_array(self.max_pool, dtype=np.int32)
        self.d_propagated = cuda.device_array(self.max_pool, dtype=np.int8)

        self.d_next_free = cuda.device_array(1, dtype=np.int32)
        self.d_n_collected = cuda.device_array(1, dtype=np.int32)
        self.d_active_count = cuda.device_array(1, dtype=np.int32)

        # Collected samples.
        self.d_collected_features = cuda.device_array((self.max_pool, N_FEATURES), dtype=np.float32)
        self.d_collected_regrets = cuda.device_array((self.max_pool, N_ACTIONS), dtype=np.float32)
        self.d_policy_features = cuda.device_array(
            (self.policy_capacity, N_FEATURES), dtype=np.float32
        )
        self.d_policy_masks = cuda.device_array(
            (self.policy_capacity, N_ACTIONS), dtype=np.float32
        )
        self.d_policy_targets = cuda.device_array(
            (self.policy_capacity, N_ACTIONS), dtype=np.float32
        )
        self.d_n_policy_collected = cuda.device_array(1, dtype=np.int32)

    def reset(self, n_traversals: int):
        """Reset game + traversal state for a new traversal batch."""
        if n_traversals > self.max_traversals:
            raise ValueError(
                f"n_traversals={n_traversals} exceeds workspace max={self.max_traversals}"
            )

        # Reinitialize first n_traversals game slots.
        rng = np.random.default_rng()
        seeds = rng.integers(1, 2**62, size=(n_traversals, 2), dtype=np.int64)
        self.d_seeds[:n_traversals].copy_to_device(seeds)

        threads = 256
        blocks = (n_traversals + threads - 1) // threads
        init_games_kernel[blocks, threads](
            self.batch.chips, self.batch.bets, self.batch.active, self.batch.hole_cards,
            self.batch.community, self.batch.deck, self.batch.deck_cursor,
            self.batch.stage, self.batch.n_raises, self.batch.player_i_index,
            self.batch.n_actions, self.batch.pot_total,
            self.batch.n_players_started_round, self.batch.history, self.batch.payout,
            self.batch.is_done,
            self.d_seeds, self.n_players, n_traversals,
            self.d_preflop, self.initial_chips,
        )

        # Reset traversal bookkeeping for the whole reusable pool so stale
        # metadata from a larger previous chunk cannot be reached after forks.
        max_pool = self.max_pool
        reset_blocks = (max_pool + threads - 1) // threads
        reset_traversal_state_kernel[reset_blocks, threads](
            self.d_parent_idx,
            self.d_parent_action,
            self.d_is_traverser_node,
            self.d_n_children_done,
            self.d_n_children_expected,
            self.d_propagated,
            max_pool,
        )

        self.d_next_free.copy_to_device(np.array([n_traversals], dtype=np.int32))
        self.d_n_collected.copy_to_device(np.array([0], dtype=np.int32))
        self.d_n_policy_collected.copy_to_device(np.array([0], dtype=np.int32))


# ---------------------------------------------------------------------------
# GPU evaluation: play N games with trained agent vs random opponents
# ---------------------------------------------------------------------------

def gpu_evaluate_vs_random(
    value_net: ValueNetwork,
    device: torch.device,
    n_games: int = 1000,
    n_players: int = 6,
    initial_chips: int = 10000,
) -> float:
    """Evaluate agent (player 0) vs random opponents using GPU simulation.

    Uses zero-copy interop and GPU action sampling — no Python per-game loop.
    """
    tables = get_gpu_tables()
    d_flush_keys, d_flush_vals, d_unsuited_keys, d_unsuited_vals, d_card_lookup, _ = tables

    batch = create_game_batch(n_games, n_players, initial_chips=initial_chips)

    d_features = cuda.device_array((n_games, N_FEATURES), dtype=np.float32)
    d_masks = cuda.device_array((n_games, N_ACTIONS), dtype=np.float32)
    d_strategies = cuda.device_array((n_games, N_ACTIONS), dtype=np.float32)
    d_actions = cuda.device_array(n_games, dtype=np.int8)

    threads = 256
    blocks = (n_games + threads - 1) // threads

    d_preflop, d_postflop = _get_orders(n_players)
    d_raise_fractions = cuda.to_device(
        np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32))

    # RNG states for action sampling.
    rng_states = create_xoroshiro128p_states(n_games, seed=np.random.randint(1, 2**31))

    value_net.eval()

    for step in range(200):
        # GPU: get features + masks.
        get_features_kernel[blocks, threads](
            batch.chips, batch.bets, batch.active,
            batch.hole_cards, batch.community,
            batch.stage, batch.n_raises, batch.player_i_index,
            batch.pot_total, batch.history,
            n_players, d_preflop, d_postflop,
            d_features, n_games, initial_chips,
        )
        get_legal_mask_kernel[blocks, threads](
            batch.active, batch.chips, batch.bets, batch.n_raises,
            batch.stage, batch.pot_total,
            batch.player_i_index, n_players,
            d_preflop, d_postflop,
            d_raise_fractions,
            d_masks, n_games,
        )
        cuda.synchronize()

        # Check how many games are still active (need stages on CPU).
        stages = batch.stage.copy_to_host()
        n_active = int(np.sum(stages < 4))
        if n_active == 0:
            break

        # NN forward pass — zero-copy: Numba device array → PyTorch tensor.
        with torch.no_grad():
            feat_t = torch.as_tensor(d_features, device=device)
            adv_t = value_net(feat_t)
        torch.cuda.synchronize()

        # Zero-copy: PyTorch tensor → Numba device array.
        d_advantages = cuda.as_cuda_array(adv_t.detach())

        # GPU: regret matching → strategy for player 0 (agent).
        regret_match_kernel[blocks, threads](
            d_advantages, d_masks, d_strategies, n_games,
        )

        # For evaluation: player 0 uses NN strategy, others use uniform.
        # We use sample_action_kernel for all (uniform = regret_match on
        # zero advantages, which produces uniform over legal).
        # But we need different strategies for agent vs random.
        # Simpler: use classify_and_sample_kernel treating player 0 as
        # "not traverser" — all players sample from their strategy.
        # For random players, their strategy should be uniform.
        # Actually, let's just sample for player 0 with NN strategy,
        # and override with uniform for others.
        #
        # Cleanest approach: two-pass.
        # 1. Sample actions for ALL games using NN strategy (handles player 0).
        sample_action_kernel[blocks, threads](
            d_strategies, d_masks, batch.stage, rng_states,
            d_actions, n_games,
        )
        cuda.synchronize()

        # Override non-player-0 actions with uniform sampling on CPU.
        # This is fast because we only copy small arrays (stages, pii, actions).
        pii = batch.player_i_index.copy_to_host()
        actions_host = d_actions.copy_to_host()
        masks_host = d_masks.copy_to_host()

        for g in range(n_games):
            if stages[g] >= 4:
                continue
            # Determine current player.
            if n_players > 2:
                preflop_order = list(range(2, n_players)) + [0, 1]
            else:
                preflop_order = [0, 1]
            postflop_order = list(range(n_players))
            if stages[g] == 0:
                pi = preflop_order[pii[g] % n_players]
            else:
                pi = postflop_order[pii[g] % n_players]

            if pi != 0:
                # Random opponent: uniform over legal actions.
                mask = masks_host[g]
                legal = np.where(mask > 0)[0]
                if len(legal) > 0:
                    actions_host[g] = np.random.choice(legal)

        # Upload modified actions and apply.
        d_actions = cuda.to_device(actions_host)
        apply_action_kernel[blocks, threads](
            batch.chips, batch.bets, batch.active,
            batch.hole_cards, batch.community, batch.deck,
            batch.deck_cursor, batch.stage, batch.n_raises,
            batch.player_i_index, batch.n_actions, batch.pot_total,
            batch.history, batch.n_players_started_round,
            d_actions, n_games, n_players,
            d_preflop, d_postflop,
            d_raise_fractions,
        )
        cuda.synchronize()

    # Compute winners.
    compute_winners_kernel[blocks, threads](
        batch.chips, batch.bets, batch.active,
        batch.hole_cards, batch.community,
        batch.payout, batch.stage, n_games, n_players,
        d_card_lookup,
        d_flush_keys, d_flush_vals, FLUSH_SIZE,
        d_unsuited_keys, d_unsuited_vals, UNSUITED_SIZE,
        initial_chips,
    )
    cuda.synchronize()

    payouts = batch.payout.copy_to_host()
    return float(payouts[:, 0].mean())


# ---------------------------------------------------------------------------
# GPU Traversal for Deep CFR regret collection
# ---------------------------------------------------------------------------

def gpu_traverse_for_player(
    traverser: int,
    n_traversals: int,
    value_net: ValueNetwork,
    buffer: ReservoirBuffer,
    iteration: int,
    n_players: int,
    device: torch.device,
    initial_chips: int = 10000,
    workspace: _GPUTraverseWorkspace | None = None,
    policy_buffer: PolicyReservoirBuffer | None = None,
):
    """Run n_traversals game tree traversals on GPU for one player.

    Fully GPU-resident wavefront traversal — no per-slot Python loops.
    Fork bookkeeping and value propagation run as GPU kernels.
    CPU work per depth: kernel launches + 2 scalar reads.
    """
    if workspace is None:
        workspace = _GPUTraverseWorkspace(
            max_traversals=n_traversals,
            n_players=n_players,
            initial_chips=initial_chips,
        )
    workspace.reset(n_traversals)

    value_net.eval()

    max_pool = workspace.max_pool
    threads = 256
    blocks_pool = (max_pool + threads - 1) // threads
    zero_i32 = np.array([0], dtype=np.int32)

    # Local aliases for readability.
    batch = workspace.batch
    d_features = workspace.d_features
    d_masks = workspace.d_masks
    d_strategies = workspace.d_strategies
    d_actions_gpu = workspace.d_actions_gpu
    d_is_traverser = workspace.d_is_traverser
    d_parent_idx = workspace.d_parent_idx
    d_parent_action = workspace.d_parent_action
    d_is_traverser_node = workspace.d_is_traverser_node
    d_traverser_features = workspace.d_traverser_features
    d_slot_strategy = workspace.d_slot_strategy
    d_child_values = workspace.d_child_values
    d_n_children_done = workspace.d_n_children_done
    d_n_children_expected = workspace.d_n_children_expected
    d_propagated = workspace.d_propagated
    d_next_free = workspace.d_next_free
    d_n_collected = workspace.d_n_collected
    d_collected_features = workspace.d_collected_features
    d_collected_regrets = workspace.d_collected_regrets
    d_policy_features = workspace.d_policy_features
    d_policy_masks = workspace.d_policy_masks
    d_policy_targets = workspace.d_policy_targets
    d_n_policy_collected = workspace.d_n_policy_collected
    d_active_count = workspace.d_active_count
    d_preflop = workspace.d_preflop
    d_postflop = workspace.d_postflop
    d_raise_fractions = workspace.d_raise_fractions
    rng_states = workspace.rng_states
    d_flush_keys = workspace.d_flush_keys
    d_flush_vals = workspace.d_flush_vals
    d_unsuited_keys = workspace.d_unsuited_keys
    d_unsuited_vals = workspace.d_unsuited_vals
    d_card_lookup = workspace.d_card_lookup

    n_active = n_traversals

    for depth in range(100):
        # 1. GPU: extract features + legal masks.
        get_features_kernel[blocks_pool, threads](
            batch.chips, batch.bets, batch.active,
            batch.hole_cards, batch.community,
            batch.stage, batch.n_raises, batch.player_i_index,
            batch.pot_total, batch.history,
            n_players, d_preflop, d_postflop,
            d_features, n_active, initial_chips,
        )
        get_legal_mask_kernel[blocks_pool, threads](
            batch.active, batch.chips, batch.bets, batch.n_raises,
            batch.stage, batch.pot_total,
            batch.player_i_index, n_players,
            d_preflop, d_postflop,
            d_raise_fractions,
            d_masks, n_active,
        )
        cuda.synchronize()

        # 2. NN forward pass — zero-copy, chunked. Only process n_active slots.
        feat_t = torch.as_tensor(d_features, device=device)
        NN_CHUNK = _nn_forward_chunk_size(value_net, device)
        if n_active <= NN_CHUNK:
            with torch.no_grad():
                adv_t = value_net(feat_t[:n_active])
        else:
            chunks = []
            for s in range(0, n_active, NN_CHUNK):
                e = min(s + NN_CHUNK, n_active)
                with torch.no_grad():
                    chunks.append(value_net(feat_t[s:e]))
            adv_t = torch.cat(chunks, dim=0)
        torch.cuda.synchronize()

        # Write NN output into a pre-allocated GPU buffer for zero-copy kernel access.
        # adv_t is (n_active, 9) — contiguous on GPU. Kernels bound to n_active.
        d_advantages = cuda.as_cuda_array(adv_t.detach())

        # 3. GPU: regret matching + classify/sample.
        blocks_active = (n_active + threads - 1) // threads
        regret_match_kernel[blocks_active, threads](
            d_advantages, d_masks, d_strategies, n_active,
        )
        classify_and_sample_kernel[blocks_active, threads](
            d_strategies, d_masks, batch.stage, batch.player_i_index,
            n_players, traverser, d_preflop, d_postflop,
            rng_states, d_actions_gpu, d_is_traverser, n_active,
        )
        if policy_buffer is not None:
            collect_policy_targets_kernel[blocks_active, threads](
                d_features, d_masks, d_strategies,
                batch.stage, batch.player_i_index,
                n_players, traverser, d_preflop, d_postflop,
                d_policy_features, d_policy_masks, d_policy_targets,
                d_n_policy_collected,
                workspace.policy_capacity,
                n_active,
            )
        # 4. GPU: compute winners for terminals.
        compute_winners_kernel[blocks_active, threads](
            batch.chips, batch.bets, batch.active,
            batch.hole_cards, batch.community,
            batch.payout, batch.stage, n_active, n_players,
            d_card_lookup,
            d_flush_keys, d_flush_vals, FLUSH_SIZE,
            d_unsuited_keys, d_unsuited_vals, UNSUITED_SIZE,
            initial_chips,
        )

        # 5. GPU: propagate terminal values up tree, collect regret samples.
        propagate_kernel[blocks_active, threads](
            batch.stage, batch.payout, traverser,
            d_parent_idx, d_parent_action,
            d_is_traverser_node, d_traverser_features, d_slot_strategy,
            d_child_values, d_n_children_done, d_n_children_expected,
            d_propagated,
            d_collected_features, d_collected_regrets, d_n_collected,
            np.float32(initial_chips), n_active,
        )

        # 6. Track current frontier before fork.
        old_next_free = n_active

        # 7. GPU: fork traverser nodes — allocate children.
        fork_kernel[blocks_active, threads](
            d_is_traverser, batch.stage,
            d_features, d_strategies, d_masks,
            d_parent_idx, d_parent_action,
            d_is_traverser_node, d_traverser_features, d_slot_strategy,
            d_n_children_expected, d_child_values, d_n_children_done,
            d_next_free, max_pool,
            d_actions_gpu, rng_states, n_active,
        )
        cuda.synchronize()

        # 8. Read new_next_free (1 scalar D→H). Clamp to max_pool since
        #    atomic counter can overshoot when multiple threads try to allocate
        #    simultaneously and some fall back.
        new_next_free = min(int(d_next_free.copy_to_host()[0]), max_pool)

        # 9. GPU: copy game state from parents to newly allocated children.
        if new_next_free > old_next_free:
            n_new = new_next_free - old_next_free
            copy_blocks = (n_new + threads - 1) // threads
            copy_from_parent_kernel[copy_blocks, threads](
                batch.chips, batch.bets, batch.active, batch.hole_cards,
                batch.community, batch.deck, batch.deck_cursor, batch.stage,
                batch.n_raises, batch.player_i_index, batch.n_actions,
                batch.pot_total, batch.n_players_started_round, batch.history,
                batch.payout, batch.is_done,
                d_parent_idx, old_next_free, new_next_free, n_players,
            )
            n_active = new_next_free

        # 10. GPU: apply actions for ALL slots (opponent sampled + fork children).
        #     Terminal/traverser slots have action=-1 and are skipped by the kernel.
        blocks_active = (n_active + threads - 1) // threads
        apply_action_kernel[blocks_active, threads](
            batch.chips, batch.bets, batch.active,
            batch.hole_cards, batch.community, batch.deck,
            batch.deck_cursor, batch.stage, batch.n_raises,
            batch.player_i_index, batch.n_actions, batch.pot_total,
            batch.history, batch.n_players_started_round,
            d_actions_gpu, n_active, n_players,
            d_preflop, d_postflop,
            d_raise_fractions,
        )
        # 11. Count non-terminal slots with a single scalar copy.
        d_active_count.copy_to_device(zero_i32)
        count_nonterminal_kernel[blocks_active, threads](
            batch.stage, d_active_count, n_active
        )
        active_count = int(d_active_count.copy_to_host()[0])
        if active_count == 0:
            break

    # Final propagation pass: handle terminals from the last depth.
    blocks_final = (n_active + threads - 1) // threads
    compute_winners_kernel[blocks_final, threads](
        batch.chips, batch.bets, batch.active,
        batch.hole_cards, batch.community,
        batch.payout, batch.stage, n_active, n_players,
        d_card_lookup,
        d_flush_keys, d_flush_vals, FLUSH_SIZE,
        d_unsuited_keys, d_unsuited_vals, UNSUITED_SIZE,
        initial_chips,
    )
    propagate_kernel[blocks_final, threads](
        batch.stage, batch.payout, traverser,
        d_parent_idx, d_parent_action,
        d_is_traverser_node, d_traverser_features, d_slot_strategy,
        d_child_values, d_n_children_done, d_n_children_expected,
        d_propagated,
        d_collected_features, d_collected_regrets, d_n_collected,
        np.float32(initial_chips), n_active,
    )
    # Copy collected samples to CPU and add to buffer.
    n_collected = min(int(d_n_collected.copy_to_host()[0]), max_pool)
    if n_collected > 0:
        h_features = d_collected_features[:n_collected].copy_to_host()
        h_regrets = d_collected_regrets[:n_collected].copy_to_host()
        buffer.add_batch(h_features, iteration, h_regrets, n_collected)
    if policy_buffer is not None:
        n_policy = min(
            int(d_n_policy_collected.copy_to_host()[0]),
            workspace.policy_capacity,
        )
        if n_policy > 0:
            h_policy_features = d_policy_features[:n_policy].copy_to_host()
            h_policy_masks = d_policy_masks[:n_policy].copy_to_host()
            h_policy_targets = d_policy_targets[:n_policy].copy_to_host()
            policy_buffer.add_batch(
                h_policy_features,
                h_policy_masks,
                h_policy_targets,
                float(max(iteration, 1)),
                n_policy,
            )

    final_next_free = int(d_next_free.copy_to_host()[0])
    n_policy_seen = 0
    if n_traversals >= 1000:
        print(f"    [pool] used {final_next_free}/{max_pool} slots "
              f"({final_next_free/max_pool*100:.0f}%), "
              f"{final_next_free/n_traversals:.0f} per trav, "
              f"{n_collected} regret samples")
        if policy_buffer is not None:
            n_policy_seen = int(d_n_policy_collected.copy_to_host()[0])
            print(f"    [policy] collected {min(n_policy_seen, workspace.policy_capacity)}"
                  f"/{workspace.policy_capacity} targets")
    elif policy_buffer is not None:
        n_policy_seen = int(d_n_policy_collected.copy_to_host()[0])

    return {
        "requested_slots": final_next_free,
        "pool_max_slots": max_pool,
        "n_traversals": n_traversals,
        "regret_samples": n_collected,
        "policy_samples": min(n_policy_seen, workspace.policy_capacity),
        "policy_capacity": workspace.policy_capacity,
    }


# ---------------------------------------------------------------------------
# GPUDeepCFRTrainer
# ---------------------------------------------------------------------------

class GPUDeepCFRTrainer:
    """Deep CFR trainer using GPU-accelerated game simulation."""

    def __init__(
        self,
        n_players: int = 6,
        buffer_capacity: int = 2_000_000,
        hidden_dim: int = 256,
        n_layers: int = 2,
        batch_size: int = 2048,
        lr: float = 0.001,
        n_training_steps: int = 1000,
        n_traversals: int = 500,
        device: torch.device | None = None,
        initial_chips: int = 10000,
        auto_scale_train_schedule: bool = True,
        train_batch_target: int = 8192,
        release_workspace_before_training: bool = True,
        traversal_pool_max_slots: int = _DEFAULT_TRAVERSAL_POOL_MAX_SLOTS,
        traversal_slots_per_traversal: int = _DEFAULT_TRAVERSAL_SLOTS_PER_TRAVERSAL,
        policy_slots_per_traversal: int = _DEFAULT_POLICY_SLOTS_PER_TRAVERSAL,
        policy_target_buffer: PolicyTargetBuffer | None = None,
        policy_target_weight: float = 0.0,
        policy_target_batch_size: int | None = None,
        average_strategy_memory_capacity: int | None = None,
        average_strategy_weight: float = 0.0,
        average_strategy_batch_size: int | None = None,
        use_betting_history: bool = True,
    ):
        self.n_players = n_players
        self.initial_chips = initial_chips
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lr = lr
        self.n_training_steps = n_training_steps
        self.n_traversals = n_traversals
        self.auto_scale_train_schedule = auto_scale_train_schedule
        self.train_batch_target = train_batch_target
        self.release_workspace_before_training = release_workspace_before_training
        self.traversal_pool_max_slots = max(1, int(traversal_pool_max_slots))
        self.traversal_slots_per_traversal = max(1, int(traversal_slots_per_traversal))
        self.policy_slots_per_traversal = max(1, int(policy_slots_per_traversal))
        self.policy_target_buffer = policy_target_buffer
        self.policy_target_weight = float(policy_target_weight)
        self.policy_target_batch_size = policy_target_batch_size
        self.average_strategy_weight = float(average_strategy_weight)
        self.average_strategy_batch_size = average_strategy_batch_size
        self.use_betting_history = bool(use_betting_history)
        self.strategy_buffer = PolicyReservoirBuffer(
            int(average_strategy_memory_capacity or buffer_capacity)
        )

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.buffers: List[ReservoirBuffer] = [
            ReservoirBuffer(buffer_capacity) for _ in range(n_players)
        ]
        self.value_net = ValueNetwork(
            N_FEATURES,
            hidden_dim,
            N_ACTIONS,
            n_layers=n_layers,
            use_betting_history=self.use_betting_history,
        ).to(self.device)
        self.average_policy_net = PolicyNetwork(
            N_FEATURES,
            hidden_dim,
            N_ACTIONS,
            n_layers=n_layers,
            use_betting_history=self.use_betting_history,
        ).to(self.device)
        self.has_average_policy_net = False
        self.iteration = 0
        self._workspace: _GPUTraverseWorkspace | None = None
        self._schedule_logged = False
        self.last_traversal_pool_stats: List[dict[str, float | int]] = []
        self.traversal_pool_stats_history: List[dict[str, float | int]] = []
        self.last_profile: dict[str, float | int] = {}
        self.profile_history: list[dict[str, float | int]] = []

    def run_iteration(self):
        """Run one CFR iteration with GPU traversal."""
        import time as _time
        self.iteration += 1
        self.value_net.eval()

        # Batch traversals to keep GPU pool memory manageable while giving
        # each traversal enough fork slots to avoid demoting traverser nodes.
        trav_batch = _traversal_batch_size(
            n_traversals=self.n_traversals,
            pool_max_slots=self.traversal_pool_max_slots,
            slots_per_traversal=self.traversal_slots_per_traversal,
        )
        if (
            self._workspace is None
            or self._workspace.max_traversals < trav_batch
            or self._workspace.n_players != self.n_players
            or self._workspace.initial_chips != self.initial_chips
            or self._workspace.slots_per_traversal != self.traversal_slots_per_traversal
            or self._workspace.policy_slots_per_traversal != self.policy_slots_per_traversal
        ):
            self._workspace = _GPUTraverseWorkspace(
                max_traversals=trav_batch,
                n_players=self.n_players,
                initial_chips=self.initial_chips,
                slots_per_traversal=self.traversal_slots_per_traversal,
                policy_slots_per_traversal=self.policy_slots_per_traversal,
            )

        t0 = _time.perf_counter()
        self.last_traversal_pool_stats = []
        for player_i in range(self.n_players):
            remaining = self.n_traversals
            while remaining > 0:
                chunk = min(remaining, trav_batch)
                stats = gpu_traverse_for_player(
                    traverser=player_i,
                    n_traversals=chunk,
                    value_net=self.value_net,
                    buffer=self.buffers[player_i],
                    iteration=self.iteration,
                    n_players=self.n_players,
                    device=self.device,
                    initial_chips=self.initial_chips,
                    workspace=self._workspace,
                    policy_buffer=(
                        self.strategy_buffer if self.average_strategy_weight > 0 else None
                    ),
                )
                self.last_traversal_pool_stats.append(stats)
                self.traversal_pool_stats_history.append(stats)
                remaining -= chunk
        t1 = _time.perf_counter()

        if self.release_workspace_before_training:
            self._release_workspace_for_training()

        # Combine buffers and retrain.
        combined = self._combine_buffers()
        effective_train_batch = 0
        effective_train_steps = 0
        if len(combined) > 0:
            train_batch = self.batch_size
            train_steps = self.n_training_steps
            if (
                self.device.type == "cuda"
                and self.auto_scale_train_schedule
                and self.batch_size > 0
                and self.n_training_steps > 0
                and self.batch_size < self.train_batch_target
            ):
                sample_budget = self.batch_size * self.n_training_steps
                train_batch = min(self.train_batch_target, sample_budget)
                train_steps = max(1, (sample_budget + train_batch - 1) // train_batch)
                if not self._schedule_logged:
                    logger.info(
                        "Auto-scaled training schedule: batch %d->%d, steps %d->%d (sample budget %d)",
                        self.batch_size,
                        train_batch,
                        self.n_training_steps,
                        train_steps,
                        sample_budget,
                    )
                    self._schedule_logged = True
            if hasattr(combined, "set_expected_sample_budget"):
                combined.set_expected_sample_budget(train_batch * train_steps)
            effective_train_batch = int(train_batch)
            effective_train_steps = int(train_steps)
            try:
                self.value_net = train_value_network(
                    buffer=combined,
                    hidden_dim=self.hidden_dim,
                    n_epochs=train_steps,
                    batch_size=train_batch,
                    lr=self.lr,
                    device=self.device,
                    n_layers=self.n_layers,
                    use_betting_history=self.use_betting_history,
                    policy_target_buffer=self.policy_target_buffer,
                    policy_target_weight=self.policy_target_weight,
                    policy_target_batch_size=self.policy_target_batch_size,
                )
                if self.average_strategy_weight > 0 and self.strategy_buffer.size > 0:
                    self.average_policy_net = train_average_policy_network(
                        self.strategy_buffer,
                        hidden_dim=self.hidden_dim,
                        n_layers=self.n_layers,
                        n_epochs=train_steps,
                        batch_size=self.average_strategy_batch_size or train_batch,
                        lr=self.lr,
                        device=self.device,
                        use_betting_history=self.use_betting_history,
                    )
                    self.has_average_policy_net = True
            finally:
                if hasattr(combined, "release_gpu_cache"):
                    combined.release_gpu_cache()
        t2 = _time.perf_counter()
        self.last_profile = _build_iteration_profile(
            iteration=self.iteration,
            n_players=self.n_players,
            n_traversals=self.n_traversals,
            traversal_stats=self.last_traversal_pool_stats,
            traverse_seconds=t1 - t0,
            train_seconds=t2 - t1,
            train_batch_size=effective_train_batch,
            train_steps=effective_train_steps,
        )
        self.profile_history.append(self.last_profile)
        print(f"  [profile] traverse={t1-t0:.1f}s  train={t2-t1:.1f}s")
        return self.last_profile

    def traversal_pool_summary(self) -> dict[str, float | int]:
        return _summarize_traversal_pool_stats(self.traversal_pool_stats_history)

    def _release_workspace_for_training(self):
        """Drop traversal buffers before replay-cache allocation/training."""
        if self._workspace is None:
            return
        self._workspace = None
        gc.collect()
        if self.device.type == "cuda":
            try:
                cuda.current_context().deallocations.clear()
            except Exception:
                logger.debug("Could not flush Numba CUDA deallocations.", exc_info=True)
            torch.cuda.empty_cache()

    def _combine_buffers(self) -> ReservoirBuffer:
        """Create a lightweight view that samples from all player buffers.

        Instead of copying 10+GB into a new array, we use a MultiBuffer
        wrapper that samples proportionally from each player's buffer.
        """
        return _MultiBufferView(self.buffers)

    def evaluate(self, n_games: int = 500) -> float:
        return gpu_evaluate_vs_random(
            self.value_net, self.device, n_games, self.n_players,
            initial_chips=self.initial_chips,
        )

    def save(self, path: str):
        torch.save(
            {
                "value_net": self.value_net.state_dict(),
                "iteration": self.iteration,
                "n_players": self.n_players,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "initial_chips": self.initial_chips,
                "search_target_weight": self.policy_target_weight,
                "search_target_size": (
                    int(self.policy_target_buffer.size)
                    if self.policy_target_buffer is not None else 0
                ),
                "average_strategy_target_size": int(self.strategy_buffer.size),
                "average_strategy_weight": self.average_strategy_weight,
                "uses_betting_history": self.use_betting_history,
                "has_average_policy_net": bool(self.has_average_policy_net),
                "buffer_sizes": [len(b) for b in self.buffers],
                **(
                    {"average_policy_net": self.average_policy_net.state_dict()}
                    if self.has_average_policy_net else {}
                ),
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device | None = None):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        trainer = cls(
            n_players=checkpoint["n_players"],
            hidden_dim=checkpoint["hidden_dim"],
            n_layers=checkpoint.get("n_layers", 2),
            initial_chips=checkpoint.get("initial_chips", 10000),
            device=device,
            use_betting_history=bool(checkpoint.get("uses_betting_history", False)),
        )
        state = _remap_legacy_value_state_dict(checkpoint["value_net"])
        missing, unexpected = trainer.value_net.load_state_dict(state, strict=False)
        allowed_missing = {
            "policy_head.weight",
            "policy_head.bias",
            "seq_proj.weight",
            "seq_proj.bias",
        }
        missing = [key for key in missing if key not in allowed_missing]
        unexpected = list(unexpected)
        if missing or unexpected:
            raise RuntimeError(
                "Could not load value_net state_dict: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if checkpoint.get("average_policy_net") is not None:
            trainer.average_policy_net.load_state_dict(checkpoint["average_policy_net"])
            trainer.has_average_policy_net = True
        trainer.iteration = checkpoint["iteration"]
        return trainer
