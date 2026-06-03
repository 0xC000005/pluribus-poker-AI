"""GPU-resident R-NaD trajectory collector over the cuda/ GameBatch substrate.

Ports `CompiledNativeRNaDCollector` (numba-CPU, native_rnad.py) onto the existing GPU game
kernels so R-NaD training budget can scale far beyond the single-core numba CPU substrate.

Satisfies the `RNaDSolver` GPU-collector contract (solver.py:76-79, 133-141):
  - `is_gpu = True`            -> solver creates a torch.Generator and uses the GPU branch
  - `n_players / n_actions / obs_dim`
  - `collect(policy_fn_t, batch_size, trajectory_max, gen) -> Trajectory` with all tensors
    already on `gen.device` (no `.to()` afterwards)

Game logic reuses the already-parity-tested kernels UNCHANGED: `get_features_kernel`,
`get_legal_mask_kernel`, `apply_action_kernel`, `compute_winners_kernel`, `init_games_kernel`,
plus `GameBatch` and the HU order arrays ([0,1] preflop, [1,0] postflop). The only net-new device
code is a trivial `current_player_kernel` that exposes the acting seat (for `player_id` and for
opponent routing). Actions are sampled in torch (inverse-CDF, exactly like `GPUTreeCollector`) and
the FULL behavior-policy vector is stored — the R-NaD loss consumes `policy[T,B,A]`, not a per-action
log-prob, so no custom sampling kernel is needed.

Semantics match `CompiledNativeRNaDCollector` (the CPU reference): every live game records one
decision per step (HU alternation), the terminal payoff (`chips - initial_chips`, normalised by
`initial_chips`, zeroed for truncated games) lands on the game's LAST recorded step, and padded
slots use the same defaults as `_pack` (legal=1, action_oh=action-0, policy=uniform, valid=0).
RNG differs from the numpy collector, so GPU vs CPU trajectories are DISTRIBUTIONALLY equal, not
bitwise (same policy as `GPUTreeCollector`).

DIAGNOSTIC compute-substrate port: same R-NaD algorithm, opt-in via `substrate="cuda"`; the numba
CPU path stays the default. Writes nothing; no Slumbot data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from numba import cuda, int32

from poker_ai.deep_cfr.cuda.game_kernels import (
    _current_player,
    apply_action_kernel,
    compute_winners_kernel,
    get_features_kernel,
    get_legal_mask_kernel,
)
from poker_ai.deep_cfr.cuda.game_state import (
    SHOWDOWN,
    GameBatch,
    _get_device_orders,
    init_games_kernel,
)
from poker_ai.deep_cfr.cuda.lookup_tables import (
    FLUSH_SIZE,
    UNSUITED_SIZE,
    get_gpu_tables,
)
from poker_ai.research.native_rnad import (
    _adapter_policy_batch,
    _normalize_legal_policy,
    _normalize_opponent_meta_strategy,
)
from poker_ai.rnad.collector import Trajectory

N_FEATURES = 126
N_ACTIONS = 9
_RAISE_FRACTIONS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)


@cuda.jit
def current_player_kernel(
    player_i_index, stage, n_players, preflop_order, postflop_order, out_seat, n_games
):
    """Write the acting SEAT for each game (PREFLOP -> preflop_order, else postflop_order)."""
    i = cuda.grid(1)
    if i >= n_games:
        return
    out_seat[i] = _current_player(
        player_i_index[i], stage[i], n_players, preflop_order, postflop_order
    )


def _masked_normalize_torch(pi: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Torch mirror of native_rnad._normalize_legal_policy (mask illegal, renorm, uniform-legal fallback).

    Uses row_sum > 1e-7 (not > 0) for float32 robustness on near-zero rows.
    """
    mask = legal > 0.0
    pi = torch.where(mask, pi.clamp(min=0.0), torch.zeros_like(pi))
    row_sum = pi.sum(dim=-1, keepdim=True)
    legal_count = mask.to(pi.dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
    fallback = mask.to(pi.dtype) / legal_count
    return torch.where(row_sum > 1e-7, pi / row_sum.clamp(min=1e-30), fallback)


@dataclass
class CUDANativeRNaDCollector:
    """GPU R-NaD trajectory collector over the cuda/ GameBatch substrate (HU full-deck)."""

    seed: int = 20260601
    initial_chips: int = 1000
    device: Any = "cuda"
    opponent_policies: list[Any] | None = None
    opponent_meta_strategy: Sequence[float] | None = None
    opponent_device: Any = None
    last_metrics: dict[str, Any] = field(default_factory=dict)

    is_gpu: bool = True
    n_players: int = 2
    n_actions: int = N_ACTIONS
    obs_dim: int = N_FEATURES

    def __post_init__(self):
        self.device = torch.device(self.device)
        if self.device.type != "cuda":
            raise ValueError("CUDANativeRNaDCollector requires a cuda device")
        self.opponent_policies = list(self.opponent_policies or [])
        self.opponent_device = torch.device(self.opponent_device or self.device)
        # Device-resident constants (uploaded once).
        self._d_preflop, self._d_postflop = _get_device_orders()[2]
        self._d_raise_fractions = cuda.to_device(_RAISE_FRACTIONS)
        tables = get_gpu_tables()
        (
            self._d_flush_keys,
            self._d_flush_vals,
            self._d_unsuited_keys,
            self._d_unsuited_vals,
            self._d_card_lookup,
            _,
        ) = tables

    @torch.no_grad()
    def collect(self, policy_fn_t, batch_size: int, trajectory_max: int, gen) -> Trajectory:
        B = int(batch_size)
        T = int(trajectory_max)
        if B <= 0:
            raise ValueError("batch_size must be positive")
        if T <= 0:
            raise ValueError("trajectory_max must be positive")
        A, D, P = self.n_actions, self.obs_dim, self.n_players
        dev = self.device
        np_dtype_seed = np.int64
        started = time.perf_counter()

        # --- seed derivation (one host sync, off the hot path; mirrors CPU rng.randint draw) ---
        draw = int(torch.randint(0, 2 ** 30, (1,), device=dev, generator=gen).item())
        per_call_seed = (int(self.seed) + draw) & 0x7FFFFFFF

        # --- population / opponent assignment (host, once per collect) ---
        opponents = list(self.opponent_policies or [])
        use_population = len(opponents) > 0
        opponent_probs = _normalize_opponent_meta_strategy(
            self.opponent_meta_strategy, len(opponents)
        )
        host_rng = np.random.default_rng(per_call_seed)
        if use_population and self.opponent_meta_strategy is not None:
            opponent_assignments = host_rng.choice(
                len(opponents), size=B, p=opponent_probs
            ).astype(np.int32, copy=False)
        elif use_population:
            opponent_assignments = (np.arange(B, dtype=np.int32) % len(opponents))
        else:
            opponent_assignments = np.zeros(B, dtype=np.int32)
        learner_seats = (np.arange(B, dtype=np.int32) % P)
        opponent_steps_by_policy = np.zeros(len(opponents), dtype=np.int64)
        opponent_game_counts = (
            np.bincount(opponent_assignments, minlength=len(opponents)).astype(np.int64)
            if use_population
            else np.zeros(0, dtype=np.int64)
        )

        # --- create + initialise the GameBatch with explicit seeds ---
        batch = GameBatch(B, P)
        seeds_np = host_rng.integers(1, 2 ** 62, size=(B, 2), dtype=np_dtype_seed)
        d_seeds = cuda.to_device(seeds_np)
        threads = 256
        blocks = (B + threads - 1) // threads
        init_games_kernel[blocks, threads](
            batch.chips, batch.bets, batch.active, batch.hole_cards,
            batch.community, batch.deck, batch.deck_cursor,
            batch.stage, batch.n_raises, batch.player_i_index,
            batch.n_actions, batch.pot_total,
            batch.n_players_started_round, batch.history, batch.payout,
            batch.is_done,
            d_seeds, P, B, self._d_preflop, int(self.initial_chips),
        )

        # --- pre-allocate kernel I/O (reused each step) ---
        d_features = cuda.device_array((B, D), dtype=np.float32)
        d_masks = cuda.device_array((B, A), dtype=np.float32)
        d_seat = cuda.device_array(B, dtype=np.int32)

        # --- trajectory buffers initialised to the _pack defaults ---
        obs_buf = torch.zeros((T, B, D), dtype=torch.float32, device=dev)
        legal_buf = torch.ones((T, B, A), dtype=torch.float32, device=dev)
        pid_buf = torch.zeros((T, B), dtype=torch.float32, device=dev)
        valid_buf = torch.zeros((T, B), dtype=torch.float32, device=dev)
        rew_buf = torch.zeros((T, B, P), dtype=torch.float32, device=dev)
        aoh_buf = torch.zeros((T, B, A), dtype=torch.float32, device=dev)
        aoh_buf[:, :, 0] = 1.0
        pol_buf = torch.full((T, B, A), 1.0 / A, dtype=torch.float32, device=dev)
        default_oh = torch.zeros(A, dtype=torch.float32, device=dev)
        default_oh[0] = 1.0
        uniform_row = torch.full((A,), 1.0 / A, dtype=torch.float32, device=dev)
        ones_row = torch.ones(A, dtype=torch.float32, device=dev)

        ar = torch.arange(B, device=dev)
        last_t = np.full(B, -1, dtype=np.int64)
        steps_recorded = 0
        learner_controlled = 0
        opponent_controlled = 0
        illegal_records = 0

        for t in range(T):
            stages_host = batch.stage.copy_to_host()
            live_host = stages_host < SHOWDOWN
            if not live_host.any():
                break

            # GPU: features, legal masks, acting seat for ALL games.
            get_features_kernel[blocks, threads](
                batch.chips, batch.bets, batch.active,
                batch.hole_cards, batch.community,
                batch.stage, batch.n_raises, batch.player_i_index,
                batch.pot_total, batch.history,
                P, self._d_preflop, self._d_postflop,
                d_features, B, int(self.initial_chips),
            )
            get_legal_mask_kernel[blocks, threads](
                batch.active, batch.chips, batch.bets, batch.n_raises,
                batch.stage, batch.pot_total,
                batch.player_i_index, P,
                self._d_preflop, self._d_postflop,
                self._d_raise_fractions,
                d_masks, B,
            )
            current_player_kernel[blocks, threads](
                batch.player_i_index, batch.stage, P,
                self._d_preflop, self._d_postflop, d_seat, B,
            )
            cuda.synchronize()

            feat_t = torch.as_tensor(d_features, device=dev)
            masks_t = torch.as_tensor(d_masks, device=dev)
            seat_t = torch.as_tensor(d_seat, device=dev)
            live_t = torch.as_tensor(live_host).to(dev)

            # Behaviour policy: learner net for all rows; overwrite opponent-controlled rows.
            pi = policy_fn_t(feat_t, masks_t)

            if use_population:
                seat_host = d_seat.copy_to_host()
                opp_ctrl = live_host & (seat_host != learner_seats)
                learner_ctrl = live_host & ~opp_ctrl
                learner_controlled += int(learner_ctrl.sum())
                for key in np.unique(opponent_assignments[opp_ctrl]):
                    sel = opp_ctrl & (opponent_assignments == key)
                    idx = np.nonzero(sel)[0]
                    if idx.size == 0:
                        continue
                    idx_t = torch.as_tensor(idx.astype(np.int64), device=dev)
                    feats = feat_t[idx_t].detach().cpu().numpy().astype(np.float32, copy=False)
                    legals = masks_t[idx_t].detach().cpu().numpy().astype(np.float32, copy=False)
                    raw = _adapter_policy_batch(
                        opponents[int(key)], feats, legals, device=self.opponent_device
                    )
                    opp_pi = _normalize_legal_policy(raw, legals)
                    pi[idx_t] = torch.as_tensor(opp_pi, device=dev)
                    opponent_steps_by_policy[int(key)] += int(idx.size)
                    opponent_controlled += int(idx.size)
            else:
                learner_controlled += int(live_host.sum())

            pi = _masked_normalize_torch(pi, masks_t)

            # Inverse-CDF action sampling (matches CPU _sample_actions / GPUTreeCollector).
            cdf = torch.cumsum(pi, dim=-1)
            cdf[:, -1] = 1.0
            r = torch.rand(B, 1, device=dev, generator=gen)
            actions = (r < cdf).to(torch.int8).argmax(dim=-1)  # [B] first j with r < cdf[j]

            oh = torch.zeros((B, A), dtype=torch.float32, device=dev)
            oh[ar, actions] = 1.0
            live_col = live_t.unsqueeze(1)

            # Record the [t] slice: live rows take real values, padded rows take _pack defaults.
            obs_buf[t] = torch.where(live_col, feat_t, torch.zeros_like(feat_t))
            legal_buf[t] = torch.where(live_col, masks_t, ones_row.expand(B, A))
            pid_buf[t] = torch.where(live_t, seat_t.to(torch.float32), torch.zeros(B, device=dev))
            valid_buf[t] = live_t.to(torch.float32)
            aoh_buf[t] = torch.where(live_col, oh, default_oh.expand(B, A))
            pol_buf[t] = torch.where(live_col, pi, uniform_row.expand(B, A))

            # illegal-action audit over live rows.
            chosen_legal = masks_t.gather(1, actions.long().unsqueeze(1)).squeeze(1)
            illegal_records += int(((live_t) & (chosen_legal <= 0.0)).sum())

            steps_recorded += int(live_host.sum())
            last_t[live_host] = t

            # Apply actions on GPU (kernel skips finished games and action<0).
            actions_i8 = actions.to(torch.int8).contiguous()
            d_actions = cuda.as_cuda_array(actions_i8)
            apply_action_kernel[blocks, threads](
                batch.chips, batch.bets, batch.active,
                batch.hole_cards, batch.community, batch.deck,
                batch.deck_cursor, batch.stage, batch.n_raises,
                batch.player_i_index, batch.n_actions, batch.pot_total,
                batch.history, batch.n_players_started_round,
                d_actions, B, P,
                self._d_preflop, self._d_postflop,
                self._d_raise_fractions,
            )
            cuda.synchronize()

        # --- terminal payoffs ---
        compute_winners_kernel[blocks, threads](
            batch.chips, batch.bets, batch.active,
            batch.hole_cards, batch.community,
            batch.payout, batch.stage, B, P,
            self._d_card_lookup,
            self._d_flush_keys, self._d_flush_vals, FLUSH_SIZE,
            self._d_unsuited_keys, self._d_unsuited_vals, UNSUITED_SIZE,
            int(self.initial_chips),
        )
        cuda.synchronize()

        payoffs = torch.as_tensor(batch.payout, device=dev).to(torch.float32) / float(
            self.initial_chips
        )
        terminal = torch.as_tensor(batch.stage, device=dev) >= SHOWDOWN
        truncated_games = int((~terminal).sum())
        payoffs = torch.where(terminal.unsqueeze(1), payoffs, torch.zeros_like(payoffs))

        # Place each game's payoff at its LAST recorded step (advanced indexing, single fused write).
        recorded = last_t >= 0
        if recorded.any():
            lt = torch.as_tensor(last_t[recorded], device=dev)
            gi = torch.as_tensor(np.nonzero(recorded)[0].astype(np.int64), device=dev)
            rew_buf[lt, gi, :] = payoffs[gi, :]

        seconds = time.perf_counter() - started
        self.last_metrics = {
            "collector": "cuda_native_rnad_collector",
            "backend": "cuda-game-batch",
            "n_games": B,
            "trajectory_max": T,
            "steps": int(steps_recorded),
            "seconds": float(seconds),
            "steps_per_second": float(steps_recorded / max(seconds, 1e-12)),
            "learner_controlled_steps": int(learner_controlled),
            "opponent_controlled_steps": int(opponent_controlled),
            "population_opponent_size": int(len(opponents)),
            "population_opponent_meta_strategy": opponent_probs.tolist(),
            "population_opponent_game_counts": opponent_game_counts.astype(int).tolist(),
            "opponent_controlled_steps_by_policy": opponent_steps_by_policy.astype(int).tolist(),
            "train_opponent_mode": "population" if use_population else "self_play",
            "truncated_games": int(truncated_games),
            "needs_python_showdown": 0,
            "illegal_records": int(illegal_records),
        }
        return Trajectory(
            obs=obs_buf, legal=legal_buf, player_id=pid_buf, valid=valid_buf,
            rewards=rew_buf, action_oh=aoh_buf, policy=pol_buf,
        )
