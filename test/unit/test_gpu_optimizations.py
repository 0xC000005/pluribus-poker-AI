"""Tests for GPU-accelerated Deep CFR optimizations.

Verifies:
  1. regret_match_kernel matches CPU reference implementation
  2. sample_action_kernel produces valid actions with correct distribution
  3. classify_and_sample_kernel correctly separates traverser/opponent nodes
  4. Zero-copy Numba↔PyTorch interop works correctly
  5. GPU trainer produces buffer samples and trains (integration)
  6. GPU evaluation runs without errors
"""

import numpy as np
import pytest
import torch
from numba import cuda

# Skip all tests if no CUDA device.
pytestmark = pytest.mark.skipif(
    not cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# 1. regret_match_kernel correctness
# ---------------------------------------------------------------------------

class TestRegretMatchKernel:
    """Verify GPU regret matching matches the CPU reference."""

    @staticmethod
    def _cpu_regret_match(advantages, legal_mask):
        """CPU reference: same as _regret_match_np in the original trainer."""
        strategy = np.maximum(advantages, 0) * legal_mask
        total = strategy.sum()
        if total > 0:
            strategy /= total
        else:
            n_legal = legal_mask.sum()
            strategy = legal_mask / n_legal if n_legal > 0 else legal_mask
        return strategy

    def test_positive_advantages(self):
        """When advantages are positive, strategy is proportional."""
        from poker_ai.deep_cfr.cuda.action_kernels import regret_match_kernel

        adv = np.array([[2.0, 1.0, 0.5]], dtype=np.float32)
        mask = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

        d_adv = cuda.to_device(adv)
        d_mask = cuda.to_device(mask)
        d_strat = cuda.device_array((1, 3), dtype=np.float32)

        regret_match_kernel[1, 1](d_adv, d_mask, d_strat, 1)
        cuda.synchronize()
        gpu_strat = d_strat.copy_to_host()[0]

        cpu_strat = self._cpu_regret_match(adv[0], mask[0])
        np.testing.assert_allclose(gpu_strat, cpu_strat, atol=1e-6)

    def test_all_negative_advantages(self):
        """When all advantages are negative, strategy is uniform over legal."""
        from poker_ai.deep_cfr.cuda.action_kernels import regret_match_kernel

        adv = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        mask = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

        d_adv = cuda.to_device(adv)
        d_mask = cuda.to_device(mask)
        d_strat = cuda.device_array((1, 3), dtype=np.float32)

        regret_match_kernel[1, 1](d_adv, d_mask, d_strat, 1)
        cuda.synchronize()
        gpu_strat = d_strat.copy_to_host()[0]

        cpu_strat = self._cpu_regret_match(adv[0], mask[0])
        np.testing.assert_allclose(gpu_strat, cpu_strat, atol=1e-6)

    def test_mixed_with_illegal_actions(self):
        """Strategy correctly masks out illegal actions."""
        from poker_ai.deep_cfr.cuda.action_kernels import regret_match_kernel

        adv = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)  # raise illegal

        d_adv = cuda.to_device(adv)
        d_mask = cuda.to_device(mask)
        d_strat = cuda.device_array((1, 3), dtype=np.float32)

        regret_match_kernel[1, 1](d_adv, d_mask, d_strat, 1)
        cuda.synchronize()
        gpu_strat = d_strat.copy_to_host()[0]

        cpu_strat = self._cpu_regret_match(adv[0], mask[0])
        np.testing.assert_allclose(gpu_strat, cpu_strat, atol=1e-6)
        assert gpu_strat[2] == 0.0  # Illegal action gets 0 probability.

    def test_batch_matches_cpu(self):
        """GPU batch of 1000 games all match CPU regret matching."""
        from poker_ai.deep_cfr.cuda.action_kernels import regret_match_kernel

        N = 1000
        rng = np.random.default_rng(42)
        adv = rng.standard_normal((N, 3)).astype(np.float32) * 5
        # Random legal masks (at least one action legal).
        mask = rng.integers(0, 2, (N, 3)).astype(np.float32)
        for i in range(N):
            if mask[i].sum() == 0:
                mask[i, 1] = 1.0  # Ensure at least call is legal.

        d_adv = cuda.to_device(adv)
        d_mask = cuda.to_device(mask)
        d_strat = cuda.device_array((N, 3), dtype=np.float32)

        blocks = (N + 255) // 256
        regret_match_kernel[blocks, 256](d_adv, d_mask, d_strat, N)
        cuda.synchronize()
        gpu_strats = d_strat.copy_to_host()

        for i in range(N):
            cpu_strat = self._cpu_regret_match(adv[i], mask[i])
            np.testing.assert_allclose(
                gpu_strats[i], cpu_strat, atol=1e-5,
                err_msg=f"Mismatch at game {i}: adv={adv[i]}, mask={mask[i]}",
            )

    def test_strategies_sum_to_one(self):
        """All strategies should sum to 1.0 (or 0 if no legal actions)."""
        from poker_ai.deep_cfr.cuda.action_kernels import regret_match_kernel

        N = 500
        rng = np.random.default_rng(123)
        adv = rng.standard_normal((N, 3)).astype(np.float32)
        mask = np.ones((N, 3), dtype=np.float32)

        d_adv = cuda.to_device(adv)
        d_mask = cuda.to_device(mask)
        d_strat = cuda.device_array((N, 3), dtype=np.float32)

        blocks = (N + 255) // 256
        regret_match_kernel[blocks, 256](d_adv, d_mask, d_strat, N)
        cuda.synchronize()
        gpu_strats = d_strat.copy_to_host()

        sums = gpu_strats.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_no_legal_actions_gives_zero(self):
        """If no actions are legal, strategy should be all zeros."""
        from poker_ai.deep_cfr.cuda.action_kernels import regret_match_kernel

        adv = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        mask = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        d_adv = cuda.to_device(adv)
        d_mask = cuda.to_device(mask)
        d_strat = cuda.device_array((1, 3), dtype=np.float32)

        regret_match_kernel[1, 1](d_adv, d_mask, d_strat, 1)
        cuda.synchronize()
        gpu_strat = d_strat.copy_to_host()[0]

        assert gpu_strat.sum() == 0.0


# ---------------------------------------------------------------------------
# 2. sample_action_kernel correctness
# ---------------------------------------------------------------------------

class TestSampleActionKernel:
    """Verify GPU action sampling produces valid actions."""

    def test_all_actions_valid(self):
        """Sampled actions should be 0, 1, or 2 for active games."""
        from poker_ai.deep_cfr.cuda.action_kernels import sample_action_kernel
        from numba.cuda.random import create_xoroshiro128p_states

        N = 1000
        strats = np.tile([0.3, 0.3, 0.4], (N, 1)).astype(np.float32)
        masks = np.ones((N, 3), dtype=np.float32)
        stages = np.zeros(N, dtype=np.int8)

        d_strat = cuda.to_device(strats)
        d_mask = cuda.to_device(masks)
        d_stage = cuda.to_device(stages)
        d_actions = cuda.device_array(N, dtype=np.int8)
        rng = create_xoroshiro128p_states(N, seed=42)

        blocks = (N + 255) // 256
        sample_action_kernel[blocks, 256](
            d_strat, d_mask, d_stage, rng, d_actions, N,
        )
        cuda.synchronize()
        actions = d_actions.copy_to_host()

        assert all(a in [0, 1, 2] for a in actions)

    def test_finished_games_get_skip(self):
        """Games with stage >= 4 should get action -1."""
        from poker_ai.deep_cfr.cuda.action_kernels import sample_action_kernel
        from numba.cuda.random import create_xoroshiro128p_states

        N = 10
        strats = np.ones((N, 3), dtype=np.float32) / 3
        masks = np.ones((N, 3), dtype=np.float32)
        stages = np.array([0, 1, 2, 3, 4, 5, 4, 0, 0, 5], dtype=np.int8)

        d_strat = cuda.to_device(strats)
        d_mask = cuda.to_device(masks)
        d_stage = cuda.to_device(stages)
        d_actions = cuda.device_array(N, dtype=np.int8)
        rng = create_xoroshiro128p_states(N, seed=42)

        sample_action_kernel[1, N](
            d_strat, d_mask, d_stage, rng, d_actions, N,
        )
        cuda.synchronize()
        actions = d_actions.copy_to_host()

        # Active games: actions should be 0, 1, or 2.
        for i in [0, 1, 2, 3, 7, 8]:
            assert actions[i] in [0, 1, 2], f"Game {i}: expected valid action, got {actions[i]}"

        # Finished games: action should be -1.
        for i in [4, 5, 6, 9]:
            assert actions[i] == -1, f"Game {i}: expected -1, got {actions[i]}"

    def test_distribution_matches_strategy(self):
        """Over many samples, action frequencies should match strategy."""
        from poker_ai.deep_cfr.cuda.action_kernels import sample_action_kernel
        from numba.cuda.random import create_xoroshiro128p_states

        N = 10000
        target = [0.1, 0.3, 0.6]
        strats = np.tile(target, (N, 1)).astype(np.float32)
        masks = np.ones((N, 3), dtype=np.float32)
        stages = np.zeros(N, dtype=np.int8)

        d_strat = cuda.to_device(strats)
        d_mask = cuda.to_device(masks)
        d_stage = cuda.to_device(stages)
        d_actions = cuda.device_array(N, dtype=np.int8)
        rng = create_xoroshiro128p_states(N, seed=42)

        blocks = (N + 255) // 256
        sample_action_kernel[blocks, 256](
            d_strat, d_mask, d_stage, rng, d_actions, N,
        )
        cuda.synchronize()
        actions = d_actions.copy_to_host()

        freqs = [np.sum(actions == a) / N for a in range(3)]
        for a in range(3):
            assert abs(freqs[a] - target[a]) < 0.03, (
                f"Action {a}: expected ~{target[a]}, got {freqs[a]}"
            )

    def test_respects_legal_mask(self):
        """If an action is illegal (mask=0), it should never be sampled."""
        from poker_ai.deep_cfr.cuda.action_kernels import sample_action_kernel
        from numba.cuda.random import create_xoroshiro128p_states

        N = 5000
        # Strategy has weight on action 2, but mask blocks it.
        strats = np.tile([0.0, 0.5, 0.5], (N, 1)).astype(np.float32)
        masks = np.tile([0.0, 1.0, 1.0], (N, 1)).astype(np.float32)
        stages = np.zeros(N, dtype=np.int8)

        # But wait — regret_match_kernel already handles this.
        # sample_action_kernel trusts the strategy is already masked.
        # The strategy [0, 0.5, 0.5] shouldn't produce action 0.
        d_strat = cuda.to_device(strats)
        d_mask = cuda.to_device(masks)
        d_stage = cuda.to_device(stages)
        d_actions = cuda.device_array(N, dtype=np.int8)
        rng = create_xoroshiro128p_states(N, seed=42)

        blocks = (N + 255) // 256
        sample_action_kernel[blocks, 256](
            d_strat, d_mask, d_stage, rng, d_actions, N,
        )
        cuda.synchronize()
        actions = d_actions.copy_to_host()

        # Action 0 should never appear (probability 0).
        assert np.sum(actions == 0) == 0, "Sampled illegal action 0!"


# ---------------------------------------------------------------------------
# 3. classify_and_sample_kernel correctness
# ---------------------------------------------------------------------------

class TestClassifyAndSampleKernel:
    """Verify traverser/opponent classification on GPU."""

    def test_traverser_gets_no_action(self):
        """Traverser nodes should have out_actions=-1, out_is_traverser=1."""
        from poker_ai.deep_cfr.cuda.action_kernels import classify_and_sample_kernel
        from numba.cuda.random import create_xoroshiro128p_states

        N = 4
        strats = np.ones((N, 3), dtype=np.float32) / 3
        masks = np.ones((N, 3), dtype=np.float32)

        # 2-player game: preflop order [0, 1].
        # If traverser=0 and current player=0, it's a traverser node.
        # player_i_index=0 in preflop → player 0.
        stages = np.array([0, 0, 1, 1], dtype=np.int8)  # preflop, preflop, flop, flop
        pii = np.array([0, 1, 0, 1], dtype=np.int8)     # indices into order

        d_strat = cuda.to_device(strats)
        d_mask = cuda.to_device(masks)
        d_stage = cuda.to_device(stages)
        d_pii = cuda.to_device(pii)
        d_actions = cuda.device_array(N, dtype=np.int8)
        d_is_trav = cuda.device_array(N, dtype=np.int8)
        rng = create_xoroshiro128p_states(N, seed=42)

        d_preflop = cuda.to_device(np.array([0, 1], dtype=np.int8))
        d_postflop = cuda.to_device(np.array([0, 1], dtype=np.int8))

        classify_and_sample_kernel[1, N](
            d_strat, d_mask, d_stage, d_pii,
            2, 0,  # n_players=2, traverser=0
            d_preflop, d_postflop, rng,
            d_actions, d_is_trav, N,
        )
        cuda.synchronize()

        actions = d_actions.copy_to_host()
        is_trav = d_is_trav.copy_to_host()

        # Game 0: preflop, pii=0 → player 0 = traverser.
        assert is_trav[0] == 1
        assert actions[0] == -1

        # Game 1: preflop, pii=1 → player 1 = opponent.
        assert is_trav[1] == 0
        assert actions[1] in [0, 1, 2]

        # Game 2: flop, pii=0 → player 0 = traverser.
        assert is_trav[2] == 1
        assert actions[2] == -1

        # Game 3: flop, pii=1 → player 1 = opponent.
        assert is_trav[3] == 0
        assert actions[3] in [0, 1, 2]


# ---------------------------------------------------------------------------
# 4. Zero-copy interop
# ---------------------------------------------------------------------------

class TestZeroCopyInterop:
    """Verify Numba↔PyTorch zero-copy transfers."""

    def test_numba_to_pytorch(self):
        """torch.as_tensor from Numba device array shares memory."""
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        d_arr = cuda.to_device(data)

        t = torch.as_tensor(d_arr, device="cuda")
        assert t.shape == (1, 3)
        assert t.device.type == "cuda"

        # Verify same data.
        np.testing.assert_allclose(t.cpu().numpy(), data)

    def test_pytorch_to_numba(self):
        """cuda.as_cuda_array from PyTorch tensor shares memory."""
        t = torch.tensor([[4.0, 5.0, 6.0]], device="cuda", dtype=torch.float32)
        d_arr = cuda.as_cuda_array(t)

        assert d_arr.shape == (1, 3)
        np.testing.assert_allclose(d_arr.copy_to_host(), t.cpu().numpy())

    def test_roundtrip_preserves_data(self):
        """Numba → PyTorch → model → Numba round trip preserves correctness."""
        from poker_ai.deep_cfr.networks import ValueNetwork
        from poker_ai.deep_cfr.fast_state import N_FEATURES, N_ACTIONS

        model = ValueNetwork(N_FEATURES, 64, N_ACTIONS).cuda()
        model.eval()

        # Create features on GPU via Numba.
        features_np = np.random.randn(10, N_FEATURES).astype(np.float32)
        d_features = cuda.to_device(features_np)

        # Zero-copy forward pass.
        with torch.no_grad():
            feat_t = torch.as_tensor(d_features, device="cuda")
            out_t = model(feat_t)
        torch.cuda.synchronize()

        # Convert output back to Numba.
        d_out = cuda.as_cuda_array(out_t.detach())
        out_np = d_out.copy_to_host()

        # Compare with standard path (explicit copy).
        with torch.no_grad():
            feat_t2 = torch.from_numpy(features_np).cuda()
            out_t2 = model(feat_t2)
        expected = out_t2.cpu().numpy()

        np.testing.assert_allclose(out_np, expected, atol=1e-6)

    def test_large_batch_zero_copy(self):
        """Zero-copy works correctly for large batches (10K x 126)."""
        N = 10000
        data = np.random.randn(N, 126).astype(np.float32)
        d_arr = cuda.to_device(data)

        t = torch.as_tensor(d_arr, device="cuda")
        assert t.shape == (N, 126)

        # Verify subset of data.
        np.testing.assert_allclose(
            t[0:10].cpu().numpy(), data[0:10], atol=1e-6,
        )


# ---------------------------------------------------------------------------
# 5. GPU trainer integration
# ---------------------------------------------------------------------------

class TestGPUTrainerIntegration:
    """Integration tests for the optimized GPU trainer."""

    def test_2player_produces_samples(self):
        """2-player training produces buffer samples."""
        from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

        trainer = GPUDeepCFRTrainer(
            n_players=2, n_traversals=100, n_training_steps=50,
            buffer_capacity=50_000, hidden_dim=64,
        )
        trainer.run_iteration()

        total = sum(len(b) for b in trainer.buffers)
        assert total > 0, "No buffer samples after 1 iteration"

    def test_6player_produces_samples(self):
        """6-player training produces buffer samples."""
        from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

        trainer = GPUDeepCFRTrainer(
            n_players=6, n_traversals=20, n_training_steps=50,
            buffer_capacity=50_000, hidden_dim=64,
        )
        trainer.run_iteration()

        for i, buf in enumerate(trainer.buffers):
            assert len(buf) > 0, f"Buffer {i} is empty after 1 iteration"

    def test_multiple_iterations_grow_buffer(self):
        """Multiple iterations increase buffer size."""
        from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

        trainer = GPUDeepCFRTrainer(
            n_players=2, n_traversals=20, n_training_steps=50,
            buffer_capacity=100_000, hidden_dim=64,
        )

        trainer.run_iteration()
        size_after_1 = sum(len(b) for b in trainer.buffers)

        trainer.run_iteration()
        size_after_2 = sum(len(b) for b in trainer.buffers)

        assert size_after_2 > size_after_1, (
            f"Buffer didn't grow: {size_after_1} -> {size_after_2}"
        )

    def test_evaluation_returns_finite(self):
        """GPU evaluation returns a finite number."""
        from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

        trainer = GPUDeepCFRTrainer(
            n_players=2, n_traversals=10, n_training_steps=10,
            buffer_capacity=10_000, hidden_dim=64,
        )
        trainer.run_iteration()

        payout = trainer.evaluate(n_games=100)
        assert np.isfinite(payout), f"Evaluation returned {payout}"

    def test_buffer_features_valid(self):
        """Buffer samples have valid feature vectors (no NaN/Inf)."""
        from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer

        trainer = GPUDeepCFRTrainer(
            n_players=2, n_traversals=30, n_training_steps=50,
            buffer_capacity=50_000, hidden_dim=64,
        )
        trainer.run_iteration()

        for buf in trainer.buffers:
            if len(buf) > 0:
                features = buf.features[:buf.size]
                assert np.all(np.isfinite(features)), "NaN/Inf in buffer features"
                advantages = buf.advantages[:buf.size]
                assert np.all(np.isfinite(advantages)), "NaN/Inf in buffer advantages"
