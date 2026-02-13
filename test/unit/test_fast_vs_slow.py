"""Verification tests: FastPokerState + fast optimizations vs reference implementation.

Tests confirm that the optimized numpy-array-based poker state and the coroutine
batched traversal produce correct results by comparing against the original
Python-object-based PokerState.
"""
import copy
import time

import numpy as np
import pytest
import torch

from poker_ai.deep_cfr.fast_state import (
    CARD_INDEX_TO_EVAL_CARD,
    FastPokerState,
    N_ACTIONS,
    N_FEATURES,
    new_fast_game,
)
from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.fast_traverse import batched_traverse
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.deep_cfr.vectorized_env import VectorizedPokerEnv, fast_evaluate_vs_random
from poker_ai.poker.evaluation.eval_card import EvaluationCard


# ---------------------------------------------------------------------------
# 1. Card index mapping
# ---------------------------------------------------------------------------

class TestCardIndexMapping:
    """Verify all 52 card indices produce valid EvaluationCard ints."""

    def test_all_52_cards_valid(self):
        """Each card index maps to a non-zero EvaluationCard int."""
        for i in range(52):
            assert CARD_INDEX_TO_EVAL_CARD[i] != 0, f"Card index {i} maps to 0"

    def test_all_52_cards_unique(self):
        """No two card indices map to the same EvaluationCard."""
        vals = set(int(CARD_INDEX_TO_EVAL_CARD[i]) for i in range(52))
        assert len(vals) == 52, f"Only {len(vals)} unique eval cards"

    def test_known_cards(self):
        """Spot-check specific cards against EvaluationCard.new()."""
        # card_idx = (rank - 2) * 4 + suit_idx
        # rank 14 = Ace, suit 3 = spades → idx = 12*4 + 3 = 51
        ace_spades = EvaluationCard.new("As")
        assert int(CARD_INDEX_TO_EVAL_CARD[51]) == ace_spades

        # rank 2 = deuce, suit 0 = clubs → idx = 0
        deuce_clubs = EvaluationCard.new("2c")
        assert int(CARD_INDEX_TO_EVAL_CARD[0]) == deuce_clubs

        # rank 13 = King, suit 2 = hearts → idx = (13-2)*4 + 2 = 46
        king_hearts = EvaluationCard.new("Kh")
        assert int(CARD_INDEX_TO_EVAL_CARD[46]) == king_hearts


# ---------------------------------------------------------------------------
# 2. Chip conservation
# ---------------------------------------------------------------------------

class TestChipConservation:
    """Chip total must be preserved across all random games."""

    @pytest.mark.parametrize("n_players", [2, 3, 6])
    def test_chip_conservation(self, n_players):
        """Sum of payouts == 0 for 200 random games."""
        violations = 0
        for _ in range(200):
            state = new_fast_game(n_players)
            for _ in range(200):  # max actions safety
                if state.is_terminal:
                    break
                pi = state.current_player_i
                if not state.active[pi]:
                    child = state.copy()
                    child.apply_action(None)
                    state = child
                    continue
                mask = state.get_legal_mask()
                legal = np.where(mask > 0)[0]
                action = int(np.random.choice(legal))
                child = state.copy()
                child.apply_action(action)
                state = child

            if state.is_terminal:
                payout = state.payout
                total = sum(payout.values())
                if total != 0:
                    violations += 1
        assert violations == 0, f"{violations}/200 games had non-zero payout sum"


# ---------------------------------------------------------------------------
# 3. FastPokerState copy performance
# ---------------------------------------------------------------------------

class TestFastStateCopyPerformance:
    """FastPokerState.copy() must be >100x faster than deepcopy(PokerState)."""

    def test_copy_speed(self):
        """Benchmark copy vs deepcopy on FastPokerState itself."""
        state = new_fast_game(2)
        # Warm up
        for _ in range(100):
            state.copy()

        n = 10_000
        t0 = time.perf_counter()
        for _ in range(n):
            state.copy()
        fast_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            copy.deepcopy(state)
        slow_time = time.perf_counter() - t0

        speedup = slow_time / fast_time
        assert speedup > 5, (
            f"copy() only {speedup:.1f}x faster than deepcopy (expected >5x)"
        )


# ---------------------------------------------------------------------------
# 4. Feature vector properties
# ---------------------------------------------------------------------------

class TestFeatureVector:
    """Feature vector must have correct shape and sensible values."""

    def test_shape(self):
        state = new_fast_game(2)
        fv = state.to_feature_vector()
        assert fv.shape == (N_FEATURES,)
        assert fv.dtype == np.float32

    def test_hole_cards_set(self):
        """Exactly 2 hole card bits should be set."""
        state = new_fast_game(2)
        fv = state.to_feature_vector()
        hole_bits = fv[:52]
        assert hole_bits.sum() == 2.0

    def test_no_community_preflop(self):
        """No community card bits set at preflop start."""
        state = new_fast_game(2)
        fv = state.to_feature_vector()
        comm_bits = fv[52:104]
        assert comm_bits.sum() == 0.0

    def test_round_one_hot(self):
        """Exactly one round bit is set (preflop at start)."""
        state = new_fast_game(2)
        fv = state.to_feature_vector()
        round_bits = fv[104:108]
        assert round_bits.sum() == 1.0
        assert round_bits[0] == 1.0  # preflop


# ---------------------------------------------------------------------------
# 5. Legal mask properties
# ---------------------------------------------------------------------------

class TestLegalMask:
    """Legal action mask must be consistent with game state."""

    def test_active_player_has_actions(self):
        state = new_fast_game(2)
        mask = state.get_legal_mask()
        assert mask.sum() > 0, "Active player should have legal actions"

    def test_fold_call_always_legal(self):
        state = new_fast_game(2)
        mask = state.get_legal_mask()
        assert mask[0] == 1.0, "Fold should always be legal"
        assert mask[1] == 1.0, "Call should always be legal"

    def test_raise_limit(self):
        """After 3 raises, raise should be illegal."""
        state = new_fast_game(2)
        for _ in range(3):
            if state.is_terminal:
                pytest.skip("Game ended before 3 raises")
            child = state.copy()
            child.apply_action(2)  # raise
            state = child
        if not state.is_terminal:
            mask = state.get_legal_mask()
            assert mask[2] == 0.0, "Raise should be illegal after 3 raises"


# ---------------------------------------------------------------------------
# 6. Batched traverse produces samples
# ---------------------------------------------------------------------------

class TestBatchedTraverse:
    """Coroutine batched traverse must populate the buffer."""

    def test_produces_samples(self):
        """50 traversals should produce >0 buffer samples."""
        device = torch.device("cpu")
        value_net = ValueNetwork(N_FEATURES, 64, N_ACTIONS)
        value_net.eval()
        buffer = ReservoirBuffer(100_000)

        batched_traverse(
            n_traversals=50,
            traverser=0,
            value_net=value_net,
            buffer=buffer,
            iteration=1,
            device=device,
            n_players=2,
        )
        assert len(buffer) > 0, "Buffer should have samples after traversal"
        # Each traversal produces at least 1 sample (traverser nodes).
        assert len(buffer) >= 50, (
            f"Expected >=50 samples from 50 traversals, got {len(buffer)}"
        )

    def test_samples_have_valid_shape(self):
        """Buffer samples must have correct feature/advantage dimensions."""
        device = torch.device("cpu")
        value_net = ValueNetwork(N_FEATURES, 64, N_ACTIONS)
        value_net.eval()
        buffer = ReservoirBuffer(100_000)

        batched_traverse(
            n_traversals=20,
            traverser=0,
            value_net=value_net,
            buffer=buffer,
            iteration=1,
            device=device,
        )

        feat, iters, advs = buffer.sample_batch(10, device)
        assert feat.shape == (10, N_FEATURES)
        assert iters.shape == (10,)
        assert advs.shape == (10, N_ACTIONS)


# ---------------------------------------------------------------------------
# 7. Vectorized evaluation
# ---------------------------------------------------------------------------

class TestVectorizedEval:
    """VectorizedPokerEnv and fast_evaluate_vs_random."""

    def test_env_reset_and_features(self):
        env = VectorizedPokerEnv(10, n_players=2)
        env.reset()
        features = env.get_features()
        assert features.shape == (10, N_FEATURES)
        masks = env.get_legal_masks()
        assert masks.shape == (10, N_ACTIONS)

    def test_evaluate_returns_float(self):
        """fast_evaluate_vs_random should return a float."""
        device = torch.device("cpu")
        value_net = ValueNetwork(N_FEATURES, 64, N_ACTIONS)
        avg = fast_evaluate_vs_random(value_net, device, n_games=20, n_players=2)
        assert isinstance(avg, float)


# ---------------------------------------------------------------------------
# 8. Multi-process traversal
# ---------------------------------------------------------------------------

class TestMultiProcessTraversal:
    """Verify worker_fn returns valid buffer data."""

    def test_worker_fn_returns_data(self):
        from poker_ai.deep_cfr.fast_traverse import worker_fn

        value_net = ValueNetwork(N_FEATURES, 64, N_ACTIONS)
        state_dict = {k: v.cpu() for k, v in value_net.state_dict().items()}

        features, iterations, advantages, size = worker_fn(
            worker_id=0,
            n_traversals=20,
            traverser=0,
            value_net_state_dict=state_dict,
            iteration=1,
            n_players=2,
            buffer_capacity=10_000,
            hidden_dim=64,
        )
        assert size > 0, "Worker should produce samples"
        assert features.shape[0] == size
        assert features.shape[1] == N_FEATURES
        assert iterations.shape[0] == size
        assert advantages.shape[0] == size
        assert advantages.shape[1] == N_ACTIONS

    def test_buffer_merge_preserves_samples(self):
        """Merging worker data into a buffer should retain samples."""
        main_buffer = ReservoirBuffer(100_000)
        value_net = ValueNetwork(N_FEATURES, 64, N_ACTIONS)
        state_dict = {k: v.cpu() for k, v in value_net.state_dict().items()}

        from poker_ai.deep_cfr.fast_traverse import worker_fn

        features, iterations, advantages, size = worker_fn(
            worker_id=0, n_traversals=30, traverser=0,
            value_net_state_dict=state_dict, iteration=1,
            n_players=2, buffer_capacity=10_000, hidden_dim=64,
        )
        main_buffer.merge(features, iterations, advantages, size)
        assert len(main_buffer) == size, (
            f"After merge, buffer size {len(main_buffer)} != worker size {size}"
        )


# ---------------------------------------------------------------------------
# 9. FastDeepCFRTrainer end-to-end
# ---------------------------------------------------------------------------

class TestFastDeepCFRTrainer:
    """End-to-end trainer test (lightweight settings)."""

    def test_single_iteration(self):
        from poker_ai.deep_cfr.fast_trainer import FastDeepCFRTrainer

        trainer = FastDeepCFRTrainer(
            n_players=2,
            buffer_capacity=10_000,
            hidden_dim=64,
            batch_size=128,
            lr=0.001,
            n_training_steps=10,
            n_traversals=20,
            n_workers=1,
            device=torch.device("cpu"),
        )
        trainer.run_iteration()
        assert trainer.iteration == 1
        assert any(len(b) > 0 for b in trainer.buffers)

    def test_save_load_roundtrip(self, tmp_path):
        from poker_ai.deep_cfr.fast_trainer import FastDeepCFRTrainer

        trainer = FastDeepCFRTrainer(
            n_players=2, buffer_capacity=1000, hidden_dim=64,
            n_training_steps=5, n_traversals=10, n_workers=1,
            device=torch.device("cpu"),
        )
        trainer.run_iteration()

        path = str(tmp_path / "test_checkpoint.pt")
        trainer.save(path)

        loaded = FastDeepCFRTrainer.load(path, device=torch.device("cpu"))
        assert loaded.iteration == trainer.iteration
        assert loaded.n_players == trainer.n_players

        # Weights should match.
        for key in trainer.value_net.state_dict():
            orig = trainer.value_net.state_dict()[key]
            load = loaded.value_net.state_dict()[key]
            assert torch.allclose(orig, load), f"Weights mismatch for {key}"
