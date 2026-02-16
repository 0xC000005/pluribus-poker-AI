"""Reservoir sampling buffer for Deep CFR training samples.

Stores (feature_vector, iteration, advantage_targets) tuples. When the
buffer exceeds capacity, new samples replace existing ones with
probability capacity/n_seen (reservoir sampling). This is critical for
convergence — a sliding window causes stalling.
"""
import numpy as np
import torch

from poker_ai.games.full_deck.state import N_FEATURES, N_ACTIONS


class ReservoirBuffer:
    """Fixed-capacity buffer with reservoir sampling.

    Pre-allocates numpy arrays for efficiency. Samples are stored as:
      - features: (capacity, N_FEATURES) float32
      - iterations: (capacity,) int32 — CFR iteration when sample was collected
      - advantages: (capacity, N_ACTIONS) float32 — regret targets per action
    """

    def __init__(self, capacity: int = 2_000_000):
        self.capacity = capacity
        self.features = np.zeros((capacity, N_FEATURES), dtype=np.float32)
        self.iterations = np.zeros(capacity, dtype=np.int32)
        self.advantages = np.zeros((capacity, N_ACTIONS), dtype=np.float32)
        self.size = 0
        self._n_seen = 0

    def add(
        self,
        features: np.ndarray,
        iteration: int,
        advantages: np.ndarray,
    ):
        """Add a single sample to the buffer.

        Uses reservoir sampling: if buffer is full, replace a random
        existing sample with probability capacity / n_seen.

        Parameters
        ----------
        features : ndarray of shape (N_FEATURES,)
        iteration : int
        advantages : ndarray of shape (N_ACTIONS,)
        """
        self._n_seen += 1
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            # Reservoir sampling.
            idx = np.random.randint(0, self._n_seen)
            if idx >= self.capacity:
                return  # Skip this sample.
        self.features[idx] = features
        self.iterations[idx] = iteration
        self.advantages[idx] = advantages

    def sample_batch(
        self, batch_size: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch as torch tensors.

        Parameters
        ----------
        batch_size : int
        device : torch.device, optional

        Returns
        -------
        features : Tensor (batch_size, N_FEATURES)
        iterations : Tensor (batch_size,)
        advantages : Tensor (batch_size, N_ACTIONS)
        """
        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))
        feat = torch.from_numpy(self.features[indices])
        iters = torch.from_numpy(self.iterations[indices].astype(np.float32))
        advs = torch.from_numpy(self.advantages[indices])
        if device is not None:
            feat = feat.to(device)
            iters = iters.to(device)
            advs = advs.to(device)
        return feat, iters, advs

    def add_batch(
        self,
        features_batch: np.ndarray,
        iteration: int,
        advantages_batch: np.ndarray,
        count: int,
    ):
        """Add multiple samples to the buffer using vectorized reservoir sampling.

        Parameters
        ----------
        features_batch : ndarray of shape (count, N_FEATURES)
        iteration : int
        advantages_batch : ndarray of shape (count, N_ACTIONS)
        count : int
            Number of valid samples in the arrays.
        """
        if count == 0:
            return

        # Phase 1: fill unfilled capacity with direct copy.
        n_direct = 0
        if self.size < self.capacity:
            n_direct = min(count, self.capacity - self.size)
            start = self.size
            self.features[start:start + n_direct] = features_batch[:n_direct]
            self.iterations[start:start + n_direct] = iteration
            self.advantages[start:start + n_direct] = advantages_batch[:n_direct]
            self.size += n_direct
            self._n_seen += n_direct

        # Phase 2: reservoir sampling for remaining samples.
        n_reservoir = count - n_direct
        if n_reservoir > 0:
            # Generate all random indices at once.
            # For sample i, n_seen will be self._n_seen + i + 1.
            n_seen_base = self._n_seen
            n_seen_values = n_seen_base + np.arange(1, n_reservoir + 1)
            rand_idx = (np.random.random(n_reservoir) * n_seen_values).astype(np.int64)

            # Keep only samples where rand_idx < capacity.
            keep_mask = rand_idx < self.capacity
            keep_positions = np.where(keep_mask)[0]

            if len(keep_positions) > 0:
                src_indices = n_direct + keep_positions
                dst_indices = rand_idx[keep_positions]
                self.features[dst_indices] = features_batch[src_indices]
                self.iterations[dst_indices] = iteration
                self.advantages[dst_indices] = advantages_batch[src_indices]

            self._n_seen += n_reservoir

    def clear(self):
        """Reset the buffer."""
        self.size = 0
        self._n_seen = 0

    def merge(
        self,
        features: np.ndarray,
        iterations: np.ndarray,
        advantages: np.ndarray,
        size: int,
    ):
        """Merge worker results into this buffer via reservoir sampling.

        Parameters
        ----------
        features : ndarray of shape (size, N_FEATURES)
        iterations : ndarray of shape (size,)
        advantages : ndarray of shape (size, N_ACTIONS)
        size : int
            Number of valid samples in the arrays.
        """
        if size == 0:
            return

        # Phase 1: fill unfilled capacity with direct copy.
        n_direct = 0
        if self.size < self.capacity:
            n_direct = min(size, self.capacity - self.size)
            start = self.size
            self.features[start:start + n_direct] = features[:n_direct]
            self.iterations[start:start + n_direct] = iterations[:n_direct]
            self.advantages[start:start + n_direct] = advantages[:n_direct]
            self.size += n_direct
            self._n_seen += n_direct

        # Phase 2: reservoir sampling for remaining samples.
        n_reservoir = size - n_direct
        if n_reservoir > 0:
            n_seen_base = self._n_seen
            n_seen_values = n_seen_base + np.arange(1, n_reservoir + 1)
            rand_idx = (np.random.random(n_reservoir) * n_seen_values).astype(np.int64)

            keep_mask = rand_idx < self.capacity
            keep_positions = np.where(keep_mask)[0]

            if len(keep_positions) > 0:
                src_indices = n_direct + keep_positions
                dst_indices = rand_idx[keep_positions]
                self.features[dst_indices] = features[src_indices]
                self.iterations[dst_indices] = iterations[src_indices]
                self.advantages[dst_indices] = advantages[src_indices]

            self._n_seen += n_reservoir

    def __len__(self) -> int:
        return self.size
