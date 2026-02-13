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

    def clear(self):
        """Reset the buffer."""
        self.size = 0
        self._n_seen = 0

    def __len__(self) -> int:
        return self.size
