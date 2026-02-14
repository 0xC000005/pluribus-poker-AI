"""GPU-compatible lookup tables for Cactus Kev hand evaluation.

Converts the Python dict-based LookupTable into sorted numpy arrays
suitable for binary search on GPU.  Tables are built once at import
and uploaded to GPU device memory on first use.

Total GPU memory: ~58 KB (fits in shared memory).
"""

import numpy as np
from numba import cuda

from poker_ai.poker.evaluation.eval_card import EvaluationCard
from poker_ai.poker.evaluation.lookup import LookupTable

# ---------------------------------------------------------------------------
# Build tables at import time (CPU)
# ---------------------------------------------------------------------------

_lt = LookupTable()

# Sorted key/value arrays for binary search on GPU.
_flush_items = sorted(_lt.flush_lookup.items())
FLUSH_KEYS = np.array([k for k, v in _flush_items], dtype=np.int32)
FLUSH_VALS = np.array([v for k, v in _flush_items], dtype=np.int32)
FLUSH_SIZE = len(FLUSH_KEYS)

_unsuited_items = sorted(_lt.unsuited_lookup.items())
UNSUITED_KEYS = np.array([k for k, v in _unsuited_items], dtype=np.int32)
UNSUITED_VALS = np.array([v for k, v in _unsuited_items], dtype=np.int32)
UNSUITED_SIZE = len(UNSUITED_KEYS)

# Card index (0-51) → EvaluationCard 32-bit integer.
# Card index = (rank - 2) * 4 + suit_idx, suits: c=0, d=1, h=2, s=3.
_SUIT_CHARS = ["c", "d", "h", "s"]
_RANK_CHARS = list(EvaluationCard.STR_RANKS)  # "23456789TJQKA"
CARD_INDEX_TO_EVAL_CARD = np.zeros(52, dtype=np.int32)
for _ci in range(52):
    _r = _ci // 4
    _s = _ci % 4
    CARD_INDEX_TO_EVAL_CARD[_ci] = EvaluationCard.new(
        f"{_RANK_CHARS[_r]}{_SUIT_CHARS[_s]}"
    )

# Primes for each rank (deuce=2, trey=3, ..., ace=41).
PRIMES = np.array(EvaluationCard.PRIMES, dtype=np.int32)

# Clean up module namespace.
del _lt, _flush_items, _unsuited_items, _ci, _r, _s

# ---------------------------------------------------------------------------
# GPU device arrays (lazy initialization)
# ---------------------------------------------------------------------------

_gpu_tables = None


def get_gpu_tables():
    """Return (d_flush_keys, d_flush_vals, d_unsuited_keys, d_unsuited_vals,
    d_card_lookup, d_primes) as device arrays.  Cached after first call."""
    global _gpu_tables
    if _gpu_tables is None:
        _gpu_tables = (
            cuda.to_device(FLUSH_KEYS),
            cuda.to_device(FLUSH_VALS),
            cuda.to_device(UNSUITED_KEYS),
            cuda.to_device(UNSUITED_VALS),
            cuda.to_device(CARD_INDEX_TO_EVAL_CARD),
            cuda.to_device(PRIMES),
        )
    return _gpu_tables
