"""CUDA device functions for poker hand evaluation.

Implements the Cactus Kev evaluator as Numba CUDA device functions
so hand ranking can run entirely on GPU.  Uses sorted-array binary
search on the flush/unsuited lookup tables (~58 KB total).

Card format: 32-bit EvaluationCard integers (same encoding as CPU).
Rank output: 1 (royal flush) to 7462 (worst high card), lower = better.
"""

from numba import cuda, int32, int64


# ---------------------------------------------------------------------------
# Binary search (device function)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _binary_search(keys, size, target):
    """Binary search sorted int32 array.  Returns value index or -1."""
    lo = int32(0)
    hi = int32(size - 1)
    while lo <= hi:
        mid = (lo + hi) >> 1
        k = keys[mid]
        if k == target:
            return mid
        elif k < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return int32(-1)


# ---------------------------------------------------------------------------
# 5-card evaluator (device function)
# ---------------------------------------------------------------------------

# Primes per rank index (deuce=0 → ace=12).
# Duplicated as literal tuple to avoid device-array issues in device functions.
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


@cuda.jit(device=True)
def _prime_product_from_hand(c0, c1, c2, c3, c4):
    """Compute prime product from 5 EvaluationCard integers."""
    p = int64(c0 & 0xFF)
    p *= int64(c1 & 0xFF)
    p *= int64(c2 & 0xFF)
    p *= int64(c3 & 0xFF)
    p *= int64(c4 & 0xFF)
    return int32(p)


@cuda.jit(device=True)
def _prime_product_from_rankbits(rankbits):
    """Compute prime product from 13-bit rank pattern."""
    product = int64(1)
    for i in range(13):
        if rankbits & (1 << i):
            product *= _PRIMES[i]
    return int32(product)


@cuda.jit(device=True)
def eval_5cards(c0, c1, c2, c3, c4,
                flush_keys, flush_vals, flush_size,
                unsuited_keys, unsuited_vals, unsuited_size):
    """Evaluate a 5-card hand.  Returns rank 1-7462 (lower = better).

    Parameters are 32-bit EvaluationCard integers and the lookup table
    device arrays from lookup_tables.get_gpu_tables().
    """
    # Check for flush: all 5 cards share a suit bit.
    if c0 & c1 & c2 & c3 & c4 & 0xF000:
        hand_or = (c0 | c1 | c2 | c3 | c4) >> 16
        prime = _prime_product_from_rankbits(hand_or)
        idx = _binary_search(flush_keys, flush_size, prime)
        if idx >= 0:
            return flush_vals[idx]
        return int32(7463)  # Should not happen.

    prime = _prime_product_from_hand(c0, c1, c2, c3, c4)
    idx = _binary_search(unsuited_keys, unsuited_size, prime)
    if idx >= 0:
        return unsuited_vals[idx]
    return int32(7463)  # Should not happen.


# ---------------------------------------------------------------------------
# 7-card evaluator (device function) — tries all C(7,5)=21 combos
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def eval_7cards(cards,
                flush_keys, flush_vals, flush_size,
                unsuited_keys, unsuited_vals, unsuited_size):
    """Evaluate a 7-card hand by checking all 21 five-card subsets.

    Parameters
    ----------
    cards : device array of 7 int32 EvaluationCard values
    (remaining params are lookup tables)

    Returns
    -------
    int32 : best rank 1-7462 (lower = better)
    """
    best = int32(7463)
    # Iterate all C(7,5) = 21 combinations.
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                for m in range(k + 1, 7):
                    for n in range(m + 1, 7):
                        rank = eval_5cards(
                            cards[i], cards[j], cards[k], cards[m], cards[n],
                            flush_keys, flush_vals, flush_size,
                            unsuited_keys, unsuited_vals, unsuited_size,
                        )
                        if rank < best:
                            best = rank
    return best


# ---------------------------------------------------------------------------
# Kernel for batch hand evaluation (for testing / standalone use)
# ---------------------------------------------------------------------------

@cuda.jit
def eval_hands_kernel(hands, out_ranks, n,
                      flush_keys, flush_vals, flush_size,
                      unsuited_keys, unsuited_vals, unsuited_size):
    """Evaluate N 7-card hands in parallel.

    Parameters
    ----------
    hands : (N, 7) int32 device array of EvaluationCard values
    out_ranks : (N,) int32 device array for results
    n : int — number of hands
    """
    i = cuda.grid(1)
    if i < n:
        # Load 7 cards into local storage.
        local_cards = cuda.local.array(7, dtype=int32)
        for c in range(7):
            local_cards[c] = hands[i, c]
        out_ranks[i] = eval_7cards(
            local_cards,
            flush_keys, flush_vals, flush_size,
            unsuited_keys, unsuited_vals, unsuited_size,
        )
