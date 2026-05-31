"""Utilities for ragged terminal-evaluation experiments."""

from __future__ import annotations

import numpy as np


def pad_ragged_2d_arrays(
    arrays: list[np.ndarray],
    *,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of `(rows, cols)` arrays into `(batch, max_rows, cols)`."""
    if not arrays:
        raise ValueError("arrays must be non-empty")
    normalized = [np.asarray(array, dtype=np.float32) for array in arrays]
    n_cols = int(normalized[0].shape[1])
    for index, array in enumerate(normalized):
        if array.ndim != 2:
            raise ValueError(f"array {index} must be 2D")
        if int(array.shape[1]) != n_cols:
            raise ValueError("all arrays must have the same column count")
    max_rows = max((int(array.shape[0]) for array in normalized), default=0)
    padded = np.full(
        (len(normalized), max_rows, n_cols),
        float(pad_value),
        dtype=np.float32,
    )
    mask = np.zeros((len(normalized), max_rows), dtype=bool)
    for index, array in enumerate(normalized):
        rows = int(array.shape[0])
        if rows:
            padded[index, :rows, :] = array
            mask[index, :rows] = True
    return padded, mask
