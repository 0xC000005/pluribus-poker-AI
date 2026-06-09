"""HUNL subgame specification + canonical hand indexing (p1_design.json STEP 3).

Two index spaces:
  * GLOBAL: the canonical C(52,2) = 1326 hole-hand index. ``GLOBAL_HANDS`` is
    ``itertools.combinations(range(52), 2)`` order -- (c0, c1) with c0 < c1,
    lexicographic. Card index convention matches the repo (CLAUDE.md):
    ``card_idx = (rank - 2) * 4 + suit`` with suits c=0, d=1, h=2, s=3.
    Spec-level ranges (``SubgameSpec.r0/r1``) always live here, so beliefs can
    be carried across boards/streets without re-indexing.
  * LOCAL: the per-board hand set the kernels run at -- exactly
    ``scripts/solver.py``'s ``StreetSolver`` enumeration
    (``itertools.combinations(sorted(set(range(52)) - set(board)), 2)``):
    river H = C(47,2) = 1081, turn H = C(48,2) = 1128. The local width is
    uniform per street, so same-street populations batch without padding.

The 1326 x 1326 global card-conflict matrix is precomputed once; every
per-board ``valid`` matrix is a cheap gather of it at the board's local->global
indices (parity-gated against StreetSolver's own conflict computation in
test_hunl_population_parity.py gate P-D).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from poker_ai.rebel.hunl.tree_builder import (
    STREETS,
    build_street_tree,
    topology_key,
)

__all__ = [
    "N_CARDS",
    "N_GLOBAL_HANDS",
    "RIVER_H",
    "TURN_H",
    "STREET_BOARD_SIZES",
    "GLOBAL_HANDS",
    "SubgameSpec",
    "hand_index",
    "board_key",
    "local_hands",
    "local_to_global",
    "global_to_local",
    "global_conflict_matrix",
    "global_valid_matrix",
    "local_valid_matrix",
    "gather_local",
    "scatter_global",
]

N_CARDS = 52
N_GLOBAL_HANDS = 1326          # C(52, 2)
RIVER_H = 1081                 # C(47, 2): 5 board cards removed
TURN_H = 1128                  # C(48, 2): 4 board cards removed
STREET_BOARD_SIZES = {"flop": 3, "turn": 4, "river": 5}

GLOBAL_HANDS = tuple(itertools.combinations(range(N_CARDS), 2))
_GLOBAL_INDEX = {hand: idx for idx, hand in enumerate(GLOBAL_HANDS)}


def hand_index(c0, c1) -> int:
    """Global 1326-index of the hole pair {c0, c1} (order-insensitive)."""
    a, b = int(c0), int(c1)
    if a > b:
        a, b = b, a
    if a == b:
        raise ValueError(f"hand cards must differ, got {c0!r}, {c1!r}")
    try:
        return _GLOBAL_INDEX[(a, b)]
    except KeyError:
        raise ValueError(f"card indices must be in [0, 52), got {c0!r}, {c1!r}") from None


def board_key(board) -> tuple:
    """Canonical (sorted, validated) board tuple usable as a cache key."""
    cards = tuple(sorted(int(c) for c in board))
    if any(c < 0 or c >= N_CARDS for c in cards):
        raise ValueError(f"board cards must be in [0, 52): {board!r}")
    if len(set(cards)) != len(cards):
        raise ValueError(f"board cards must be distinct: {board!r}")
    return cards


@lru_cache(maxsize=1)
def global_conflict_matrix() -> np.ndarray:
    """[1326, 1326] bool: True where the two hands share at least one card."""
    hands = np.asarray(GLOBAL_HANDS, dtype=np.int32)  # [1326, 2]
    conflict = (
        hands[:, np.newaxis, :, np.newaxis] == hands[np.newaxis, :, np.newaxis, :]
    ).any(axis=(2, 3))
    return conflict


@lru_cache(maxsize=1)
def global_valid_matrix() -> np.ndarray:
    """[1326, 1326] float32 card-conflict mask (1 = valid pair, 0 = conflict).

    The global twin of StreetSolver's per-board ``valid``: gathering it at a
    board's local->global indices reproduces the per-board matrix exactly.
    """
    return (~global_conflict_matrix()).astype(np.float32)


@lru_cache(maxsize=4096)
def _local_hands_cached(key: tuple) -> tuple:
    remaining = sorted(set(range(N_CARDS)) - set(key))
    return tuple(itertools.combinations(remaining, 2))


def local_hands(board) -> list:
    """Per-board hand list in StreetSolver order (board cards removed)."""
    return list(_local_hands_cached(board_key(board)))


@lru_cache(maxsize=4096)
def _local_to_global_cached(key: tuple) -> np.ndarray:
    arr = np.asarray(
        [_GLOBAL_INDEX[hand] for hand in _local_hands_cached(key)],
        dtype=np.int32,
    )
    arr.setflags(write=False)
    return arr


def local_to_global(board) -> np.ndarray:
    """[H_local] int32: global index of each local hand (read-only view)."""
    return _local_to_global_cached(board_key(board))


@lru_cache(maxsize=4096)
def _global_to_local_cached(key: tuple) -> np.ndarray:
    l2g = _local_to_global_cached(key)
    arr = np.full(N_GLOBAL_HANDS, -1, dtype=np.int32)
    arr[l2g] = np.arange(l2g.shape[0], dtype=np.int32)
    arr.setflags(write=False)
    return arr


def global_to_local(board) -> np.ndarray:
    """[1326] int32: local index per global hand, -1 where the hand hits the board."""
    return _global_to_local_cached(board_key(board))


def local_valid_matrix(board) -> np.ndarray:
    """[H, H] float32 conflict mask for the board, gathered from the global matrix."""
    l2g = local_to_global(board)
    return np.ascontiguousarray(global_valid_matrix()[np.ix_(l2g, l2g)])


def gather_local(board, global_values) -> np.ndarray:
    """Gather a [1326] global per-hand vector down to the board's local hands."""
    values = np.asarray(global_values)
    if values.shape[-1] != N_GLOBAL_HANDS:
        raise ValueError(
            f"expected trailing dim {N_GLOBAL_HANDS}, got {values.shape}")
    return np.ascontiguousarray(values[..., local_to_global(board)])


def scatter_global(board, local_values, fill=0.0) -> np.ndarray:
    """Scatter a local per-hand vector back to the [1326] global index."""
    l2g = local_to_global(board)
    values = np.asarray(local_values, dtype=np.float64)
    if values.shape != (l2g.shape[0],):
        raise ValueError(
            f"expected shape ({l2g.shape[0]},) for board {board_key(board)}, "
            f"got {values.shape}")
    out = np.full(N_GLOBAL_HANDS, float(fill), dtype=np.float64)
    out[l2g] = values
    return out


@dataclass
class SubgameSpec:
    """One street-entry HUNL subgame: public state + both players' beliefs.

    ``r0``/``r1`` are float ranges over the CANONICAL GLOBAL 1326 hand index
    (entries for hands that hit ``board`` are simply ignored by the local
    gather). ``first_to_act`` is 0 if player 0 ("hero" in StreetSolver terms)
    opens the street, 1 otherwise; HU postflop the BB acts first.
    """

    street: str
    board: tuple
    pot: int
    stack0: int
    stack1: int
    first_to_act: int
    r0: np.ndarray
    r1: np.ndarray

    def __post_init__(self):
        self.street = str(self.street).lower()
        if self.street not in STREETS:
            raise ValueError(f"street must be one of {STREETS}, got {self.street!r}")
        self.board = tuple(int(c) for c in self.board)
        key = board_key(self.board)
        expected = STREET_BOARD_SIZES.get(self.street)
        if expected is not None and len(key) != expected:
            raise ValueError(
                f"{self.street} board must have {expected} cards, got {len(key)}")
        self.pot = int(self.pot)
        self.stack0 = int(self.stack0)
        self.stack1 = int(self.stack1)
        self.first_to_act = int(self.first_to_act)
        if self.first_to_act not in (0, 1):
            raise ValueError(f"first_to_act must be 0 or 1, got {self.first_to_act!r}")
        if self.pot < 0 or self.stack0 < 0 or self.stack1 < 0:
            raise ValueError(
                f"negative chips: pot={self.pot} stack0={self.stack0} stack1={self.stack1}")
        for name in ("r0", "r1"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (N_GLOBAL_HANDS,):
                raise ValueError(
                    f"{name} must have shape ({N_GLOBAL_HANDS},), got {arr.shape}")
            if not np.isfinite(arr).all() or (arr < 0).any():
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, arr)

    # -- index plumbing -----------------------------------------------------

    @property
    def n_hands(self) -> int:
        """Local hand-set width H (1081 river / 1128 turn)."""
        return local_to_global(self.board).shape[0]

    def local_hands(self) -> list:
        return local_hands(self.board)

    def local_ranges(self):
        """(r0_local, r1_local) float64 [H] in the board's local hand order."""
        return (gather_local(self.board, self.r0).astype(np.float64),
                gather_local(self.board, self.r1).astype(np.float64))

    def local_valid(self) -> np.ndarray:
        return local_valid_matrix(self.board)

    # -- tree plumbing ------------------------------------------------------

    def tree(self) -> dict:
        """Lazily built (LRU-shared, treat read-only) street-entry betting tree."""
        return build_street_tree(
            self.pot, self.stack0, self.stack1, self.first_to_act, self.street)

    def topology_key(self) -> str:
        return topology_key(self.tree())

    def bucket_key(self) -> tuple:
        """Fused-batch grouping key: exact tree shape + local hand width."""
        return (self.topology_key(), self.street, self.n_hands)
