"""Lazy street-local HUNL betting-tree construction (p1_design.json STEP 1).

Reuses the trusted abstraction in ``scripts/solver.py`` (``StreetSolver._build_node``:
6 pot-fraction raise buckets + all-in, 3-raise cap, min-raise legality, all-in
truncation, first-to-act check branching) WITHOUT constructing a full solver:
``StreetSolver.__new__`` + setting ``hero_first`` -- the only instance attribute
``_build_node`` reads -- yields the exact same tree shape with no board, no hand
enumeration, and no terminal matrices. The flat tree arrays come from the equally
trusted ``scripts/fast_cfr.py::build_tree_arrays``.

Key facts this module encodes (from the design's bucketing scheme):
  * The BOARD never affects tree shape; only (pot, stacks, first_to_act) do,
    through action legality. ``street`` is part of the cache key only.
  * Two subgames are batchable in one fused same-topology GPU solve iff their
    ``fast_cfr._same_topology_signature`` tuples match exactly; ``topology_key``
    is a stable blake2b digest of that signature.

Both ``scripts/solver.py`` and ``scripts/fast_cfr.py`` are protected surfaces:
they are imported and reused here, never edited.
"""
from __future__ import annotations

import functools
import hashlib
import os
import sys

import numpy as np

# scripts/ holds solver.py + fast_cfr.py and they import each other by bare name
# (same pattern as poker_ai/rebel/turn_river.py).
_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "scripts",
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import fast_cfr  # noqa: E402  (protected surface; imported, never edited)
import solver as _solver  # noqa: E402  (protected surface; imported, never edited)

__all__ = [
    "STREETS",
    "build_betting_tree",
    "build_street_tree",
    "topology_key",
    "street_tree_cache_info",
    "street_tree_cache_clear",
    "fast_cfr",
]

STREETS = ("preflop", "flop", "turn", "river")

_TOPOLOGY_KEY_VERSION = b"hunl-topology-v1"


def _bare_street_solver(hero_first: bool) -> "_solver.StreetSolver":
    """Instantiate StreetSolver WITHOUT running __init__.

    ``StreetSolver._build_node`` reads exactly one instance attribute:
    ``self.hero_first`` (for the first-to-act check-branching rule). No board,
    hand set, or terminal matrix is touched, so a bare instance reproduces the
    tree SHAPE of a fully constructed solver at zero cost.
    """
    shell = _solver.StreetSolver.__new__(_solver.StreetSolver)
    shell.hero_first = bool(hero_first)
    return shell


def build_betting_tree(pot, s0, s1, first_to_act, n_raises=0, to_call=0):
    """Board-free betting tree from the trusted StreetSolver chip arithmetic.

    Args:
        pot: chips already in the pot at this node.
        s0: player-0 (``hero`` in StreetSolver terms) remaining stack.
        s1: player-1 (``villain``) remaining stack.
        first_to_act: 0 if player 0 opens the street, 1 if player 1 does.
        n_raises: raises already consumed this street (root of a fresh
            street is 0; the preflop trunk uses this to model the blinds).
        to_call: chips the first actor must call (0 at a fresh postflop
            street; 50 for the preflop small blind).

    Returns:
        The flat tree-array dict from ``fast_cfr.build_tree_arrays``.
    """
    pot, s0, s1 = int(pot), int(s0), int(s1)
    n_raises, to_call = int(n_raises), int(to_call)
    if first_to_act not in (0, 1):
        raise ValueError(f"first_to_act must be 0 or 1, got {first_to_act!r}")
    if pot < 0 or s0 < 0 or s1 < 0 or to_call < 0 or n_raises < 0:
        raise ValueError(
            f"negative chip arithmetic: pot={pot} s0={s0} s1={s1} "
            f"to_call={to_call} n_raises={n_raises}"
        )
    shell = _bare_street_solver(hero_first=(first_to_act == 0))
    root = shell._build_node(pot, s0, s1, n_raises, int(first_to_act), to_call)
    return fast_cfr.build_tree_arrays(root)


@functools.lru_cache(maxsize=8192)
def _cached_street_tree(pot: int, s0: int, s1: int, first_to_act: int, street: str):
    return build_betting_tree(pot, s0, s1, first_to_act, n_raises=0, to_call=0)


def build_street_tree(pot, s0, s1, first_to_act, street):
    """LRU-cached street-entry betting tree (fresh street: no raises, no to_call).

    ``street`` does NOT affect tree shape (shape is board-free and street-free);
    it is part of the cache key so river/turn populations stay distinguishable
    downstream. The returned dict is SHARED across callers: treat it as
    read-only (the one sanctioned mutation is fast_cfr's own
    ``setdefault('_level_edge_groups', ...)`` memoization).
    """
    street = str(street).lower()
    if street not in STREETS:
        raise ValueError(f"street must be one of {STREETS}, got {street!r}")
    return _cached_street_tree(int(pot), int(s0), int(s1), int(first_to_act), street)


def street_tree_cache_info():
    return _cached_street_tree.cache_info()


def street_tree_cache_clear():
    _cached_street_tree.cache_clear()


def topology_key(tree) -> str:
    """Stable blake2b digest over ``fast_cfr._same_topology_signature(tree)``.

    Two trees may be fused into one batched same-topology GPU solve
    (``fast_cfr.solve_cfr_levelsync_torch_batched_same_topology``) iff their
    keys are equal -- the digest covers exactly the arrays that
    ``fast_cfr._validate_same_topology_batch`` compares:
    (n_nodes, n_actions, player[], parent_idx[], children[], terminal_type[]).
    Per-node pots/stacks are deliberately EXCLUDED (the batched kernel takes
    them per element), so members of one bucket may differ in board, pot,
    stacks, and ranges.
    """
    signature = fast_cfr._same_topology_signature(tree)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_TOPOLOGY_KEY_VERSION)
    digest.update(int(signature[0]).to_bytes(8, "little", signed=True))
    digest.update(int(signature[1]).to_bytes(8, "little", signed=True))
    for array in signature[2:]:
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.int32))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()
