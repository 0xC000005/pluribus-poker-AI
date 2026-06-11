"""Suit-isomorphism canonicalization for HUNL subgames (river PBS net v1).

Relabeling the four suits by any bijection is a LOSSLESS symmetry of the
game (P0b bundle: exact symmetry, not abstraction): card strength is
suit-invariant, so permuting suits maps boards to boards, the canonical
C(52,2)=1326 hand index to itself, and exact subgame solutions to exact
subgame solutions, while pots, stacks and actor order are untouched.

This module provides the canonical-suit mapping the v1 net consumes:

  * ``canonicalize(board, r0, r1) -> (canon_board, perm)`` picks, among the 24
    suit permutations, the one mapping ``board`` to its lexicographically
    minimal relabeling; residual ties (boards with a nontrivial suit
    stabilizer, e.g. monotone boards or absent suits) are broken
    DETERMINISTICALLY on the permuted range bytes so the canonical FORM
    (board, r0, r1) is a fixed point of the map (idempotency, exact-tested in
    test_hunl_suit_iso.py).
  * ``permute_global(values, perm)`` applies the induced 1326-hand
    permutation to range / CFV / weight vectors -- a pure index permutation,
    bit-exact and bit-exactly invertible via ``invert_perm``.
  * ``canonicalize_spec(spec)`` lifts the map to :class:`SubgameSpec`.

Inference contract (the v1 net): canonicalize the input, run the net in
canonical space, then map the predicted CFVs back with
``permute_global(y, invert_perm(perm))``. Targets generated in canonical
space are exactly the relabeled targets of the raw spec (soundness gate:
suit-isomorphic subgames solved independently agree in canonical space).

Card index convention (CLAUDE.md): ``card_idx = (rank - 2) * 4 + suit`` with
suits c=0, d=1, h=2, s=3 -- a suit perm acts on ``card_idx % 4`` only.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

from poker_ai.rebel.hunl import subgame_spec as sgs

__all__ = [
    "SUIT_PERMS",
    "IDENTITY_PERM",
    "invert_perm",
    "permute_card",
    "permute_board",
    "hand_perm",
    "permute_global",
    "canonical_board_and_perms",
    "canonical_board",
    "canonicalize",
    "canonicalize_spec",
]

N_SUITS = 4
SUIT_PERMS = tuple(itertools.permutations(range(N_SUITS)))  # all 24, lex order
IDENTITY_PERM = (0, 1, 2, 3)


def invert_perm(perm) -> tuple:
    """Inverse suit permutation: ``invert_perm(p)[p[s]] == s``."""
    perm = tuple(perm)
    inv = [0] * N_SUITS
    for old_suit, new_suit in enumerate(perm):
        inv[new_suit] = old_suit
    return tuple(inv)


def permute_card(card, perm) -> int:
    """Relabel one card's suit: rank invariant, ``suit -> perm[suit]``."""
    card = int(card)
    return (card // N_SUITS) * N_SUITS + perm[card % N_SUITS]


def permute_board(board, perm) -> tuple:
    """Suit-relabeled board as a canonical sorted tuple."""
    return tuple(sorted(permute_card(c, perm) for c in board))


@lru_cache(maxsize=32)
def _hand_perm_cached(perm: tuple) -> np.ndarray:
    out = np.empty(sgs.N_GLOBAL_HANDS, dtype=np.int32)
    for i, (c0, c1) in enumerate(sgs.GLOBAL_HANDS):
        out[i] = sgs.hand_index(permute_card(c0, perm), permute_card(c1, perm))
    out.setflags(write=False)
    return out


def hand_perm(perm) -> np.ndarray:
    """[1326] int32 (read-only): new global index of hand i under ``perm``."""
    return _hand_perm_cached(tuple(perm))


def permute_global(values, perm) -> np.ndarray:
    """Apply the induced hand permutation to a [..., 1326] vector.

    Pure index permutation (``out[..., hand_perm[i]] = values[..., i]``):
    bit-exact, dtype-preserving, inverted exactly by
    ``permute_global(out, invert_perm(perm))``.
    """
    values = np.asarray(values)
    if values.shape[-1] != sgs.N_GLOBAL_HANDS:
        raise ValueError(
            f"expected trailing dim {sgs.N_GLOBAL_HANDS}, got {values.shape}")
    out = np.empty_like(values)
    out[..., hand_perm(perm)] = values
    return out


@lru_cache(maxsize=65536)
def _canonical_board_and_perms_cached(key: tuple):
    best = None
    best_perms = []
    for perm in SUIT_PERMS:
        candidate = permute_board(key, perm)
        if best is None or candidate < best:
            best = candidate
            best_perms = [perm]
        elif candidate == best:
            best_perms.append(perm)
    return best, tuple(best_perms)


def canonical_board_and_perms(board):
    """(canonical board, all perms achieving it). The canonical board is the
    lexicographic minimum of the board's 24-element suit orbit; the tying
    perms form a coset of the canonical board's suit stabilizer."""
    return _canonical_board_and_perms_cached(sgs.board_key(board))


def canonical_board(board) -> tuple:
    return canonical_board_and_perms(board)[0]


def canonicalize(board, r0=None, r1=None):
    """Canonical suit relabeling of one (board[, ranges]) input.

    Returns ``(canon_board, perm)`` with ``permute_board(board, perm) ==
    canon_board``. When several perms reach the canonical board (suit-
    symmetric boards), ties are broken deterministically on the BYTES of the
    permuted ``r0`` then ``r1`` (a value-based total order, so the canonical
    form is idempotent); any remaining tie means the tying perms produce
    bit-identical canonical ranges, and the lexicographically smallest perm
    is returned.
    """
    canon, perms = canonical_board_and_perms(board)
    if len(perms) == 1 or (r0 is None and r1 is None):
        return canon, perms[0]
    best_key = None
    best_perm = None
    for perm in perms:
        key = tuple(
            permute_global(np.asarray(r), perm).tobytes()
            for r in (r0, r1) if r is not None)
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm
    return canon, best_perm


def canonicalize_spec(spec):
    """(canonical SubgameSpec, perm) for one spec; lossless relabeling.

    Pot, stacks and ``first_to_act`` are suit-invariant and carried through
    unchanged, so the betting-tree topology (and the population solver's
    bucketing) is identical for the canonical spec.
    """
    canon_board_, perm = canonicalize(spec.board, spec.r0, spec.r1)
    if perm == IDENTITY_PERM:
        return spec, perm
    canon = sgs.SubgameSpec(
        street=spec.street,
        board=canon_board_,
        pot=spec.pot,
        stack0=spec.stack0,
        stack1=spec.stack1,
        first_to_act=spec.first_to_act,
        r0=permute_global(spec.r0, perm),
        r1=permute_global(spec.r1, perm),
    )
    return canon, perm
