"""Unified range-vs-range CFR+ solver for heads-up no-limit hold'em.

Handles both turn (equity-based leaf evaluation) and river (exact showdown).
Uses the same 9-action abstraction as training for consistency:
  0=fold, 1=check/call, 2=0.25x pot, 3=0.5x pot, 4=0.75x pot,
  5=1.0x pot, 6=1.5x pot, 7=2.0x pot, 8=all-in
"""
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from poker_ai.poker.evaluation.eval_card import EvaluationCard
from poker_ai.poker.evaluation.evaluator import Evaluator
from fast_cfr import (
    build_tree_arrays,
    get_average_strategy,
    prune_hands,
    solve_cfr,
    solve_cfr_torch,
)

BIG_BLIND = 100
# Use full training RAISE_FRACTIONS mapping for action indices 2..7.
BET_FRACS = {2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0, 6: 1.5, 7: 2.0}
ALLIN_ACTION = 8

_EVALUATOR = Evaluator()
_SUIT_CHARS = ['c', 'd', 'h', 's']
_RANK_CHARS = list(EvaluationCard.STR_RANKS)
_CARD_TO_EVAL = np.zeros(52, dtype=np.int32)
for _ci in range(52):
    _CARD_TO_EVAL[_ci] = EvaluationCard.new(
        _RANK_CHARS[_ci // 4] + _SUIT_CHARS[_ci % 4])


def resolve_solver_backend(backend='auto', device=None):
    """Resolve public backend names to the concrete CFR implementation."""
    if backend == 'auto':
        # The current torch-CUDA backend is experimental and often slower than
        # NumPy because the CFR tree recurrence is still Python-driven. Keep
        # auto on the measured-fast reference backend until the solver is fused.
        return 'cpu', None
    if backend == 'torch-cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("torch-cuda solver backend requested but CUDA is unavailable.")
        return 'torch', 'cuda'
    if backend == 'torch-cpu':
        return 'torch', 'cpu'
    if backend in ('cpu', 'torch'):
        return backend, device
    raise ValueError(f"Unknown solver backend: {backend}")


@dataclass
class Node:
    player: int       # 0=hero, 1=villain, -1=terminal
    pot: int
    stacks: tuple     # (hero, villain)
    to_call: int
    n_raises: int
    terminal_type: str = ''
    children: dict = field(default_factory=dict)

    @property
    def is_terminal(self):
        return self.player == -1


class StreetSolver:
    """Range-vs-range CFR+ solver for a single street."""

    def __init__(self, board, pot, hero_stack, villain_stack, hero_first,
                 active_indices=None):
        self.board = board
        self.pot_start = pot
        self.hero_stack_start = hero_stack
        self.villain_stack_start = villain_stack
        self.hero_first = hero_first

        # Enumerate possible hands (excluding board cards).
        remaining = sorted(set(range(52)) - set(board))
        all_hands = list(itertools.combinations(remaining, 2))
        self.full_n = len(all_hands)
        if active_indices is None:
            self.hands = all_hands
        else:
            self.hands = [all_hands[int(i)] for i in active_indices]
        self.n = len(self.hands)
        self.hand_to_idx = {h: i for i, h in enumerate(self.hands)}

        # Card conflict matrix.
        ha = np.array(self.hands, dtype=np.int32)
        c0 = ha[:, np.newaxis, :]
        c1 = ha[np.newaxis, :, :]
        conflict = (c0[:, :, :, np.newaxis] == c1[:, :, np.newaxis, :]).any(axis=(2, 3))
        self.valid = (~conflict).astype(np.float32)

        # Compute terminal evaluation matrices based on street.
        if len(board) == 5:
            self._compute_showdown_matrices(board)
        elif len(board) == 4:
            self._compute_equity_matrices(board)
        else:
            raise ValueError(f"Expected 4 or 5 board cards, got {len(board)}")

        # Build tree.
        first = 0 if hero_first else 1
        self.root = self._build_node(pot, hero_stack, villain_stack, 0, first, 0)
        self._tree = build_tree_arrays(self.root)

    # ------------------------------------------------------------------
    # Terminal evaluation matrices
    # ------------------------------------------------------------------

    def _compute_showdown_matrices(self, board):
        """Exact river showdown — ground truth payoffs."""
        board_eval = [int(_CARD_TO_EVAL[c]) for c in board]
        ranks = np.zeros(self.n, dtype=np.int32)
        for i, (c1, c2) in enumerate(self.hands):
            ranks[i] = _EVALUATOR.evaluate(
                [int(_CARD_TO_EVAL[c1]), int(_CARD_TO_EVAL[c2])], board_eval)

        ri = ranks[:, np.newaxis]
        rj = ranks[np.newaxis, :]
        result = np.sign(rj - ri).astype(np.float32)

        self.win_m = (result > 0) * self.valid
        self.lose_m = (result < 0) * self.valid
        self.tie_m = np.maximum(
            (result == 0) * self.valid - np.eye(self.n, dtype=np.float32), 0)

    def _compute_equity_matrices(self, board_4):
        """Turn equity: win/lose/tie averaged over all river runouts."""
        n = self.n
        board_eval = [int(_CARD_TO_EVAL[c]) for c in board_4]
        river_cards = sorted(set(range(52)) - set(board_4))

        win_sum = np.zeros((n, n), dtype=np.float32)
        lose_sum = np.zeros((n, n), dtype=np.float32)
        tie_sum = np.zeros((n, n), dtype=np.float32)
        count = np.zeros((n, n), dtype=np.float32)

        # Precompute which hands contain each possible river card.
        hand_contains = {}
        for card in river_cards:
            mask = np.ones(n, dtype=np.float32)
            for i, hand in enumerate(self.hands):
                if card in hand:
                    mask[i] = 0.0
            hand_contains[card] = mask

        for river_card in river_cards:
            river_eval = int(_CARD_TO_EVAL[river_card])
            full_board = board_eval + [river_eval]

            ranks = np.zeros(n, dtype=np.int32)
            for i, (c1, c2) in enumerate(self.hands):
                if river_card in (c1, c2):
                    ranks[i] = 999999
                else:
                    ranks[i] = _EVALUATOR.evaluate(
                        [int(_CARD_TO_EVAL[c1]), int(_CARD_TO_EVAL[c2])],
                        full_board)

            ri = ranks[:, np.newaxis]
            rj = ranks[np.newaxis, :]
            result = np.sign(rj - ri).astype(np.float32)

            rc_mask = hand_contains[river_card]
            river_valid = self.valid * rc_mask[:, np.newaxis] * rc_mask[np.newaxis, :]

            win_sum += (result > 0) * river_valid
            lose_sum += (result < 0) * river_valid
            tie_rv = (result == 0) * river_valid
            np.fill_diagonal(tie_rv, 0)
            tie_sum += tie_rv
            count += river_valid

        safe_count = np.maximum(count, 1)
        self.win_m = win_sum / safe_count
        self.lose_m = lose_sum / safe_count
        self.tie_m = tie_sum / safe_count

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _build_node(self, pot, hs, vs, nr, player, tc):
        if hs == 0 and vs == 0:
            return Node(-1, pot, (hs, vs), 0, nr, terminal_type='showdown')

        acting = hs if player == 0 else vs
        node = Node(player, pot, (hs, vs), tc, nr)

        if acting == 0:
            if tc > 0:
                node.player = -1; node.terminal_type = 'showdown'; return node
            first = 0 if self.hero_first else 1
            if player != first:
                node.player = -1; node.terminal_type = 'showdown'; return node
            node.children[1] = self._build_node(pot, hs, vs, nr, 1-player, 0)
            return node

        # Fold (only if facing a bet).
        if tc > 0:
            ft = 'hero_fold' if player == 0 else 'villain_fold'
            node.children[0] = Node(-1, pot, (hs, vs), 0, nr, terminal_type=ft)

        # Check/Call.
        ac = min(tc, acting)
        cp = pot + ac
        ch, cv = (hs - ac, vs) if player == 0 else (hs, vs - ac)
        if tc > 0:
            node.children[1] = Node(-1, cp, (ch, cv), 0, nr, terminal_type='showdown')
        else:
            first = 0 if self.hero_first else 1
            if player != first:
                node.children[1] = Node(-1, cp, (ch, cv), 0, nr, terminal_type='showdown')
            else:
                node.children[1] = self._build_node(cp, ch, cv, nr, 1-player, 0)

        # Bets/raises.
        if nr < 3 and acting > tc:
            for ai, frac in BET_FRACS.items():
                rb = max(int(frac * pot), BIG_BLIND)
                tot = tc + rb
                if tot > acting:
                    continue
                bp = pot + tot
                bh, bv = (hs - tot, vs) if player == 0 else (hs, vs - tot)
                node.children[ai] = self._build_node(
                    bp, bh, bv, nr+1, 1-player, rb)

            # All-in.
            ar = acting - tc
            if ar > 0:
                ap = pot + acting
                ah, av = (0, vs) if player == 0 else (hs, 0)
                node.children[ALLIN_ACTION] = self._build_node(
                    ap, ah, av, nr+1, 1-player, ar)

        return node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, n_iterations=100, hero_range=None, villain_range=None,
              backend='cpu', device=None):
        hr = hero_range.astype(np.float32) if hero_range is not None else None
        vr = villain_range.astype(np.float32) if villain_range is not None else None
        backend, device = resolve_solver_backend(backend, device)
        if backend == 'cpu':
            solver_fn = solve_cfr
            kwargs = {}
        elif backend == 'torch':
            solver_fn = solve_cfr_torch
            kwargs = {'device': device or 'cuda'}
        else:
            raise ValueError(f"Unknown solver backend: {backend}")
        started = time.perf_counter()
        self._regret_sum, self._strategy_sum = solver_fn(
            self._tree, self.n,
            self.win_m, self.lose_m, self.tie_m, self.valid,
            self.pot_start, self.hero_stack_start, self.villain_stack_start,
            n_iterations=n_iterations,
            hero_range=hr, villain_range=vr,
            **kwargs,
        )
        self.last_solve_ms = (time.perf_counter() - started) * 1000.0

    def get_strategy(self, hand, node=None):
        if node is None:
            node = self.root
        if node.is_terminal:
            return {}
        h = tuple(sorted(hand))
        idx = self.hand_to_idx.get(h)
        if idx is None:
            return {}
        node_idx = self._tree['all_nodes'].index(node)
        actions = sorted(node.children.keys())
        return get_average_strategy(self._strategy_sum, node_idx, actions, idx)

    def navigate(self, action_seq):
        node = self.root
        for a in action_seq:
            if node.is_terminal or a not in node.children:
                return None
            node = node.children[a]
        return node

    def find_closest_bet(self, node, slumbot_street_bet_to):
        actor = node.player
        start = self.hero_stack_start if actor == 0 else self.villain_stack_start
        best_a, best_d = None, float('inf')
        for a in sorted(node.children.keys()):
            if a <= 1:
                continue
            d = abs((start - node.children[a].stacks[actor]) - slumbot_street_bet_to)
            if d < best_d:
                best_d = d
                best_a = a
        return best_a


# ---------------------------------------------------------------------------
# High-level interface for play_slumbot.py
# ---------------------------------------------------------------------------

def solve_street(
    our_cards_idx, board_idx, pot, hero_stack, villain_stack, hero_first,
    action_str='', n_iterations=100, hero_range=None, villain_range=None,
    backend='cpu', device=None, range_prune_threshold=0.0,
):
    """Solve a street (turn or river) and return action.

    Returns (solver_action_idx, strategy_dict, solver, node).
    """
    active_indices = None
    if range_prune_threshold > 0 and (hero_range is not None or villain_range is not None):
        remaining = sorted(set(range(52)) - set(board_idx))
        full_hands = list(itertools.combinations(remaining, 2))
        _, active_indices, hero_range, villain_range = prune_hands(
            full_hands,
            hero_range=hero_range,
            villain_range=villain_range,
            keep_hand=tuple(sorted(our_cards_idx)),
            threshold=range_prune_threshold,
        )

    solver = StreetSolver(
        board_idx, pot, hero_stack, villain_stack, hero_first,
        active_indices=active_indices)
    solver.solve(n_iterations, hero_range=hero_range, villain_range=villain_range,
                 backend=backend, device=device)

    nav = _parse_nav(action_str, solver)
    node = solver.navigate(nav)
    our_hand = tuple(sorted(our_cards_idx))

    if node is None or node.is_terminal:
        return 1, {1: 1.0}, solver, None

    strategy = solver.get_strategy(our_hand, node)
    if not strategy:
        return 1, {1: 1.0}, solver, node

    actions = list(strategy.keys())
    probs = np.array([strategy[a] for a in actions], dtype=np.float64)
    probs = np.maximum(probs, 0)
    total = probs.sum()
    if total <= 0:
        return 1, {1: 1.0}, solver, node
    probs /= total
    action = int(np.random.choice(actions, p=probs))
    return action, strategy, solver, node


def _parse_nav(action_str, solver):
    """Parse a Slumbot action string into solver navigation indices."""
    if not action_str:
        return []
    nav = []
    node = solver.root
    i = 0
    while i < len(action_str):
        if node is None or node.is_terminal:
            break
        c = action_str[i]
        i += 1
        if c in ('k', 'c'):
            nav.append(1)
            node = node.children.get(1)
        elif c == 'f':
            nav.append(0)
            break
        elif c == 'b':
            j = i
            while i < len(action_str) and action_str[i].isdigit():
                i += 1
            sbt = int(action_str[j:i])
            a = solver.find_closest_bet(node, sbt)
            if a is not None:
                nav.append(a)
                node = node.children.get(a)
            else:
                break
    return nav


def solver_action_to_slumbot(action_idx, solver_node, solver, parsed):
    """Convert solver action index to Slumbot action string."""
    if action_idx == 0:
        return 'f'
    if action_idx == 1:
        return 'c' if solver_node.to_call > 0 else 'k'

    child = solver_node.children[action_idx]
    hero_street_bet = solver.hero_stack_start - child.stacks[0]

    min_rb = max(parsed.get('last_bet_size', 0), BIG_BLIND)
    min_to = parsed['street_last_bet_to'] + min_rb
    hero_street_bet = max(hero_street_bet, min_to)
    hero_street_bet = min(hero_street_bet, solver.hero_stack_start)

    if hero_street_bet <= parsed['street_last_bet_to']:
        return 'c'
    return f'b{hero_street_bet}'
