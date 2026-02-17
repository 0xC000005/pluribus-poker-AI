"""Range-vs-range CFR+ turn solver for heads-up no-limit hold'em.

Same structure as river_solver.py but with equity-based leaf evaluation.
Terminal showdown nodes use expected value averaged over all possible
river cards instead of exact showdown.

Uses optimized iterative CFR with batched GEMM (fast_cfr module).
"""
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from poker_ai.poker.evaluation.eval_card import EvaluationCard
from poker_ai.poker.evaluation.evaluator import Evaluator
from fast_cfr import build_tree_arrays, solve_cfr, get_average_strategy

BIG_BLIND = 100
BET_FRACS = {2: 0.33, 3: 0.75, 4: 1.5}
ALLIN_ACTION = 5
MAX_ALLIN_OVERBET = 4.0  # only include all-in when ≤ 4× pot

_EVALUATOR = Evaluator()
_SUIT_CHARS = ['c', 'd', 'h', 's']
_RANK_CHARS = list(EvaluationCard.STR_RANKS)
_CARD_TO_EVAL = np.zeros(52, dtype=np.int32)
for _ci in range(52):
    _CARD_TO_EVAL[_ci] = EvaluationCard.new(
        _RANK_CHARS[_ci // 4] + _SUIT_CHARS[_ci % 4])


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


class TurnSolver:
    """Range-vs-range CFR+ for the turn with equity-based leaf evaluation."""

    def __init__(self, board_4, pot, hero_stack, villain_stack, hero_first,
                 active_hands=None):
        assert len(board_4) == 4, f"Turn solver needs exactly 4 board cards, got {len(board_4)}"
        self.board = board_4
        self.pot_start = pot
        self.hero_stack_start = hero_stack
        self.villain_stack_start = villain_stack
        self.hero_first = hero_first

        # Enumerate hands (optionally filtered).
        if active_hands is not None:
            self.hands = active_hands
        else:
            remaining = sorted(set(range(52)) - set(board_4))
            self.hands = list(itertools.combinations(remaining, 2))
        self.n = len(self.hands)
        self.hand_to_idx = {h: i for i, h in enumerate(self.hands)}

        # Card conflict matrix.
        ha = np.array(self.hands, dtype=np.int32)
        c0 = ha[:, np.newaxis, :]
        c1 = ha[np.newaxis, :, :]
        conflict = (c0[:, :, :, np.newaxis] == c1[:, :, np.newaxis, :]).any(axis=(2, 3))
        self.valid = (~conflict).astype(np.float32)

        # Precompute equity matrices (averaged over river runouts).
        self._compute_equity_matrices(board_4)

        # Build tree.
        first = 0 if hero_first else 1
        self.root = self._build_node(pot, hero_stack, villain_stack, 0, first, 0)

        # Flatten tree for fast CFR.
        self._tree = build_tree_arrays(self.root)

    def _compute_equity_matrices(self, board_4):
        """Precompute avg win/lose/tie over all river runouts."""
        n = self.n
        board_eval = [int(_CARD_TO_EVAL[c]) for c in board_4]
        river_cards = sorted(set(range(52)) - set(board_4))

        win_sum = np.zeros((n, n), dtype=np.float32)
        lose_sum = np.zeros((n, n), dtype=np.float32)
        tie_sum = np.zeros((n, n), dtype=np.float32)
        count = np.zeros((n, n), dtype=np.float32)

        # Precompute which hands contain each card.
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
        self.equity_win = win_sum / safe_count
        self.equity_lose = lose_sum / safe_count
        self.equity_tie = tie_sum / safe_count

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

        if tc > 0:
            ft = 'hero_fold' if player == 0 else 'villain_fold'
            node.children[0] = Node(-1, pot, (hs, vs), 0, nr, terminal_type=ft)

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

        if nr < 3 and acting > tc:
            for ai, frac in BET_FRACS.items():
                rb = max(int(frac * pot), BIG_BLIND)
                tot = tc + rb
                if tot > acting:
                    continue
                bp = pot + tot
                bh, bv = (hs - tot, vs) if player == 0 else (hs, vs - tot)
                node.children[ai] = self._build_node(bp, bh, bv, nr+1, 1-player, rb)

            ar = acting - tc
            if ar > 0 and pot > 0 and ar / pot <= MAX_ALLIN_OVERBET:
                ap = pot + acting
                ah, av = (0, vs) if player == 0 else (hs, 0)
                node.children[ALLIN_ACTION] = self._build_node(
                    ap, ah, av, nr+1, 1-player, ar)

        return node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, n_iterations=100, hero_range=None, villain_range=None):
        hr = hero_range.astype(np.float32) if hero_range is not None else None
        vr = villain_range.astype(np.float32) if villain_range is not None else None
        self._regret_sum, self._strategy_sum = solve_cfr(
            self._tree, self.n,
            self.equity_win, self.equity_lose, self.equity_tie, self.valid,
            self.pot_start, self.hero_stack_start, self.villain_stack_start,
            n_iterations=n_iterations,
            hero_range=hr, villain_range=vr,
        )

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
                continue  # skip fold and check/call
            d = abs((start - node.children[a].stacks[actor]) - slumbot_street_bet_to)
            if d < best_d:
                best_d = d
                best_a = a
        return best_a


# ---------------------------------------------------------------------------
# High-level interface for play_slumbot.py
# ---------------------------------------------------------------------------

def turn_solve(
    our_cards_idx, board_idx_4, pot_before_turn,
    hero_stack, villain_stack, hero_first,
    turn_action_str='', n_iterations=100,
    hero_range=None, villain_range=None,
    active_hands=None,
):
    """Solve turn and return (solver_action_idx, strategy_dict, solver, node)."""
    solver = TurnSolver(
        board_idx_4, pot_before_turn,
        hero_stack, villain_stack, hero_first,
        active_hands=active_hands,
    )
    solver.solve(n_iterations, hero_range=hero_range, villain_range=villain_range)

    nav = _parse_turn_nav(turn_action_str, solver)
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


def _parse_turn_nav(turn_str, solver):
    if not turn_str:
        return []
    nav = []
    node = solver.root
    i = 0
    while i < len(turn_str):
        if node is None or node.is_terminal:
            break
        c = turn_str[i]
        i += 1
        if c in ('k', 'c'):
            nav.append(1)
            node = node.children.get(1)
        elif c == 'f':
            nav.append(0)
            break
        elif c == 'b':
            j = i
            while i < len(turn_str) and turn_str[i].isdigit():
                i += 1
            sbt = int(turn_str[j:i])
            a = solver.find_closest_bet(node, sbt)
            if a is not None:
                nav.append(a)
                node = node.children.get(a)
            else:
                break
    return nav


def turn_solver_action_to_slumbot(action_idx, solver_node, solver, parsed):
    """Convert turn solver action to Slumbot action string."""
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
