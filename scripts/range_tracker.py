"""Bayesian range tracker for heads-up no-limit hold'em.

Tracks both players' hand distributions throughout a hand using the
blueprint strategy (value network + regret matching) as a prior for
Bayesian updates. When the opponent takes an action, their range is
narrowed by multiplying by P(action | hand) from the blueprint.

The hero range tracks what the opponent *believes* about our hand
distribution — needed so subgame solvers don't degenerate to the
1-hand-vs-range problem.

Usage:
    tracker = RangeTracker(our_cards_idx, value_net, device)
    tracker.update_board([flop1, flop2, flop3])
    tracker.update_opponent_action(action_idx, action_str, opp_pos, parsed)
    tracker.update_hero_action(action_idx, action_str, hero_pos, parsed)
    # Get ranges for solver
    opp_range = tracker.opponent_range
    hero_range = tracker.hero_range
"""
import itertools

import numpy as np
import torch

# Inline constants (same as play_slumbot.py).
N_FEATURES = 126
N_ACTIONS = 9
RAISE_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000
NUM_STREETS = 4


def regret_match(advantages, legal_mask):
    """Convert advantages to strategy via regret matching."""
    positive = np.maximum(advantages, 0) * legal_mask
    total = positive.sum()
    if total > 0:
        return positive / total
    return legal_mask / legal_mask.sum()


# ---------------------------------------------------------------------------
# Helpers ported from play_slumbot.py to avoid circular imports.
# ---------------------------------------------------------------------------

def _compute_bets(action_str, client_pos):
    """Compute total chips bet by each player. Returns (our_total, opp_total)."""
    our_total = SMALL_BLIND if client_pos == 1 else BIG_BLIND
    opp_total = SMALL_BLIND if client_pos == 0 else BIG_BLIND
    our_street_bet = our_total
    opp_street_bet = opp_total
    acting_pos = 1  # SB first preflop
    i = 0
    while i < len(action_str):
        c = action_str[i]
        i += 1
        if c == '/':
            our_street_bet = 0
            opp_street_bet = 0
            acting_pos = 0
            continue
        if c == 'k':
            acting_pos = 1 - acting_pos
        elif c == 'c':
            if acting_pos == client_pos:
                diff = opp_street_bet - our_street_bet
                our_total += diff
                our_street_bet = opp_street_bet
            else:
                diff = our_street_bet - opp_street_bet
                opp_total += diff
                opp_street_bet = our_street_bet
            acting_pos = 1 - acting_pos
        elif c == 'f':
            break
        elif c == 'b':
            j = i
            while i < len(action_str) and action_str[i].isdigit():
                i += 1
            new_sb = int(action_str[j:i])
            if acting_pos == client_pos:
                our_total += new_sb - our_street_bet
                our_street_bet = new_sb
            else:
                opp_total += new_sb - opp_street_bet
                opp_street_bet = new_sb
            acting_pos = 1 - acting_pos
    return our_total, opp_total


def _count_raises_current_street(action_str):
    streets = action_str.split('/')
    current = streets[-1] if streets else ''
    return current.count('b')


def _get_our_street_bet(current_street_actions, client_pos, st):
    if st == 0:
        bets = [BIG_BLIND, SMALL_BLIND]
    else:
        bets = [0, 0]
    acting_pos = 1 if st == 0 else 0
    i = 0
    while i < len(current_street_actions):
        c = current_street_actions[i]
        i += 1
        if c == 'k':
            acting_pos = 1 - acting_pos
        elif c == 'c':
            bets[acting_pos] = bets[1 - acting_pos]
            acting_pos = 1 - acting_pos
        elif c == 'b':
            j = i
            while i < len(current_street_actions) and current_street_actions[i].isdigit():
                i += 1
            bets[acting_pos] = int(current_street_actions[j:i])
            acting_pos = 1 - acting_pos
        elif c == 'f':
            break
    return bets[client_pos]


def _get_legal_mask(parsed, action_str, client_pos):
    """Build a (9,) legal mask from parsed state."""
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    our_bet, opp_bet = _compute_bets(action_str, client_pos)
    pot_total = our_bet + opp_bet
    our_chips = STACK_SIZE - our_bet

    if parsed['last_bet_size'] > 0:
        mask[0] = 1.0
    mask[1] = 1.0

    n_raises = _count_raises_current_street(action_str)
    if n_raises < 3 and our_chips > 0:
        streets = action_str.split('/')
        current_street = streets[-1] if streets else ''
        our_sb = _get_our_street_bet(current_street, client_pos, parsed['st'])
        to_call = parsed['street_last_bet_to'] - our_sb

        for fi, frac in enumerate(RAISE_FRACTIONS):
            raise_amount = int(frac * pot_total) + to_call
            if raise_amount >= BIG_BLIND and raise_amount <= our_chips:
                mask[2 + fi] = 1.0

        if our_chips > 0:
            mask[8] = 1.0

    if parsed['last_bet_size'] == 0:
        mask[0] = 0.0

    return mask


def _build_features_batch(hands, board_idx, action_str, client_pos, parsed):
    """Build (n_hands, 126) feature array.

    Only hole cards differ per hand; everything else is shared.
    """
    n = len(hands)
    features = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Board (same for all).
    for c in board_idx:
        features[:, 52 + c] = 1.0

    # Street one-hot (same for all).
    st = parsed['st']
    if st < 4:
        features[:, 104 + st] = 1.0

    # Scalars (same for all).
    our_bet, opp_bet = _compute_bets(action_str, client_pos)
    our_chips = STACK_SIZE - our_bet
    pot_total = our_bet + opp_bet
    total_chips_all = STACK_SIZE * 2

    features[:, 108] = pot_total / total_chips_all
    features[:, 109] = our_chips / STACK_SIZE
    features[:, 110] = our_bet / STACK_SIZE
    features[:, 111] = 1.0
    our_player_idx = 1 - client_pos
    features[:, 112] = our_player_idx / 1.0
    features[:, 113] = _count_raises_current_street(action_str) / 3.0

    # Action history (same for all).
    streets = action_str.split('/')
    for rd in range(4):
        if rd >= len(streets):
            break
        s = streets[rd]
        n_calls = s.count('c') + s.count('k')
        n_raises = s.count('b')
        n_folds = s.count('f')
        offset = 114 + rd * 3
        features[:, offset] = n_calls / 2.0
        features[:, offset + 1] = n_raises / 3.0
        features[:, offset + 2] = n_folds / 2.0

    # Hole cards (different per hand).
    for i, (c1, c2) in enumerate(hands):
        features[i, c1] = 1.0
        features[i, c2] = 1.0

    return features


def _parse_action(action):
    """Minimal re-implementation of parse_action from play_slumbot.py."""
    st = 0
    street_last_bet_to = BIG_BLIND
    total_last_bet_to = BIG_BLIND
    last_bet_size = BIG_BLIND - SMALL_BLIND
    last_bettor = 0
    pos = 1

    if not action:
        return {
            'st': st, 'pos': pos,
            'street_last_bet_to': street_last_bet_to,
            'total_last_bet_to': total_last_bet_to,
            'last_bet_size': last_bet_size,
            'last_bettor': last_bettor,
        }

    check_or_call_ends_street = False
    i = 0
    while i < len(action):
        if st >= NUM_STREETS:
            return {'error': 'Unexpected'}
        c = action[i]
        i += 1

        if c == 'k':
            if check_or_call_ends_street:
                if st < NUM_STREETS - 1 and i < len(action):
                    if action[i] != '/':
                        return {'error': 'Missing slash'}
                    i += 1
                if st == NUM_STREETS - 1:
                    pos = -1
                else:
                    pos = 0
                    st += 1
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            last_bet_size = 0
            last_bettor = -1

        elif c == 'c':
            if total_last_bet_to == STACK_SIZE:
                for st1 in range(st, NUM_STREETS - 1):
                    if i < len(action):
                        if action[i] == '/':
                            i += 1
                st = NUM_STREETS - 1
                pos = -1
                last_bet_size = 0
                return {
                    'st': st, 'pos': pos,
                    'street_last_bet_to': street_last_bet_to,
                    'total_last_bet_to': total_last_bet_to,
                    'last_bet_size': last_bet_size,
                    'last_bettor': last_bettor,
                }
            if check_or_call_ends_street:
                if st < NUM_STREETS - 1 and i < len(action):
                    if action[i] != '/':
                        return {'error': 'Missing slash'}
                    i += 1
                if st == NUM_STREETS - 1:
                    pos = -1
                else:
                    pos = 0
                    st += 1
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            last_bet_size = 0
            last_bettor = -1

        elif c == 'f':
            pos = -1
            return {
                'st': st, 'pos': pos,
                'street_last_bet_to': street_last_bet_to,
                'total_last_bet_to': total_last_bet_to,
                'last_bet_size': last_bet_size,
                'last_bettor': last_bettor,
            }

        elif c == 'b':
            j = i
            while i < len(action) and action[i].isdigit():
                i += 1
            new_street_last_bet_to = int(action[j:i])
            new_last_bet_size = new_street_last_bet_to - street_last_bet_to
            total_last_bet_to += new_last_bet_size
            last_bet_size = new_last_bet_size
            street_last_bet_to = new_street_last_bet_to
            last_bettor = pos
            pos = (pos + 1) % 2
            check_or_call_ends_street = True

    return {
        'st': st, 'pos': pos,
        'street_last_bet_to': street_last_bet_to,
        'total_last_bet_to': total_last_bet_to,
        'last_bet_size': last_bet_size,
        'last_bettor': last_bettor,
    }


# ---------------------------------------------------------------------------
# Action mapping: Slumbot action → our 9-action index
# ---------------------------------------------------------------------------

def map_slumbot_action_to_idx(action_char, bet_to_amount, action_str_before,
                               acting_pos, parsed_before):
    """Map an observed Slumbot action to the closest discrete action index.

    action_char: 'f', 'k', 'c', or 'b'
    bet_to_amount: for 'b', the bet-to amount on this street (int)
    action_str_before: action string BEFORE this action was taken
    acting_pos: Slumbot position (0=BB, 1=SB) of the player who acted
    parsed_before: parse_action(action_str_before)
    """
    if action_char == 'f':
        return 0
    if action_char in ('k', 'c'):
        return 1

    # Bet action: compute pot fraction.
    our_bet, opp_bet = _compute_bets(action_str_before, acting_pos)
    pot = our_bet + opp_bet
    our_chips = STACK_SIZE - our_bet
    street_last_bet_to = parsed_before['street_last_bet_to']
    raise_by = bet_to_amount - street_last_bet_to

    # Check all-in.
    streets = action_str_before.split('/')
    current_street = streets[-1] if streets else ''
    our_sb = _get_our_street_bet(current_street, acting_pos, parsed_before['st'])
    chips_left = STACK_SIZE - our_bet
    bet_from_stack = bet_to_amount - our_sb
    if bet_from_stack >= chips_left * 0.95:  # near all-in
        return 8

    # Map to closest raise fraction.
    frac = raise_by / pot if pot > 0 else 0
    fracs = list(RAISE_FRACTIONS)
    best = min(range(len(fracs)), key=lambda i: abs(fracs[i] - frac))
    return best + 2


# ---------------------------------------------------------------------------
# Action string walker: extract individual actions from a Slumbot action str
# ---------------------------------------------------------------------------

def walk_actions(action_str):
    """Yield (action_str_before, acting_pos, action_char, bet_to_or_0)
    for each action in the Slumbot action string.

    acting_pos: 0=BB, 1=SB (Slumbot convention).
    """
    pos = 1  # SB first preflop
    st = 0
    check_or_call_ends = False
    i = 0
    while i < len(action_str):
        before = action_str[:i]
        c = action_str[i]
        if c == '/':
            st += 1
            pos = 0
            check_or_call_ends = False
            i += 1
            continue

        if c == 'b':
            j = i + 1
            while j < len(action_str) and action_str[j].isdigit():
                j += 1
            bet_to = int(action_str[i+1:j])
            yield (before, pos, 'b', bet_to)
            pos = 1 - pos
            check_or_call_ends = True
            i = j
        elif c == 'k':
            yield (before, pos, 'k', 0)
            if check_or_call_ends:
                # Street ends
                pass  # The '/' will handle street change
            else:
                check_or_call_ends = True
            pos = 1 - pos
            i += 1
        elif c == 'c':
            yield (before, pos, 'c', 0)
            if check_or_call_ends:
                pass
            else:
                check_or_call_ends = True
            pos = 1 - pos
            i += 1
        elif c == 'f':
            yield (before, pos, 'f', 0)
            i += 1
            break
        else:
            i += 1


# ---------------------------------------------------------------------------
# RangeTracker
# ---------------------------------------------------------------------------

class RangeTracker:
    """Bayesian range tracker for HU NLHE.

    Maintains probability distributions over both players' possible hands,
    updated after each observed action using the blueprint strategy.

    Attributes:
        hands: List of (c1, c2) tuples — all possible opponent hands.
        hand_to_idx: Dict mapping hand tuple → index.
        opponent_range: ndarray (n_hands,) — P(opponent has hand_i).
        hero_range: ndarray (n_hands,) — P(hero has hand_i) from opponent's view.
    """

    def __init__(self, our_cards_idx, value_net, device):
        """
        our_cards_idx: list of 2 ints (0-51), our hole cards
        value_net: trained ValueNetwork
        device: torch device
        """
        self.our_cards = tuple(sorted(our_cards_idx))
        self.value_net = value_net
        self.device = device

        # All possible hands for the opponent (excluding our cards).
        remaining = sorted(set(range(52)) - set(our_cards_idx))
        self.hands = list(itertools.combinations(remaining, 2))
        self.n = len(self.hands)
        self.hand_to_idx = {h: i for i, h in enumerate(self.hands)}

        # Opponent range: starts uniform.
        self.opponent_range = np.ones(self.n, dtype=np.float64) / self.n

        # Hero range: starts uniform over the same hand set.
        # This represents what the opponent thinks we could have.
        self.hero_range = np.ones(self.n, dtype=np.float64) / self.n

        # Track which board cards have been revealed.
        self.board_cards = []

    def update_board(self, new_board_cards_idx):
        """Zero out hands that contain newly revealed board cards."""
        for c in new_board_cards_idx:
            if c in self.board_cards:
                continue
            self.board_cards.append(c)
            for i, hand in enumerate(self.hands):
                if c in hand:
                    self.opponent_range[i] = 0.0
                    self.hero_range[i] = 0.0
        # Renormalize.
        os = self.opponent_range.sum()
        if os > 0:
            self.opponent_range /= os
        hs = self.hero_range.sum()
        if hs > 0:
            self.hero_range /= hs

    def update_opponent_action(self, action_idx, action_str_before,
                                opp_pos, parsed_before, board_idx):
        """Opponent took action_idx. Narrow their range via Bayes rule.

        P(hand | action) ∝ P(action | hand) × P(hand)
        where P(action | hand) = blueprint_strategy(hand, state)[action_idx]

        Uses epsilon-greedy floor to prevent catastrophic zeroing:
        the blueprint might assign 0 to an action for a hand that should
        be possible (e.g., AA never raises 0.75x in a weak blueprint).
        """
        strategies = self._batch_blueprint(
            self.hands, board_idx, action_str_before, opp_pos, parsed_before)
        likelihood = strategies[:, action_idx]
        # Floor: mix with uniform to prevent exact zeros.
        eps = 0.01
        likelihood = (1.0 - eps) * likelihood + eps * (1.0 / N_ACTIONS)
        self.opponent_range *= likelihood
        s = self.opponent_range.sum()
        if s > 0:
            self.opponent_range /= s

    def update_hero_action(self, action_idx, action_str_before,
                            hero_pos, parsed_before, board_idx):
        """We took action_idx. Update opponent's belief about our range.

        Same Bayes rule but applied to the hero range.
        """
        strategies = self._batch_blueprint(
            self.hands, board_idx, action_str_before, hero_pos, parsed_before)
        likelihood = strategies[:, action_idx]
        eps = 0.01
        likelihood = (1.0 - eps) * likelihood + eps * (1.0 / N_ACTIONS)
        self.hero_range *= likelihood
        s = self.hero_range.sum()
        if s > 0:
            self.hero_range /= s

    def _batch_blueprint(self, hands, board_idx, action_str, pos, parsed):
        """Get blueprint strategy for all hands at this game state.

        Returns (n_hands, 9) strategy array.
        """
        features = _build_features_batch(
            hands, board_idx, action_str, pos, parsed)
        legal_mask = _get_legal_mask(parsed, action_str, pos)

        # Batch inference.
        feat_t = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            advantages = self.value_net(feat_t).cpu().numpy()

        # Regret match each hand (vectorized where possible).
        n = len(hands)
        strategies = np.zeros((n, N_ACTIONS), dtype=np.float64)
        for i in range(n):
            strategies[i] = regret_match(advantages[i], legal_mask)

        return strategies

    def get_solver_ranges(self, solver_hands, solver_hand_to_idx):
        """Map tracked ranges to a solver's hand indexing.

        solver_hands: list of (c1, c2) tuples from the solver
        solver_hand_to_idx: dict mapping hand → solver index

        Returns (hero_range, villain_range) in solver's indexing.
        """
        n_solver = len(solver_hands)
        hero_range = np.zeros(n_solver, dtype=np.float64)
        villain_range = np.zeros(n_solver, dtype=np.float64)

        for s_idx, hand in enumerate(solver_hands):
            h = tuple(sorted(hand))
            t_idx = self.hand_to_idx.get(h)
            if t_idx is not None:
                villain_range[s_idx] = self.opponent_range[t_idx]
                hero_range[s_idx] = self.hero_range[t_idx]

        # Normalize.
        vs = villain_range.sum()
        if vs > 0:
            villain_range /= vs
        hs = hero_range.sum()
        if hs > 0:
            hero_range /= hs

        return hero_range, villain_range


def update_tracker_from_actions(tracker, action_str, client_pos, board_idx):
    """Walk through an action string and update the tracker for NEW actions only.

    Tracks how many characters have been processed via tracker._processed_len.
    Safe to call repeatedly with growing action strings.
    """
    if not hasattr(tracker, '_processed_len'):
        tracker._processed_len = 0
        tracker._processed_board_len = 0

    opp_pos = 1 - client_pos

    # Board updates: update_board is idempotent (zeroes already-zero hands).
    streets = action_str.split('/')
    board_cards_to_reveal = []
    if len(streets) > 1 and len(board_idx) >= 3:
        board_cards_to_reveal = list(board_idx[:3])
    if len(streets) > 2 and len(board_idx) >= 4:
        board_cards_to_reveal = list(board_idx[:4])
    if len(streets) > 3 and len(board_idx) >= 5:
        board_cards_to_reveal = list(board_idx[:5])
    if len(board_cards_to_reveal) > tracker._processed_board_len:
        new_cards = board_cards_to_reveal[tracker._processed_board_len:]
        if new_cards:
            tracker.update_board(new_cards)
        tracker._processed_board_len = len(board_cards_to_reveal)

    # Only process new actions (characters after _processed_len).
    if len(action_str) <= tracker._processed_len:
        return

    # Walk all actions but skip already-processed ones.
    char_pos = 0
    for (before, acting_pos, action_char, bet_to) in walk_actions(action_str):
        # Compute char_pos of this action in the string.
        action_start = len(before)
        if action_start < tracker._processed_len:
            continue  # Already processed.

        parsed_before = _parse_action(before)
        if 'error' in parsed_before:
            continue
        action_idx = map_slumbot_action_to_idx(
            action_char, bet_to, before, acting_pos, parsed_before)

        if acting_pos == opp_pos:
            tracker.update_opponent_action(
                action_idx, before, opp_pos, parsed_before, board_idx)
        else:
            tracker.update_hero_action(
                action_idx, before, client_pos, parsed_before, board_idx)

    tracker._processed_len = len(action_str)
