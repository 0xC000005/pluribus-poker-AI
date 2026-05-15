"""Play our trained 2-player Deep CFR agent against Slumbot.

Translates between our 9-action discrete space and Slumbot's continuous
bet-size API. Tracks session results over many hands.

Usage:
    python scripts/play_slumbot.py --model models/slumbot_2p_final.pt --hands 200
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(line_buffering=True)

# Inline constants to avoid importing poker_ai (which eagerly loads sklearn/scipy/etc).
N_FEATURES = 126
N_ACTIONS = 9
RAISE_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

from poker_ai.deep_cfr.networks import ValueNetwork, PolicyNetwork
from solver import resolve_solver_backend, solve_street, solver_action_to_slumbot
from range_tracker import (
    RangeTracker,
    map_slumbot_action_to_idx,
    update_tracker_from_actions,
)

# ---------------------------------------------------------------------------
# Slumbot API constants
# ---------------------------------------------------------------------------

HOST = 'slumbot.com'
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000
NUM_STREETS = 4

# Card encoding: Slumbot uses "Ac", "Kh" etc.
# Our features use card_idx = (rank - 2) * 4 + suit_idx
_RANK_MAP = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}
_SUIT_MAP = {'c': 0, 'd': 1, 'h': 2, 's': 3}


def card_str_to_index(card_str: str) -> int:
    """Convert Slumbot card string (e.g. 'Ac') to 0-51 index."""
    rank = _RANK_MAP[card_str[0]]
    suit = _SUIT_MAP[card_str[1]]
    return (rank - 2) * 4 + suit


# ---------------------------------------------------------------------------
# Slumbot action parser (faithful port of official ParseAction)
# ---------------------------------------------------------------------------

def parse_action(action):
    """Parse Slumbot action string into game state dict."""
    st = 0
    street_last_bet_to = BIG_BLIND
    total_last_bet_to = BIG_BLIND
    last_bet_size = BIG_BLIND - SMALL_BLIND
    last_bettor = 0
    pos = 1  # SB acts first preflop

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


def count_actions_in_street(action_str, street_idx):
    """Count calls, raises, folds in a given street of the action string."""
    streets = action_str.split('/')
    if street_idx >= len(streets):
        return 0, 0, 0
    s = streets[street_idx]
    # In our training, checks (action 1 with no bet) are recorded as CALL.
    # Slumbot uses 'k' for checks and 'c' for calls — count both.
    n_calls = s.count('c') + s.count('k')
    n_folds = s.count('f')
    n_raises = s.count('b')
    return n_calls, n_raises, n_folds


def count_raises_current_street(action_str):
    """Count number of raises (b) on the current street."""
    streets = action_str.split('/')
    current = streets[-1] if streets else ''
    return current.count('b')


# ---------------------------------------------------------------------------
# Feature vector builder (from Slumbot game state)
# ---------------------------------------------------------------------------

def build_features(
    hole_cards,       # list of 2 card strings, e.g. ['Ac', 'Kh']
    board,            # list of 0-5 card strings
    action_str,       # Slumbot action string
    client_pos,       # 0=BB, 1=SB
    parsed,           # output of parse_action()
) -> np.ndarray:
    """Build 126-dim feature vector matching our training encoding.

    Feature layout (same as FastPokerState.to_feature_vector):
      [0:52]   - hole cards (52-dim binary)
      [52:104] - community cards (52-dim binary)
      [104:108]- current street (4-dim one-hot)
      [108:114]- scalar features
      [114:126]- per-round action summary (4 rounds x 3)
    """
    features = np.zeros(N_FEATURES, dtype=np.float32)

    # Hole cards.
    for card_str in hole_cards:
        features[card_str_to_index(card_str)] = 1.0

    # Community cards.
    for card_str in board:
        features[52 + card_str_to_index(card_str)] = 1.0

    # Current street one-hot.
    st = parsed['st']
    if st < 4:
        features[104 + st] = 1.0

    # Scalar features.
    # In our model: player 0 = SB, player 1 = BB
    # Slumbot: client_pos 0 = BB, client_pos 1 = SB
    # Map: our_player = 1 - client_pos (if client_pos=0 (BB), our_player=1; if client_pos=1 (SB), our_player=0)
    # But for features, we encode from current player's perspective.
    n_players = 2
    total_chips_all = STACK_SIZE * n_players

    # Compute our chips remaining and bet amount from the action.
    our_total_bet = parsed['total_last_bet_to'] if parsed.get('last_bettor') == client_pos else 0
    # Actually, we need to track each player's total bet across all streets.
    # Simpler: compute from the pot.
    # Our total bet = what we've put in across all streets.
    # For pot calculation, each player's contribution needs tracking.
    # Use a simpler approach: total pot = sum of all bets, and approximate.

    # Track per-player total bets from action string.
    our_bet, opp_bet = _compute_bets(action_str, client_pos)
    our_chips = STACK_SIZE - our_bet
    pot_total = our_bet + opp_bet

    features[108] = pot_total / total_chips_all                    # pot ratio
    features[109] = our_chips / STACK_SIZE                         # our chips ratio
    features[110] = our_bet / STACK_SIZE                           # our bet ratio
    features[111] = 1.0                                            # active ratio (always 1.0 in HU)
    our_player_idx = 1 - client_pos  # our internal player index
    features[112] = our_player_idx / max(n_players - 1, 1)        # position
    features[113] = count_raises_current_street(action_str) / 3.0  # n_raises

    # Per-round action summary.
    for rd in range(4):
        n_calls, n_raises, n_folds = count_actions_in_street(action_str, rd)
        offset = 114 + rd * 3
        features[offset] = n_calls / max(n_players, 1)
        features[offset + 1] = n_raises / 3.0
        features[offset + 2] = n_folds / max(n_players, 1)

    return features


def _compute_bets(action_str, client_pos):
    """Compute total chips bet by each player from action string.

    client_pos: 0=BB, 1=SB.
    Returns (our_total_bet, opp_total_bet).
    """
    # In Slumbot: position 1 = SB posts 50, position 0 = BB posts 100.
    # pos tracks who acts next. Betting starts with pos=1 (SB) preflop.
    our_total = SMALL_BLIND if client_pos == 1 else BIG_BLIND
    opp_total = SMALL_BLIND if client_pos == 0 else BIG_BLIND

    # Track per-street bets for each position.
    our_street_bet = our_total  # preflop initial
    opp_street_bet = opp_total

    # Walk through actions.
    acting_pos = 1  # SB acts first preflop
    st = 0
    i = 0
    while i < len(action_str):
        c = action_str[i]
        i += 1

        if c == '/':
            st += 1
            our_street_bet = 0
            opp_street_bet = 0
            acting_pos = 0  # postflop, pos 0 acts first
            continue

        if c == 'k':
            acting_pos = 1 - acting_pos

        elif c == 'c':
            # Call = match the other player's street bet.
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
            new_street_bet = int(action_str[j:i])
            if acting_pos == client_pos:
                diff = new_street_bet - our_street_bet
                our_total += diff
                our_street_bet = new_street_bet
            else:
                diff = new_street_bet - opp_street_bet
                opp_total += diff
                opp_street_bet = new_street_bet
            acting_pos = 1 - acting_pos

    return our_total, opp_total


def _compute_bets_before_street(action_str, client_pos, target_street):
    """Compute total bets for each player up to (not including) target_street.

    Returns (our_total_bet, opp_total_bet) at the start of target_street.
    """
    # Process action_str only up to the target_street boundary.
    streets = action_str.split('/')
    # Rejoin only streets before target_street.
    pre_streets = streets[:target_street]
    if not pre_streets:
        # No actions before this street.
        return (BIG_BLIND if client_pos == 0 else SMALL_BLIND,
                BIG_BLIND if client_pos == 1 else SMALL_BLIND)
    pre_action = '/'.join(pre_streets)
    return _compute_bets(pre_action, client_pos)


SOLVER_ACTION_NAMES = {
    0: 'fold', 1: 'chk/call',
    2: '0.25xpot', 3: '0.5xpot', 4: '0.75xpot', 5: '1.0xpot',
    6: '1.5xpot', 7: '2.0xpot', 8: 'all-in',
}


# ---------------------------------------------------------------------------
# Action translation: our discrete action -> Slumbot format
# ---------------------------------------------------------------------------

def get_legal_mask_from_parsed(parsed, action_str, client_pos):
    """Build a (9,) legal mask from the parsed Slumbot state."""
    mask = np.zeros(N_ACTIONS, dtype=np.float32)

    our_bet, opp_bet = _compute_bets(action_str, client_pos)
    pot_total = our_bet + opp_bet
    our_chips = STACK_SIZE - our_bet

    # Can always fold if there's a bet to us.
    if parsed['last_bet_size'] > 0:
        mask[0] = 1.0  # fold
    # Can always call/check.
    mask[1] = 1.0  # call or check

    # Raise actions (2-7) and all-in (8).
    # Must match training legal mask logic exactly (fast_state.py / game_kernels.py):
    #   raise_amount = int(frac * pot_total) + to_call
    #   legal if raise_amount covers the no-limit minimum raise and fits stack
    n_raises = count_raises_current_street(action_str)
    if n_raises < 3 and our_chips > 0:
        streets = action_str.split('/')
        current_street = streets[-1] if streets else ''
        our_street_bet = _get_our_street_bet(current_street, client_pos, parsed['st'])
        street_last_bet_to = parsed['street_last_bet_to']
        to_call = street_last_bet_to - our_street_bet
        min_raise_chips = BIG_BLIND if to_call <= 0 else to_call + max(to_call, BIG_BLIND)

        for fi, frac in enumerate(RAISE_FRACTIONS):
            # Match training: raise_amount = frac * pot + to_call (total chips from stack)
            raise_amount = int(frac * pot_total) + to_call
            if raise_amount >= min_raise_chips and raise_amount <= our_chips:
                mask[2 + fi] = 1.0

        # All-in always legal if we have chips.
        if our_chips > 0:
            mask[8] = 1.0

    # If no fold possible (no outstanding bet), remove fold.
    if parsed['last_bet_size'] == 0:
        mask[0] = 0.0

    return mask


def _get_our_street_bet(current_street_actions, client_pos, st):
    """Compute our bet on the current street from action substring."""
    if st == 0:
        # Preflop: SB=50, BB=100 initially.
        bets = [BIG_BLIND, SMALL_BLIND]  # [pos0=BB, pos1=SB]
    else:
        bets = [0, 0]

    acting_pos = 1 if st == 0 else 0  # SB first preflop, pos 0 first postflop
    i = 0
    while i < len(current_street_actions):
        c = current_street_actions[i]
        i += 1
        if c == 'k':
            acting_pos = 1 - acting_pos
        elif c == 'c':
            # Call: match the other player's bet level.
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


def action_to_slumbot(action_idx, parsed, action_str, client_pos):
    """Convert our discrete action index to Slumbot incremental action string."""
    if action_idx == 0:
        return 'f'

    if action_idx == 1:
        if parsed['last_bet_size'] > 0:
            return 'c'
        else:
            return 'k'

    # Raise actions (2-7) and all-in (8).
    our_bet, opp_bet = _compute_bets(action_str, client_pos)
    pot_total = our_bet + opp_bet
    our_chips = STACK_SIZE - our_bet

    streets = action_str.split('/')
    current_street = streets[-1] if streets else ''
    st = parsed['st']
    our_street_bet = _get_our_street_bet(current_street, client_pos, st)
    street_last_bet_to = parsed['street_last_bet_to']
    to_call = street_last_bet_to - our_street_bet

    if action_idx == 8:
        # All-in: bet everything.
        new_street_bet = our_street_bet + our_chips
        # If our all-in doesn't exceed the current bet, just call (call-all-in).
        if new_street_bet <= street_last_bet_to:
            return 'c'
        return f'b{new_street_bet}'

    # Fractional raise (actions 2-7).
    frac = RAISE_FRACTIONS[action_idx - 2]
    raise_by = int(frac * pot_total)
    new_street_bet = street_last_bet_to + raise_by
    # Enforce minimum raise.
    min_raise_by = max(parsed['last_bet_size'], BIG_BLIND)
    min_raise_to = street_last_bet_to + min_raise_by
    new_street_bet = max(new_street_bet, min_raise_to)
    # Cap at our total chips.
    max_street_bet = our_street_bet + our_chips
    new_street_bet = min(new_street_bet, max_street_bet)
    # If capped bet doesn't exceed current bet, just call.
    if new_street_bet <= street_last_bet_to:
        return 'c'
    return f'b{new_street_bet}'


# ---------------------------------------------------------------------------
# Regret matching (CPU)
# ---------------------------------------------------------------------------

def regret_match(advantages, legal_mask):
    """Convert advantages to strategy via regret matching.

    Must match the training regret_match (deep_cfr.py) exactly:
    positive advantages normalized, else uniform over legal actions.
    """
    positive = np.maximum(advantages, 0) * legal_mask
    total = positive.sum()
    if total > 0:
        return positive / total
    return legal_mask / legal_mask.sum()


def _policy_calibration_target_streets(value_net):
    calibration = getattr(value_net, "policy_calibration", None)
    if not isinstance(calibration, dict):
        return ()
    streets = calibration.get("target_streets")
    if not streets:
        return ()
    return tuple(sorted({int(street) for street in streets}))


def _feature_street(features):
    street_one_hot = np.asarray(features, dtype=np.float32)[104:108]
    if float(np.max(street_one_hot)) <= 0.0:
        return -1
    return int(np.argmax(street_one_hot))


def effective_strategy_source(value_net, features, strategy_source):
    if strategy_source != "policy-head-covered":
        return strategy_source
    covered_streets = _policy_calibration_target_streets(value_net)
    if not covered_streets:
        raise RuntimeError(
            "policy-head-covered requires policy_calibration.target_streets metadata"
        )
    return "policy-head" if _feature_street(features) in covered_streets else "regret"


def network_strategy(value_net, features, legal_mask, device, strategy_source="regret"):
    """Return (advantages, strategy) from the requested learned policy source."""
    strategy_source = effective_strategy_source(value_net, features, strategy_source)
    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        if strategy_source == "policy-head":
            adv_t, logits_t = value_net.forward_with_policy(feat_t)
            advantages = adv_t.cpu().numpy()[0]
            logits = logits_t.cpu().numpy()[0].astype(np.float64)
        elif strategy_source == "average-policy":
            average_policy_net = getattr(value_net, "average_policy_net", None)
            if average_policy_net is None:
                raise RuntimeError("average-policy strategy source requires average_policy_net")
            logits = average_policy_net(feat_t).cpu().numpy()[0].astype(np.float64)
            advantages = value_net(feat_t).cpu().numpy()[0]
        else:
            if strategy_source != "regret":
                raise ValueError(f"Unknown strategy source: {strategy_source}")
            advantages = value_net(feat_t).cpu().numpy()[0]

        if strategy_source in {"policy-head", "average-policy"}:
            masked_logits = np.where(legal_mask > 0, logits, -1e9)
            shifted = masked_logits - np.max(masked_logits)
            probs = np.exp(shifted) * legal_mask
            total = probs.sum()
            strategy = probs / total if total > 0 else legal_mask / legal_mask.sum()
            return advantages, strategy
    return advantages, regret_match(advantages, legal_mask)


# ---------------------------------------------------------------------------
# Slumbot API
# ---------------------------------------------------------------------------

def api_new_hand(token):
    data = {'token': token} if token else {}
    r = requests.post(f'https://{HOST}/slumbot/api/new_hand', json=data).json()
    if 'error_msg' in r:
        print(f"API error: {r['error_msg']}")
        sys.exit(1)
    return r


def api_act(token, incr):
    data = {'token': token, 'incr': incr}
    r = requests.post(f'https://{HOST}/slumbot/api/act', json=data).json()
    if 'error_msg' in r:
        print(f"\n  API error on '{incr}': {r['error_msg']}")
        # Try folding to recover the hand gracefully.
        r2 = requests.post(f'https://{HOST}/slumbot/api/act',
                           json={'token': token, 'incr': 'f'}).json()
        if 'error_msg' not in r2:
            return r2
        # If fold also fails, return the original error with winnings=0.
        return {'token': token, 'winnings': 0}
    return r


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------

ACTION_NAMES = [
    "fold", "call/chk",
    "r0.25x", "r0.5x", "r0.75x", "r1.0x", "r1.5x", "r2.0x",
    "all-in",
]


class ActionDiagnostics:
    """Track cheap live-play diagnostics for Slumbot distribution shift."""

    def __init__(self):
        self.decision_policy = 0
        self.decision_solver = 0
        self.decision_fallback = 0
        self.parse_errors = 0
        self.api_errors = 0
        self.action_mix = {name: 0 for name in ACTION_NAMES}
        self.action_mix["solver"] = 0
        self.increment_mix = {"f": 0, "k": 0, "c": 0, "b": 0}
        self.mapping_drifts = []
        self.solver_latencies_ms = []
        self.solver_hand_counts = []
        self.solver_full_hand_counts = []
        self.solver_cache_hits = 0

    def _record_increment(self, incr):
        if not incr:
            return
        key = incr[0]
        if key in self.increment_mix:
            self.increment_mix[key] += 1

    def _record_mapping_drift(self, action_idx, incr, action_str, client_pos, parsed):
        if not incr:
            return
        action_char = incr[0]
        bet_to = int(incr[1:]) if action_char == "b" and incr[1:].isdigit() else 0
        mapped = map_slumbot_action_to_idx(
            action_char, bet_to, action_str, client_pos, parsed,
        )
        intended_weight = sum(weight for idx, weight in mapped if idx == action_idx)
        drift = max(0.0, min(1.0, 1.0 - intended_weight))
        self.mapping_drifts.append(drift)

    def record_policy_action(self, action_idx, incr, action_str, client_pos, parsed):
        self.decision_policy += 1
        self.action_mix[ACTION_NAMES[action_idx]] += 1
        self._record_increment(incr)
        self._record_mapping_drift(action_idx, incr, action_str, client_pos, parsed)

    def record_solver_action(self, incr, *, latency_ms=None, n_hands=None,
                             full_n_hands=None, cached=False):
        self.decision_solver += 1
        self.action_mix["solver"] += 1
        self._record_increment(incr)
        if cached:
            self.solver_cache_hits += 1
        if latency_ms is not None:
            self.solver_latencies_ms.append(float(latency_ms))
        if n_hands is not None:
            self.solver_hand_counts.append(int(n_hands))
        if full_n_hands is not None:
            self.solver_full_hand_counts.append(int(full_n_hands))

    def record_fallback(self, incr):
        self.decision_fallback += 1
        self._record_increment(incr)

    def record_parse_error(self):
        self.parse_errors += 1

    def record_api_error(self):
        self.api_errors += 1

    def as_summary(self):
        n_drift = len(self.mapping_drifts)
        mean_drift = float(np.mean(self.mapping_drifts)) if n_drift else 0.0
        max_drift = float(np.max(self.mapping_drifts)) if n_drift else 0.0
        n_solver_latency = len(self.solver_latencies_ms)
        mean_solver_latency = (
            float(np.mean(self.solver_latencies_ms)) if n_solver_latency else 0.0
        )
        max_solver_latency = (
            float(np.max(self.solver_latencies_ms)) if n_solver_latency else 0.0
        )
        mean_solver_hands = (
            float(np.mean(self.solver_hand_counts)) if self.solver_hand_counts else 0.0
        )
        mean_solver_full_hands = (
            float(np.mean(self.solver_full_hand_counts))
            if self.solver_full_hand_counts else 0.0
        )
        mean_prune_ratio = (
            mean_solver_hands / mean_solver_full_hands
            if mean_solver_full_hands > 0 else 0.0
        )
        return {
            "decision_total": (
                self.decision_policy + self.decision_solver + self.decision_fallback
            ),
            "decision_policy": self.decision_policy,
            "decision_solver": self.decision_solver,
            "decision_fallback": self.decision_fallback,
            "parse_errors": self.parse_errors,
            "api_errors": self.api_errors,
            "action_mix": dict(self.action_mix),
            "increment_mix": dict(self.increment_mix),
            "mapping_drift_n": n_drift,
            "mapping_drift_mean": round(mean_drift, 3),
            "mapping_drift_max": round(max_drift, 3),
            "solver_latency_n": n_solver_latency,
            "solver_latency_mean_ms": round(mean_solver_latency, 1),
            "solver_latency_max_ms": round(max_solver_latency, 1),
            "solver_cache_hits": self.solver_cache_hits,
            "solver_mean_hands": round(mean_solver_hands, 1),
            "solver_mean_full_hands": round(mean_solver_full_hands, 1),
            "solver_mean_prune_ratio": round(mean_prune_ratio, 4),
        }

    def format_summary_lines(self):
        summary = self.as_summary()
        action_parts = " ".join(
            f"{name}={summary['action_mix'][name]}"
            for name in [*ACTION_NAMES, "solver"]
        )
        increment_parts = " ".join(
            f"{name}={summary['increment_mix'][name]}" for name in ["f", "k", "c", "b"]
        )
        return [
            "  Decisions: "
            f"total={summary['decision_total']} "
            f"policy={summary['decision_policy']} "
            f"solver={summary['decision_solver']} "
            f"fallback={summary['decision_fallback']} "
            f"parse_errors={summary['parse_errors']} "
            f"api_errors={summary['api_errors']}",
            f"  Action mix: {action_parts}",
            f"  Increments: {increment_parts}",
            "  Mapping drift: "
            f"n={summary['mapping_drift_n']} "
            f"mean={summary['mapping_drift_mean']:.3f} "
            f"max={summary['mapping_drift_max']:.3f}",
            "  Solver perf: "
            f"n={summary['solver_latency_n']} "
            f"mean_ms={summary['solver_latency_mean_ms']:.1f} "
            f"max_ms={summary['solver_latency_max_ms']:.1f} "
            f"cache_hits={summary['solver_cache_hits']} "
            f"mean_hands={summary['solver_mean_hands']:.1f}/"
            f"{summary['solver_mean_full_hands']:.1f} "
            f"prune_ratio={summary['solver_mean_prune_ratio']:.4f}",
        ]


# Simple cache for subgame solves keyed by (street, board, action_str, stacks, hero_first).
_SOLVER_CACHE = {}


def _base_policy_action(hole_cards, board, action_str, client_pos, parsed,
                         value_net, device, greedy, no_allin, verbose,
                         diagnostics=None, strategy_source="regret"):
    """Select action using trained base policy (for pre-river streets)."""
    features = build_features(hole_cards, board, action_str, client_pos, parsed)
    legal_mask = get_legal_mask_from_parsed(parsed, action_str, client_pos)

    if no_allin:
        legal_mask[8] = 0

    advantages, strategy = network_strategy(
        value_net, features, legal_mask, device, strategy_source=strategy_source)

    legal_actions = np.where(legal_mask > 0)[0]
    if len(legal_actions) == 0:
        incr = 'k' if parsed['last_bet_size'] == 0 else 'c'
        if diagnostics is not None:
            diagnostics.record_fallback(incr)
        return incr

    effective_source = effective_strategy_source(value_net, features, strategy_source)
    if greedy:
        if effective_source in {"policy-head", "average-policy"}:
            action_idx = int(np.argmax(strategy))
        else:
            masked_adv = advantages * legal_mask + (1 - legal_mask) * (-1e9)
            action_idx = int(np.argmax(masked_adv))
    else:
        probs = np.array([strategy[a] for a in legal_actions], dtype=np.float64)
        if probs.sum() > 0:
            probs /= probs.sum()
            action_idx = int(np.random.choice(legal_actions, p=probs))
        else:
            action_idx = int(np.random.choice(legal_actions))

    incr = action_to_slumbot(action_idx, parsed, action_str, client_pos)
    if diagnostics is not None:
        diagnostics.record_policy_action(action_idx, incr, action_str, client_pos, parsed)
    if verbose:
        print(f" [{ACTION_NAMES[action_idx]}→{incr}]", end="", flush=True)
    return incr


def _solver_action(hole_cards, board, action_str, client_pos, parsed,
                    verbose, tracker=None, diagnostics=None,
                    solver_backend='auto', solver_budget_profile='live'):
    """Select action using real-time CFR+ solver (turn or river)."""
    import itertools

    st = parsed['st']  # 2=turn, 3=river
    n_board = 4 if st == 2 else 5

    # Compute game state at start of this street.
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        action_str, client_pos, target_street=st)
    pot = our_bet_pre + opp_bet_pre
    hero_stack = STACK_SIZE - our_bet_pre
    villain_stack = STACK_SIZE - opp_bet_pre
    hero_first = (client_pos == 0)

    our_cards_idx = [card_str_to_index(c) for c in hole_cards]
    board_idx = [card_str_to_index(c) for c in board[:n_board]]

    # Extract current street actions.
    streets = action_str.split('/')
    street_str = streets[st] if len(streets) > st else ''

    # Get tracked ranges; solve_street prunes low-probability hands before CFR.
    hero_range = None
    villain_range = None
    if tracker is not None:
        remaining = sorted(set(range(52)) - set(board_idx))
        all_hands = list(itertools.combinations(remaining, 2))
        all_hand_to_idx = {h: i for i, h in enumerate(all_hands)}
        hero_range, villain_range = tracker.get_solver_ranges(all_hands, all_hand_to_idx)

    # Cache key: street, board, action string for this street, stacks, hero_first.
    cache_key = (
        st,
        tuple(board_idx),
        street_str,
        int(pot), int(hero_stack), int(villain_stack),
        bool(hero_first),
        solver_budget_profile,
    )

    incr_cached = _SOLVER_CACHE.get(cache_key)
    if incr_cached is not None:
        incr = incr_cached
        if diagnostics is not None:
            diagnostics.record_solver_action(incr, cached=True)
        if verbose:
            label = "TURN-SOLVE" if st == 2 else "RIVER-SOLVE"
            print(f" [{label}:CACHED>{incr}]", end="", flush=True)
        return incr

    # Adaptive iterations: increase when facing large to_call or deep stacks.
    streets = action_str.split('/')
    current_street = streets[-1] if streets else ''
    our_street_bet = _get_our_street_bet(current_street, client_pos, parsed['st'])
    to_call = parsed['street_last_bet_to'] - our_street_bet
    iters = _solver_iterations_for_profile(
        solver_budget_profile,
        to_call=to_call,
        pot=pot,
        hero_stack=hero_stack,
        villain_stack=villain_stack,
    )

    solve_started = time.perf_counter()
    solver_action, strategy, solver, node = solve_street(
        our_cards_idx, board_idx, pot, hero_stack, villain_stack, hero_first,
        action_str=street_str, n_iterations=iters,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=solver_backend,
        range_prune_threshold=1e-4,
    )
    solve_latency_ms = (time.perf_counter() - solve_started) * 1000.0

    # Convert solver action to Slumbot format.
    if node is None or node.is_terminal:
        incr = 'k' if parsed['last_bet_size'] == 0 else 'c'
    else:
        incr = solver_action_to_slumbot(solver_action, node, solver, parsed)

    if verbose:
        strat_str = ' '.join(f'{SOLVER_ACTION_NAMES[a]}:{p:.0%}'
                             for a, p in sorted(strategy.items()))
        label = "TURN-SOLVE" if st == 2 else "RIVER-SOLVE"
        backend_name, backend_device = resolve_solver_backend(solver_backend)
        backend_label = backend_device or backend_name
        if solver_budget_profile != "live":
            backend_label = f"{backend_label}:{solver_budget_profile}"
        print(f" [{label}:{backend_label}:{SOLVER_ACTION_NAMES[solver_action]}>{incr} ({strat_str})]",
              end="", flush=True)

    # Cache result for identical future states in this session.
    _SOLVER_CACHE[cache_key] = incr
    if diagnostics is not None:
        diagnostics.record_solver_action(
            incr,
            latency_ms=solve_latency_ms,
            n_hands=solver.n,
            full_n_hands=solver.full_n,
        )
    return incr


def _solver_iterations_for_profile(profile, *, to_call, pot, hero_stack, villain_stack):
    """Return the live resolver CFR+ iteration budget for an opt-in profile."""
    if profile == "live":
        iters = 150
        deep_stack_floor = 250
        medium_pressure_iters = 250
        high_pressure_iters = 350
    elif profile == "fast-live":
        iters = 100
        deep_stack_floor = 150
        medium_pressure_iters = 150
        high_pressure_iters = 250
    else:
        raise ValueError(f"Unknown solver budget profile: {profile}")

    if to_call > 0:
        pressure = to_call / max(pot, 1)
        if pressure >= 0.25:
            iters = medium_pressure_iters
        if pressure >= 0.5:
            iters = high_pressure_iters
    if max(hero_stack, villain_stack) >= 10000:
        iters = max(iters, deep_stack_floor)
    return iters


def play_hand(value_net, token, device, verbose=False, greedy=False,
              no_allin=False, use_solver=True, diagnostics=None,
              strategy_source="regret", solver_backend='auto',
              solver_budget_profile='live'):
    """Play one hand against Slumbot. Returns (token, winnings)."""
    r = api_new_hand(token)
    token = r.get('token', token)

    client_pos = r['client_pos']
    hole_cards = r['hole_cards']

    if verbose:
        pos_name = "BB" if client_pos == 0 else "SB"
        print(f"  pos={pos_name} cards={hole_cards}", end="", flush=True)

    if r.get('winnings') is not None:
        if verbose:
            print(f" | bot folded preflop | {r['winnings']:+d}")
        return token, r['winnings']

    # Create range tracker for this hand.
    tracker = None
    if use_solver:
        our_cards_idx = [card_str_to_index(c) for c in hole_cards]
        tracker = RangeTracker(
            our_cards_idx, value_net, device, strategy_source=strategy_source)

    while r.get('winnings') is None:
        action_str = r.get('action', '')
        board = r.get('board', [])
        parsed = parse_action(action_str)

        if 'error' in parsed:
            print(f"\n  PARSE ERROR: {parsed['error']} in '{action_str}'")
            if diagnostics is not None:
                diagnostics.record_parse_error()
            # Fold to recover.
            r = api_act(token, 'f')
            token = r.get('token', token)
            break

        # Update range tracker with all actions so far.
        if tracker is not None:
            board_idx = [card_str_to_index(c) for c in board]
            update_tracker_from_actions(tracker, action_str, client_pos, board_idx)

        # ----- Turn/River: use real-time CFR+ solver -----
        if parsed['st'] >= 2 and use_solver:
            incr = _solver_action(
                hole_cards, board, action_str, client_pos, parsed, verbose,
                tracker=tracker, diagnostics=diagnostics,
                solver_backend=solver_backend,
                solver_budget_profile=solver_budget_profile,
            )
        else:
            # ----- Preflop/Flop: use base policy (trained model) -----
            incr = _base_policy_action(
                hole_cards, board, action_str, client_pos, parsed,
                value_net, device, greedy, no_allin, verbose,
                diagnostics=diagnostics, strategy_source=strategy_source,
            )

        r = api_act(token, incr)
        token = r.get('token', token)

    w = r.get('winnings', 0)
    if verbose:
        board = r.get('board', [])
        bot_cards = r.get('bot_hole_cards', [])
        board_str = ' '.join(board) if board else ''
        print(f" | {board_str} | {w:+d}")

    return token, w


def _remap_legacy_state_dict(state: dict) -> dict:
    """Remap legacy `net.<i>.{weight,bias}` keys to the current layout.

    Older checkpoints stored ValueNetwork as a plain nn.Sequential named
    `net` (trunk layers interleaved with the final output head). The current
    architecture splits these into `trunk` + `adv_head`.
    """
    if not any(k.startswith('net.') for k in state):
        return state
    linear_indices = sorted({int(k.split('.')[1]) for k in state if k.startswith('net.')})
    *trunk_idxs, head_idx = linear_indices
    remapped = {}
    for k, v in state.items():
        if not k.startswith('net.'):
            remapped[k] = v
            continue
        _, idx, param = k.split('.', 2)
        idx = int(idx)
        if idx == head_idx:
            remapped[f'adv_head.{param}'] = v
        else:
            remapped[f'trunk.{idx}.{param}'] = v
    return remapped


def main():
    parser = argparse.ArgumentParser(description='Play against Slumbot')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--hands', type=int, default=200, help='Number of hands to play')
    parser.add_argument('--verbose', action='store_true', help='Print each hand')
    parser.add_argument('--greedy', action='store_true', help='Deterministic (argmax) action selection')
    parser.add_argument('--no-allin', action='store_true', help='Disable all-in action')
    parser.add_argument('--no-solver', action='store_true', help='Disable river CFR solver')
    parser.add_argument(
        '--solver-backend',
        choices=(
            'auto',
            'cpu',
            'torch-cuda',
            'torch-cpu',
            'torch-levelsync-cuda',
            'torch-levelsync-cpu',
        ),
        default='auto',
        help='Turn/river CFR+ backend. auto uses the stable reference CPU path.',
    )
    parser.add_argument(
        '--solver-budget-profile',
        choices=('live', 'fast-live'),
        default='live',
        help='Turn/river CFR+ iteration profile. live preserves the default budget.',
    )
    parser.add_argument(
        '--strategy-source',
        choices=('regret', 'policy-head', 'average-policy', 'policy-head-covered'),
        default='regret',
        help='Learned blueprint source for non-solver decisions and range tracking.',
    )
    args = parser.parse_args()

    mode_str = "greedy" if args.greedy else "sampled"
    if args.no_allin:
        mode_str += "+no-allin"
    if not args.no_solver:
        mode_str += "+turn+river-solver"
        mode_str += f"+solver-{args.solver_backend}"
        if args.solver_budget_profile != "live":
            mode_str += f"+budget-{args.solver_budget_profile}"
    if args.strategy_source != "regret":
        mode_str += f"+{args.strategy_source}"
    print("=" * 60)
    print(f"Playing {args.hands} hands vs Slumbot ({mode_str})")
    print(f"Model: {args.model}")
    print("=" * 60)

    # Load model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    n_layers = checkpoint.get('n_layers', 2)
    uses_betting_history = bool(checkpoint.get('uses_betting_history', False))
    value_net = ValueNetwork(
        N_FEATURES,
        hidden_dim,
        N_ACTIONS,
        n_layers=n_layers,
        use_betting_history=uses_betting_history,
    ).to(device)
    state = _remap_legacy_state_dict(checkpoint['value_net'])
    missing, unexpected = value_net.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    policy_calibration = checkpoint.get('policy_calibration')
    value_net.policy_calibration = (
        dict(policy_calibration) if isinstance(policy_calibration, dict) else {}
    )
    # policy_head/seq_proj absent in legacy checkpoints; they are unused at
    # inference time (forward() returns only adv from trunk+adv_head).
    if checkpoint.get('average_policy_net') is not None:
        average_policy_net = PolicyNetwork(
            N_FEATURES,
            hidden_dim,
            N_ACTIONS,
            n_layers=n_layers,
            use_betting_history=uses_betting_history,
        ).to(device)
        average_policy_net.load_state_dict(checkpoint['average_policy_net'])
        average_policy_net.eval()
        value_net.average_policy_net = average_policy_net
    elif args.strategy_source == "average-policy":
        raise RuntimeError(
            "Checkpoint does not contain average_policy_net; use --strategy-source regret "
            "or train with average-strategy collection."
        )
    value_net.eval()
    print(f"Loaded model (iter {checkpoint['iteration']}, hidden={hidden_dim}, layers={n_layers})")
    print()

    token = None
    total_winnings = 0
    results = []
    diagnostics = ActionDiagnostics()

    for h in range(args.hands):
        if args.verbose:
            print(f"Hand {h+1:3d}:", end="")

        token, w = play_hand(value_net, token, device, verbose=args.verbose,
                             greedy=args.greedy, no_allin=args.no_allin,
                             use_solver=not args.no_solver,
                             diagnostics=diagnostics,
                             strategy_source=args.strategy_source,
                             solver_backend=args.solver_backend,
                             solver_budget_profile=args.solver_budget_profile)
        total_winnings += w
        results.append(w)

        if (h + 1) % 50 == 0:
            avg = np.mean(results)
            se = np.std(results) / np.sqrt(len(results))
            print(f"  --- {h+1} hands: {total_winnings:+d} total, "
                  f"{avg:+.0f} +/- {1.96*se:.0f} chips/hand (95% CI)")

    print()
    print("=" * 60)
    avg = np.mean(results)
    se = np.std(results) / np.sqrt(len(results))
    mbb_per_hand = avg / BIG_BLIND * 1000  # milli-big-blinds per hand
    print(f"FINAL: {args.hands} hands | {total_winnings:+d} chips")
    print(f"  Avg: {avg:+.0f} +/- {1.96*se:.0f} chips/hand")
    print(f"  Rate: {mbb_per_hand:+.0f} mbb/hand")
    print(f"  Win rate: {np.mean(np.array(results) > 0)*100:.1f}%")
    for line in diagnostics.format_summary_lines():
        print(line)
    print("=" * 60)


if __name__ == '__main__':
    main()
