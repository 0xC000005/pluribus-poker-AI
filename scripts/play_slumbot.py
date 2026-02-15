"""Play our trained 2-player Deep CFR agent against Slumbot.

Translates between our 9-action discrete space and Slumbot's continuous
bet-size API. Tracks session results over many hands.

Usage:
    python scripts/play_slumbot.py --model models/slumbot_2p_final.pt --hands 200
"""
import argparse
import sys
import time

import numpy as np
import requests
import torch

sys.stdout.reconfigure(line_buffering=True)

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES, RAISE_FRACTIONS
from poker_ai.deep_cfr.networks import ValueNetwork

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
    n_calls = s.count('c')
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
    n_raises = count_raises_current_street(action_str)
    if n_raises < 3 and our_chips > 0:
        street_last_bet_to = parsed['street_last_bet_to']
        streets = action_str.split('/')
        current_street = streets[-1] if streets else ''
        our_street_bet = _get_our_street_bet(current_street, client_pos, parsed['st'])

        to_call = street_last_bet_to - our_street_bet
        # Min raise-by: at least last_bet_size and at least BIG_BLIND.
        min_raise_by = max(parsed['last_bet_size'], BIG_BLIND)
        min_raise_to = street_last_bet_to + min_raise_by

        for fi, frac in enumerate(RAISE_FRACTIONS):
            raise_by = int(frac * pot_total)
            new_street_bet = street_last_bet_to + raise_by
            # Clamp up to min raise.
            new_street_bet = max(new_street_bet, min_raise_to)
            raise_total = new_street_bet - our_street_bet  # chips from our stack
            if raise_total <= our_chips and new_street_bet > street_last_bet_to:
                mask[2 + fi] = 1.0

        # All-in always legal if we have chips beyond calling.
        if our_chips > to_call:
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
    return f'b{new_street_bet}'


# ---------------------------------------------------------------------------
# Regret matching (CPU)
# ---------------------------------------------------------------------------

def regret_match(advantages, legal_mask):
    """Convert advantages to strategy via regret matching."""
    strategy = np.maximum(advantages, 0) * legal_mask
    total = strategy.sum()
    if total > 0:
        strategy /= total
    else:
        n_legal = legal_mask.sum()
        if n_legal > 0:
            strategy = legal_mask / n_legal
    return strategy


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
        print(f"API error on '{incr}': {r['error_msg']}")
        sys.exit(1)
    return r


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------

ACTION_NAMES = [
    "fold", "call/chk",
    "r0.25x", "r0.5x", "r0.75x", "r1.0x", "r1.5x", "r2.0x",
    "all-in",
]


def play_hand(value_net, token, device, verbose=False):
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

    while r.get('winnings') is None:
        action_str = r.get('action', '')
        board = r.get('board', [])
        parsed = parse_action(action_str)

        if 'error' in parsed:
            print(f"\n  PARSE ERROR: {parsed['error']} in '{action_str}'")
            # Fold to recover.
            r = api_act(token, 'f')
            token = r.get('token', token)
            break

        # Build features and get strategy from NN.
        features = build_features(hole_cards, board, action_str, client_pos, parsed)
        legal_mask = get_legal_mask_from_parsed(parsed, action_str, client_pos)

        feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            advantages = value_net(feat_t).cpu().numpy()[0]

        strategy = regret_match(advantages, legal_mask)

        # Sample action from strategy.
        legal_actions = np.where(legal_mask > 0)[0]
        if len(legal_actions) == 0:
            # Shouldn't happen, but check/call as fallback.
            incr = 'k' if parsed['last_bet_size'] == 0 else 'c'
        else:
            probs = np.array([strategy[a] for a in legal_actions], dtype=np.float64)
            if probs.sum() > 0:
                probs /= probs.sum()
                action_idx = int(np.random.choice(legal_actions, p=probs))
            else:
                action_idx = int(np.random.choice(legal_actions))

            incr = action_to_slumbot(action_idx, parsed, action_str, client_pos)

            if verbose:
                print(f" [{ACTION_NAMES[action_idx]}→{incr}]", end="", flush=True)

        r = api_act(token, incr)
        token = r.get('token', token)

    w = r.get('winnings', 0)
    if verbose:
        board = r.get('board', [])
        bot_cards = r.get('bot_hole_cards', [])
        board_str = ' '.join(board) if board else ''
        print(f" | {board_str} | {w:+d}")

    return token, w


def main():
    parser = argparse.ArgumentParser(description='Play against Slumbot')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--hands', type=int, default=200, help='Number of hands to play')
    parser.add_argument('--verbose', action='store_true', help='Print each hand')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Playing {args.hands} hands vs Slumbot")
    print(f"Model: {args.model}")
    print("=" * 60)

    # Load model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    value_net = ValueNetwork(N_FEATURES, hidden_dim, N_ACTIONS).to(device)
    value_net.load_state_dict(checkpoint['value_net'])
    value_net.eval()
    print(f"Loaded model (iter {checkpoint['iteration']}, hidden={hidden_dim})")
    print()

    token = None
    total_winnings = 0
    results = []

    for h in range(args.hands):
        if args.verbose:
            print(f"Hand {h+1:3d}:", end="")

        token, w = play_hand(value_net, token, device, verbose=args.verbose)
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
    print("=" * 60)


if __name__ == '__main__':
    main()
