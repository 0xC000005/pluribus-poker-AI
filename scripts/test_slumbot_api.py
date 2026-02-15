"""Test Slumbot API - play hands with check/call to verify protocol understanding."""
import requests
import sys

HOST = 'slumbot.com'
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000
NUM_STREETS = 4

sys.stdout.reconfigure(line_buffering=True)


def parse_action(action):
    """Parse Slumbot action string. Returns state dict.

    Faithful port of Slumbot's official ParseAction.
    Bet sizes are per-street chip amounts, not raise-by amounts.
    """
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
            if last_bet_size > 0:
                return {'error': 'Illegal check'}
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

        elif c == 'c':
            if last_bet_size == 0:
                return {'error': 'Illegal call'}
            if total_last_bet_to == STACK_SIZE:
                # Call of all-in
                for st1 in range(st, NUM_STREETS - 1):
                    if i < len(action):
                        if action[i] != '/':
                            return {'error': 'Missing slash after all-in call'}
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
            if i == j:
                return {'error': 'Missing bet size'}
            new_street_last_bet_to = int(action[j:i])
            new_last_bet_size = new_street_last_bet_to - street_last_bet_to
            last_bet_size = new_last_bet_size
            total_last_bet_to += last_bet_size
            street_last_bet_to = new_street_last_bet_to
            last_bettor = pos
            pos = (pos + 1) % 2
            check_or_call_ends_street = True

        elif c == '/':
            # This shouldn't happen — slashes are consumed after checks/calls
            return {'error': f'Unexpected slash at position {i-1}'}
        else:
            return {'error': f'Unexpected character: {c}'}

    return {
        'st': st, 'pos': pos,
        'street_last_bet_to': street_last_bet_to,
        'total_last_bet_to': total_last_bet_to,
        'last_bet_size': last_bet_size,
        'last_bettor': last_bettor,
    }


def new_hand(token):
    data = {'token': token} if token else {}
    r = requests.post(f'https://{HOST}/slumbot/api/new_hand', json=data).json()
    if 'error_msg' in r:
        print(f"API error: {r['error_msg']}")
        sys.exit(1)
    return r


def act(token, incr):
    data = {'token': token, 'incr': incr}
    r = requests.post(f'https://{HOST}/slumbot/api/act', json=data).json()
    if 'error_msg' in r:
        print(f"API error: {r['error_msg']}")
        sys.exit(1)
    return r


def main():
    token = None
    total_winnings = 0
    num_hands = 20

    for h in range(num_hands):
        r = new_hand(token)
        token = r.get('token', token)

        client_pos = r['client_pos']
        pos_name = "BB" if client_pos == 0 else "SB"
        hole_cards = r['hole_cards']

        print(f"Hand {h+1:2d} | {pos_name} | {hole_cards[0]} {hole_cards[1]}", end="", flush=True)

        if r.get('winnings') is not None:
            w = r['winnings']
            total_winnings += w
            print(f" | Bot folded | {w:+d} | total: {total_winnings:+d}")
            continue

        # Play check/call
        moves = 0
        while r.get('winnings') is None:
            action_str = r.get('action', '')
            a = parse_action(action_str)

            if 'error' in a:
                print(f"\n  PARSE ERROR: {a['error']} in action '{action_str}'")
                sys.exit(1)

            # Determine whose turn it is
            # client_pos: 0=BB=player0, 1=SB=player1
            # a['pos']: position of next actor (0 or 1)
            # We act when a['pos'] == client_pos
            if a['pos'] != client_pos:
                print(f"\n  ERROR: Not our turn? pos={a['pos']} client_pos={client_pos} action='{action_str}'")
                sys.exit(1)

            if a['last_bettor'] != -1 and a['last_bet_size'] > 0:
                incr = 'c'
            else:
                incr = 'k'

            r = act(token, incr)
            token = r.get('token', token)
            moves += 1

        w = r['winnings']
        total_winnings += w
        board = r.get('board', [])
        bot_cards = r.get('bot_hole_cards', [])
        action_str = r.get('action', '')

        board_str = ' '.join(board) if board else 'none'
        print(f" | {board_str} | {action_str}")
        print(f"         bot={bot_cards} | {w:+d} chips | total: {total_winnings:+d}")

    print(f"\n{'='*60}")
    print(f"{num_hands} hands | Total: {total_winnings:+d} | Avg: {total_winnings/num_hands:+.0f}/hand")


if __name__ == '__main__':
    main()
