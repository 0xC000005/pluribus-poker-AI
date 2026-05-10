"""Test that play_slumbot.py build_features matches GPU get_features_kernel.

Sets up identical game states in both systems and compares 126-dim feature
vectors element by element. Catches training/play mismatches that would
cause the model to see different inputs than it was trained on.
"""
import sys
import os
import numpy as np

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.play_slumbot import (
    build_features,
    parse_action,
    card_str_to_index,
    count_actions_in_street,
    count_raises_current_street,
    _compute_bets,
    STACK_SIZE,
    SMALL_BLIND,
    BIG_BLIND,
    N_FEATURES,
)

# GPU imports.
from numba import cuda
from poker_ai.deep_cfr.cuda.game_state import (
    GameBatch,
    _get_device_orders,
    PREFLOP, FLOP, TURN, RIVER,
)
from poker_ai.deep_cfr.cuda.game_kernels import get_features_kernel

# ---------------------------------------------------------------------------
# Helpers: set up a single GPU game state manually
# ---------------------------------------------------------------------------

INITIAL_CHIPS = 20000  # Slumbot 200BB stacks.
N_PLAYERS = 2


def make_gpu_batch():
    """Create a 1-game, 2-player GameBatch with blank state."""
    batch = GameBatch(1, N_PLAYERS)
    return batch


def set_gpu_state(
    batch,
    hole_cards_p0,  # list of 2 card indices for player 0 (SB)
    hole_cards_p1,  # list of 2 card indices for player 1 (BB)
    community,      # list of card indices (up to 5), -1 for empty
    chips,          # [p0_chips, p1_chips]
    bets,           # [p0_bet, p1_bet]
    active,         # [p0_active, p1_active]
    stage,          # PREFLOP/FLOP/TURN/RIVER
    n_raises,       # int
    player_i_index, # int (index into order array)
    pot_total,      # int
    history,        # (4, 3) array: [rd][calls/raises/folds]
):
    """Upload a specific game state to GPU batch slot 0."""
    h_chips = np.array([chips], dtype=np.int32)
    h_bets = np.array([bets], dtype=np.int32)
    h_active = np.array([active], dtype=np.int8)

    h_hole = np.full((1, N_PLAYERS, 2), -1, dtype=np.int8)
    h_hole[0, 0, 0] = hole_cards_p0[0]
    h_hole[0, 0, 1] = hole_cards_p0[1]
    h_hole[0, 1, 0] = hole_cards_p1[0]
    h_hole[0, 1, 1] = hole_cards_p1[1]

    h_community = np.full((1, 5), -1, dtype=np.int8)
    for i, c in enumerate(community):
        h_community[0, i] = c

    h_stage = np.array([stage], dtype=np.int8)
    h_n_raises = np.array([n_raises], dtype=np.int8)
    h_pii = np.array([player_i_index], dtype=np.int8)
    h_pot = np.array([pot_total], dtype=np.int32)
    h_history = np.array([history], dtype=np.int8)

    batch.chips.copy_to_device(h_chips)
    batch.bets.copy_to_device(h_bets)
    batch.active.copy_to_device(h_active)
    batch.hole_cards.copy_to_device(h_hole)
    batch.community.copy_to_device(h_community)
    batch.stage.copy_to_device(h_stage)
    batch.n_raises.copy_to_device(h_n_raises)
    batch.player_i_index.copy_to_device(h_pii)
    batch.pot_total.copy_to_device(h_pot)
    batch.history.copy_to_device(h_history)


def get_gpu_features(batch):
    """Run get_features_kernel and return (126,) float32 array."""
    d_features = cuda.device_array((1, N_FEATURES), dtype=np.float32)
    orders = _get_device_orders()
    d_preflop, d_postflop = orders[N_PLAYERS]

    get_features_kernel[1, 1](
        batch.chips, batch.bets, batch.active,
        batch.hole_cards, batch.community,
        batch.stage, batch.n_raises, batch.player_i_index,
        batch.pot_total, batch.history,
        N_PLAYERS, d_preflop, d_postflop,
        d_features, 1, INITIAL_CHIPS,
    )
    cuda.synchronize()
    return d_features.copy_to_host()[0]


# ---------------------------------------------------------------------------
# Feature index labels for readable diffs
# ---------------------------------------------------------------------------

def feature_name(idx):
    if idx < 52:
        rank = idx // 4 + 2
        suit = ['c', 'd', 'h', 's'][idx % 4]
        rank_str = {10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(rank, str(rank))
        return f"hole[{rank_str}{suit}]"
    if idx < 104:
        ci = idx - 52
        rank = ci // 4 + 2
        suit = ['c', 'd', 'h', 's'][ci % 4]
        rank_str = {10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(rank, str(rank))
        return f"comm[{rank_str}{suit}]"
    if idx < 108:
        return f"street_onehot[{['preflop','flop','turn','river'][idx-104]}]"
    names = ['pot_ratio', 'chips_ratio', 'bet_ratio', 'active_ratio', 'position', 'n_raises_ratio']
    if idx < 114:
        return f"scalar[{names[idx-108]}]"
    rd = (idx - 114) // 3
    tp = (idx - 114) % 3
    tp_name = ['calls', 'raises', 'folds'][tp]
    rd_name = ['preflop', 'flop', 'turn', 'river'][rd]
    return f"hist[{rd_name}_{tp_name}]"


def compare_features(gpu_feat, play_feat, scenario_name):
    """Compare two feature vectors and report differences."""
    mismatches = []
    for i in range(N_FEATURES):
        if not np.isclose(gpu_feat[i], play_feat[i], atol=1e-5):
            mismatches.append((i, feature_name(i), gpu_feat[i], play_feat[i]))

    if mismatches:
        print(f"\n  FAIL: {scenario_name} — {len(mismatches)} mismatches:")
        for idx, name, gv, pv in mismatches:
            print(f"    [{idx:3d}] {name:30s}  GPU={gv:.6f}  Play={pv:.6f}  diff={gv-pv:+.6f}")
        return False
    else:
        print(f"  PASS: {scenario_name}")
        return True


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def test_preflop_sb_first_to_act():
    """Scenario A: Preflop, SB first to act (no action yet).

    GPU state: player 0 = SB, player 1 = BB.
    2-player preflop order: [0, 1] → player_i_index=0 means current player is 0 (SB).

    Slumbot: client_pos=1 means we are SB. No action yet.
    """
    # Cards: SB has Ac Kh.
    sb_cards = ['Ac', 'Kh']
    sb_card_idx = [card_str_to_index(c) for c in sb_cards]

    # BB has random cards (doesn't matter for feature computation from SB perspective).
    bb_cards = [0, 1]  # arbitrary

    batch = make_gpu_batch()
    history = np.zeros((4, 3), dtype=np.int8)
    set_gpu_state(
        batch,
        hole_cards_p0=sb_card_idx,  # player 0 = SB
        hole_cards_p1=bb_cards,
        community=[-1, -1, -1, -1, -1],
        chips=[INITIAL_CHIPS - SMALL_BLIND, INITIAL_CHIPS - BIG_BLIND],
        bets=[SMALL_BLIND, BIG_BLIND],
        active=[1, 1],
        stage=PREFLOP,
        n_raises=0,
        player_i_index=0,  # first in preflop order [0,1] → player 0 (SB)
        pot_total=SMALL_BLIND + BIG_BLIND,
        history=history,
    )
    gpu_feat = get_gpu_features(batch)

    # Play features: client_pos=1 (SB), no action.
    action_str = ''
    parsed = parse_action(action_str)
    play_feat = build_features(sb_cards, [], action_str, client_pos=1, parsed=parsed)

    return compare_features(gpu_feat, play_feat, "A: Preflop SB first to act")


def test_preflop_bb_facing_raise():
    """Scenario B: Preflop, BB facing a raise to 200.

    GPU: SB (player 0) raised. Now player_i_index=1 → player 1 (BB) to act.
    Action: SB raised to 200 (total bet from SB's stack).

    Slumbot: client_pos=0 (BB). Action string: "b200" (SB bet to 200 on the street).
    pos=0 means it's BB's turn.
    """
    bb_cards = ['Js', 'Td']
    bb_card_idx = [card_str_to_index(c) for c in bb_cards]
    sb_cards_idx = [card_str_to_index('7h'), card_str_to_index('8c')]

    # After SB raises to 200 on the street:
    # SB (p0): bet=200, chips=INITIAL_CHIPS-200
    # BB (p1): bet=100 (posted BB), chips=INITIAL_CHIPS-100
    # pot = 200 + 100 = 300
    # history: preflop has 1 raise
    batch = make_gpu_batch()
    history = np.zeros((4, 3), dtype=np.int8)
    history[0, 1] = 1  # 1 raise in preflop

    set_gpu_state(
        batch,
        hole_cards_p0=sb_cards_idx,
        hole_cards_p1=bb_card_idx,
        community=[-1, -1, -1, -1, -1],
        chips=[INITIAL_CHIPS - 200, INITIAL_CHIPS - BIG_BLIND],
        bets=[200, BIG_BLIND],
        active=[1, 1],
        stage=PREFLOP,
        n_raises=1,
        player_i_index=1,  # second in preflop order [0,1] → player 1 (BB)
        pot_total=300,
        history=history,
    )
    gpu_feat = get_gpu_features(batch)

    # Play: client_pos=0 (BB), SB raised to 200.
    action_str = 'b200'
    parsed = parse_action(action_str)
    play_feat = build_features(bb_cards, [], action_str, client_pos=0, parsed=parsed)

    return compare_features(gpu_feat, play_feat, "B: Preflop BB facing raise to 200")


def test_postflop_both_checked_preflop():
    """Scenario C: Postflop after limp/check, BB first to act.

    Preflop: SB calls (limps), BB checks. Street ends.
    Flop dealt. Postflop order [1,0], player_i_index=0 → player 1 (BB) to act.

    GPU state after preflop SB call + BB check:
    - SB (p0): bet=100 (called BB), chips=INITIAL_CHIPS-100
    - BB (p1): bet=100 (posted), chips=INITIAL_CHIPS-100
    - pot = 200
    - Stage: FLOP
    - History: preflop calls=2 (SB call + BB check counted as call)
    - Actually, in the GPU the SB calls (action 1) → history[0,0]+=1 (call)
    - Then BB... wait. After SB calls, the BB doesn't act (betting is finished).
    - Let me re-examine: SB is first to act preflop. SB calls (limps to 100).
    - Then n_actions becomes >= n_players_started_round? No, n_actions=1, n_players_started_round=2.
    - Then BB checks (action 1, call). n_actions=2 >= 2 and bets equal. Street ends.
    - History: preflop calls=2

    Slumbot: SB limps = 'c' (call to match BB). BB checks = 'k'.
    Action: "ck/"  → preflop: c (SB call), k (BB check), then / = new street.
    Wait, no. In Slumbot format:
    - SB acts first preflop (pos=1). SB calls = 'c'. check_or_call_ends_street becomes True.
    - BB acts (pos=0). BB checks = 'k'. check_or_call_ends_street is True, so street ends.
    - Action string to flop first-to-act: 'ck/' (the '/' separates streets).
    - Actually, parse_action splits by processing chars. After 'c' ends the street
      followed by 'k', that's wrong — 'c' from SB means calling the BB, and then
      check_or_call_ends_street=True, so the next 'k' would end the street.
    - Wait: actually after SB calls preflop (matching BB), the check_or_call_ends_street
      is already True only if it was set by a previous check. Let me re-read.
    - parse_action: Initially check_or_call_ends_street=False.
    - SB does 'c': it's a call. check_or_call_ends_street is False, so we go to else
      branch: pos = (pos+1) % 2, check_or_call_ends_street = True.
    - BB does 'k': it's a check. check_or_call_ends_street is True, so street ends.

    Slumbot and the training stack both use BB first postflop in heads-up.
    This scenario must therefore compare BB/client_pos=0 features.
    """
    flop_cards = ['Qh', '9c', '4d']
    flop_idx = [card_str_to_index(c) for c in flop_cards]

    sb_cards = ['Ac', 'Kh']
    sb_card_idx = [card_str_to_index(c) for c in sb_cards]
    bb_cards = ['2c', '2d']
    bb_card_idx = [card_str_to_index(c) for c in bb_cards]

    batch = make_gpu_batch()
    history = np.zeros((4, 3), dtype=np.int8)
    history[0, 0] = 2  # preflop: 2 calls (SB limp + BB check)

    # Postflop: bets reset to 0 (new street).
    # Chips: both put in 100 preflop, so each has INITIAL_CHIPS - 100.
    # Pot: 200.
    set_gpu_state(
        batch,
        hole_cards_p0=sb_card_idx,
        hole_cards_p1=bb_card_idx,
        community=flop_idx + [-1, -1],
        chips=[INITIAL_CHIPS - 100, INITIAL_CHIPS - 100],
        bets=[100, 100],  # Total bets across all streets.
        active=[1, 1],
        stage=FLOP,
        n_raises=0,
        player_i_index=0,  # postflop order [1,0] → player 1 (BB)
        pot_total=200,
        history=history,
    )
    gpu_feat = get_gpu_features(batch)

    # Play: BB's turn on flop.
    # client_pos=0 (BB). Action string: 'ck/' (SB called, BB checked, flop).
    action_str = 'ck/'
    parsed = parse_action(action_str)
    play_feat = build_features(bb_cards, flop_cards, action_str, client_pos=0, parsed=parsed)

    return compare_features(gpu_feat, play_feat, "C: Postflop BB first to act (checked preflop)")


def test_postflop_after_bet_and_call():
    """Scenario D: Postflop, after a bet and call on flop.

    Preflop: SB raises to 200, BB calls. Flop: BB bets 200, SB calls.
    Now on the turn, BB acts first.

    GPU state:
    - Preflop: SB raise (history[0,1]+=1), BB call (history[0,0]+=1)
    - SB chips: 20000 - 200 = 19800 after preflop
    - BB chips: 20000 - 200 = 19800 after preflop (called 200)
    - Pot after preflop: 400
    - Flop: BB bets 200 more (raise action in training? or... need to think)
    - Actually in training, a "bet" on flop is a raise action (action 2-7) since
    - there's no outstanding bet. Raise amount = frac * pot + to_call.
    - With frac=0.5, pot=400, to_call=0: raise_chips = 200. That works.
    - So BB does action 3 (frac=0.5): history[1,1]+=1 (flop raise)
    - BB chips: 19800 - 200 = 19600, BB bets: 200+200=400
    - Pot: 400 + 200 = 600
    - SB calls: history[1,0]+=1 (flop call)
    - SB chips: 19800 - 200 = 19600, SB bets: 200+200=400
    - Pot: 600 + 200 = 800
    - Now on turn, bets stay (accumulated), n_raises reset to 0.

    Wait: bets array - does it reset per street in GPU?
    Looking at _advance in game_kernels.py: when a street ends (lines 126-156),
    it resets n_actions, n_raises, player_i_index, n_players_started_round.
    But it does NOT reset bets. So bets accumulate across streets.

    Slumbot: preflop 'b200c/' flop 'b200c/' turn.
    Wait: Slumbot preflop SB raise to 200 = 'b200', BB call = 'c', then '/'.
    Flop: SB bets 200 on the street = 'b200', BB calls = 'c', then '/'.
    Full action: 'b200c/b200c/'
    """
    turn_card = ['2s']
    flop_cards = ['Qh', '9c', '4d']
    community = flop_cards + turn_card
    comm_idx = [card_str_to_index(c) for c in community]

    sb_cards = ['Ac', 'Kh']
    sb_card_idx = [card_str_to_index(c) for c in sb_cards]
    bb_cards = ['2c', '2d']
    bb_card_idx = [card_str_to_index(c) for c in bb_cards]

    batch = make_gpu_batch()
    history = np.zeros((4, 3), dtype=np.int8)
    history[0, 0] = 1  # preflop: 1 call (BB)
    history[0, 1] = 1  # preflop: 1 raise (SB)
    history[1, 0] = 1  # flop: 1 call (BB)
    history[1, 1] = 1  # flop: 1 raise (SB bet)

    # After preflop+flop: each player put in 400 total.
    # Turn: bets still at 400 each (accumulated).
    set_gpu_state(
        batch,
        hole_cards_p0=sb_card_idx,
        hole_cards_p1=bb_card_idx,
        community=comm_idx + [-1],
        chips=[INITIAL_CHIPS - 400, INITIAL_CHIPS - 400],
        bets=[400, 400],
        active=[1, 1],
        stage=TURN,
        n_raises=0,
        player_i_index=0,  # postflop order [1,0] → player 1 (BB)
        pot_total=800,
        history=history,
    )
    gpu_feat = get_gpu_features(batch)

    # Play: client_pos=0 (BB). Action: 'b200c/b200c/'
    action_str = 'b200c/b200c/'
    parsed = parse_action(action_str)
    play_feat = build_features(bb_cards, community, action_str, client_pos=0, parsed=parsed)

    return compare_features(gpu_feat, play_feat, "D: Turn after bet-call on flop")


def test_turn_after_checks():
    """Scenario E: Turn after check-check on flop, bet on turn, opponent to act.

    Slumbot action: 'ck/kk/b200'
    - Preflop: SB (pos 1) calls, BB (pos 0) checks. Street ends.
    - Flop: BB (pos 0) checks, SB (pos 1) checks. Street ends.
    - Turn: BB (pos 0) bets 200. SB (pos 1) to act.
    We play as SB (client_pos=1).

    In Slumbot, postflop pos 0 = BB acts first. So 'b200' on the turn is
    BB's bet. Now SB faces the bet.

    GPU mapping: our_player_idx = 1 - client_pos = 1 - 1 = 0.
    So we (SB) are player 0 in the GPU. BB is player 1.
    But in GPU, player 0 = SB posts small blind, player 1 = BB posts big blind.

    GPU state from SB's (player 0) perspective:
    - SB (p0): preflop put in 100 (called), turn bet nothing yet = 100 total
    - BB (p1): preflop put in 100 (blind), turn bet 200 = 300 total
    - Pot: 100 + 100 + 200 = 400
    - Current player: SB (player 0), player_i_index depends on postflop order.
      GPU postflop order [1,0]: player_i_index=1 → player 0 (SB). So player_i_index=1.
      But wait, BB already acted (bet 200). After BB acts, advance moves to next player.
      In GPU postflop order [1,0]: BB is player 1, at index 0. After acting, advance
      goes to index (0+1)%2 = 1, which is player 0 (SB). So player_i_index=1. OK.

    History: preflop 2 calls, flop 2 calls (checks), turn 1 raise.
    n_raises on turn = 1.
    """
    turn_card = ['2s']
    flop_cards = ['Qh', '9c', '4d']
    community = flop_cards + turn_card
    comm_idx = [card_str_to_index(c) for c in community]

    sb_cards = ['Ac', 'Kh']
    sb_card_idx = [card_str_to_index(c) for c in sb_cards]
    bb_card_idx = [card_str_to_index('Js'), card_str_to_index('Td')]

    batch = make_gpu_batch()
    history = np.zeros((4, 3), dtype=np.int8)
    history[0, 0] = 2  # preflop: 2 calls (SB limp + BB check)
    history[1, 0] = 2  # flop: 2 checks (both check)
    history[2, 1] = 1  # turn: 1 raise (BB bet)

    # SB (p0): bet 100 total. BB (p1): bet 300 total.
    set_gpu_state(
        batch,
        hole_cards_p0=sb_card_idx,
        hole_cards_p1=bb_card_idx,
        community=comm_idx + [-1],
        chips=[INITIAL_CHIPS - 100, INITIAL_CHIPS - 300],
        bets=[100, 300],
        active=[1, 1],
        stage=TURN,
        n_raises=1,
        player_i_index=1,  # postflop order [1,0] → player 0 (SB) to act
        pot_total=400,
        history=history,
    )
    gpu_feat = get_gpu_features(batch)

    # Play: client_pos=1 (SB). Action: 'ck/kk/b200'
    action_str = 'ck/kk/b200'
    parsed = parse_action(action_str)
    play_feat = build_features(sb_cards, community, action_str, client_pos=1, parsed=parsed)

    return compare_features(gpu_feat, play_feat, "E: Turn, SB facing BB bet after check-check flop")


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------

def test_preflop_bb_after_limp():
    """SB limps (calls), BB to act.

    GPU: SB called (action 1). history[0,0]=1.
    player_i_index=1 → player 1 (BB).
    Bets: SB=100, BB=100. Pot=200. Chips: both 19900.

    Slumbot: 'c' (SB called). client_pos=0 (BB).
    """
    bb_cards = ['Qs', 'Jh']
    bb_card_idx = [card_str_to_index(c) for c in bb_cards]
    sb_card_idx = [card_str_to_index('2c'), card_str_to_index('3d')]

    batch = make_gpu_batch()
    history = np.zeros((4, 3), dtype=np.int8)
    history[0, 0] = 1  # SB called/limped

    set_gpu_state(
        batch,
        hole_cards_p0=sb_card_idx,
        hole_cards_p1=bb_card_idx,
        community=[-1, -1, -1, -1, -1],
        chips=[INITIAL_CHIPS - 100, INITIAL_CHIPS - 100],
        bets=[100, 100],
        active=[1, 1],
        stage=PREFLOP,
        n_raises=0,
        player_i_index=1,
        pot_total=200,
        history=history,
    )
    gpu_feat = get_gpu_features(batch)

    action_str = 'c'
    parsed = parse_action(action_str)
    play_feat = build_features(bb_cards, [], action_str, client_pos=0, parsed=parsed)

    return compare_features(gpu_feat, play_feat, "F: Preflop BB after SB limp")


def test_river_complex():
    """River scenario with multiple raises across streets.

    Preflop: SB raises to 300, BB calls.
    Flop: SB bets 300, BB raises to 900, SB calls.
    Turn: Both check.
    River: BB to act.

    GPU:
    - Preflop: history[0,1]=1 (SB raise), history[0,0]=1 (BB call)
    - Flop: history[1,1]=2 (SB bet + BB raise), history[1,0]=1 (SB call)
    - Turn: history[2,0]=2 (two checks)
    - River: n_raises=0, player_i_index=0 (BB)

    Bets accumulated: SB=300+900=1200, BB=300+900=1200. Pot=2400.

    Slumbot: 'b300c/b300b900c/kk/'
    """
    river_card = ['3h']
    turn_card = ['2s']
    flop_cards = ['Qh', '9c', '4d']
    community = flop_cards + turn_card + river_card
    comm_idx = [card_str_to_index(c) for c in community]

    sb_cards = ['Ac', 'Kh']
    sb_card_idx = [card_str_to_index(c) for c in sb_cards]
    bb_cards = ['Js', 'Td']
    bb_card_idx = [card_str_to_index(c) for c in bb_cards]

    batch = make_gpu_batch()
    history = np.zeros((4, 3), dtype=np.int8)
    history[0, 1] = 1  # preflop: 1 raise (SB)
    history[0, 0] = 1  # preflop: 1 call (BB)
    history[1, 1] = 2  # flop: 2 raises (SB bet + BB raise)
    history[1, 0] = 1  # flop: 1 call (SB call)
    history[2, 0] = 2  # turn: 2 checks

    set_gpu_state(
        batch,
        hole_cards_p0=sb_card_idx,
        hole_cards_p1=bb_card_idx,
        community=comm_idx,
        chips=[INITIAL_CHIPS - 1200, INITIAL_CHIPS - 1200],
        bets=[1200, 1200],
        active=[1, 1],
        stage=RIVER,
        n_raises=0,
        player_i_index=0,
        pot_total=2400,
        history=history,
    )
    gpu_feat = get_gpu_features(batch)

    action_str = 'b300c/b300b900c/kk/'
    parsed = parse_action(action_str)
    play_feat = build_features(bb_cards, community, action_str, client_pos=0, parsed=parsed)

    return compare_features(gpu_feat, play_feat, "G: River complex multi-street")


# ---------------------------------------------------------------------------
# Targeted issue tests
# ---------------------------------------------------------------------------

def test_check_k_vs_call_counting():
    """Verify that Slumbot 'k' (check) counts the same as checks in training.

    In training, a check is action 1 (call with no outstanding bet), and it
    increments history[rd, 0] (calls counter). In Slumbot, checks are 'k'
    and calls are 'c'. The play script's count_actions_in_street should count
    both 'k' and 'c' as calls.

    Test: Flop with two checks. history[1,0] should be 2.
    """
    # Slumbot action: 'ck/kk/' → preflop SB call + BB check, flop BB check + SB check.
    # On flop street ('kk'): 2 checks.
    n_calls, n_raises, n_folds = count_actions_in_street('ck/kk/', 1)
    ok1 = (n_calls == 2 and n_raises == 0 and n_folds == 0)

    # Verify 'c' also counts: preflop ('ck'): 1 call + 1 check = 2 calls.
    n_calls2, n_raises2, n_folds2 = count_actions_in_street('ck/kk/', 0)
    ok2 = (n_calls2 == 2 and n_raises2 == 0 and n_folds2 == 0)

    # Mixed: preflop with raise then call: 'b200c' = 0 calls? No: 1 call.
    n_calls3, n_raises3, n_folds3 = count_actions_in_street('b200c/kk/', 0)
    ok3 = (n_calls3 == 1 and n_raises3 == 1 and n_folds3 == 0)

    passed = ok1 and ok2 and ok3
    if passed:
        print("  PASS: H: Check 'k' counted as call in history")
    else:
        print(f"  FAIL: H: Check 'k' counting")
        print(f"    Flop 'kk': calls={n_calls} (expected 2)")
        print(f"    Preflop 'ck': calls={n_calls2} (expected 2)")
        print(f"    Preflop 'b200c': calls={n_calls3}, raises={n_raises3} (expected 1, 1)")
    return passed


def test_position_encoding_2player():
    """Verify position encoding for 2-player is correct.

    GPU: features[112] = float(pi) / (n_players - 1)
    - SB = player 0 → position = 0.0
    - BB = player 1 → position = 1.0

    Play: features[112] = our_player_idx / max(n_players - 1, 1)
    - client_pos=1 (SB) → our_player_idx = 1 - 1 = 0 → position = 0.0
    - client_pos=0 (BB) → our_player_idx = 1 - 0 = 1 → position = 1.0
    """
    # SB perspective (client_pos=1).
    action_str = ''
    parsed = parse_action(action_str)
    feat_sb = build_features(['Ac', 'Kh'], [], action_str, client_pos=1, parsed=parsed)
    sb_pos = feat_sb[112]

    # BB perspective (client_pos=0, after SB raises).
    action_str = 'b200'
    parsed = parse_action(action_str)
    feat_bb = build_features(['Js', 'Td'], [], action_str, client_pos=0, parsed=parsed)
    bb_pos = feat_bb[112]

    ok = np.isclose(sb_pos, 0.0) and np.isclose(bb_pos, 1.0)
    if ok:
        print("  PASS: I: Position encoding (SB=0.0, BB=1.0)")
    else:
        print(f"  FAIL: I: Position encoding — SB pos={sb_pos:.4f} (expected 0.0), BB pos={bb_pos:.4f} (expected 1.0)")
    return ok


def test_normalization_divisors():
    """Verify pot/chip/bet normalizations use correct divisors.

    GPU kernel:
    - pot_ratio = pot_total / (initial_chips * n_players)  → /40000
    - chips_ratio = chips[pi] / initial_chips              → /20000
    - bet_ratio = bets[pi] / initial_chips                 → /20000
    - active_ratio = active_count / n_players              → /2
    - n_raises_ratio = n_raises / 3.0

    Play:
    - pot_ratio = pot_total / (STACK_SIZE * n_players)     → /40000
    - chips_ratio = our_chips / STACK_SIZE                 → /20000
    - bet_ratio = our_bet / STACK_SIZE                     → /20000
    - active_ratio = 1.0                                   (always 1.0 in HU)
    - n_raises_ratio = n_raises / 3.0
    """
    # Use a specific scenario: SB raises to 500 preflop, BB to act.
    # Pot = 500 + 100 = 600.
    action_str = 'b500'
    parsed = parse_action(action_str)
    feat = build_features(['Js', 'Td'], [], action_str, client_pos=0, parsed=parsed)

    # Expected: pot=600, our_chips(BB)=20000-100=19900, our_bet(BB)=100.
    expected_pot_ratio = 600.0 / 40000.0     # 0.015
    expected_chips = 19900.0 / 20000.0       # 0.995
    expected_bet = 100.0 / 20000.0           # 0.005
    expected_active = 1.0
    expected_position = 1.0                  # BB = player 1
    expected_n_raises = 1.0 / 3.0            # 1 raise on this street

    checks = [
        (108, 'pot_ratio', expected_pot_ratio),
        (109, 'chips_ratio', expected_chips),
        (110, 'bet_ratio', expected_bet),
        (111, 'active_ratio', expected_active),
        (112, 'position', expected_position),
        (113, 'n_raises_ratio', expected_n_raises),
    ]

    all_ok = True
    for idx, name, expected in checks:
        if not np.isclose(feat[idx], expected, atol=1e-5):
            print(f"    [{idx}] {name}: got {feat[idx]:.6f}, expected {expected:.6f}")
            all_ok = False

    if all_ok:
        print("  PASS: J: Normalization divisors (STACK_SIZE=20000)")
    else:
        print("  FAIL: J: Normalization divisors")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Feature Encoding Verification: GPU Training vs Play Script")
    print("=" * 60)
    print(f"INITIAL_CHIPS={INITIAL_CHIPS}, N_PLAYERS={N_PLAYERS}")
    print()

    print("--- GPU vs Play feature vector comparisons ---")
    results = []
    results.append(test_preflop_sb_first_to_act())
    results.append(test_preflop_bb_facing_raise())
    results.append(test_postflop_both_checked_preflop())
    results.append(test_postflop_after_bet_and_call())
    results.append(test_turn_after_checks())
    results.append(test_preflop_bb_after_limp())
    results.append(test_river_complex())

    print()
    print("--- Targeted issue checks ---")
    results.append(test_check_k_vs_call_counting())
    results.append(test_position_encoding_2player())
    results.append(test_normalization_divisors())

    print()
    n_pass = sum(results)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} passed")

    if n_pass < n_total:
        print("\nDIAGNOSTICS:")
        print("  Check feature indices above for specific mismatches.")
        print("  Common issues:")
        print("  - Position encoding: our_player_idx mapping")
        print("  - Pot/chip normalization divisors")
        print("  - History counting: 'k' vs 'c' for checks")
        print("  - Bet accumulation across streets")
        sys.exit(1)
    else:
        print("\nAll features match between GPU training and play script.")
        sys.exit(0)


if __name__ == '__main__':
    main()
