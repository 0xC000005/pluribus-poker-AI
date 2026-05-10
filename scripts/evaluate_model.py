"""Evaluate a trained 6-player Deep CFR model."""
import numpy as np
import torch

from cuda_env import configure_numba_cuda_env

configure_numba_cuda_env()
from poker_ai.deep_cfr.cuda.gpu_trainer import GPUDeepCFRTrainer
from poker_ai.deep_cfr.fast_state import FastPokerState, new_fast_game, N_ACTIONS, N_FEATURES, RAISE_FRACTIONS
from poker_ai.deep_cfr.networks import ValueNetwork

ACTION_NAMES = [
    "fold", "call",
    "raise 0.25x", "raise 0.5x", "raise 0.75x",
    "raise 1.0x", "raise 1.5x", "raise 2.0x",
    "all-in",
]


def regret_match(advantages, legal_mask):
    """CPU regret matching."""
    strategy = np.maximum(advantages, 0) * legal_mask
    total = strategy.sum()
    if total > 0:
        strategy /= total
    else:
        n_legal = legal_mask.sum()
        if n_legal > 0:
            strategy = legal_mask / n_legal
    return strategy


def play_game_with_strategy(value_net, n_players=6, device="cuda"):
    """Play one game, return payouts and action log."""
    state = new_fast_game(n_players)
    actions_taken = []

    for _ in range(500):  # safety limit
        if state.is_terminal:
            break
        pi = state.current_player_i
        if not state.active[pi]:
            child = state.copy()
            child.apply_action(None)
            state = child
            continue

        mask = state.get_legal_mask()
        features = state.to_feature_vector()
        feat_t = torch.from_numpy(features).unsqueeze(0).to(device)

        with torch.no_grad():
            adv = value_net(feat_t).cpu().numpy()[0]

        strategy = regret_match(adv, mask)
        legal = np.where(mask > 0)[0]

        if strategy.sum() > 0:
            action = np.random.choice(N_ACTIONS, p=strategy)
        else:
            action = np.random.choice(legal)

        actions_taken.append((pi, int(action), state.stage))
        child = state.copy()
        child.apply_action(int(action))
        state = child

    return state.payout if state.is_terminal else None, actions_taken


def play_vs_random(value_net, agent_seat, n_players=6, device="cuda"):
    """Play one game: agent at seat `agent_seat`, random at all other seats."""
    state = new_fast_game(n_players)

    for _ in range(500):
        if state.is_terminal:
            break
        pi = state.current_player_i
        if not state.active[pi]:
            child = state.copy()
            child.apply_action(None)
            state = child
            continue

        mask = state.get_legal_mask()
        legal = np.where(mask > 0)[0]

        if pi == agent_seat:
            features = state.to_feature_vector()
            feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
            with torch.no_grad():
                adv = value_net(feat_t).cpu().numpy()[0]
            strategy = regret_match(adv, mask)
            if strategy.sum() > 0:
                action = np.random.choice(N_ACTIONS, p=strategy)
            else:
                action = np.random.choice(legal)
        else:
            action = np.random.choice(legal)

        child = state.copy()
        child.apply_action(int(action))
        state = child

    if state.is_terminal:
        return state.payout[agent_seat]
    return 0


def main():
    model_path = "models/deep_cfr_9action_6p_final.pt"
    print("=" * 60)
    print(f"Evaluating: {model_path}")
    print("=" * 60)

    checkpoint = torch.load(model_path, map_location="cuda", weights_only=False)
    hidden_dim = checkpoint.get("hidden_dim", 256)
    value_net = ValueNetwork(N_FEATURES, hidden_dim, N_ACTIONS).cuda()
    value_net.load_state_dict(checkpoint["value_net"])
    value_net.eval()
    print(f"Loaded model (iter {checkpoint['iteration']}, hidden={hidden_dim})\n")

    # ---------------------------------------------------------------
    # 1. Large-scale eval vs random (agent rotates through all seats)
    # ---------------------------------------------------------------
    print("--- Eval vs Random (10,000 games, rotating seats) ---")
    n_games = 10000
    n_players = 6
    payouts = []
    for g in range(n_games):
        seat = g % n_players
        p = play_vs_random(value_net, seat, n_players)
        payouts.append(p)
        if (g + 1) % 2000 == 0:
            avg = np.mean(payouts)
            se = np.std(payouts) / np.sqrt(len(payouts))
            print(f"  {g+1:5d} games: {avg:+.0f} +/- {1.96*se:.0f} chips/game")

    avg = np.mean(payouts)
    se = np.std(payouts) / np.sqrt(len(payouts))
    print(f"\n  RESULT: {avg:+.0f} +/- {1.96*se:.0f} chips/game vs random (95% CI)")
    print(f"  Win rate: {np.mean(np.array(payouts) > 0)*100:.1f}%")
    print(f"  Lose rate: {np.mean(np.array(payouts) < 0)*100:.1f}%")
    print(f"  Break even: {np.mean(np.array(payouts) == 0)*100:.1f}%")

    # ---------------------------------------------------------------
    # 2. Strategy distribution (self-play)
    # ---------------------------------------------------------------
    print("\n--- Strategy Distribution (1,000 self-play games) ---")
    action_counts = np.zeros(N_ACTIONS)
    stage_action_counts = np.zeros((4, N_ACTIONS))  # preflop/flop/turn/river
    total_games = 0
    total_actions = 0

    for _ in range(1000):
        payout, actions = play_game_with_strategy(value_net)
        if payout is not None:
            total_games += 1
            for pi, action, stage in actions:
                action_counts[action] += 1
                if stage < 4:
                    stage_action_counts[stage, action] += 1
                total_actions += 1

    print(f"  Games: {total_games}, Actions: {total_actions}")
    print(f"  Avg actions/game: {total_actions/max(total_games,1):.1f}\n")

    # Overall action distribution
    if action_counts.sum() > 0:
        pcts = action_counts / action_counts.sum() * 100
        print("  Overall action distribution:")
        for i, name in enumerate(ACTION_NAMES):
            if pcts[i] > 0.1:
                print(f"    {name:15s}: {pcts[i]:5.1f}%  ({int(action_counts[i]):,})")

    # Per-stage breakdown
    stage_names = ["Preflop", "Flop", "Turn", "River"]
    print("\n  Per-stage breakdown:")
    for s, sname in enumerate(stage_names):
        total = stage_action_counts[s].sum()
        if total == 0:
            continue
        pcts = stage_action_counts[s] / total * 100
        parts = []
        for i, name in enumerate(ACTION_NAMES):
            if pcts[i] > 0.5:
                parts.append(f"{name}={pcts[i]:.0f}%")
        print(f"    {sname:8s}: {', '.join(parts)}")

    # ---------------------------------------------------------------
    # 3. Per-position eval vs random
    # ---------------------------------------------------------------
    print("\n--- Per-Position Performance (2,000 games/seat) ---")
    pos_names = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
    for seat in range(n_players):
        seat_payouts = []
        for _ in range(2000):
            p = play_vs_random(value_net, seat, n_players)
            seat_payouts.append(p)
        avg = np.mean(seat_payouts)
        se = np.std(seat_payouts) / np.sqrt(len(seat_payouts))
        print(f"  {pos_names[seat]:3s} (seat {seat}): {avg:+7.0f} +/- {1.96*se:.0f}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
