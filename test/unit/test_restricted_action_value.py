import random

import numpy as np
import torch

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.games.full_deck.state import new_game
from poker_ai.poker.card import Card
from poker_ai.research.restricted_action_value import (
    RestrictedActionValueConfig,
    evaluate_restricted_action_values,
    sample_seeded_hole_cards,
    score_legal_actions_by_showdown_equity,
)


def _set_hole_cards(state, player_i: int, cards: list[Card]):
    state.players[player_i].cards = list(cards)
    known = {card for player in state.players for card in player.cards}
    state._table.dealer.deck._cards_in_deck = [
        card for card in state._table.dealer.deck._cards_in_deck if card not in known
    ]
    return state


def test_showdown_action_values_choose_pressure_with_premium_preflop():
    random.seed(7)
    np.random.seed(7)
    state = new_game(2, initial_chips=1000)
    _set_hole_cards(state, 0, [Card("ace", "spades"), Card("ace", "hearts")])

    values = score_legal_actions_by_showdown_equity(
        state,
        player_i=0,
        n_equity_samples=800,
        seed=7,
    )

    assert values["best_action"] == "all_in"
    assert values["action_values"]["all_in"] > values["action_values"]["call"]
    assert values["action_values"]["call"] > values["action_values"]["fold"]
    assert values["equity"] > 0.75


def test_showdown_action_values_fold_trash_when_facing_large_raise():
    random.seed(11)
    np.random.seed(11)
    state = new_game(2, initial_chips=1000)
    _set_hole_cards(state, 1, [Card("7", "clubs"), Card("2", "diamonds")])
    state = state.apply_action("raise_2.0")

    values = score_legal_actions_by_showdown_equity(
        state,
        player_i=1,
        n_equity_samples=800,
        seed=11,
    )

    assert values["best_action"] == "fold"
    assert values["action_values"]["fold"] > values["action_values"]["call"]
    assert values["equity"] < 0.45


def test_restricted_action_value_positive_controls_pass():
    metrics = evaluate_restricted_action_values(
        RestrictedActionValueConfig(
            n_roots=24,
            n_equity_samples=256,
            initial_chips=1000,
            seed=20260515,
            include_positive_controls=True,
        )
    )

    assert metrics["algorithm"] == "restricted_showdown_action_value"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["promotion"] is False
    assert metrics["positive_controls"]["premium_pressure"]["passed"] is True
    assert metrics["positive_controls"]["trash_vs_large_raise"]["passed"] is True
    assert metrics["oracle_best_mean_payoff"] >= metrics["call_mean_payoff"]


def test_seeded_root_hole_cards_are_deterministic_and_unique():
    first = sample_seeded_hole_cards(seed=20260516, root_idx=3)
    second = sample_seeded_hole_cards(seed=20260516, root_idx=3)
    other = sample_seeded_hole_cards(seed=20260516, root_idx=4)

    assert first == second
    assert len(set(first)) == 2
    assert first != other


def test_restricted_action_value_cli_builds_config():
    from scripts.eval_restricted_action_values import build_config

    cfg = build_config(
        [
            "--n-roots",
            "17",
            "--n-equity-samples",
            "64",
            "--initial-chips",
            "2000",
            "--seed",
            "123",
            "--no-positive-controls",
        ]
    )

    assert cfg.n_roots == 17
    assert cfg.n_equity_samples == 64
    assert cfg.initial_chips == 2000
    assert cfg.seed == 123
    assert cfg.include_positive_controls is False


def test_restricted_action_value_scores_checkpoint_oracle_gap(tmp_path):
    checkpoint = tmp_path / "allin.pt"
    net = ValueNetwork(N_FEATURES, hidden_dim=16, output_dim=N_ACTIONS, n_layers=1)
    with torch.no_grad():
        for param in net.parameters():
            param.zero_()
        net.adv_head.bias[8] = 1.0
    torch.save(
        {
            "value_net": net.state_dict(),
            "hidden_dim": 16,
            "n_layers": 1,
            "n_players": 2,
            "initial_chips": 1000,
            "iteration": 1,
        },
        checkpoint,
    )

    metrics = evaluate_restricted_action_values(
        RestrictedActionValueConfig(
            n_roots=8,
            n_equity_samples=128,
            initial_chips=1000,
            seed=20260516,
            include_positive_controls=False,
            checkpoint=str(checkpoint),
            strategy_source="regret",
            device="cpu",
        )
    )

    assert metrics["checkpoint"] == str(checkpoint)
    assert metrics["checkpoint_action_counts"]["all_in"] == 8
    assert 0.0 <= metrics["checkpoint_oracle_match_rate"] <= 1.0
    assert metrics["checkpoint_mean_oracle_gap"] >= 0.0
    assert -1.0 <= metrics["checkpoint_mean_advantage_value_corr"] <= 1.0
