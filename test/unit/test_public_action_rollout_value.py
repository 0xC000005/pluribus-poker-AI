import numpy as np

from poker_ai.deep_cfr.fast_state import FastPokerState
from poker_ai.research import public_action_rollout_value as rollout_module
from poker_ai.research.public_action_rollout_value import (
    PublicActionRolloutConfig,
    build_public_world_state,
    call_policy,
    evaluate_public_action_rollout_values,
    score_first_actions_across_worlds,
    sample_public_worlds,
    select_deployed_root_action,
)


def _always_call_policy(state: FastPokerState, rng: np.random.Generator) -> int:
    mask = state.get_legal_mask()
    if mask[1] > 0:
        return 1
    legal = np.flatnonzero(mask > 0)
    return int(legal[0])


def test_build_public_world_state_preserves_public_sampling_contract():
    hero = (50, 51)
    opponent = (0, 1)
    tail = [2, 3, 4, 5, 6, 7]

    state = build_public_world_state(
        hero_cards=hero,
        opponent_cards=opponent,
        deck_tail=tail,
        initial_chips=1000,
    )

    assert state.current_player_i == 0
    assert tuple(int(c) for c in state.hole_cards[0]) == hero
    assert tuple(int(c) for c in state.hole_cards[1]) == opponent
    assert int(state.deck_cursor) == 4
    assert len(set(int(c) for c in state.deck_order)) == 52

    state.apply_action(1)
    state.apply_action(1)

    assert state.stage == FastPokerState.FLOP
    assert [int(c) for c in state.community[:3]] == tail[:3]


def test_sample_public_worlds_excludes_known_cards_and_reuses_hero_cards():
    hero = (50, 51)
    worlds = sample_public_worlds(hero_cards=hero, n_worlds=16, seed=20260521)

    assert len(worlds) == 16
    for world in worlds:
        cards = [*hero, *world.opponent_cards, *world.deck_tail]
        assert len(cards) == 52
        assert len(set(cards)) == 52
        assert tuple(world.hero_cards) == hero
        assert not set(world.opponent_cards).intersection(hero)


def test_score_first_actions_uses_public_worlds_and_scores_fold_exactly():
    hero = (50, 51)
    worlds = sample_public_worlds(hero_cards=hero, n_worlds=8, seed=20260522)

    result = score_first_actions_across_worlds(
        worlds=worlds,
        continuation_policy=_always_call_policy,
        seed=20260523,
        max_steps_per_hand=128,
    )

    assert result.legal_actions
    assert result.action_values[0] == -50.0
    assert result.best_action in result.action_values
    assert result.action_values[result.best_action] >= result.action_values[0]


def test_score_first_actions_reuses_one_state_build_per_public_world(monkeypatch):
    hero = (50, 51)
    worlds = sample_public_worlds(hero_cards=hero, n_worlds=4, seed=20260526)
    calls = 0
    original_build = rollout_module.build_public_world_state

    def counted_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "build_public_world_state", counted_build)

    result = score_first_actions_across_worlds(
        worlds=worlds,
        continuation_policy=_always_call_policy,
        seed=20260527,
        max_steps_per_hand=128,
    )

    assert result.legal_actions
    assert calls <= len(worlds) + 1


def test_score_first_actions_specializes_builtin_call_policy_masks(monkeypatch):
    hero = (50, 51)
    worlds = sample_public_worlds(hero_cards=hero, n_worlds=4, seed=20260528)
    calls = 0
    original_get_legal_mask = FastPokerState.get_legal_mask

    def counted_get_legal_mask(self):
        nonlocal calls
        calls += 1
        return original_get_legal_mask(self)

    monkeypatch.setattr(FastPokerState, "get_legal_mask", counted_get_legal_mask)

    result = score_first_actions_across_worlds(
        worlds=worlds,
        continuation_policy=call_policy,
        seed=20260529,
        max_steps_per_hand=128,
    )

    assert result.legal_actions
    assert result.action_values[0] == -50.0
    assert calls <= 1


def test_public_rollout_positive_control_prefers_aa_pressure_to_fold():
    hero_aces = (50, 51)
    worlds = sample_public_worlds(hero_cards=hero_aces, n_worlds=64, seed=20260524)

    result = score_first_actions_across_worlds(
        worlds=worlds,
        continuation_policy=_always_call_policy,
        seed=20260525,
        max_steps_per_hand=128,
    )

    assert result.action_values[8] > result.action_values[0]
    assert result.best_action != 0


def test_select_deployed_root_action_matches_regret_greedy_advantage_argmax():
    legal_mask = np.array([1, 1, 0, 1], dtype=np.float32)
    advantages = np.array([-4.0, -1.0, 10.0, -2.0], dtype=np.float32)
    strategy = legal_mask / legal_mask.sum()

    selected = select_deployed_root_action(
        advantages=advantages,
        strategy=strategy,
        legal_mask=legal_mask,
        strategy_source="regret",
        greedy=True,
    )

    assert selected == 1


def test_evaluate_can_use_separate_continuation_checkpoint(monkeypatch):
    calls = []

    class FakeLoaded:
        def __init__(self, path: str):
            self.value_net = object()
            self.metadata = {
                "checkpoint": path,
                "has_policy_head": False,
                "has_average_policy_net": False,
            }

    def fake_load(path, device):
        calls.append(str(path))
        return FakeLoaded(str(path))

    monkeypatch.setattr(
        "poker_ai.research.public_action_rollout_value.load_value_network_checkpoint",
        fake_load,
    )

    metrics = evaluate_public_action_rollout_values(
        PublicActionRolloutConfig(
            n_roots=0,
            checkpoint="candidate.pt",
            continuation_checkpoint="continuation.pt",
            include_positive_controls=False,
            device="cpu",
        )
    )

    assert calls == ["candidate.pt", "continuation.pt"]
    assert metrics["checkpoint"] == "candidate.pt"
    assert metrics["continuation_checkpoint"] == "continuation.pt"


def test_evaluate_can_use_continuation_checkpoint_population(monkeypatch):
    calls = []

    class FakeLoaded:
        def __init__(self, path: str):
            self.value_net = object()
            self.metadata = {
                "checkpoint": path,
                "has_policy_head": False,
                "has_average_policy_net": False,
            }

    def fake_load(path, device):
        calls.append(str(path))
        return FakeLoaded(str(path))

    monkeypatch.setattr(
        "poker_ai.research.public_action_rollout_value.load_value_network_checkpoint",
        fake_load,
    )

    metrics = evaluate_public_action_rollout_values(
        PublicActionRolloutConfig(
            n_roots=0,
            checkpoint="candidate.pt",
            continuation_checkpoints=("a.pt", "b.pt"),
            include_positive_controls=False,
            device="cpu",
        )
    )

    assert calls == ["candidate.pt", "a.pt", "b.pt"]
    assert metrics["continuation_checkpoints"] == ["a.pt", "b.pt"]
