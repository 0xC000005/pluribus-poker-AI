import pytest


pytest.importorskip("pettingzoo")

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.rllib_native import RllibNativeNoLimitHoldemAECEnv


def test_rllib_native_env_uses_rllib_action_mask_observation_keys():
    env = RllibNativeNoLimitHoldemAECEnv(seed=20260716, max_steps_per_hand=16)
    env.reset()

    space = env.observation_space("player_0")
    obs = env.observe(env.agent_selection)

    assert set(space.spaces) == {"action_mask", "observations"}
    assert space["action_mask"].shape == (N_ACTIONS,)
    assert space["observations"].shape == (N_FEATURES,)
    assert set(obs) == {"action_mask", "observations"}
    assert obs["action_mask"].shape == (N_ACTIONS,)
    assert obs["observations"].shape == (N_FEATURES,)
    assert env.action_space("player_0").n == N_ACTIONS
