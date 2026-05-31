import numpy as np
import pytest

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES


pytest.importorskip("pettingzoo")


from poker_ai.research.native_pettingzoo import NativeNoLimitHoldemAECEnv  # noqa: E402


def test_native_pettingzoo_env_exposes_native_9_action_mask():
    env = NativeNoLimitHoldemAECEnv(seed=20260634, initial_chips=1000, max_steps_per_hand=32)

    env.reset(seed=20260634)
    obs, reward, terminated, truncated, info = env.last()

    assert env.possible_agents == ["player_0", "player_1"]
    assert env.agent_selection == "player_0"
    assert env.action_space(env.agent_selection).n == N_ACTIONS
    assert obs["observation"].shape == (N_FEATURES,)
    assert obs["observation"].dtype == np.float32
    assert obs["action_mask"].shape == (N_ACTIONS,)
    assert obs["action_mask"].dtype == np.int8
    assert obs["action_mask"].sum() >= 2
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["player_i"] == 0


def test_native_pettingzoo_env_steps_to_next_poker_actor():
    env = NativeNoLimitHoldemAECEnv(seed=20260635, initial_chips=1000, max_steps_per_hand=32)
    env.reset(seed=20260635)
    obs, *_ = env.last()
    legal_action = 1 if bool(obs["action_mask"][1]) else int(np.flatnonzero(obs["action_mask"])[0])

    env.step(legal_action)
    next_obs, _reward, terminated, truncated, info = env.last()

    assert env.agent_selection in {"player_0", "player_1"}
    assert next_obs["observation"].shape == (N_FEATURES,)
    assert next_obs["action_mask"].shape == (N_ACTIONS,)
    assert terminated is False
    assert truncated is False
    assert info["player_i"] in {0, 1}


def test_native_pettingzoo_env_wraps_with_tianshou_pettingzoo_env():
    pytest.importorskip("tianshou")
    from tianshou.env import PettingZooEnv

    env = PettingZooEnv(
        NativeNoLimitHoldemAECEnv(seed=20260636, initial_chips=1000, max_steps_per_hand=32)
    )

    obs, info = env.reset(seed=20260636)

    assert obs["agent_id"] == "player_0"
    assert obs["obs"].shape == (N_FEATURES,)
    assert len(obs["mask"]) == N_ACTIONS
    assert any(obs["mask"])
    assert info["player_i"] == 0


def test_native_pettingzoo_env_supports_fast_state_backend():
    env = NativeNoLimitHoldemAECEnv(
        seed=20260759,
        initial_chips=1000,
        max_steps_per_hand=32,
        state_backend="fast-state",
    )

    env.reset(seed=20260759)
    obs, reward, terminated, truncated, info = env.last()
    legal_action = 1 if bool(obs["action_mask"][1]) else int(np.flatnonzero(obs["action_mask"])[0])
    env.step(legal_action)
    next_obs, _next_reward, _next_terminated, _next_truncated, next_info = env.last()

    assert info["state_backend"] == "fast-state"
    assert next_info["state_backend"] == "fast-state"
    assert obs["observation"].shape == (N_FEATURES,)
    assert next_obs["observation"].shape == (N_FEATURES,)
    assert obs["action_mask"].sum() >= 2
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
