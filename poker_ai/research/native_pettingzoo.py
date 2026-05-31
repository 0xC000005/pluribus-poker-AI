"""PettingZoo adapter for the native full-deck HU NLHE environment.

The adapter exists so maintained multi-agent RL libraries can train on the
repo's native 9-action game contract. It deliberately contains no RL algorithm
logic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from poker_ai.deep_cfr.fast_state import new_fast_game
from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, new_game
from poker_ai.research.native_nfsp import get_legal_mask


class NativeNoLimitHoldemAECEnv(AECEnv):
    """AEC wrapper over the native heads-up no-limit Texas Hold'em state."""

    metadata = {"name": "poker_ai_native_hu_nlhe_v0", "render_modes": []}

    def __init__(
        self,
        *,
        seed: int = 20260526,
        initial_chips: int = 1000,
        max_steps_per_hand: int = 256,
        state_backend: str = "full-deck",
    ) -> None:
        self.possible_agents = ["player_0", "player_1"]
        self.agents = list(self.possible_agents)
        self.initial_chips = int(initial_chips)
        self.max_steps_per_hand = int(max_steps_per_hand)
        self.state_backend = str(state_backend)
        if self.state_backend not in {"full-deck", "fast-state"}:
            raise ValueError("state_backend must be one of: full-deck, fast-state")
        self._seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.state_obj = None
        self.n_steps = 0
        self.agent_selection = self.possible_agents[0]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: self._info_for_agent(agent) for agent in self.possible_agents}

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(N_FEATURES,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(0, 1, shape=(N_ACTIONS,), dtype=np.int8),
            }
        )

    @lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return spaces.Discrete(N_ACTIONS)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
        self.agents = list(self.possible_agents)
        self.state_obj = self._new_state()
        self.n_steps = 0
        self.agent_selection = self._agent_name(self._current_player_i())
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: self._info_for_agent(agent) for agent in self.possible_agents}

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        if self.state_obj is None or self.state_obj.is_terminal:
            features = np.zeros(N_FEATURES, dtype=np.float32)
            mask = np.zeros(N_ACTIONS, dtype=np.int8)
        else:
            features = self._feature_vector()
            if agent == self.agent_selection:
                mask = self._legal_mask().astype(np.int8, copy=False)
            else:
                mask = np.zeros(N_ACTIONS, dtype=np.int8)
        return {"observation": features, "action_mask": mask}

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            self._was_dead_step(action)
            return
        if self.state_obj is None:
            raise RuntimeError("reset must be called before step")
        if action is None:
            raise ValueError("live poker agents require an integer action")

        legal_mask = self._legal_mask().astype(bool)
        action_i = int(action)
        if action_i < 0 or action_i >= N_ACTIONS or not bool(legal_mask[action_i]):
            raise ValueError(f"illegal action {action_i} for {agent}")

        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        self._apply_action(action_i)
        self.n_steps += 1

        terminated = bool(self.state_obj.is_terminal)
        truncated = bool(self.n_steps >= self.max_steps_per_hand and not terminated)
        if terminated:
            for player_i, name in enumerate(self.possible_agents):
                self.rewards[name] = float(self.state_obj.payout.get(player_i, 0)) / float(
                    self.initial_chips
                )
                self.terminations[name] = True
        elif truncated:
            for name in self.possible_agents:
                self.truncations[name] = True
        else:
            self.agent_selection = self._agent_name(self._current_player_i())

        self.infos = {name: self._info_for_agent(name) for name in self.possible_agents}
        self._accumulate_rewards()

    def state(self) -> np.ndarray:
        if self.state_obj is None or self.state_obj.is_terminal:
            return np.zeros(N_FEATURES, dtype=np.float32)
        return self._feature_vector()

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _agent_name(self, player_i: int) -> str:
        return self.possible_agents[int(player_i)]

    def _player_i(self, agent: str) -> int:
        return int(agent.split("_", 1)[1])

    def _info_for_agent(self, agent: str) -> dict[str, Any]:
        player_i = self._player_i(agent)
        return {
            "player_i": player_i,
            "is_current_player": agent == self.agent_selection,
            "n_steps": int(self.n_steps),
            "state_backend": self.state_backend,
        }

    def _new_state(self):
        if self.state_backend == "fast-state":
            return new_fast_game(2, initial_chips=self.initial_chips)
        return new_game(2, initial_chips=self.initial_chips)

    def _legal_mask(self) -> np.ndarray:
        if self.state_obj is None:
            return np.zeros(N_ACTIONS, dtype=np.float32)
        if self.state_backend == "fast-state":
            return self.state_obj.get_legal_mask()
        return get_legal_mask(self.state_obj)

    def _feature_vector(self) -> np.ndarray:
        assert self.state_obj is not None
        return self.state_obj.to_feature_vector().astype(np.float32, copy=False)

    def _apply_action(self, action_i: int) -> None:
        assert self.state_obj is not None
        if self.state_backend == "fast-state":
            self.state_obj.apply_action(int(action_i))
        else:
            self.state_obj = self.state_obj.apply_action(INDEX_TO_ACTION[int(action_i)])

    def _current_player_i(self) -> int:
        assert self.state_obj is not None
        if self.state_backend == "fast-state":
            return int(self.state_obj.current_player_i)
        return int(self.state_obj.player_i)
