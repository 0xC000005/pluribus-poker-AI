"""RLlib adapters for the native full-deck poker environment."""

from __future__ import annotations

from functools import lru_cache

from gymnasium import spaces

from poker_ai.research.native_pettingzoo import NativeNoLimitHoldemAECEnv


class RllibNativeNoLimitHoldemAECEnv(NativeNoLimitHoldemAECEnv):
    """Native HU NLHE PettingZoo env with RLlib action-mask observation keys."""

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        base = super().observation_space(agent)
        return spaces.Dict(
            {
                "observations": base["observation"],
                "action_mask": base["action_mask"],
            }
        )

    def observe(self, agent: str):
        obs = super().observe(agent)
        return {
            "observations": obs["observation"],
            "action_mask": obs["action_mask"],
        }
