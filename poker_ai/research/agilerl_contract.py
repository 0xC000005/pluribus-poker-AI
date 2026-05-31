"""AgileRL/PettingZoo compatibility probes for native poker autoresearch."""

from __future__ import annotations

import importlib.metadata
import traceback
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.native_pettingzoo import NativeNoLimitHoldemAECEnv


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


class AgileRLActionMaskInfoWrapper(ParallelEnv):
    """Move PettingZoo action masks from observations into infos for AgileRL."""

    metadata = {"name": "poker_ai_native_hu_nlhe_agilerl_v0", "render_modes": []}

    def __init__(self, env: ParallelEnv) -> None:
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(getattr(env, "agents", self.possible_agents))
        self.render_mode = getattr(env, "render_mode", None)

    def observation_space(self, agent: str):
        return spaces.Box(-np.inf, np.inf, shape=(N_FEATURES,), dtype=np.float32)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return self._move_masks_to_infos(observations, infos)

    def step(self, actions: dict[str, Any]):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        stripped, next_infos = self._move_masks_to_infos(observations, infos)
        return stripped, rewards, terminations, truncations, next_infos

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def state(self):
        if hasattr(self.env, "state"):
            return self.env.state()
        return np.zeros(N_FEATURES, dtype=np.float32)

    @staticmethod
    def _move_masks_to_infos(
        observations: dict[str, Any],
        infos: dict[str, dict[str, Any]] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        infos = {str(agent): dict(info) for agent, info in (infos or {}).items()}
        stripped: dict[str, np.ndarray] = {}
        for agent, obs in observations.items():
            agent_name = str(agent)
            if isinstance(obs, dict) and "observation" in obs:
                stripped[agent_name] = np.asarray(obs["observation"], dtype=np.float32)
                action_mask = np.asarray(obs.get("action_mask", []), dtype=np.int8)
            else:
                stripped[agent_name] = np.asarray(obs, dtype=np.float32)
                action_mask = np.ones(N_ACTIONS, dtype=np.int8)
            agent_info = infos.setdefault(agent_name, {})
            action_mask = action_mask.astype(np.int8, copy=False)
            agent_info["action_mask"] = action_mask.tolist()
            agent_info["env_defined_actions"] = (
                None
                if int(action_mask.sum()) > 0
                else np.asarray([0], dtype=np.int64)
            )
        return stripped, infos


def make_agilerl_native_parallel_env(
    *,
    seed: int = 20260527,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
) -> AgileRLActionMaskInfoWrapper:
    """Build the native poker env in AgileRL's expected parallel mask format."""
    from pettingzoo.utils.conversions import turn_based_aec_to_parallel

    base_env = NativeNoLimitHoldemAECEnv(
        seed=int(seed),
        initial_chips=int(initial_chips),
        max_steps_per_hand=int(max_steps_per_hand),
    )
    return AgileRLActionMaskInfoWrapper(turn_based_aec_to_parallel(base_env))


def _shape_list(value: Any) -> list[int]:
    return [int(dim) for dim in np.asarray(value).shape]


def _parallel_env_shape_diagnostics(env: ParallelEnv, *, seed: int) -> dict[str, Any]:
    """Capture the tensor/mask contract at the native adapter boundary."""
    observations, infos = env.reset(seed=int(seed))
    reset_observation_shapes = {
        str(agent): _shape_list(obs) for agent, obs in observations.items()
    }
    reset_action_mask_shapes: dict[str, list[int]] = {}
    reset_action_mask_sums: dict[str, int] = {}
    sample_action_values: dict[str, int] = {}
    for agent in env.agents:
        agent_name = str(agent)
        action_mask = np.asarray(infos[agent_name].get("action_mask", []), dtype=np.int8)
        reset_action_mask_shapes[agent_name] = _shape_list(action_mask)
        reset_action_mask_sums[agent_name] = int(action_mask.sum())
        sample_action_values[agent_name] = (
            int(env.action_space(agent).sample(action_mask))
            if reset_action_mask_sums[agent_name] > 0
            else 0
        )

    next_observations, rewards, terminations, truncations, next_infos = env.step(
        sample_action_values
    )
    return {
        "agent_order": [str(agent) for agent in env.possible_agents],
        "reset_observation_shapes": reset_observation_shapes,
        "reset_action_mask_shapes": reset_action_mask_shapes,
        "reset_action_mask_sums": reset_action_mask_sums,
        "sample_action_values": sample_action_values,
        "step_observation_shapes": {
            str(agent): _shape_list(obs) for agent, obs in next_observations.items()
        },
        "step_action_mask_shapes": {
            str(agent): _shape_list(next_infos[str(agent)].get("action_mask", []))
            for agent in next_infos
        },
        "step_reward_values": {str(agent): float(value) for agent, value in rewards.items()},
        "step_termination_values": {
            str(agent): bool(value) for agent, value in terminations.items()
        },
        "step_truncation_values": {
            str(agent): bool(value) for agent, value in truncations.items()
        },
    }


def probe_native_agilerl_contract(
    *,
    seed: int = 20260527,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
) -> dict[str, Any]:
    """Report whether the native AEC poker env can feed AgileRL-style MARL."""
    env = make_agilerl_native_parallel_env(
        seed=int(seed),
        initial_chips=int(initial_chips),
        max_steps_per_hand=int(max_steps_per_hand),
    )
    observations, infos = env.reset(seed=int(seed))

    mask_sums: dict[str, int] = {}
    observation_shapes: dict[str, list[int]] = {}
    sampled_actions: dict[str, int] = {}
    for agent, obs in observations.items():
        action_mask = np.asarray(infos[str(agent)]["action_mask"], dtype=np.int8)
        mask_sums[str(agent)] = int(action_mask.sum())
        observation_shapes[str(agent)] = [int(dim) for dim in obs.shape]
        if mask_sums[str(agent)] > 0:
            sampled_actions[str(agent)] = int(env.action_space(agent).sample(action_mask))
        else:
            sampled_actions[str(agent)] = 0

    active_mask_agents = sum(1 for value in mask_sums.values() if value > 0)
    inactive_zero_mask_agents = sum(1 for value in mask_sums.values() if value == 0)
    action_n = int(env.action_space(env.agents[0]).n) if env.agents else 0
    observation_dim = (
        int(next(iter(observations.values())).shape[0])
        if observations
        else 0
    )
    compatible = (
        action_n == N_ACTIONS
        and observation_dim == N_FEATURES
        and active_mask_agents == 1
        and inactive_zero_mask_agents == max(0, len(env.agents) - 1)
    )
    return {
        "algorithm": "agilerl_pettingzoo_contract_probe",
        "role": "maintained_library_parallel_env_contract",
        "environment": "poker_ai:pettingzoo_full_deck_hu_nlhe",
        "parallel_conversion": "turn_based_aec_to_parallel",
        "mask_location": "infos.action_mask",
        "agilerl_available": _package_version("agilerl") is not None,
        "agilerl_version": _package_version("agilerl"),
        "pettingzoo_version": _package_version("pettingzoo"),
        "num_agents": int(len(env.agents)),
        "agents": [str(agent) for agent in env.agents],
        "num_actions": action_n,
        "num_features": observation_dim,
        "expected_num_actions": N_ACTIONS,
        "expected_num_features": N_FEATURES,
        "observation_shapes": observation_shapes,
        "action_mask_sums": mask_sums,
        "sampled_actions": sampled_actions,
        "active_mask_agents": int(active_mask_agents),
        "inactive_zero_mask_agents": int(inactive_zero_mask_agents),
        "native_parallel_contract_ok": bool(compatible),
        "recommendation": (
            "candidate_agilerl_parallel_spike"
            if compatible
            else "adapter_contract_fix_required"
        ),
        "promotion": False,
        "warning": (
            "This validates only the native PettingZoo parallel contract for "
            "maintained MARL libraries; it is not training or strength evidence."
        ),
    }


def probe_agilerl_offpolicy_algorithms(
    *,
    algorithms: tuple[str, ...] = ("MADDPG", "MATD3"),
    seed: int = 20260528,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    max_steps: int = 32,
    evo_steps: int = 16,
    batch_size: int = 8,
    hidden_dim: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    """Probe AgileRL off-policy MARL algorithms on the native poker adapter."""
    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
    from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy
    from agilerl.utils.utils import create_population

    device_info = resolve_device(device)
    resolved_device = str(device_info["resolved_device"])
    results: dict[str, dict[str, Any]] = {}
    for algo in algorithms:
        env = make_agilerl_native_parallel_env(
            seed=int(seed),
            initial_chips=int(initial_chips),
            max_steps_per_hand=int(max_steps_per_hand),
        )
        observation_spaces = {
            str(agent): env.observation_space(agent) for agent in env.possible_agents
        }
        action_spaces = {str(agent): env.action_space(agent) for agent in env.possible_agents}
        init_hp = {
            "AGENT_IDS": [str(agent) for agent in env.possible_agents],
            "BATCH_SIZE": int(batch_size),
            "LR_ACTOR": 1e-4,
            "LR_CRITIC": 1e-3,
            "LEARN_STEP": int(max(1, batch_size)),
            "GAMMA": 0.99,
            "TAU": 0.01,
        }
        net_config = {
            "encoder_config": {
                "hidden_size": [int(hidden_dim), int(hidden_dim)],
                "init_layers": False,
            },
            "head_config": {"hidden_size": [int(hidden_dim)], "init_layers": False},
        }
        diagnostic_env = make_agilerl_native_parallel_env(
            seed=int(seed),
            initial_chips=int(initial_chips),
            max_steps_per_hand=int(max_steps_per_hand),
        )
        try:
            contract_diagnostics = _parallel_env_shape_diagnostics(
                diagnostic_env,
                seed=int(seed),
            )
        finally:
            diagnostic_env.close()
        result: dict[str, Any] = {
            "constructs": False,
            "train_smoke_ok": False,
            "train_error_type": None,
            "train_error": None,
            "train_error_traceback": None,
            "contract_diagnostics": contract_diagnostics,
            "recommendation": "adapter_or_library_contract_fix_required",
        }
        try:
            population = create_population(
                algo=str(algo),
                net_config=net_config,
                INIT_HP=init_hp,
                observation_space=observation_spaces,
                action_space=action_spaces,
                population_size=1,
                num_envs=1,
                device=resolved_device,
            )
            result["constructs"] = True
            memory = MultiAgentReplayBuffer(
                memory_size=max(int(max_steps) * 4, int(batch_size) * 4, 64),
                field_names=["state", "action", "reward", "next_state", "done"],
                agent_ids=[str(agent) for agent in env.possible_agents],
                device=resolved_device,
            )
            train_multi_agent_off_policy(
                env=env,
                env_name="poker_ai_native_hu_nlhe",
                algo=str(algo),
                pop=population,
                memory=memory,
                sum_scores=True,
                INIT_HP=init_hp,
                max_steps=int(max_steps),
                evo_steps=int(evo_steps),
                eval_steps=max(1, int(evo_steps)),
                learning_delay=0,
                verbose=False,
            )
            result["train_smoke_ok"] = True
            result["recommendation"] = "candidate_bounded_training"
        except Exception as exc:  # pragma: no cover - exception text is the artifact.
            result["train_error_type"] = type(exc).__name__
            result["train_error"] = str(exc)[:500]
            result["train_error_traceback"] = traceback.format_exc(limit=8)
        results[str(algo)] = result
        env.close()

    compatible = any(result["train_smoke_ok"] for result in results.values())
    return {
        "algorithm": "agilerl_offpolicy_contract_probe",
        "role": "maintained_library_offpolicy_marl_contract",
        "environment": "poker_ai:pettingzoo_full_deck_hu_nlhe",
        **device_info,
        "agilerl_version": _package_version("agilerl"),
        "pettingzoo_version": _package_version("pettingzoo"),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "max_steps": int(max_steps),
        "evo_steps": int(evo_steps),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "results": results,
        "any_train_smoke_ok": bool(compatible),
        "promotion": False,
        "recommendation": (
            "candidate_bounded_training"
            if compatible
            else "do_not_use_offpolicy_agilerl_until_contract_fixed"
        ),
        "warning": (
            "This probes maintained AgileRL off-policy algorithms only. It is "
            "contract evidence, not strength or Slumbot evidence."
        ),
    }
