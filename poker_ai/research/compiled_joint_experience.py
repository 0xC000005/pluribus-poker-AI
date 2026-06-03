"""Compiled joint-experience collection for local population response training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import random
import time

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import FastPokerState
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.compiled_fast_rollout import (
    CompiledFastStateBatch,
    compiled_apply_actions,
    compiled_feature_vectors,
    compiled_legal_masks,
)
from poker_ai.research.joint_experience import _normalize_meta_strategy
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.native_rollout_substrate import _new_seeded_fast_state


@dataclass
class CompiledJointPolicy:
    kind: str
    checkpoint_path: str
    algorithm: str
    module: torch.nn.Module


class _RainbowQPolicy(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, num_atoms: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "support",
            torch.linspace(-1.0, 1.0, int(num_atoms), dtype=torch.float32),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.model(features)
        distribution = output[0] if isinstance(output, tuple) else output
        return torch.sum(distribution * self.support.view(1, 1, -1), dim=-1)


class _RouterActionScorePolicy(torch.nn.Module):
    """Route each observation to one member policy and return its action scores."""

    def __init__(self, router: torch.nn.Module, members: Sequence[CompiledJointPolicy]) -> None:
        super().__init__()
        if not members:
            raise ValueError("compiled policy-router requires at least one member policy")
        self.router = router
        self.members = torch.nn.ModuleList([member.module for member in members])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        router_scores = self.router(features)
        if router_scores.ndim != 2 or router_scores.shape[1] != len(self.members):
            raise ValueError("policy-router must return shape (batch, n_members)")
        selected = torch.argmax(router_scores, dim=1)
        routed_scores = features.new_empty((features.shape[0], N_ACTIONS))
        for member_i, member in enumerate(self.members):
            rows = selected == int(member_i)
            if not bool(torch.any(rows)):
                continue
            output = member(features[rows])
            scores = output[0] if isinstance(output, tuple) else output
            if scores.ndim != 2 or scores.shape[1] != N_ACTIONS:
                raise ValueError("policy-router member must return shape (batch, N_ACTIONS)")
            routed_scores[rows] = scores
        return routed_scores


def _as_device(device: str | torch.device) -> tuple[torch.device, dict[str, Any]]:
    if isinstance(device, torch.device):
        return device, {
            "requested_device": str(device),
            "resolved_device": str(device),
            "device_policy": "explicit_torch_device",
        }
    device_info = resolve_device(str(device))
    return torch.device(device_info["resolved_device"]), device_info


def _policy_scores(policy: CompiledJointPolicy, features: np.ndarray, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        output = policy.module(x)
        scores = output[0] if isinstance(output, tuple) else output
        if scores.ndim != 2 or scores.shape[1] != N_ACTIONS:
            raise ValueError("compiled joint policy must return shape (batch, N_ACTIONS)")
        return scores


def _terminal_next_observation() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros(N_FEATURES, dtype=np.float32),
        np.ones(N_ACTIONS, dtype=np.bool_),
    )


def collect_compiled_joint_experience(
    policies: Sequence[CompiledJointPolicy],
    *,
    meta_strategy: Sequence[float],
    n_hands: int = 128,
    batch_size: int = 128,
    seed: int = 20260618,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    exploration_epsilon: float = 0.0,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Collect replay-ready local trajectories under a population meta-policy."""

    if not policies:
        raise ValueError("policies must contain at least one CompiledJointPolicy")
    n_hands = int(n_hands)
    batch_size = int(batch_size)
    max_steps = int(max_steps_per_hand)
    if n_hands <= 0:
        raise ValueError("n_hands must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps_per_hand must be positive")
    epsilon = float(exploration_epsilon)
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("exploration_epsilon must be in [0, 1]")
    chips = int(initial_chips)
    if chips <= 0:
        raise ValueError("initial_chips must be positive")

    resolved_device, device_info = _as_device(device)
    meta = _normalize_meta_strategy(meta_strategy, len(policies))
    rng = np.random.default_rng(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    for policy in policies:
        policy.module.to(resolved_device)
        policy.module.eval()
        for parameter in policy.module.parameters():
            parameter.requires_grad_(False)

    seat_policy_indices = rng.choice(
        len(policies),
        size=(n_hands, 2),
        p=meta,
    ).astype(np.int64)
    states = [_new_seeded_fast_state(seed, hand_i, chips) for hand_i in range(n_hands)]
    compiled = CompiledFastStateBatch.from_fast_states(states)
    steps_per_hand = np.zeros(n_hands, dtype=np.int32)

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    next_legal_masks: list[np.ndarray] = []
    actions: list[int] = []
    dones: list[bool] = []
    player_indices: list[int] = []
    behavior_policy_indices: list[int] = []
    hand_indices: list[int] = []
    step_indices: list[int] = []
    forward_calls = 0
    exploratory_actions = 0
    needs_python_showdown = 0
    started = time.perf_counter()

    while True:
        live_indices = [
            hand_i
            for hand_i in range(n_hands)
            if int(compiled.stage[hand_i]) < FastPokerState.SHOWDOWN
            and int(steps_per_hand[hand_i]) < max_steps
        ]
        if not live_indices:
            break

        all_features = compiled_feature_vectors(compiled)
        all_masks = compiled_legal_masks(compiled).astype(bool, copy=False)
        current_players = compiled.current_players()
        action_array = np.full(n_hands, -1, dtype=np.int16)
        pending: list[tuple[int, int, int, int, np.ndarray, np.ndarray, int]] = []

        for start in range(0, len(live_indices), batch_size):
            chunk = live_indices[start : start + batch_size]
            grouped: dict[int, list[int]] = {}
            for hand_i in chunk:
                player_i = int(current_players[hand_i])
                policy_i = int(seat_policy_indices[hand_i, player_i])
                grouped.setdefault(policy_i, []).append(int(hand_i))

            for policy_i, group in grouped.items():
                features = all_features[group].astype(np.float32, copy=False)
                masks = all_masks[group]
                scores = _policy_scores(policies[policy_i], features, resolved_device)
                mask_t = torch.as_tensor(masks, dtype=torch.bool, device=resolved_device)
                masked_scores = scores.masked_fill(~mask_t, -1.0e30)
                selected = torch.argmax(masked_scores, dim=1).detach().cpu().numpy()
                forward_calls += 1
                for row_i, hand_i in enumerate(group):
                    player_i = int(current_players[hand_i])
                    action_idx = int(selected[row_i])
                    if epsilon > 0.0 and float(rng.random()) < epsilon:
                        legal_actions = np.flatnonzero(masks[row_i])
                        if legal_actions.size > 0:
                            action_idx = int(rng.choice(legal_actions))
                            exploratory_actions += 1
                    action_array[hand_i] = action_idx
                    pending.append(
                        (
                            int(hand_i),
                            int(steps_per_hand[hand_i]),
                            player_i,
                            int(policy_i),
                            features[row_i].astype(np.float32, copy=False),
                            masks[row_i].astype(np.bool_, copy=False),
                            action_idx,
                        )
                    )
                    steps_per_hand[hand_i] += 1

        result = compiled_apply_actions(compiled, action_array)
        needs_python_showdown += int(result["needs_python_showdown"])
        post_features = compiled_feature_vectors(compiled)
        post_masks = compiled_legal_masks(compiled).astype(bool, copy=False)
        terminal = compiled.stage >= FastPokerState.SHOWDOWN
        for hand_i, step_i, player_i, policy_i, features, mask, action_idx in pending:
            done = bool(terminal[hand_i] or int(steps_per_hand[hand_i]) >= max_steps)
            if bool(terminal[hand_i]):
                next_features, next_mask = _terminal_next_observation()
            else:
                next_features = post_features[hand_i].astype(np.float32, copy=False)
                next_mask = post_masks[hand_i].astype(np.bool_, copy=False)
            observations.append(features)
            legal_masks.append(mask)
            next_observations.append(next_features)
            next_legal_masks.append(next_mask)
            actions.append(int(action_idx))
            dones.append(done)
            player_indices.append(int(player_i))
            behavior_policy_indices.append(int(policy_i))
            hand_indices.append(int(hand_i))
            step_indices.append(int(step_i))

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    terminal = compiled.stage >= FastPokerState.SHOWDOWN
    truncated_hands = int(np.sum(~terminal))
    payoffs = (compiled.chips.astype(np.float32) - float(chips)) / float(chips)
    payoffs[~terminal, :] = 0.0
    returns = np.array(
        [payoffs[hand_i, player_i] for hand_i, player_i in zip(hand_indices, player_indices, strict=False)],
        dtype=np.float32,
    )

    if observations:
        observation_array = np.stack(observations).astype(np.float32, copy=False)
        next_observation_array = np.stack(next_observations).astype(np.float32, copy=False)
        mask_array = np.stack(legal_masks).astype(np.bool_, copy=False)
        next_mask_array = np.stack(next_legal_masks).astype(np.bool_, copy=False)
    else:
        observation_array = np.zeros((0, N_FEATURES), dtype=np.float32)
        next_observation_array = np.zeros((0, N_FEATURES), dtype=np.float32)
        mask_array = np.zeros((0, N_ACTIONS), dtype=np.bool_)
        next_mask_array = np.zeros((0, N_ACTIONS), dtype=np.bool_)

    return {
        "algorithm": "compiled_joint_experience_meta_policy_dataset",
        "role": "compiled_psro_joint_experience_response_oracle_data_contract",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "backend": "compiled-fast-state",
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "policy_kinds": [policy.kind for policy in policies],
        "policy_checkpoints": [policy.checkpoint_path for policy in policies],
        "policy_algorithms": [policy.algorithm for policy in policies],
        "meta_strategy": meta,
        "n_hands": n_hands,
        "n_transitions": int(len(actions)),
        "truncated_hands": int(truncated_hands),
        "initial_chips": chips,
        "max_steps_per_hand": max_steps,
        "exploration_epsilon": float(epsilon),
        "exploratory_actions": int(exploratory_actions),
        "exploratory_action_fraction": float(exploratory_actions / max(len(actions), 1)),
        "batch_size": batch_size,
        "seconds": float(seconds),
        "transitions_per_second": float(len(actions) / max(seconds, 1e-12)),
        "policy_forward_calls": int(forward_calls),
        "mean_decisions_per_forward": float(len(actions) / max(forward_calls, 1)),
        "needs_python_showdown": int(needs_python_showdown),
        "observations": observation_array,
        "next_observations": next_observation_array,
        "legal_masks": mask_array,
        "next_legal_masks": next_mask_array,
        "actions": np.asarray(actions, dtype=np.int64),
        "dones": np.asarray(dones, dtype=np.bool_),
        "player_indices": np.asarray(player_indices, dtype=np.int64),
        "behavior_policy_indices": np.asarray(behavior_policy_indices, dtype=np.int64),
        "hand_indices": np.asarray(hand_indices, dtype=np.int64),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
        "returns": returns,
        "hand_policy_indices": seat_policy_indices,
        "hand_returns": payoffs.astype(np.float32, copy=False),
        "uses_slumbot_training_data": False,
        "promotion": False,
        **device_info,
    }


def load_compiled_joint_policy(
    checkpoint: str | Path,
    *,
    kind: str,
    device: str | torch.device = "auto",
) -> CompiledJointPolicy:
    """Load a policy checkpoint into the batched compiled collector contract."""

    resolved_device, _device_info = _as_device(device)
    normalized_kind = str(kind).strip().lower()
    if normalized_kind in {"tianshou-rainbow", "rainbow", "tianshou_rainbow"}:
        from scripts.run_tianshou_rainbow_native_control import (  # noqa: PLC0415
            _RainbowDistributionNet,
            _rainbow_state_dicts_by_seat_from_payload,
        )

        payload = torch.load(str(checkpoint), map_location=resolved_device, weights_only=False)
        if int(payload.get("num_actions", -1)) != N_ACTIONS:
            raise ValueError("Rainbow checkpoint action count does not match native contract")
        if int(payload.get("num_features", -1)) != N_FEATURES:
            raise ValueError("Rainbow checkpoint feature count does not match native contract")
        hidden_dim = int(payload.get("hidden_dim", 128))
        num_atoms = int(payload.get("num_atoms", 51))
        model = _RainbowDistributionNet(
            hidden_dim=hidden_dim,
            num_atoms=num_atoms,
            device=resolved_device,
        ).to(resolved_device)
        state_dicts = _rainbow_state_dicts_by_seat_from_payload(payload)
        model.load_state_dict(state_dicts.get(0, state_dicts[sorted(state_dicts)[0]]))
        return CompiledJointPolicy(
            kind="tianshou-rainbow",
            checkpoint_path=str(checkpoint),
            algorithm=str(payload.get("algorithm", "tianshou_rainbow_dqn")),
            module=_RainbowQPolicy(model, num_atoms=num_atoms).to(resolved_device),
        )
    if normalized_kind in {"native-ppo", "native_ppo"}:
        from poker_ai.research.native_ppo_policy import _load_policy_network  # noqa: PLC0415

        payload, policy, feature_mode = _load_policy_network(
            str(checkpoint),
            resolved_device,
            strategy_source="auto",
        )
        if feature_mode != "flat":
            raise ValueError("compiled joint policy requires flat native-PPO features")
        return CompiledJointPolicy(
            kind="native-ppo",
            checkpoint_path=str(checkpoint),
            algorithm=str(payload.get("algorithm", "native_ppo_policy")),
            module=policy.to(resolved_device),
        )
    if normalized_kind in {"native-nfsp", "native_nfsp"}:
        from poker_ai.research.native_nfsp import (  # noqa: PLC0415
            _load_native_checkpoint_networks,
        )

        payload, _q_net, avg_net = _load_native_checkpoint_networks(
            str(checkpoint),
            resolved_device,
        )
        return CompiledJointPolicy(
            kind="native-nfsp",
            checkpoint_path=str(checkpoint),
            algorithm=str(payload.get("algorithm", "native_nfsp")),
            module=avg_net.to(resolved_device),
        )
    if normalized_kind in {"policy-router", "policy_router"}:
        from poker_ai.research.policy_router import load_policy_router_checkpoint  # noqa: PLC0415

        payload, router = load_policy_router_checkpoint(checkpoint, device=resolved_device)
        member_kinds = [str(value) for value in payload.get("member_policy_kinds", [])]
        member_checkpoints = [str(value) for value in payload.get("member_checkpoints", [])]
        if len(member_kinds) != len(member_checkpoints) or not member_kinds:
            raise ValueError("policy-router checkpoint has invalid member policy metadata")
        normalized_member_kinds = {str(kind_i).strip().lower().replace("_", "-") for kind_i in member_kinds}
        if "policy-router" in normalized_member_kinds:
            raise ValueError("policy-router members may not themselves be policy-router checkpoints")
        members = [
            load_compiled_joint_policy(member_checkpoint, kind=member_kind, device=resolved_device)
            for member_kind, member_checkpoint in zip(member_kinds, member_checkpoints, strict=True)
        ]
        module = _RouterActionScorePolicy(router, members).to(resolved_device)
        return CompiledJointPolicy(
            kind="policy-router",
            checkpoint_path=str(checkpoint),
            algorithm=str(payload.get("algorithm", "policy_population_router")),
            module=module,
        )
    raise ValueError(f"unsupported compiled joint policy kind: {kind}")
