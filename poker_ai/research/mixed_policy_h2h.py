"""Duplicate-swapped H2H evaluation for mixed native policy checkpoint formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import random
import time

import numpy as np
import torch

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, new_game
from poker_ai.research.native_rollout_substrate import fast_state_from_full_deck_state
from poker_ai.research.native_nfsp import (
    get_legal_mask,
    resolve_device,
    select_action as sample_action,
)


ActionProbsFn = Callable[[np.ndarray, np.ndarray, torch.device], np.ndarray]
ActionForStateFn = Callable[
    [Any, np.ndarray, np.ndarray, torch.device, np.random.Generator],
    int,
]
SUPPORTED_POLICY_KINDS = (
    "npi",
    "native-nfsp",
    "native-ppo",
    "tianshou-rainbow",
    "tianshou-rainbow-softmax",
    "tianshou-ppo",
    "agilerl-ippo",
    "rllib-ppo",
    "policy-router",
)
_KIND_ALIASES = {
    "rainbow": "tianshou-rainbow",
    "tianshou_rainbow": "tianshou-rainbow",
    "rainbow-softmax": "tianshou-rainbow-softmax",
    "tianshou_rainbow_softmax": "tianshou-rainbow-softmax",
    "native_ppo": "native-ppo",
    "ppo": "tianshou-ppo",
    "tianshou_ppo": "tianshou-ppo",
    "agilerl_ippo": "agilerl-ippo",
    "ippo": "agilerl-ippo",
    "rllib_ppo": "rllib-ppo",
    "rllib": "rllib-ppo",
    "policy_router": "policy-router",
    "router": "policy-router",
    "conflux-router": "policy-router",
    "conflux_router": "policy-router",
}


@dataclass(frozen=True)
class PolicyAdapter:
    kind: str
    checkpoint_path: str
    algorithm: str
    action_probs_fn: ActionProbsFn
    initial_chips: int = 1000
    max_steps_per_hand: int = 256
    action_for_state_fn: ActionForStateFn | None = None

    def probs(
        self,
        features: np.ndarray,
        legal_mask: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        return self.action_probs_fn(features, legal_mask, device)

    def select_action(
        self,
        *,
        state: Any,
        features: np.ndarray,
        legal_mask: np.ndarray,
        device: torch.device,
        rng: np.random.Generator,
    ) -> int:
        """Select a legal action, allowing stateful policy backends when needed."""
        if self.action_for_state_fn is not None:
            action_idx = int(
                self.action_for_state_fn(state, features, legal_mask, device, rng)
            )
            if 0 <= action_idx < N_ACTIONS and bool(legal_mask[action_idx]):
                return action_idx
        return int(sample_action(self.probs(features, legal_mask, device), legal_mask, rng=rng))


@dataclass(frozen=True)
class PolicyMetaStrategyAdapter:
    members: tuple[PolicyAdapter, ...]
    weights: tuple[float, ...]
    checkpoint_path: str
    algorithm: str = "native_policy_meta_strategy"
    kind: str = "meta-strategy"
    initial_chips: int = 1000
    max_steps_per_hand: int = 256

    @property
    def size(self) -> int:
        return len(self.members)

    def sample_member_index(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.size, p=np.asarray(self.weights, dtype=np.float64)))

    def probs(
        self,
        features: np.ndarray,
        legal_mask: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        legal = np.asarray(legal_mask, dtype=np.float32)
        if float(legal.sum()) <= 0.0:
            return np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32)
        mixed = np.zeros(N_ACTIONS, dtype=np.float64)
        for weight, member in zip(self.weights, self.members, strict=True):
            if float(weight) <= 0.0:
                continue
            mixed += float(weight) * member.probs(features, legal, device).astype(np.float64)
        mixed = mixed.astype(np.float32) * legal
        total = float(mixed.sum())
        return mixed / total if total > 0.0 else legal / float(legal.sum())

    def select_action(
        self,
        *,
        state: Any,
        features: np.ndarray,
        legal_mask: np.ndarray,
        device: torch.device,
        rng: np.random.Generator,
    ) -> int:
        return int(sample_action(self.probs(features, legal_mask, device), legal_mask, rng=rng))


LoadedPolicyAdapter = PolicyAdapter | PolicyMetaStrategyAdapter


def make_policy_meta_strategy_adapter(
    members: Sequence[PolicyAdapter],
    *,
    weights: Sequence[float],
    checkpoint_path: str = "native-policy-meta-strategy",
) -> PolicyMetaStrategyAdapter:
    """Create a normalized per-hand checkpoint-population policy adapter."""
    member_tuple = tuple(members)
    if not member_tuple:
        raise ValueError("meta-strategy requires at least one member policy")
    raw_weights = np.asarray(list(weights), dtype=np.float64)
    if raw_weights.shape != (len(member_tuple),):
        raise ValueError("meta-strategy weights must match member count")
    if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0.0):
        raise ValueError("meta-strategy weights must be finite and non-negative")
    total = float(raw_weights.sum())
    if total <= 0.0:
        raise ValueError("meta-strategy requires positive total weight")
    normalized = raw_weights / total
    initial_chips = int(member_tuple[0].initial_chips)
    max_steps = int(max(member.max_steps_per_hand for member in member_tuple))
    return PolicyMetaStrategyAdapter(
        members=member_tuple,
        weights=tuple(float(value) for value in normalized.tolist()),
        checkpoint_path=str(checkpoint_path),
        initial_chips=initial_chips,
        max_steps_per_hand=max_steps,
    )


def _normalize_policy_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    return _KIND_ALIASES.get(normalized, normalized)


def _checkpoint_runtime_config(payload: dict) -> dict:
    config = dict(payload.get("config", {}))
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict):
        config = {**metrics, **config}
    return config


def _one_hot_action_probs(action_idx: int, legal_mask: np.ndarray) -> np.ndarray:
    probs = np.zeros(N_ACTIONS, dtype=np.float32)
    action_idx = int(action_idx)
    if 0 <= action_idx < N_ACTIONS:
        probs[action_idx] = 1.0
    masked = probs * np.asarray(legal_mask, dtype=np.float32)
    if float(masked.sum()) <= 1e-8:
        legal = np.asarray(legal_mask, dtype=np.float32)
        total = float(legal.sum())
        return legal / total if total > 0.0 else np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32)
    return masked / float(masked.sum())


class _RainbowDistributionNet(torch.nn.Module):
    def __init__(self, *, hidden_dim: int, num_atoms: int, device: torch.device) -> None:
        super().__init__()
        self.num_atoms = int(num_atoms)
        self.device = device
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), N_ACTIONS * int(num_atoms)),
        )

    def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.net(x).view(-1, N_ACTIONS, self.num_atoms)
        return torch.softmax(logits, dim=-1)


class _RainbowQPolicy(torch.nn.Module):
    kind = "tianshou-rainbow-softmax"

    def __init__(self, model: torch.nn.Module, *, num_atoms: int, algorithm: str) -> None:
        super().__init__()
        self.model = model
        self.algorithm = str(algorithm)
        self.register_buffer(
            "support",
            torch.linspace(-1.0, 1.0, int(num_atoms), dtype=torch.float32),
        )

    def forward(self, features: np.ndarray | torch.Tensor) -> torch.Tensor:
        distribution = self.model(features)
        return torch.sum(distribution * self.support.view(1, 1, -1), dim=-1)


def _rainbow_state_dicts_by_seat_from_payload(payload: dict) -> dict[int, dict]:
    if payload.get("shared_model_state_dict") is not None:
        shared = payload["shared_model_state_dict"]
        return {0: shared, 1: shared}
    if "agent_model_state_dicts" in payload:
        agent_state_dicts = payload["agent_model_state_dicts"]
        fallback_key = "player_0" if "player_0" in agent_state_dicts else sorted(agent_state_dicts)[0]
        return {
            seat: agent_state_dicts.get(f"player_{seat}", agent_state_dicts[fallback_key])
            for seat in (0, 1)
        }
    if "model_state_dict" in payload:
        shared = payload["model_state_dict"]
        return {0: shared, 1: shared}
    raise ValueError("Rainbow checkpoint is missing a loadable model state dict")


def _load_rainbow_q_policy(checkpoint_path: str | Path, device: torch.device) -> tuple[dict, _RainbowQPolicy]:
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("Rainbow checkpoint action count does not match native contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("Rainbow checkpoint feature count does not match native contract")
    hidden_dim = int(payload.get("hidden_dim", 128))
    num_atoms = int(payload.get("num_atoms", 51))
    model = _RainbowDistributionNet(
        hidden_dim=hidden_dim,
        num_atoms=num_atoms,
        device=device,
    ).to(device)
    state_dicts = _rainbow_state_dicts_by_seat_from_payload(payload)
    model.load_state_dict(state_dicts.get(0, state_dicts[sorted(state_dicts)[0]]))
    policy = _RainbowQPolicy(
        model,
        num_atoms=num_atoms,
        algorithm=str(payload.get("algorithm", "tianshou_rainbow_dqn")),
    ).to(device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return dict(payload), policy


def _q_softmax_action_probs(
    policy: torch.nn.Module,
    features: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    legal = np.asarray(legal_mask, dtype=np.float32)
    if float(legal.sum()) <= 0.0:
        return np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32)
    with torch.no_grad():
        q_values = policy(
            torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        ).detach().float().reshape(-1)
    mask = torch.as_tensor(legal > 0, dtype=torch.bool, device=q_values.device)
    logits = q_values.masked_fill(~mask, -1.0e30)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
    probs *= legal
    total = float(probs.sum())
    return probs / total if total > 0.0 else legal / float(legal.sum())


def _resolve_rllib_module_checkpoint(checkpoint_path: str | Path) -> Path:
    """Return the shared RLModule subcheckpoint for an RLlib Algorithm checkpoint."""
    path = Path(checkpoint_path).resolve()
    shared = path / "learner_group" / "learner" / "rl_module" / "shared"
    if shared.exists():
        return shared
    default_policy = path / "learner_group" / "learner" / "rl_module" / "default_policy"
    if default_policy.exists():
        return default_policy
    if (path / "module_state.pkl").exists():
        return path
    raise ValueError(
        "RLlib PPO checkpoint must be an Algorithm checkpoint containing "
        "learner_group/learner/rl_module/shared, "
        "learner_group/learner/rl_module/default_policy, or a direct RLModule checkpoint"
    )


def _rllib_ppo_action_probs(
    module: Any,
    features: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run an RLlib action-masked RLModule and return masked probabilities."""
    legal = np.asarray(legal_mask, dtype=np.float32)
    if float(legal.sum()) <= 0.0:
        return np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32)
    obs = {
        "observations": torch.as_tensor(
            np.asarray(features, dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=device,
        ),
        "action_mask": torch.as_tensor(
            legal[None, :],
            dtype=torch.int8,
            device=device,
        ),
    }
    with torch.no_grad():
        output = module.forward_inference({"obs": obs})
        logits = output["action_dist_inputs"].detach().float().reshape(-1)
    mask = torch.as_tensor(legal, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(~mask, -1.0e30)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
    probs *= legal
    total = float(probs.sum())
    if total <= 1e-8:
        return legal / float(legal.sum())
    return probs / total


def load_policy_adapter(
    checkpoint_path: str,
    *,
    kind: str,
    device: torch.device,
) -> PolicyAdapter:
    """Load a supported native policy checkpoint as an action-probability adapter."""
    kind = _normalize_policy_kind(kind)
    if kind == "npi":
        from poker_ai.research.neural_policy_iteration import (  # noqa: PLC0415
            _load_policy_iteration_checkpoint,
            _network_policy,
        )

        policy_net, payload = _load_policy_iteration_checkpoint(checkpoint_path, device)
        config = dict(payload.get("config", {}))
        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm=str(payload.get("algorithm", "neural_policy_iteration")),
            action_probs_fn=lambda features, legal_mask, resolved_device: _network_policy(
                policy_net,
                features,
                legal_mask,
                resolved_device,
            ),
            initial_chips=int(config.get("initial_chips", 1000)),
            max_steps_per_hand=int(config.get("max_steps_per_hand", 256)),
        )
    if kind == "native-nfsp":
        from poker_ai.research.native_nfsp import (  # noqa: PLC0415
            _load_native_checkpoint_networks,
            _network_probs,
        )

        payload, _q_net, avg_net = _load_native_checkpoint_networks(checkpoint_path, device)
        config = dict(payload.get("config", {}))
        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm=str(payload.get("algorithm", "native_nfsp")),
            action_probs_fn=lambda features, legal_mask, resolved_device: _network_probs(
                avg_net,
                features,
                legal_mask,
                resolved_device,
            ),
            initial_chips=int(config.get("initial_chips", 1000)),
            max_steps_per_hand=int(config.get("max_steps_per_hand", 256)),
        )
    if kind == "native-ppo":
        from poker_ai.research.native_ppo_policy import (  # noqa: PLC0415
            _load_policy_network,
            _network_probs,
            _policy_feature_vector,
        )

        payload, policy, feature_mode = _load_policy_network(
            checkpoint_path,
            device,
            strategy_source="auto",
        )
        config = _checkpoint_runtime_config(dict(payload))

        def _native_ppo_probs(
            features: np.ndarray,
            legal_mask: np.ndarray,
            resolved_device: torch.device,
        ) -> np.ndarray:
            if feature_mode != "flat":
                raise ValueError(
                    "mixed native PPO adapter currently supports flat feature checkpoints only"
                )
            return _network_probs(policy, features, legal_mask, resolved_device)

        def _native_ppo_action(
            state: Any,
            _features: np.ndarray,
            legal_mask: np.ndarray,
            resolved_device: torch.device,
            rng: np.random.Generator,
        ) -> int:
            if feature_mode != "flat" and not hasattr(state, "current_player"):
                raise ValueError(
                    f"native PPO feature_mode={feature_mode!r} requires full-deck state evaluation"
                )
            policy_features = _policy_feature_vector(state, feature_mode)
            probs = _network_probs(policy, policy_features, legal_mask, resolved_device)
            return int(sample_action(probs, legal_mask, rng=rng))

        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm=str(payload.get("algorithm", "native_ppo_policy")),
            action_probs_fn=_native_ppo_probs,
            initial_chips=int(config.get("initial_chips", 1000)),
            max_steps_per_hand=int(config.get("max_steps_per_hand", 256)),
            action_for_state_fn=_native_ppo_action,
        )
    if kind == "tianshou-rainbow":
        from scripts.run_tianshou_rainbow_native_control import (  # noqa: PLC0415
            _load_rainbow_checkpoint_policy,
            _rainbow_greedy_action,
        )

        payload, policy = _load_rainbow_checkpoint_policy(checkpoint_path, device)
        config = _checkpoint_runtime_config(dict(payload))
        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm=str(payload.get("algorithm", "tianshou_rainbow_dqn")),
            action_probs_fn=lambda features, legal_mask, _resolved_device: _one_hot_action_probs(
                _rainbow_greedy_action(policy, features, legal_mask),
                legal_mask,
            ),
            initial_chips=int(config.get("initial_chips", 1000)),
            max_steps_per_hand=int(config.get("max_steps_per_hand", 256)),
        )
    if kind == "tianshou-rainbow-softmax":
        payload, policy = _load_rainbow_q_policy(checkpoint_path, device)
        config = _checkpoint_runtime_config(dict(payload))
        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm=str(payload.get("algorithm", "tianshou_rainbow_dqn")),
            action_probs_fn=lambda features, legal_mask, resolved_device: _q_softmax_action_probs(
                policy,
                features,
                legal_mask,
                resolved_device,
            ),
            initial_chips=int(config.get("initial_chips", 1000)),
            max_steps_per_hand=int(config.get("max_steps_per_hand", 256)),
        )
    if kind == "tianshou-ppo":
        from scripts.run_tianshou_ppo_native_control import (  # noqa: PLC0415
            _ppo_action_probs,
            _load_ppo_checkpoint_policy,
        )

        payload, policy = _load_ppo_checkpoint_policy(checkpoint_path, device)
        config = _checkpoint_runtime_config(dict(payload))
        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm=str(payload.get("algorithm", "tianshou_ppo")),
            action_probs_fn=lambda features, legal_mask, _resolved_device: _ppo_action_probs(
                policy,
                features,
                legal_mask,
            ),
            initial_chips=int(config.get("initial_chips", 1000)),
            max_steps_per_hand=int(config.get("max_steps_per_hand", 256)),
        )
    if kind == "agilerl-ippo":
        from agilerl.algorithms import IPPO  # noqa: PLC0415
        from scripts.run_agilerl_ippo_native_control import (  # noqa: PLC0415
            _agilerl_checkpoint_action,
        )

        agent = IPPO.load(str(checkpoint_path), device=str(device))
        if hasattr(agent, "set_training_mode"):
            agent.set_training_mode(False)

        def _uniform_probs(
            _features: np.ndarray,
            legal_mask: np.ndarray,
            _resolved_device: torch.device,
        ) -> np.ndarray:
            legal = np.asarray(legal_mask, dtype=np.float32)
            total = float(legal.sum())
            if total <= 0.0:
                return np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32)
            return legal / total

        def _stateful_action(
            state: Any,
            features: np.ndarray,
            legal_mask: np.ndarray,
            _resolved_device: torch.device,
            _rng: np.random.Generator,
        ) -> int:
            player_i = getattr(state, "player_i", getattr(state, "current_player_i", 0))
            return int(
                _agilerl_checkpoint_action(
                    agent,
                    player_i=int(player_i),
                    features=features,
                    legal_mask=legal_mask,
                )
            )

        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm="agilerl_ippo",
            action_probs_fn=_uniform_probs,
            initial_chips=1000,
            max_steps_per_hand=256,
            action_for_state_fn=_stateful_action,
        )
    if kind == "rllib-ppo":
        from ray.rllib.core.rl_module import RLModule  # noqa: PLC0415

        module_checkpoint = _resolve_rllib_module_checkpoint(checkpoint_path)
        module = RLModule.from_checkpoint(str(module_checkpoint))
        if hasattr(module, "to"):
            module = module.to(device)
        if hasattr(module, "eval"):
            module.eval()
        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm="rllib_ppo_action_masked",
            action_probs_fn=lambda features, legal_mask, resolved_device: _rllib_ppo_action_probs(
                module,
                features,
                legal_mask,
                resolved_device,
            ),
            initial_chips=1000,
            max_steps_per_hand=256,
        )
    if kind == "policy-router":
        from poker_ai.research.policy_router import (  # noqa: PLC0415
            load_policy_router_checkpoint,
            router_policy_scores,
        )

        payload, router = load_policy_router_checkpoint(checkpoint_path, device=device)
        member_kinds = [str(value) for value in payload.get("member_policy_kinds", [])]
        member_checkpoints = [str(value) for value in payload.get("member_checkpoints", [])]
        if len(member_kinds) != len(member_checkpoints) or not member_kinds:
            raise ValueError("policy-router checkpoint has invalid member policy metadata")
        if any(_normalize_policy_kind(kind_i) == "policy-router" for kind_i in member_kinds):
            raise ValueError("policy-router members may not themselves be policy-router checkpoints")
        members = tuple(
            load_policy_adapter(checkpoint, kind=kind_i, device=device)
            for kind_i, checkpoint in zip(member_kinds, member_checkpoints, strict=True)
        )
        max_steps = int(max(member.max_steps_per_hand for member in members))
        initial_chips = int(members[0].initial_chips)

        def _select_member(features: np.ndarray, resolved_device: torch.device) -> PolicyAdapter:
            scores = router_policy_scores(router, features, resolved_device)
            member_i = int(np.argmax(scores))
            if member_i < 0 or member_i >= len(members):
                raise ValueError("policy-router selected an invalid member")
            return members[member_i]

        def _router_probs(
            features: np.ndarray,
            legal_mask: np.ndarray,
            resolved_device: torch.device,
        ) -> np.ndarray:
            member = _select_member(features, resolved_device)
            return member.probs(features, legal_mask, resolved_device)

        def _router_action(
            state: Any,
            features: np.ndarray,
            legal_mask: np.ndarray,
            resolved_device: torch.device,
            rng: np.random.Generator,
        ) -> int:
            member = _select_member(features, resolved_device)
            return member.select_action(
                state=state,
                features=features,
                legal_mask=legal_mask,
                device=resolved_device,
                rng=rng,
            )

        return PolicyAdapter(
            kind=kind,
            checkpoint_path=str(checkpoint_path),
            algorithm=str(payload.get("algorithm", "policy_population_router")),
            action_probs_fn=_router_probs,
            initial_chips=initial_chips,
            max_steps_per_hand=max_steps,
            action_for_state_fn=_router_action,
        )
    raise ValueError(f"kind must be one of: {', '.join(SUPPORTED_POLICY_KINDS)}")


def _hand_policy(
    adapter: LoadedPolicyAdapter,
    rng: np.random.Generator,
    sample_counts: list[int] | None,
) -> PolicyAdapter:
    if isinstance(adapter, PolicyMetaStrategyAdapter):
        member_i = adapter.sample_member_index(rng)
        if sample_counts is not None:
            sample_counts[member_i] += 1
        return adapter.members[member_i]
    return adapter


def _mixture_metric_fields(
    prefix: str,
    adapter: LoadedPolicyAdapter,
    sample_counts: list[int] | None,
) -> dict[str, Any]:
    if not isinstance(adapter, PolicyMetaStrategyAdapter):
        return {}
    return {
        f"{prefix}_mixture_size": int(adapter.size),
        f"{prefix}_mixture_weights": [float(weight) for weight in adapter.weights],
        f"{prefix}_mixture_checkpoints": [
            str(member.checkpoint_path) for member in adapter.members
        ],
        f"{prefix}_mixture_kinds": [str(member.kind) for member in adapter.members],
        f"{prefix}_mixture_algorithms": [str(member.algorithm) for member in adapter.members],
        f"{prefix}_mixture_sample_counts": (
            [int(count) for count in sample_counts]
            if sample_counts is not None
            else [0 for _member in adapter.members]
        ),
    }


def evaluate_loaded_policies_head_to_head(
    candidate: LoadedPolicyAdapter,
    baseline: LoadedPolicyAdapter,
    *,
    n_games: int = 1000,
    device: str = "auto",
    seed: int = 20260527,
    initial_chips: int | None = None,
    max_steps_per_hand: int | None = None,
    min_lower95_candidate_payoff: float | None = None,
    eval_state_backend: str = "full-deck",
) -> dict:
    """Evaluate two loaded policies with duplicate-swapped seats."""
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    chips = int(initial_chips or candidate.initial_chips or baseline.initial_chips or 1000)
    max_steps = int(
        max_steps_per_hand
        or candidate.max_steps_per_hand
        or baseline.max_steps_per_hand
        or 256
    )
    if eval_state_backend not in {"full-deck", "fast-state-canonical-deal"}:
        raise ValueError("eval_state_backend must be one of: full-deck, fast-state-canonical-deal")
    pair_payoffs: list[float] = []
    candidate_payoffs: list[float] = []
    candidate_sample_counts = (
        [0 for _member in candidate.members]
        if isinstance(candidate, PolicyMetaStrategyAdapter)
        else None
    )
    baseline_sample_counts = (
        [0 for _member in baseline.members]
        if isinstance(baseline, PolicyMetaStrategyAdapter)
        else None
    )
    total_steps = 0
    started = time.perf_counter()
    pair_idx = 0
    while len(candidate_payoffs) < int(n_games):
        game_seed = int(seed) + pair_idx
        action_seed = int(seed) + 1_000_000 + pair_idx
        swapped_payoffs: list[float] = []
        for candidate_seat in (0, 1):
            if len(candidate_payoffs) >= int(n_games):
                break
            random.seed(game_seed)
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)
            rng = np.random.default_rng(action_seed)
            candidate_policy = _hand_policy(candidate, rng, candidate_sample_counts)
            baseline_policy = _hand_policy(baseline, rng, baseline_sample_counts)
            policies = {
                int(candidate_seat): candidate_policy,
                int(1 - candidate_seat): baseline_policy,
            }
            state = new_game(2, initial_chips=chips)
            if eval_state_backend == "fast-state-canonical-deal":
                state = fast_state_from_full_deck_state(state, future_deal_mode="random")
            steps = 0
            while not state.is_terminal and steps < max_steps:
                player_i = int(getattr(state, "player_i", getattr(state, "current_player_i", 0)))
                adapter = policies[player_i]
                features = state.to_feature_vector().astype(np.float32, copy=False)
                legal_mask = (
                    state.get_legal_mask()
                    if eval_state_backend == "fast-state-canonical-deal"
                    else get_legal_mask(state)
                )
                action_idx = adapter.select_action(
                    state=state,
                    features=features,
                    legal_mask=legal_mask,
                    device=resolved_device,
                    rng=rng,
                )
                if eval_state_backend == "fast-state-canonical-deal":
                    state.apply_action(int(action_idx))
                else:
                    state = state.apply_action(INDEX_TO_ACTION[int(action_idx)])
                steps += 1
            total_steps += steps
            payoff = float(state.payout.get(int(candidate_seat), 0)) / float(chips)
            candidate_payoffs.append(payoff)
            swapped_payoffs.append(payoff)
        if swapped_payoffs:
            pair_payoffs.append(float(np.mean(swapped_payoffs)))
        pair_idx += 1
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payoff_arr = np.asarray(pair_payoffs, dtype=np.float64)
    mean = float(payoff_arr.mean()) if payoff_arr.size else 0.0
    std = float(payoff_arr.std(ddof=1)) if payoff_arr.size > 1 else 0.0
    se = std / float(np.sqrt(payoff_arr.size)) if payoff_arr.size else 0.0
    lower95 = float(mean - 1.96 * se)
    upper95 = float(mean + 1.96 * se)
    passed = True
    if min_lower95_candidate_payoff is not None:
        passed = bool(lower95 >= float(min_lower95_candidate_payoff))
    return {
        "algorithm": "mixed_native_policy_h2h",
        "role": "duplicate_swapped_mixed_native_policy_league",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "num_actions": N_ACTIONS,
        "candidate_checkpoint": candidate.checkpoint_path,
        "baseline_checkpoint": baseline.checkpoint_path,
        "candidate_kind": candidate.kind,
        "baseline_kind": baseline.kind,
        "candidate_algorithm": candidate.algorithm,
        "baseline_algorithm": baseline.algorithm,
        "eval_state_backend": str(eval_state_backend),
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_training_data": False,
        "promotion": False,
        "n_games": int(n_games),
        "n_pairs": int(len(pair_payoffs)),
        "initial_chips": int(chips),
        "max_steps_per_hand": int(max_steps),
        "eval_seconds": float(elapsed),
        "eval_steps": int(total_steps),
        "eval_games_per_second": float(int(n_games) / max(elapsed, 1e-9)),
        "mean_candidate_payoff": mean,
        "std_candidate_payoff": std,
        "lower95_candidate_payoff": lower95,
        "upper95_candidate_payoff": upper95,
        "min_lower95_candidate_payoff": (
            None if min_lower95_candidate_payoff is None else float(min_lower95_candidate_payoff)
        ),
        "passed": bool(passed),
        **device_info,
        **_mixture_metric_fields("candidate", candidate, candidate_sample_counts),
        **_mixture_metric_fields("baseline", baseline, baseline_sample_counts),
    }


def evaluate_mixed_policy_head_to_head(
    *,
    candidate_checkpoint: str,
    candidate_kind: str,
    baseline_checkpoint: str,
    baseline_kind: str,
    n_games: int = 1000,
    device: str = "auto",
    seed: int = 20260527,
    initial_chips: int | None = None,
    max_steps_per_hand: int | None = None,
    min_lower95_candidate_payoff: float | None = None,
    eval_state_backend: str = "full-deck",
) -> dict:
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate = load_policy_adapter(
        candidate_checkpoint,
        kind=candidate_kind,
        device=resolved_device,
    )
    baseline = load_policy_adapter(
        baseline_checkpoint,
        kind=baseline_kind,
        device=resolved_device,
    )
    return evaluate_loaded_policies_head_to_head(
        candidate,
        baseline,
        n_games=n_games,
        device=device,
        seed=seed,
        initial_chips=initial_chips,
        max_steps_per_hand=max_steps_per_hand,
        min_lower95_candidate_payoff=min_lower95_candidate_payoff,
        eval_state_backend=eval_state_backend,
    )


def evaluate_meta_strategy_head_to_head(
    *,
    candidate_checkpoints: Sequence[str],
    candidate_kinds: Sequence[str],
    candidate_weights: Sequence[float],
    baseline_checkpoint: str,
    baseline_kind: str,
    n_games: int = 1000,
    device: str = "auto",
    seed: int = 20260527,
    initial_chips: int | None = None,
    max_steps_per_hand: int | None = None,
    min_lower95_candidate_payoff: float | None = None,
    eval_state_backend: str = "full-deck",
    meta_strategy_source: str = "empirical-game",
) -> dict:
    """Evaluate a fixed per-hand checkpoint meta-strategy against one baseline."""
    if len(candidate_checkpoints) != len(candidate_kinds):
        raise ValueError("candidate_checkpoints and candidate_kinds must have the same length")
    if len(candidate_checkpoints) != len(candidate_weights):
        raise ValueError("candidate_checkpoints and candidate_weights must have the same length")
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    members = [
        load_policy_adapter(checkpoint, kind=kind, device=resolved_device)
        for checkpoint, kind in zip(candidate_checkpoints, candidate_kinds, strict=True)
    ]
    candidate = make_policy_meta_strategy_adapter(
        members,
        weights=candidate_weights,
        checkpoint_path=str(meta_strategy_source),
    )
    baseline = load_policy_adapter(
        baseline_checkpoint,
        kind=baseline_kind,
        device=resolved_device,
    )
    metrics = evaluate_loaded_policies_head_to_head(
        candidate,
        baseline,
        n_games=n_games,
        device=device,
        seed=seed,
        initial_chips=initial_chips,
        max_steps_per_hand=max_steps_per_hand,
        min_lower95_candidate_payoff=min_lower95_candidate_payoff,
        eval_state_backend=eval_state_backend,
    )
    metrics.update(
        {
            "candidate_meta_strategy_source": str(meta_strategy_source),
            "candidate_meta_strategy_raw_weights": [
                float(weight) for weight in candidate_weights
            ],
            "candidate_meta_strategy_normalized_weights": [
                float(weight) for weight in candidate.weights
            ],
        }
    )
    return metrics
