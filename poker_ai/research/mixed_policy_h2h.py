"""Duplicate-swapped H2H evaluation for mixed native policy checkpoint formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import random
import time

import numpy as np
import torch

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, new_game
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
    "tianshou-ppo",
    "agilerl-ippo",
    "rllib-ppo",
)
_KIND_ALIASES = {
    "rainbow": "tianshou-rainbow",
    "tianshou_rainbow": "tianshou-rainbow",
    "native_ppo": "native-ppo",
    "ppo": "tianshou-ppo",
    "tianshou_ppo": "tianshou-ppo",
    "agilerl_ippo": "agilerl-ippo",
    "ippo": "agilerl-ippo",
    "rllib_ppo": "rllib-ppo",
    "rllib": "rllib-ppo",
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
    raise ValueError(f"kind must be one of: {', '.join(SUPPORTED_POLICY_KINDS)}")


def evaluate_loaded_policies_head_to_head(
    candidate: PolicyAdapter,
    baseline: PolicyAdapter,
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
            policies = {
                int(candidate_seat): candidate,
                int(1 - candidate_seat): baseline,
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
        min_lower95_candidate_payoff=min_lower95_candidate_payoff,
        eval_state_backend=eval_state_backend,
    )
