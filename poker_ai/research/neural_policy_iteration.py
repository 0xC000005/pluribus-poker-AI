"""AlphaZero-style neural self-play policy-iteration contract smoke.

This module is deliberately small. It establishes the loop shape we want before
claiming strength: a stochastic neural policy is the actor, local self-play
creates training states, and a policy-improvement teacher returns legal mixed
policy targets for the neural actor to learn.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.games.full_deck.state import (
    INDEX_TO_ACTION,
    N_ACTIONS,
    N_FEATURES,
    card_to_index,
    new_game,
)
from poker_ai.research.game_theoretic_rl import legal_softmax
from poker_ai.research.native_nfsp import get_legal_mask, resolve_device, select_action


@dataclass(frozen=True)
class NeuralPolicyIterationConfig:
    self_play_hands: int = 16
    max_self_play_hands: int | None = None
    min_searchable_self_play_states: int = 0
    max_improvement_targets: int = 128
    train_steps: int = 32
    hidden_dim: int = 64
    batch_size: int = 64
    parallel_self_play_hands: int = 1
    self_play_state_backend: str = "full_deck"
    lr: float = 1e-3
    value_loss_weight: float = 0.25
    policy_sample_weighting: str = "uniform"
    teacher_mode: str = "legal_mixed"
    preferred_action: int = 1
    preferred_action_prob: float = 0.7
    cfr_iterations: int = 5
    cfr_backend: str = "cpu"
    cfr_device: str | None = None
    cfr_batch_roots: bool = False
    cfr_batch_min_roots: int = 2
    rollout_worlds: int = 8
    rollout_temperature: float = 100.0
    rollout_continuation_policy: str = "call"
    min_resolver_targets: int = 0
    min_target_streets: int = 0
    max_target_top_action_fraction: float = 1.0
    initial_chips: int = 1000
    max_steps_per_hand: int = 256
    seed: int = 20260526
    device: str = "auto"
    initial_checkpoint_path: str | None = None
    checkpoint_path: str | None = None

    @property
    def feature_dim(self) -> int:
        return N_FEATURES


@dataclass(frozen=True)
class NeuralPolicyIterationPublicWorldGateConfig:
    checkpoint_path: str
    n_roots: int = 32
    n_worlds: int = 16
    seed: int = 20260526
    continuation_policy: str = "call"
    max_steps_per_hand: int = 128
    initial_chips: int = 1000
    small_blind: int = 50
    big_blind: int = 100
    device: str = "auto"


@dataclass(frozen=True)
class NeuralPolicyIterationCFRGateConfig:
    checkpoint_path: str
    state_source_checkpoint_path: str | None = None
    n_roots: int = 32
    max_self_play_hands: int = 1024
    max_steps_per_hand: int = 128
    cfr_iterations: int = 5
    cfr_backend: str = "cpu"
    cfr_device: str | None = None
    initial_chips: int = 1000
    seed: int = 20260526
    device: str = "auto"


@dataclass(frozen=True)
class SelfPlayPolicySample:
    features: np.ndarray
    legal_mask: np.ndarray
    behavior_policy: np.ndarray
    action: int
    player: int
    value_target: float
    source: str = "local_self_play"
    state: object | None = None


class _PolicyNet(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class _ValueNet(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


class LegalMixedPolicyImprovementTeacher:
    """Contract test-double for a search teacher.

    The production teacher should be public-belief CFR/resolving. This class is
    intentionally only a legal mixed-target source so the neural self-play loop
    can be tested independently from resolver quality.
    """

    def __init__(self, preferred_action: int = 1, preferred_action_prob: float = 0.7):
        self.preferred_action = int(preferred_action)
        self.preferred_action_prob = float(np.clip(preferred_action_prob, 0.0, 1.0))

    def improve_one(self, legal_mask: np.ndarray) -> np.ndarray:
        mask = (np.asarray(legal_mask, dtype=np.float32).reshape(N_ACTIONS) > 0).astype(
            np.float32
        )
        legal_indices = np.flatnonzero(mask > 0)
        if legal_indices.size <= 0:
            raise ValueError("legal_mask must contain at least one legal action")
        if legal_indices.size == 1:
            return mask / float(mask.sum())
        if self.preferred_action not in set(int(i) for i in legal_indices):
            return mask / float(mask.sum())
        target = np.zeros(N_ACTIONS, dtype=np.float32)
        target[self.preferred_action] = self.preferred_action_prob
        rest = [int(i) for i in legal_indices if int(i) != self.preferred_action]
        rest_mass = max(1.0 - self.preferred_action_prob, 0.0)
        for action in rest:
            target[action] = rest_mass / float(len(rest))
        total = float(target.sum())
        if total <= 1e-8:
            return mask / float(mask.sum())
        return (target / total).astype(np.float32)

    def improve(self, samples: list[SelfPlayPolicySample]) -> np.ndarray:
        return np.stack([self.improve_one(sample.legal_mask) for sample in samples]).astype(
            np.float32
        )


class PublicBeliefCFRPolicyImprovementTeacher:
    """Use one-street public-belief CFR as the policy-improvement teacher."""

    def __init__(
        self,
        *,
        n_iterations: int = 5,
        backend: str = "cpu",
        device: str | None = None,
        batch_roots: bool = False,
        batch_min_roots: int = 2,
        fallback_teacher: LegalMixedPolicyImprovementTeacher | None = None,
    ):
        self.n_iterations = int(n_iterations)
        self.backend = str(backend)
        self.device = device
        self.batch_roots = bool(batch_roots)
        self.batch_min_roots = max(int(batch_min_roots), 2)
        self.fallback_teacher = fallback_teacher or LegalMixedPolicyImprovementTeacher()
        self.last_stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, float | int]:
        return {
            "resolver_targets": 0,
            "fallback_targets": 0,
            "ineligible_no_state": 0,
            "ineligible_street": 0,
            "ineligible_mid_street": 0,
            "solver_failures": 0,
            "zero_mass_solver_targets": 0,
            "batched_solver_groups": 0,
            "batched_solver_roots": 0,
            "batchable_solver_groups": 0,
            "largest_batchable_solver_group": 0,
            "serial_solver_roots": 0,
            "batched_solver_failures": 0,
            "solver_context_roots": 0,
            "solver_context_failures": 0,
            "solver_context_sec": 0.0,
            "batched_solver_sec": 0.0,
            "serial_solver_sec": 0.0,
        }

    def _fallback(self, legal_mask: np.ndarray, reason: str) -> np.ndarray:
        self.last_stats["fallback_targets"] += 1
        self.last_stats[reason] = self.last_stats.get(reason, 0) + 1
        return self.fallback_teacher.improve_one(legal_mask)

    @staticmethod
    def _load_solve_street():
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from solver import solve_street  # noqa: PLC0415

        return solve_street

    @staticmethod
    def _load_solver_primitives():
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from fast_cfr import solve_cfr_levelsync_torch_batched_same_topology  # noqa: PLC0415
        from solver import StreetSolver, resolve_solver_backend  # noqa: PLC0415

        return StreetSolver, resolve_solver_backend, solve_cfr_levelsync_torch_batched_same_topology

    @staticmethod
    def _is_street_root(state: object) -> bool:
        stage = str(getattr(state, "betting_stage", ""))
        if stage not in {"turn", "river"}:
            return False
        history = getattr(state, "_history", {})
        return len(history.get(stage, [])) == 0

    @staticmethod
    def _state_solver_inputs(state: object) -> dict:
        acting_player = int(state.player_i)
        villain_i = 1 - acting_player
        board_idx = [card_to_index(card) for card in state.community_cards]
        our_cards_idx = [card_to_index(card) for card in state.current_player.cards]
        pot = int(getattr(getattr(state, "_table"), "pot").total)
        hero_stack = int(state.players[acting_player].n_chips)
        villain_stack = int(state.players[villain_i].n_chips)
        return {
            "our_cards_idx": our_cards_idx,
            "board_idx": board_idx,
            "pot": pot,
            "hero_stack": hero_stack,
            "villain_stack": villain_stack,
            "hero_first": True,
        }

    @staticmethod
    def _topology_key(tree: dict, *, n_hands: int) -> tuple:
        n_nodes = int(tree["n_nodes"])
        terminal_type = np.asarray(
            tree.get("terminal_type", np.zeros(n_nodes, dtype=np.int32)),
            dtype=np.int32,
        )
        return (
            n_nodes,
            int(tree["n_actions"]),
            int(n_hands),
            np.asarray(tree["player"], dtype=np.int32).tobytes(),
            np.asarray(tree["parent_idx"], dtype=np.int32).tobytes(),
            np.asarray(tree["children"], dtype=np.int32).tobytes(),
            terminal_type.tobytes(),
        )

    def _build_solver_context(self, sample: SelfPlayPolicySample, index: int) -> dict:
        StreetSolver, _resolve_solver_backend, _batch_solver = self._load_solver_primitives()
        state = sample.state
        if state is None:
            raise ValueError("ineligible_no_state")
        if str(getattr(state, "betting_stage", "")) not in {"turn", "river"}:
            raise ValueError("ineligible_street")
        if not self._is_street_root(state):
            raise ValueError("ineligible_mid_street")
        inputs = self._state_solver_inputs(state)
        solver = StreetSolver(
            inputs["board_idx"],
            inputs["pot"],
            inputs["hero_stack"],
            inputs["villain_stack"],
            inputs["hero_first"],
        )
        return {
            "index": int(index),
            "sample": sample,
            "solver": solver,
            "our_hand": tuple(sorted(int(card) for card in inputs["our_cards_idx"])),
            "legal_mask": np.asarray(sample.legal_mask, dtype=np.float32).reshape(N_ACTIONS),
            "topology_key": self._topology_key(solver._tree, n_hands=solver.n),
        }

    @staticmethod
    def _target_from_strategy_sum(context: dict, strategy_sum: np.ndarray) -> np.ndarray | None:
        solver = context["solver"]
        solver._strategy_sum = np.asarray(strategy_sum, dtype=np.float32)
        strategy = solver.get_strategy(context["our_hand"], solver.root)
        target = np.zeros(N_ACTIONS, dtype=np.float32)
        legal_mask = context["legal_mask"]
        for action, probability in strategy.items():
            action_idx = int(action)
            if 0 <= action_idx < N_ACTIONS and legal_mask[action_idx] > 0:
                target[action_idx] = max(float(probability), 0.0)
        total = float(target.sum())
        if total <= 1e-8:
            return None
        return (target / total).astype(np.float32)

    def _solve_contexts_serial(
        self,
        contexts: list[dict],
        targets: list[np.ndarray | None],
    ) -> None:
        _StreetSolver, resolve_solver_backend, _batch_solver = self._load_solver_primitives()
        backend, device = resolve_solver_backend(self.backend, self.device)
        for context in contexts:
            started = time.perf_counter()
            fallback_reason = "zero_mass_solver_targets"
            try:
                solver = context["solver"]
                solver.solve(
                    n_iterations=max(int(self.n_iterations), 1),
                    backend=backend,
                    device=device,
                )
                target = self._target_from_strategy_sum(context, solver._strategy_sum)
            except Exception:
                self.last_stats["solver_failures"] += 1
                fallback_reason = "solver_failures"
                target = None
            self.last_stats["serial_solver_sec"] += time.perf_counter() - started
            self.last_stats["serial_solver_roots"] += 1
            index = int(context["index"])
            if target is None:
                targets[index] = self._fallback(context["legal_mask"], fallback_reason)
            else:
                targets[index] = target
                self.last_stats["resolver_targets"] += 1

    def _solve_context_group_batched(
        self,
        contexts: list[dict],
        targets: list[np.ndarray | None],
    ) -> bool:
        if len(contexts) < self.batch_min_roots:
            return False
        _StreetSolver, resolve_solver_backend, batch_solver = self._load_solver_primitives()
        backend, device = resolve_solver_backend(self.backend, self.device)
        if backend != "torch-levelsync":
            return False
        started = time.perf_counter()
        try:
            solvers = [context["solver"] for context in contexts]
            _regret_sum, strategy_sums = batch_solver(
                [solver._tree for solver in solvers],
                solvers[0].n,
                [solver.win_m for solver in solvers],
                [solver.lose_m for solver in solvers],
                [solver.tie_m for solver in solvers],
                [solver.valid for solver in solvers],
                [solver.pot_start for solver in solvers],
                [solver.hero_stack_start for solver in solvers],
                [solver.villain_stack_start for solver in solvers],
                n_iterations=max(int(self.n_iterations), 1),
                device=device or "cuda",
            )
            for context, strategy_sum in zip(contexts, strategy_sums, strict=True):
                target = self._target_from_strategy_sum(context, strategy_sum)
                if target is None:
                    targets[int(context["index"])] = self._fallback(
                        context["legal_mask"],
                        "zero_mass_solver_targets",
                    )
                else:
                    targets[int(context["index"])] = target
                    self.last_stats["resolver_targets"] += 1
            self.last_stats["batched_solver_groups"] += 1
            self.last_stats["batched_solver_roots"] += len(contexts)
            return True
        except Exception:
            self.last_stats["batched_solver_failures"] += 1
            return False
        finally:
            self.last_stats["batched_solver_sec"] += time.perf_counter() - started

    def improve_one(self, sample: SelfPlayPolicySample) -> np.ndarray:
        legal_mask = np.asarray(sample.legal_mask, dtype=np.float32).reshape(N_ACTIONS)
        state = sample.state
        if state is None:
            return self._fallback(legal_mask, "ineligible_no_state")
        if str(getattr(state, "betting_stage", "")) not in {"turn", "river"}:
            return self._fallback(legal_mask, "ineligible_street")
        if not self._is_street_root(state):
            return self._fallback(legal_mask, "ineligible_mid_street")
        try:
            solve_street = self._load_solve_street()
            _, strategy, _, _ = solve_street(
                **self._state_solver_inputs(state),
                action_str="",
                n_iterations=max(int(self.n_iterations), 1),
                backend=self.backend,
                device=self.device,
            )
        except Exception:
            return self._fallback(legal_mask, "solver_failures")

        target = np.zeros(N_ACTIONS, dtype=np.float32)
        for action, probability in strategy.items():
            action_idx = int(action)
            if 0 <= action_idx < N_ACTIONS and legal_mask[action_idx] > 0:
                target[action_idx] = max(float(probability), 0.0)
        total = float(target.sum())
        if total <= 1e-8:
            return self._fallback(legal_mask, "zero_mass_solver_targets")
        self.last_stats["resolver_targets"] += 1
        return (target / total).astype(np.float32)

    def improve(self, samples: list[SelfPlayPolicySample]) -> np.ndarray:
        self.last_stats = self._empty_stats()
        if self.batch_roots:
            targets: list[np.ndarray | None] = [None] * len(samples)
            contexts: list[dict] = []
            for index, sample in enumerate(samples):
                legal_mask = np.asarray(sample.legal_mask, dtype=np.float32).reshape(N_ACTIONS)
                context_start = time.perf_counter()
                try:
                    contexts.append(self._build_solver_context(sample, index))
                    self.last_stats["solver_context_roots"] += 1
                except ValueError as exc:
                    reason = str(exc)
                    if reason not in self.last_stats:
                        reason = "solver_failures"
                    targets[index] = self._fallback(legal_mask, reason)
                    if reason == "solver_failures":
                        self.last_stats["solver_context_failures"] += 1
                except Exception:
                    targets[index] = self._fallback(legal_mask, "solver_failures")
                    self.last_stats["solver_context_failures"] += 1
                finally:
                    self.last_stats["solver_context_sec"] += time.perf_counter() - context_start
            groups: dict[tuple, list[dict]] = {}
            for context in contexts:
                groups.setdefault(context["topology_key"], []).append(context)
            group_sizes = [len(group) for group in groups.values()]
            self.last_stats["batchable_solver_groups"] = int(
                sum(1 for size in group_sizes if size >= self.batch_min_roots)
            )
            self.last_stats["largest_batchable_solver_group"] = int(
                max(group_sizes, default=0)
            )
            for group in groups.values():
                if not self._solve_context_group_batched(group, targets):
                    self._solve_contexts_serial(group, targets)
            return np.stack([target for target in targets if target is not None]).astype(
                np.float32
            )
        return np.stack([self.improve_one(sample) for sample in samples]).astype(np.float32)


class PublicWorldRolloutPolicyImprovementTeacher:
    """Dense public-world search teacher for early self-play roots."""

    def __init__(
        self,
        *,
        n_worlds: int = 8,
        temperature: float = 100.0,
        continuation_policy: str = "call",
        seed: int = 20260526,
        fallback_teacher: LegalMixedPolicyImprovementTeacher | None = None,
    ):
        self.n_worlds = int(n_worlds)
        self.temperature = max(float(temperature), 1e-6)
        self.continuation_policy = str(continuation_policy)
        self.seed = int(seed)
        self.fallback_teacher = fallback_teacher or LegalMixedPolicyImprovementTeacher()
        self.last_stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "rollout_targets": 0,
            "fallback_targets": 0,
            "rollout_ineligible_no_state": 0,
            "rollout_ineligible_not_preflop_root": 0,
            "rollout_failures": 0,
            "rollout_zero_mass_targets": 0,
        }

    @staticmethod
    def _is_preflop_root(state: object) -> bool:
        if hasattr(state, "stage") and hasattr(state, "PREFLOP"):
            return int(getattr(state, "stage")) == int(getattr(state, "PREFLOP")) and int(
                getattr(state, "n_actions", 0)
            ) == 0
        if str(getattr(state, "betting_stage", "")) != "pre_flop":
            return False
        history = getattr(state, "_history", {})
        return len(history.get("pre_flop", [])) == 0

    def _fallback(self, legal_mask: np.ndarray, reason: str) -> np.ndarray:
        self.last_stats["fallback_targets"] += 1
        self.last_stats[reason] = self.last_stats.get(reason, 0) + 1
        return self.fallback_teacher.improve_one(legal_mask)

    def _continuation(self):
        from poker_ai.research.public_action_rollout_value import (  # noqa: PLC0415
            call_policy,
            masked_uniform_policy,
        )

        if self.continuation_policy == "call":
            return call_policy
        if self.continuation_policy == "uniform":
            return masked_uniform_policy
        raise ValueError("rollout_continuation_policy must be one of: call, uniform")

    def improve_one(self, sample: SelfPlayPolicySample) -> np.ndarray:
        legal_mask = np.asarray(sample.legal_mask, dtype=np.float32).reshape(N_ACTIONS)
        state = sample.state
        if state is None:
            return self._fallback(legal_mask, "rollout_ineligible_no_state")
        if not self._is_preflop_root(state):
            return self._fallback(legal_mask, "rollout_ineligible_not_preflop_root")
        try:
            from poker_ai.research.public_action_rollout_value import (  # noqa: PLC0415
                sample_public_worlds,
                score_first_actions_across_worlds,
            )

            if hasattr(state, "hole_cards") and hasattr(state, "current_player_i"):
                player_i = int(getattr(state, "current_player_i"))
                hero_cards = tuple(int(card) for card in state.hole_cards[player_i])
            else:
                hero_cards = tuple(card_to_index(card) for card in state.current_player.cards)
            worlds = sample_public_worlds(
                hero_cards=hero_cards,
                n_worlds=max(int(self.n_worlds), 1),
                seed=self.seed + int(sample.player),
            )
            result = score_first_actions_across_worlds(
                worlds=worlds,
                continuation_policy=self._continuation(),
                seed=self.seed,
                max_steps_per_hand=64,
                initial_chips=int(getattr(state, "_initial_n_chips", 1000)),
                small_blind=int(getattr(state, "small_blind", 50)),
                big_blind=int(getattr(state, "big_blind", 100)),
            )
        except Exception:
            return self._fallback(legal_mask, "rollout_failures")

        target = np.zeros(N_ACTIONS, dtype=np.float32)
        legal_values = []
        legal_actions = []
        for action, value in result.action_values.items():
            action_idx = int(action)
            if 0 <= action_idx < N_ACTIONS and legal_mask[action_idx] > 0:
                legal_actions.append(action_idx)
                legal_values.append(float(value))
        if not legal_actions:
            return self._fallback(legal_mask, "rollout_zero_mass_targets")
        values = np.asarray(legal_values, dtype=np.float64) / self.temperature
        values = values - float(np.max(values))
        probs = np.exp(values)
        total = float(probs.sum())
        if total <= 1e-12 or not np.isfinite(total):
            return self._fallback(legal_mask, "rollout_zero_mass_targets")
        probs = probs / total
        for action_idx, probability in zip(legal_actions, probs):
            target[int(action_idx)] = float(probability)
        self.last_stats["rollout_targets"] += 1
        return target.astype(np.float32)

    def improve(self, samples: list[SelfPlayPolicySample]) -> np.ndarray:
        self.last_stats = self._empty_stats()
        return np.stack([self.improve_one(sample) for sample in samples]).astype(np.float32)


class PolicyContinuationPublicWorldRolloutTeacher(PublicWorldRolloutPolicyImprovementTeacher):
    """Public-world rollout teacher that uses the current neural policy to continue.

    This is the closer imperfect-information analogue of AlphaZero's
    policy-improvement loop: the neural actor creates states, local search
    scores the immediate choices, and future rollout decisions use the same
    neural actor instead of a fixed call/uniform continuation.
    """

    def __init__(
        self,
        *,
        policy_net: nn.Module,
        device: torch.device,
        n_worlds: int = 8,
        temperature: float = 100.0,
        seed: int = 20260526,
        fallback_teacher: LegalMixedPolicyImprovementTeacher | None = None,
    ):
        super().__init__(
            n_worlds=n_worlds,
            temperature=temperature,
            continuation_policy="policy",
            seed=seed,
            fallback_teacher=fallback_teacher,
        )
        self.policy_net = policy_net
        self.device = device

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        stats = PublicWorldRolloutPolicyImprovementTeacher._empty_stats()
        stats["policy_continuation_targets"] = 0
        return stats

    def _continuation(self):
        def _policy(state: object, rng: np.random.Generator) -> int:
            legal_mask = state.get_legal_mask()
            features = state.to_feature_vector().astype(np.float32, copy=False)
            probs = _network_policy(self.policy_net, features, legal_mask, self.device)
            return select_action(probs, legal_mask, rng=rng)

        return _policy

    def improve_one(self, sample: SelfPlayPolicySample) -> np.ndarray:
        before = int(self.last_stats.get("rollout_targets", 0))
        target = super().improve_one(sample)
        after = int(self.last_stats.get("rollout_targets", 0))
        if after > before:
            self.last_stats["policy_continuation_targets"] += 1
        return target


class PolicyContinuationPublicStateRolloutTeacher(PolicyContinuationPublicWorldRolloutTeacher):
    """All-street public-state rollout teacher using the current neural policy."""

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "public_state_rollout_targets": 0,
            "policy_continuation_targets": 0,
            "fallback_targets": 0,
            "public_state_rollout_ineligible_no_state": 0,
            "public_state_rollout_terminal_state": 0,
            "public_state_rollout_failures": 0,
            "public_state_rollout_zero_mass_targets": 0,
        }

    def _fallback(self, legal_mask: np.ndarray, reason: str) -> np.ndarray:
        self.last_stats["fallback_targets"] += 1
        self.last_stats[reason] = self.last_stats.get(reason, 0) + 1
        return self.fallback_teacher.improve_one(legal_mask)

    def improve_one(self, sample: SelfPlayPolicySample) -> np.ndarray:
        legal_mask = np.asarray(sample.legal_mask, dtype=np.float32).reshape(N_ACTIONS)
        state = sample.state
        if state is None:
            return self._fallback(legal_mask, "public_state_rollout_ineligible_no_state")
        if bool(getattr(state, "is_terminal", False)):
            return self._fallback(legal_mask, "public_state_rollout_terminal_state")
        try:
            from poker_ai.research.decision_value_actor import (  # noqa: PLC0415
                score_first_actions_for_public_state,
            )

            result = score_first_actions_for_public_state(
                state,
                n_worlds=max(int(self.n_worlds), 1),
                continuation_policy=self._continuation(),
                seed=self.seed + int(sample.player),
                max_steps_per_hand=64,
            )
        except Exception:
            return self._fallback(legal_mask, "public_state_rollout_failures")

        target = np.zeros(N_ACTIONS, dtype=np.float32)
        legal_actions: list[int] = []
        legal_values: list[float] = []
        for action, value in result.action_values.items():
            action_idx = int(action)
            if 0 <= action_idx < N_ACTIONS and legal_mask[action_idx] > 0:
                legal_actions.append(action_idx)
                legal_values.append(float(value))
        if not legal_actions:
            return self._fallback(legal_mask, "public_state_rollout_zero_mass_targets")
        values = np.asarray(legal_values, dtype=np.float64) / self.temperature
        values = values - float(np.max(values))
        probs = np.exp(values)
        total = float(probs.sum())
        if total <= 1e-12 or not np.isfinite(total):
            return self._fallback(legal_mask, "public_state_rollout_zero_mass_targets")
        probs = probs / total
        for action_idx, probability in zip(legal_actions, probs):
            target[int(action_idx)] = float(probability)
        self.last_stats["public_state_rollout_targets"] += 1
        self.last_stats["policy_continuation_targets"] += 1
        return target.astype(np.float32)


class PublicWorldValuePolicyImprovementTeacher:
    """One-step public-world search teacher backed by the learned value net."""

    def __init__(
        self,
        *,
        value_net: nn.Module,
        device: torch.device,
        n_worlds: int = 8,
        temperature: float = 1.0,
        seed: int = 20260526,
        use_policy_prior: bool = False,
        fallback_teacher: LegalMixedPolicyImprovementTeacher | None = None,
    ):
        self.value_net = value_net
        self.device = device
        self.n_worlds = int(n_worlds)
        self.temperature = max(float(temperature), 1e-6)
        self.seed = int(seed)
        self.use_policy_prior = bool(use_policy_prior)
        self.fallback_teacher = fallback_teacher or LegalMixedPolicyImprovementTeacher()
        self.last_stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "value_search_targets": 0,
            "fallback_targets": 0,
            "value_search_ineligible_no_state": 0,
            "value_search_ineligible_not_preflop_root": 0,
            "value_search_failures": 0,
            "value_search_zero_mass_targets": 0,
        }

    def _fallback(self, legal_mask: np.ndarray, reason: str) -> np.ndarray:
        self.last_stats["fallback_targets"] += 1
        self.last_stats[reason] = self.last_stats.get(reason, 0) + 1
        return self.fallback_teacher.improve_one(legal_mask)

    def _score_child_for_actor(self, child: object, *, actor: int, initial_chips: int) -> float:
        if bool(getattr(child, "is_terminal")):
            payout = getattr(child, "payout")
            return float(payout.get(int(actor), 0)) / max(float(initial_chips), 1.0)
        current_player = int(getattr(child, "current_player_i"))
        features = child.to_feature_vector().reshape(1, -1).astype(np.float32)
        with torch.no_grad():
            value = float(
                self.value_net(torch.from_numpy(features).to(self.device)).detach().cpu().numpy()[0]
            )
        return value if current_player == int(actor) else -value

    def improve_one(self, sample: SelfPlayPolicySample) -> np.ndarray:
        legal_mask = np.asarray(sample.legal_mask, dtype=np.float32).reshape(N_ACTIONS)
        state = sample.state
        if state is None:
            return self._fallback(legal_mask, "value_search_ineligible_no_state")
        if not PublicWorldRolloutPolicyImprovementTeacher._is_preflop_root(state):
            return self._fallback(legal_mask, "value_search_ineligible_not_preflop_root")
        try:
            from poker_ai.research.public_action_rollout_value import (  # noqa: PLC0415
                build_public_world_state,
                sample_public_worlds,
            )

            if hasattr(state, "hole_cards") and hasattr(state, "current_player_i"):
                player_i = int(getattr(state, "current_player_i"))
                hero_cards = tuple(int(card) for card in state.hole_cards[player_i])
            else:
                hero_cards = tuple(card_to_index(card) for card in state.current_player.cards)
            worlds = sample_public_worlds(
                hero_cards=hero_cards,
                n_worlds=max(int(self.n_worlds), 1),
                seed=self.seed + int(sample.player),
            )
            initial_chips = int(getattr(state, "_initial_n_chips", 1000))
            small_blind = int(getattr(state, "small_blind", 50))
            big_blind = int(getattr(state, "big_blind", 100))
            legal_actions = [
                int(action)
                for action in np.flatnonzero(legal_mask > 0)
            ]
            action_values: dict[int, float] = {}
            self.value_net.eval()
            for action in legal_actions:
                values: list[float] = []
                for world in worlds:
                    root = build_public_world_state(
                        hero_cards=world.hero_cards,
                        opponent_cards=world.opponent_cards,
                        deck_tail=world.deck_tail,
                        initial_chips=initial_chips,
                        small_blind=small_blind,
                        big_blind=big_blind,
                    )
                    actor = int(root.current_player_i)
                    child = root.copy()
                    child.apply_action(int(action))
                    values.append(
                        self._score_child_for_actor(
                            child,
                            actor=actor,
                            initial_chips=initial_chips,
                        )
                    )
                action_values[int(action)] = float(np.mean(values)) if values else 0.0
        except Exception:
            return self._fallback(legal_mask, "value_search_failures")

        target = np.zeros(N_ACTIONS, dtype=np.float32)
        if not action_values:
            return self._fallback(legal_mask, "value_search_zero_mass_targets")
        legal_actions = sorted(action_values)
        values = np.asarray([action_values[action] for action in legal_actions], dtype=np.float64)
        values = values / self.temperature
        if self.use_policy_prior:
            prior = np.asarray(sample.behavior_policy, dtype=np.float64).reshape(N_ACTIONS)
            prior_values = np.asarray([max(float(prior[action]), 1e-12) for action in legal_actions])
            values = values + np.log(prior_values)
        values = values - float(np.max(values))
        probs = np.exp(values)
        total = float(probs.sum())
        if total <= 1e-12 or not np.isfinite(total):
            return self._fallback(legal_mask, "value_search_zero_mass_targets")
        probs = probs / total
        for action_idx, probability in zip(legal_actions, probs):
            target[int(action_idx)] = float(probability)
        self.last_stats["value_search_targets"] += 1
        return target.astype(np.float32)

    def improve(self, samples: list[SelfPlayPolicySample]) -> np.ndarray:
        self.last_stats = self._empty_stats()
        return np.stack([self.improve_one(sample) for sample in samples]).astype(np.float32)


class SampledStateValuePolicyImprovementTeacher:
    """Multi-street one-sample search teacher backed by the learned value net.

    The sampled self-play state is one private-world draw from the local
    self-play distribution. The target remains an information-state target
    because the policy/value nets only consume each acting player's observation.
    """

    def __init__(
        self,
        *,
        value_net: nn.Module,
        device: torch.device,
        temperature: float = 1.0,
        use_policy_prior: bool = False,
        fallback_teacher: LegalMixedPolicyImprovementTeacher | None = None,
    ):
        self.value_net = value_net
        self.device = device
        self.temperature = max(float(temperature), 1e-6)
        self.use_policy_prior = bool(use_policy_prior)
        self.fallback_teacher = fallback_teacher or LegalMixedPolicyImprovementTeacher()
        self.last_stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "state_value_targets": 0,
            "fallback_targets": 0,
            "state_value_ineligible_no_state": 0,
            "state_value_failures": 0,
            "state_value_zero_mass_targets": 0,
        }

    def _fallback(self, legal_mask: np.ndarray, reason: str) -> np.ndarray:
        self.last_stats["fallback_targets"] += 1
        self.last_stats[reason] = self.last_stats.get(reason, 0) + 1
        return self.fallback_teacher.improve_one(legal_mask)

    def _score_child_for_actor(self, child: object, *, actor: int, initial_chips: int) -> float:
        if bool(getattr(child, "is_terminal")):
            payout = getattr(child, "payout")
            return float(payout.get(int(actor), 0)) / max(float(initial_chips), 1.0)
        current_player = _self_play_current_player_i(child)
        features = child.to_feature_vector().reshape(1, -1).astype(np.float32)
        with torch.no_grad():
            value = float(
                self.value_net(torch.from_numpy(features).to(self.device)).detach().cpu().numpy()[0]
            )
        return value if current_player == int(actor) else -value

    def improve_one(self, sample: SelfPlayPolicySample) -> np.ndarray:
        legal_mask = np.asarray(sample.legal_mask, dtype=np.float32).reshape(N_ACTIONS)
        state = sample.state
        if state is None:
            return self._fallback(legal_mask, "state_value_ineligible_no_state")
        legal_actions = [int(action) for action in np.flatnonzero(legal_mask > 0)]
        if not legal_actions:
            return self._fallback(legal_mask, "state_value_zero_mass_targets")
        try:
            actor = int(sample.player)
            initial_chips = int(getattr(state, "_initial_n_chips", getattr(state, "initial_chips", 1000)))
            self.value_net.eval()
            action_values = {
                int(action): self._score_child_for_actor(
                    _apply_self_play_action(state, int(action)),
                    actor=actor,
                    initial_chips=initial_chips,
                )
                for action in legal_actions
            }
        except Exception:
            return self._fallback(legal_mask, "state_value_failures")

        target = np.zeros(N_ACTIONS, dtype=np.float32)
        values = np.asarray([action_values[action] for action in legal_actions], dtype=np.float64)
        values = values / self.temperature
        if self.use_policy_prior:
            prior = np.asarray(sample.behavior_policy, dtype=np.float64).reshape(N_ACTIONS)
            prior_values = np.asarray([max(float(prior[action]), 1e-12) for action in legal_actions])
            values = values + np.log(prior_values)
        values = values - float(np.max(values))
        probs = np.exp(values)
        total = float(probs.sum())
        if total <= 1e-12 or not np.isfinite(total):
            return self._fallback(legal_mask, "state_value_zero_mass_targets")
        probs = probs / total
        for action_idx, probability in zip(legal_actions, probs):
            target[int(action_idx)] = float(probability)
        self.last_stats["state_value_targets"] += 1
        return target.astype(np.float32)

    def improve(self, samples: list[SelfPlayPolicySample]) -> np.ndarray:
        self.last_stats = self._empty_stats()
        return np.stack([self.improve_one(sample) for sample in samples]).astype(np.float32)


def _seed_all(seed: int) -> np.random.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    return np.random.default_rng(int(seed))


def _network_policy(
    policy_net: nn.Module,
    features: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
        logits = policy_net(x).detach().cpu().numpy().reshape(-1)
    return legal_softmax(logits, legal_mask)


def _batched_network_policies(
    policy_net: nn.Module,
    features: np.ndarray,
    legal_masks: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    features_arr = np.asarray(features, dtype=np.float32)
    masks_arr = np.asarray(legal_masks, dtype=np.float32)
    if features_arr.ndim != 2:
        raise ValueError("features must be a 2D array")
    if masks_arr.shape != (features_arr.shape[0], N_ACTIONS):
        raise ValueError(f"legal_masks must have shape (batch, {N_ACTIONS})")
    with torch.no_grad():
        x = torch.from_numpy(features_arr).to(device)
        mask = torch.from_numpy(masks_arr).to(device)
        logits = policy_net(x).masked_fill(mask <= 0, -1e4)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
    probs *= masks_arr
    totals = probs.sum(axis=1, keepdims=True)
    legal_counts = np.maximum(masks_arr.sum(axis=1, keepdims=True), 1.0)
    uniform = masks_arr / legal_counts
    return np.where(totals > 1e-8, probs / np.maximum(totals, 1e-8), uniform).astype(
        np.float32
    )


def _build_policy_net(cfg: NeuralPolicyIterationConfig, device: torch.device) -> _PolicyNet:
    return _PolicyNet(cfg.hidden_dim, cfg.feature_dim).to(device)


def _build_value_net(cfg: NeuralPolicyIterationConfig, device: torch.device) -> _ValueNet:
    return _ValueNet(cfg.hidden_dim, cfg.feature_dim).to(device)


def _load_policy_value_iteration_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[_PolicyNet, _ValueNet, dict]:
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = dict(payload.get("config", {}))
    hidden_dim = int(config.get("hidden_dim", 64))
    feature_dim = int(config.get("feature_dim", N_FEATURES))
    policy_net = _PolicyNet(hidden_dim=hidden_dim, input_dim=feature_dim).to(device)
    policy_net.load_state_dict(payload["policy_net_state_dict"])
    value_net = _ValueNet(hidden_dim=hidden_dim, input_dim=feature_dim).to(device)
    if "value_net_state_dict" not in payload:
        raise ValueError(f"checkpoint lacks value_net_state_dict: {checkpoint_path}")
    value_net.load_state_dict(payload["value_net_state_dict"])
    policy_net.eval()
    value_net.eval()
    return policy_net, value_net, payload


def _validate_initial_checkpoint_matches_config(
    payload: dict,
    cfg: NeuralPolicyIterationConfig,
    checkpoint_path: str,
) -> None:
    config = dict(payload.get("config", {}))
    checkpoint_hidden_dim = int(config.get("hidden_dim", cfg.hidden_dim))
    checkpoint_feature_dim = int(config.get("feature_dim", N_FEATURES))
    if checkpoint_hidden_dim != int(cfg.hidden_dim):
        raise ValueError(
            "initial checkpoint hidden_dim "
            f"{checkpoint_hidden_dim} does not match requested hidden_dim {int(cfg.hidden_dim)}: "
            f"{checkpoint_path}"
        )
    if checkpoint_feature_dim != int(cfg.feature_dim):
        raise ValueError(
            "initial checkpoint feature_dim "
            f"{checkpoint_feature_dim} does not match requested feature_dim {int(cfg.feature_dim)}: "
            f"{checkpoint_path}"
        )


def _append_finished_hand_samples(
    *,
    samples: list[SelfPlayPolicySample],
    records: list[tuple[object, int, np.ndarray, np.ndarray, np.ndarray, int]],
    payout: dict[int, int],
    initial_chips: int,
) -> None:
    scale = max(float(initial_chips), 1.0)
    for record_state, player, features, legal_mask, probs, action in records:
        samples.append(
            SelfPlayPolicySample(
                features=np.asarray(features, dtype=np.float32),
                legal_mask=np.asarray(legal_mask, dtype=np.float32),
                behavior_policy=np.asarray(probs, dtype=np.float32),
                action=int(action),
                player=int(player),
                value_target=float(payout.get(int(player), 0)) / scale,
                state=record_state,
            )
        )


def _new_self_play_state(cfg: NeuralPolicyIterationConfig):
    backend = str(cfg.self_play_state_backend)
    if backend == "full_deck":
        return new_game(2, initial_chips=int(cfg.initial_chips))
    if backend == "fast":
        from poker_ai.deep_cfr.fast_state import new_fast_game  # noqa: PLC0415

        return new_fast_game(2, initial_chips=int(cfg.initial_chips))
    raise ValueError("self_play_state_backend must be one of: full_deck, fast")


def _self_play_current_player_i(state: object) -> int:
    if hasattr(state, "current_player_i"):
        return int(getattr(state, "current_player_i"))
    return int(getattr(state, "player_i"))


def _self_play_legal_mask(state: object) -> np.ndarray:
    if hasattr(state, "get_legal_mask"):
        return np.asarray(state.get_legal_mask(), dtype=np.float32)
    return get_legal_mask(state)


def _apply_self_play_action(state: object, action: int) -> object:
    if hasattr(state, "get_legal_mask"):
        child = state.copy()
        child.apply_action(int(action))
        return child
    return state.apply_action(INDEX_TO_ACTION[int(action)])


def collect_stochastic_self_play_samples(
    cfg: NeuralPolicyIterationConfig,
    policy_net: nn.Module | None = None,
) -> list[SelfPlayPolicySample]:
    samples, _stats = collect_stochastic_self_play_samples_with_stats(cfg, policy_net=policy_net)
    return samples


def collect_stochastic_self_play_samples_with_stats(
    cfg: NeuralPolicyIterationConfig,
    policy_net: nn.Module | None = None,
) -> tuple[list[SelfPlayPolicySample], dict[str, float]]:
    rng = _seed_all(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    actor = policy_net if policy_net is not None else _build_policy_net(cfg, device)
    actor.to(device)
    actor.eval()

    samples: list[SelfPlayPolicySample] = []
    stats = {
        "self_play_feature_mask_sec": 0.0,
        "self_play_policy_inference_sec": 0.0,
        "self_play_action_select_sec": 0.0,
        "self_play_env_step_sec": 0.0,
        "self_play_finalize_sec": 0.0,
    }
    parallel_hands = max(1, int(cfg.parallel_self_play_hands))
    min_hands = max(int(cfg.self_play_hands), 0)
    max_hands = cfg.max_self_play_hands
    if max_hands is None:
        max_hands = min_hands
    max_hands = max(int(max_hands), min_hands)
    min_searchable = max(int(cfg.min_searchable_self_play_states), 0)

    def needs_more_hands(started: int) -> bool:
        if started < min_hands:
            return True
        if min_searchable <= 0:
            return False
        return (
            started < max_hands
            and _count_public_belief_cfr_eligible(samples) < min_searchable
        )

    if parallel_hands > 1:
        active: list[dict] = []
        started = 0
        finished = 0
        while active or needs_more_hands(started):
            while len(active) < parallel_hands and needs_more_hands(started):
                active.append(
                    {
                        "state": _new_self_play_state(cfg),
                        "records": [],
                        "steps": 0,
                    }
                )
                started += 1
            if not active:
                break
            batch_features: list[np.ndarray] = []
            batch_masks: list[np.ndarray] = []
            batch_indices: list[int] = []
            feature_start = time.perf_counter()
            for hand_idx, hand in enumerate(active):
                state = hand["state"]
                if state.is_terminal or int(hand["steps"]) >= int(cfg.max_steps_per_hand):
                    continue
                batch_features.append(state.to_feature_vector().astype(np.float32, copy=False))
                batch_masks.append(_self_play_legal_mask(state))
                batch_indices.append(hand_idx)
            stats["self_play_feature_mask_sec"] += time.perf_counter() - feature_start
            if batch_indices:
                inference_start = time.perf_counter()
                probs_batch = _batched_network_policies(
                    actor,
                    np.stack(batch_features),
                    np.stack(batch_masks),
                    device,
                )
                stats["self_play_policy_inference_sec"] += (
                    time.perf_counter() - inference_start
                )
                for row_idx, hand_idx in enumerate(batch_indices):
                    hand = active[hand_idx]
                    state = hand["state"]
                    features = batch_features[row_idx]
                    legal_mask = batch_masks[row_idx]
                    probs = probs_batch[row_idx]
                    action_start = time.perf_counter()
                    action = select_action(probs, legal_mask, rng=rng)
                    stats["self_play_action_select_sec"] += (
                        time.perf_counter() - action_start
                    )
                    hand["records"].append(
                        (
                            state,
                            _self_play_current_player_i(state),
                            features,
                            legal_mask,
                            probs,
                            action,
                        )
                    )
                    env_start = time.perf_counter()
                    hand["state"] = _apply_self_play_action(state, int(action))
                    stats["self_play_env_step_sec"] += time.perf_counter() - env_start
                    hand["steps"] = int(hand["steps"]) + 1
            survivors: list[dict] = []
            for hand in active:
                state = hand["state"]
                if state.is_terminal or int(hand["steps"]) >= int(cfg.max_steps_per_hand):
                    finalize_start = time.perf_counter()
                    _append_finished_hand_samples(
                        samples=samples,
                        records=hand["records"],
                        payout=state.payout,
                        initial_chips=int(cfg.initial_chips),
                    )
                    stats["self_play_finalize_sec"] += (
                        time.perf_counter() - finalize_start
                    )
                    finished += 1
                else:
                    survivors.append(hand)
            active = survivors
        return samples, stats

    started = 0
    while needs_more_hands(started):
        started += 1
        state = _new_self_play_state(cfg)
        hand_records: list[tuple[object, int, np.ndarray, np.ndarray, np.ndarray, int]] = []
        steps = 0
        while not state.is_terminal and steps < int(cfg.max_steps_per_hand):
            player = _self_play_current_player_i(state)
            feature_start = time.perf_counter()
            features = state.to_feature_vector().astype(np.float32, copy=False)
            legal_mask = _self_play_legal_mask(state)
            stats["self_play_feature_mask_sec"] += time.perf_counter() - feature_start
            inference_start = time.perf_counter()
            probs = _network_policy(actor, features, legal_mask, device)
            stats["self_play_policy_inference_sec"] += time.perf_counter() - inference_start
            action_start = time.perf_counter()
            action = select_action(probs, legal_mask, rng=rng)
            stats["self_play_action_select_sec"] += time.perf_counter() - action_start
            hand_records.append((state, player, features, legal_mask, probs, action))
            env_start = time.perf_counter()
            state = _apply_self_play_action(state, int(action))
            stats["self_play_env_step_sec"] += time.perf_counter() - env_start
            steps += 1
        finalize_start = time.perf_counter()
        _append_finished_hand_samples(
            samples=samples,
            records=hand_records,
            payout=state.payout,
            initial_chips=int(cfg.initial_chips),
        )
        stats["self_play_finalize_sec"] += time.perf_counter() - finalize_start
    return samples, stats


def _train_on_policy_improvement_targets(
    policy_net: nn.Module,
    value_net: nn.Module,
    samples: list[SelfPlayPolicySample],
    targets: np.ndarray,
    cfg: NeuralPolicyIterationConfig,
    device: torch.device,
    rng: np.random.Generator,
    *,
    value_samples: list[SelfPlayPolicySample] | None = None,
) -> dict[str, float]:
    value_training_samples = list(value_samples) if value_samples is not None else samples
    if not samples and not value_training_samples:
        return {
            "policy_loss": float("nan"),
            "value_loss": float("nan"),
            "total_loss": float("nan"),
            "policy_sample_weight_mean": 0.0,
            "policy_sample_weight_max": 0.0,
            "policy_training_samples": 0,
            "value_training_samples": 0,
        }

    policy_features = (
        torch.tensor(np.stack([s.features for s in samples]), device=device)
        if samples
        else torch.empty((0, cfg.feature_dim), device=device)
    )
    policy_masks = (
        torch.tensor(np.stack([s.legal_mask for s in samples]), device=device)
        if samples
        else torch.empty((0, N_ACTIONS), device=device)
    )
    target_policies = torch.tensor(targets.astype(np.float32), device=device)
    policy_sample_weights = torch.tensor(
        _policy_target_sample_weights(targets, mode=cfg.policy_sample_weighting),
        device=device,
    )
    value_features = torch.tensor(
        np.stack([s.features for s in value_training_samples]),
        device=device,
    )
    target_values = torch.tensor(
        [float(s.value_target) for s in value_training_samples],
        device=device,
    )
    optimizer = optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=float(cfg.lr),
    )
    n_policy = int(policy_features.shape[0])
    n_value = int(value_features.shape[0])
    policy_batch_size = max(1, min(int(cfg.batch_size), n_policy)) if n_policy else 0
    value_batch_size = max(1, min(int(cfg.batch_size), n_value)) if n_value else 0
    final_policy_loss = 0.0
    final_value_loss = 0.0
    final_total_loss = 0.0
    policy_net.train()
    value_net.train()
    for _ in range(max(int(cfg.train_steps), 0)):
        if n_policy:
            policy_indices = rng.choice(
                n_policy,
                size=policy_batch_size,
                replace=n_policy < policy_batch_size,
            )
            policy_index_tensor = torch.tensor(policy_indices, dtype=torch.long, device=device)
            x_policy = policy_features[policy_index_tensor]
            mask = policy_masks[policy_index_tensor]
            y_policy = target_policies[policy_index_tensor]
            y_weight = policy_sample_weights[policy_index_tensor]
            logits = policy_net(x_policy).masked_fill(mask <= 0, -1e4)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            per_sample_policy_loss = -(y_policy * log_probs).sum(dim=-1)
            policy_loss = (per_sample_policy_loss * y_weight).sum() / torch.clamp(
                y_weight.sum(),
                min=1e-8,
            )
        else:
            policy_loss = torch.zeros((), device=device)
        value_indices = rng.choice(
            n_value,
            size=value_batch_size,
            replace=n_value < value_batch_size,
        )
        value_index_tensor = torch.tensor(value_indices, dtype=torch.long, device=device)
        x_value = value_features[value_index_tensor]
        y_value = target_values[value_index_tensor]
        value_pred = value_net(x_value)
        value_loss = nn.functional.mse_loss(value_pred, y_value)
        loss = policy_loss + float(cfg.value_loss_weight) * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_policy_loss = float(policy_loss.detach().cpu())
        final_value_loss = float(value_loss.detach().cpu())
        final_total_loss = float(loss.detach().cpu())
    return {
        "policy_loss": final_policy_loss,
        "value_loss": final_value_loss,
        "total_loss": final_total_loss,
        "policy_sample_weight_mean": (
            float(policy_sample_weights.mean().detach().cpu()) if n_policy else 0.0
        ),
        "policy_sample_weight_max": (
            float(policy_sample_weights.max().detach().cpu()) if n_policy else 0.0
        ),
        "policy_training_samples": int(n_policy),
        "value_training_samples": int(n_value),
    }


def _policy_target_sample_weights(targets: np.ndarray, *, mode: str) -> np.ndarray:
    targets_arr = np.asarray(targets, dtype=np.float32)
    n = int(targets_arr.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    mode_str = str(mode)
    if mode_str == "uniform":
        return np.ones(n, dtype=np.float32)
    if mode_str != "inverse_target_top":
        raise ValueError("policy_sample_weighting must be one of: uniform, inverse_target_top")
    top_actions = np.argmax(targets_arr, axis=1)
    counts = Counter(int(action) for action in top_actions)
    weights = np.asarray([1.0 / float(counts[int(action)]) for action in top_actions], dtype=np.float32)
    mean = float(weights.mean())
    if mean <= 1e-8:
        return np.ones(n, dtype=np.float32)
    return (weights / mean).astype(np.float32)


def _target_stats(targets: np.ndarray, samples: list[SelfPlayPolicySample]) -> dict[str, object]:
    if targets.size == 0:
        return {
            "mean_target_entropy": 0.0,
            "mean_legal_actions": 0.0,
            "max_illegal_mass": 0.0,
            "target_top_action_counts": {},
            "target_top_action_counts_by_street": {},
            "target_count_by_street": {},
            "max_target_top_action_fraction_by_street": {},
            "max_target_top_action_fraction": 0.0,
        }
    legal_masks = np.stack([s.legal_mask for s in samples]).astype(np.float32)
    clipped = np.clip(targets, 1e-8, 1.0)
    entropy = -np.sum(targets * np.log(clipped), axis=1)
    illegal_mass = targets * (legal_masks <= 0)
    target_top_counts = Counter(int(action) for action in np.argmax(targets, axis=1))
    max_target_top_count = max(target_top_counts.values(), default=0)
    n_targets = int(targets.shape[0])
    top_counts_by_street: dict[str, Counter[int]] = defaultdict(Counter)
    target_count_by_street: Counter[str] = Counter()
    for sample, target in zip(samples, targets):
        street = _street_name(sample.state)
        target_count_by_street[street] += 1
        top_counts_by_street[street][int(np.argmax(target))] += 1
    max_fraction_by_street = {
        street: float(max(counts.values(), default=0) / max(target_count_by_street[street], 1))
        for street, counts in top_counts_by_street.items()
    }
    return {
        "mean_target_entropy": float(np.mean(entropy)),
        "mean_legal_actions": float(np.mean(legal_masks.sum(axis=1))),
        "max_illegal_mass": float(np.max(illegal_mass)),
        "target_top_action_counts": dict(sorted(target_top_counts.items())),
        "target_top_action_counts_by_street": {
            street: {str(action): int(count) for action, count in sorted(counts.items())}
            for street, counts in sorted(top_counts_by_street.items())
        },
        "target_count_by_street": dict(sorted(target_count_by_street.items())),
        "max_target_top_action_fraction_by_street": dict(sorted(max_fraction_by_street.items())),
        "max_target_top_action_fraction": (
            float(max_target_top_count / n_targets) if n_targets else 0.0
        ),
    }


def _select_improvement_samples(
    samples: list[SelfPlayPolicySample],
    *,
    max_targets: int,
    teacher_mode: str,
) -> list[SelfPlayPolicySample]:
    limit = max(0, min(int(max_targets), len(samples)))
    if limit == 0:
        return []
    if teacher_mode != "public_belief_cfr":
        if teacher_mode in {
            "sampled_state_value",
            "sampled_state_value_prior",
            "policy_public_state_rollout",
        }:
            eligible = [sample for sample in samples if sample.state is not None]
            by_street: dict[str, list[SelfPlayPolicySample]] = defaultdict(list)
            for sample in eligible:
                by_street[_street_name(sample.state)].append(sample)
            ordered_streets = [
                street
                for street in ("pre_flop", "flop", "turn", "river")
                if by_street.get(street)
            ]
            ordered_streets.extend(
                street
                for street in sorted(by_street)
                if street not in set(ordered_streets)
            )
            selected: list[SelfPlayPolicySample] = []
            cursor = 0
            while len(selected) < limit and ordered_streets:
                made_progress = False
                for street in ordered_streets:
                    street_samples = by_street[street]
                    if cursor < len(street_samples):
                        selected.append(street_samples[cursor])
                        made_progress = True
                        if len(selected) >= limit:
                            break
                if not made_progress:
                    break
                cursor += 1
            return selected
        if teacher_mode in {
            "public_world_rollout",
            "policy_public_world_rollout",
            "public_world_value",
            "public_world_value_prior",
        }:
            eligible = [
                sample
                for sample in samples
                if sample.state is not None
                and PublicWorldRolloutPolicyImprovementTeacher._is_preflop_root(sample.state)
            ]
            return list(eligible[:limit])
        return list(samples[:limit])
    eligible = [
        sample
        for sample in samples
        if _is_public_belief_cfr_eligible(sample)
    ]
    return list(eligible[:limit])


def _count_public_belief_cfr_eligible(samples: list[SelfPlayPolicySample]) -> int:
    return sum(1 for sample in samples if _is_public_belief_cfr_eligible(sample))


def _teacher_batching_summary(samples: list[SelfPlayPolicySample]) -> dict:
    eligible = [sample for sample in samples if _is_public_belief_cfr_eligible(sample)]
    street_counts: Counter[str] = Counter()
    legal_pattern_counts: Counter[tuple[int, ...]] = Counter()
    public_shape_counts: Counter[tuple] = Counter()
    for sample in eligible:
        state = sample.state
        street = str(getattr(state, "betting_stage", "unknown"))
        street_counts[street] += 1
        legal_pattern = tuple(int(v > 0) for v in np.asarray(sample.legal_mask).reshape(-1))
        legal_pattern_counts[legal_pattern] += 1
        board_len = len(getattr(state, "community_cards", []))
        pot = int(getattr(getattr(state, "_table"), "pot").total)
        acting_player = int(state.player_i)
        villain_i = 1 - acting_player
        hero_stack = int(state.players[acting_player].n_chips)
        villain_stack = int(state.players[villain_i].n_chips)
        public_shape_counts[
            (
                street,
                int(board_len),
                int(pot),
                int(hero_stack),
                int(villain_stack),
                legal_pattern,
            )
        ] += 1
    return {
        "eligible_teacher_states": int(len(eligible)),
        "eligible_street_counts": dict(street_counts),
        "legal_mask_pattern_groups": int(len(legal_pattern_counts)),
        "largest_legal_mask_pattern_group": int(max(legal_pattern_counts.values(), default=0)),
        "public_shape_groups": int(len(public_shape_counts)),
        "largest_public_shape_group": int(max(public_shape_counts.values(), default=0)),
    }


def _is_public_belief_cfr_eligible(sample: SelfPlayPolicySample) -> bool:
    return (
        sample.state is not None
        and PublicBeliefCFRPolicyImprovementTeacher._is_street_root(sample.state)
    )


def _is_public_world_rollout_eligible(sample: SelfPlayPolicySample) -> bool:
    return (
        sample.state is not None
        and PublicWorldRolloutPolicyImprovementTeacher._is_preflop_root(sample.state)
    )


def _street_name(state: object | None) -> str:
    if state is None:
        return "unknown"
    betting_stage = getattr(state, "betting_stage", None)
    if betting_stage is not None:
        return str(betting_stage)
    stage = getattr(state, "stage", None)
    if stage is None:
        return "unknown"
    stage_names = {
        int(getattr(state, "PREFLOP", 0)): "pre_flop",
        int(getattr(state, "FLOP", 1)): "flop",
        int(getattr(state, "TURN", 2)): "turn",
        int(getattr(state, "RIVER", 3)): "river",
        int(getattr(state, "SHOWDOWN", 4)): "show_down",
        int(getattr(state, "TERMINAL", 5)): "terminal",
    }
    return stage_names.get(int(stage), "unknown")


def run_neural_policy_iteration_pilot(cfg: NeuralPolicyIterationConfig) -> dict:
    if (
        str(cfg.teacher_mode) == "public_belief_cfr"
        and str(cfg.self_play_state_backend) != "full_deck"
    ):
        raise ValueError("public_belief_cfr requires self_play_state_backend='full_deck'")
    start = time.perf_counter()
    rng = _seed_all(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    initial_payload: dict | None = None
    if cfg.initial_checkpoint_path:
        policy_net, value_net, initial_payload = _load_policy_value_iteration_checkpoint(
            cfg.initial_checkpoint_path,
            device,
        )
        _validate_initial_checkpoint_matches_config(
            initial_payload,
            cfg,
            cfg.initial_checkpoint_path,
        )
    else:
        policy_net = _build_policy_net(cfg, device)
        value_net = _build_value_net(cfg, device)

    collection_start = time.perf_counter()
    samples, collection_stats = collect_stochastic_self_play_samples_with_stats(
        cfg,
        policy_net=policy_net,
    )
    self_play_collection_sec = time.perf_counter() - collection_start
    selected = _select_improvement_samples(
        samples,
        max_targets=cfg.max_improvement_targets,
        teacher_mode=cfg.teacher_mode,
    )
    fallback_teacher = LegalMixedPolicyImprovementTeacher(
        preferred_action=cfg.preferred_action,
        preferred_action_prob=cfg.preferred_action_prob,
    )
    if cfg.teacher_mode == "legal_mixed":
        teacher = fallback_teacher
        teacher_mode = "legal_mixed_contract_test_double"
        uses_resolver_teacher = False
        cfr_role = "policy_improvement_teacher_interface"
        teacher_stats: dict[str, int] = {}
    elif cfg.teacher_mode == "public_belief_cfr":
        teacher = PublicBeliefCFRPolicyImprovementTeacher(
            n_iterations=cfg.cfr_iterations,
            backend=cfg.cfr_backend,
            device=cfg.cfr_device,
            batch_roots=cfg.cfr_batch_roots,
            batch_min_roots=cfg.cfr_batch_min_roots,
            fallback_teacher=fallback_teacher,
        )
        teacher_mode = "public_belief_cfr"
        uses_resolver_teacher = True
        cfr_role = "public_belief_cfr_policy_improvement_teacher"
        teacher_stats = teacher.last_stats
    elif cfg.teacher_mode == "public_world_rollout":
        teacher = PublicWorldRolloutPolicyImprovementTeacher(
            n_worlds=cfg.rollout_worlds,
            temperature=cfg.rollout_temperature,
            continuation_policy=cfg.rollout_continuation_policy,
            seed=cfg.seed,
            fallback_teacher=fallback_teacher,
        )
        teacher_mode = "public_world_rollout"
        uses_resolver_teacher = False
        cfr_role = "local_public_world_policy_improvement_teacher"
        teacher_stats = teacher.last_stats
    elif cfg.teacher_mode == "policy_public_world_rollout":
        teacher = PolicyContinuationPublicWorldRolloutTeacher(
            policy_net=policy_net,
            device=device,
            n_worlds=cfg.rollout_worlds,
            temperature=cfg.rollout_temperature,
            seed=cfg.seed,
            fallback_teacher=fallback_teacher,
        )
        teacher_mode = "policy_public_world_rollout"
        uses_resolver_teacher = False
        cfr_role = "current_neural_policy_public_world_policy_improvement_teacher"
        teacher_stats = teacher.last_stats
    elif cfg.teacher_mode == "policy_public_state_rollout":
        teacher = PolicyContinuationPublicStateRolloutTeacher(
            policy_net=policy_net,
            device=device,
            n_worlds=cfg.rollout_worlds,
            temperature=cfg.rollout_temperature,
            seed=cfg.seed,
            fallback_teacher=fallback_teacher,
        )
        teacher_mode = "policy_public_state_rollout"
        uses_resolver_teacher = False
        cfr_role = "all_street_current_neural_policy_public_state_improvement_teacher"
        teacher_stats = teacher.last_stats
    elif cfg.teacher_mode in {"public_world_value", "public_world_value_prior"}:
        use_policy_prior = cfg.teacher_mode == "public_world_value_prior"
        teacher = PublicWorldValuePolicyImprovementTeacher(
            value_net=value_net,
            device=device,
            n_worlds=cfg.rollout_worlds,
            temperature=cfg.rollout_temperature,
            seed=cfg.seed,
            use_policy_prior=use_policy_prior,
            fallback_teacher=fallback_teacher,
        )
        teacher_mode = str(cfg.teacher_mode)
        uses_resolver_teacher = False
        cfr_role = (
            "prior_guided_learned_value_public_world_policy_improvement_teacher"
            if use_policy_prior
            else "learned_value_public_world_policy_improvement_teacher"
        )
        teacher_stats = teacher.last_stats
    elif cfg.teacher_mode in {"sampled_state_value", "sampled_state_value_prior"}:
        use_policy_prior = cfg.teacher_mode == "sampled_state_value_prior"
        teacher = SampledStateValuePolicyImprovementTeacher(
            value_net=value_net,
            device=device,
            temperature=cfg.rollout_temperature,
            use_policy_prior=use_policy_prior,
            fallback_teacher=fallback_teacher,
        )
        teacher_mode = str(cfg.teacher_mode)
        uses_resolver_teacher = False
        cfr_role = (
            "prior_guided_sampled_state_value_policy_improvement_teacher"
            if use_policy_prior
            else "sampled_state_value_policy_improvement_teacher"
        )
        teacher_stats = teacher.last_stats
    else:
        raise ValueError(
            "teacher_mode must be one of: legal_mixed, public_belief_cfr, "
            "public_world_rollout, policy_public_world_rollout, "
            "policy_public_state_rollout, public_world_value, public_world_value_prior, "
            "sampled_state_value, sampled_state_value_prior"
        )
    teacher_start = time.perf_counter()
    targets = teacher.improve(selected) if selected else np.zeros((0, N_ACTIONS), dtype=np.float32)
    teacher_target_sec = time.perf_counter() - teacher_start
    if isinstance(teacher, PublicBeliefCFRPolicyImprovementTeacher):
        teacher_stats = dict(teacher.last_stats)
    if isinstance(teacher, PublicWorldRolloutPolicyImprovementTeacher):
        teacher_stats = dict(teacher.last_stats)
    if isinstance(teacher, PublicWorldValuePolicyImprovementTeacher):
        teacher_stats = dict(teacher.last_stats)
    if isinstance(teacher, SampledStateValuePolicyImprovementTeacher):
        teacher_stats = dict(teacher.last_stats)
    training_start = time.perf_counter()
    losses = _train_on_policy_improvement_targets(
        policy_net,
        value_net,
        selected,
        targets,
        cfg,
        device,
        rng,
        value_samples=samples,
    )
    training_sec = time.perf_counter() - training_start
    stats = _target_stats(targets, selected)
    batching_summary = _teacher_batching_summary(selected)

    metrics = {
        "algorithm": "neural_self_play_policy_iteration",
        "role": "alphazero_style_policy_iteration_contract_smoke",
        "neural_policy_role": "main_stochastic_self_play_actor",
        "cfr_role": cfr_role,
        "teacher_mode": teacher_mode,
        "stochastic_policy_contract": "sample_mixed_strategy_not_argmax_by_default",
        "uses_slumbot_training_data": False,
        "uses_resolver_teacher": uses_resolver_teacher,
        "promotion": False,
        "initialized_from_checkpoint": cfg.initial_checkpoint_path is not None,
        "initial_checkpoint_path": cfg.initial_checkpoint_path,
        "parent_checkpoint_algorithm": str(initial_payload.get("algorithm", ""))
        if initial_payload is not None
        else "",
        "checkpoint_path": cfg.checkpoint_path,
        "self_play_states": int(len(samples)),
        "searchable_self_play_states": int(
            _count_public_belief_cfr_eligible(samples)
        ),
        "rollout_eligible_self_play_states": int(
            sum(1 for sample in samples if _is_public_world_rollout_eligible(sample))
        ),
        "self_play_hands": int(cfg.self_play_hands),
        "self_play_state_backend": str(cfg.self_play_state_backend),
        "max_self_play_hands": None
        if cfg.max_self_play_hands is None
        else int(cfg.max_self_play_hands),
        "min_searchable_self_play_states": int(cfg.min_searchable_self_play_states),
        "rollout_worlds": int(cfg.rollout_worlds),
        "rollout_temperature": float(cfg.rollout_temperature),
        "rollout_continuation_policy": str(cfg.rollout_continuation_policy),
        "cfr_iterations": int(cfg.cfr_iterations),
        "cfr_backend": str(cfg.cfr_backend),
        "cfr_device": cfg.cfr_device,
        "cfr_batch_roots": bool(cfg.cfr_batch_roots),
        "cfr_batch_min_roots": int(cfg.cfr_batch_min_roots),
        "improvement_targets": int(len(selected)),
        "dropped_ineligible_self_play_states": int(max(len(samples) - len(selected), 0))
        if cfg.teacher_mode
        in {
            "public_belief_cfr",
            "public_world_rollout",
            "public_world_value",
            "public_world_value_prior",
            "sampled_state_value",
            "sampled_state_value_prior",
        }
        else 0,
        "min_resolver_targets": int(cfg.min_resolver_targets),
        "min_target_streets": int(cfg.min_target_streets),
        "num_actions": N_ACTIONS,
        "feature_dim": int(cfg.feature_dim),
        "parallel_self_play_hands": int(cfg.parallel_self_play_hands),
        "train_steps": int(cfg.train_steps),
        "policy_sample_weighting": str(cfg.policy_sample_weighting),
        "self_play_collection_sec": float(self_play_collection_sec),
        **collection_stats,
        "teacher_target_sec": float(teacher_target_sec),
        "training_sec": float(training_sec),
        "elapsed_sec": float(time.perf_counter() - start),
        **device_info,
        **losses,
        **stats,
        **batching_summary,
        **teacher_stats,
    }
    gate_failures: list[str] = []
    resolver_targets = int(metrics.get("resolver_targets", 0))
    if cfg.teacher_mode == "public_belief_cfr" and resolver_targets < int(cfg.min_resolver_targets):
        gate_failures.append(
            f"resolver_targets {resolver_targets} < min_resolver_targets {int(cfg.min_resolver_targets)}"
        )
    if int(cfg.min_target_streets) > 0:
        n_target_streets = len(metrics.get("target_count_by_street", {}))
        if n_target_streets < int(cfg.min_target_streets):
            gate_failures.append(
                f"target_streets {n_target_streets} < min_target_streets {int(cfg.min_target_streets)}"
            )
    max_top_fraction = float(metrics.get("max_target_top_action_fraction", 0.0))
    if max_top_fraction > float(cfg.max_target_top_action_fraction):
        gate_failures.append(
            "max_target_top_action_fraction "
            f"{max_top_fraction:.6f} > {float(cfg.max_target_top_action_fraction):.6f}"
        )
    metrics["gate_failures"] = gate_failures
    metrics["passed"] = len(gate_failures) == 0
    if cfg.checkpoint_path:
        path = Path(cfg.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "role": metrics["role"],
                "config": asdict(cfg),
                "metrics": metrics,
                "parent_checkpoint_path": cfg.initial_checkpoint_path,
                "policy_net_state_dict": policy_net.state_dict(),
                "value_net_state_dict": value_net.state_dict(),
            },
            path,
        )
    return metrics


def _load_policy_iteration_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[_PolicyNet, dict]:
    policy_net, _value_net, payload = _load_policy_value_iteration_checkpoint(
        checkpoint_path,
        device,
    )
    return policy_net, payload


def _public_world_continuation_policy(name: str):
    from poker_ai.research.public_action_rollout_value import (  # noqa: PLC0415
        call_policy,
        masked_uniform_policy,
    )

    if str(name) == "call":
        return call_policy
    if str(name) == "uniform":
        return masked_uniform_policy
    raise ValueError("continuation_policy must be one of: call, uniform")


def evaluate_neural_policy_iteration_public_world_gate(
    cfg: NeuralPolicyIterationPublicWorldGateConfig,
) -> dict:
    """Held-out public-world root diagnostic for policy-iteration checkpoints."""
    started = time.perf_counter()
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    policy_net, payload = _load_policy_iteration_checkpoint(cfg.checkpoint_path, device)
    from poker_ai.research.public_action_rollout_value import (  # noqa: PLC0415
        build_public_world_state,
        sample_public_worlds,
        sample_seeded_hero_cards,
        score_first_actions_across_worlds,
    )

    continuation = _public_world_continuation_policy(cfg.continuation_policy)
    rows: list[dict] = []
    policy_evs: list[float] = []
    oracle_evs: list[float] = []
    selected_evs: list[float] = []
    policy_minus_selected_evs: list[float] = []
    oracle_gaps: list[float] = []
    selected_actions: list[int] = []
    oracle_actions: list[int] = []
    for root_idx in range(int(cfg.n_roots)):
        hero_cards = sample_seeded_hero_cards(seed=int(cfg.seed), root_idx=root_idx)
        worlds = sample_public_worlds(
            hero_cards=hero_cards,
            n_worlds=int(cfg.n_worlds),
            seed=int(cfg.seed) + 1009 * (root_idx + 1),
        )
        root = build_public_world_state(
            hero_cards=worlds[0].hero_cards,
            opponent_cards=worlds[0].opponent_cards,
            deck_tail=worlds[0].deck_tail,
            initial_chips=int(cfg.initial_chips),
            small_blind=int(cfg.small_blind),
            big_blind=int(cfg.big_blind),
        )
        features = root.to_feature_vector().reshape(1, -1).astype(np.float32)
        legal_mask = root.get_legal_mask().reshape(1, -1).astype(np.float32)
        policy = _batched_network_policies(policy_net, features, legal_mask, device)[0]
        result = score_first_actions_across_worlds(
            worlds=worlds,
            continuation_policy=continuation,
            seed=int(cfg.seed) + 2003 * (root_idx + 1),
            max_steps_per_hand=int(cfg.max_steps_per_hand),
            initial_chips=int(cfg.initial_chips),
            small_blind=int(cfg.small_blind),
            big_blind=int(cfg.big_blind),
        )
        policy_ev = float(
            sum(float(policy[action]) * float(result.action_values[action])
                for action in result.legal_actions)
        )
        selected_action = int(max(result.legal_actions, key=lambda action: policy[action]))
        selected_ev = float(result.action_values[selected_action])
        oracle_ev = float(result.best_action_value)
        gap = float(oracle_ev - policy_ev)
        policy_minus_selected = float(policy_ev - selected_ev)
        policy_evs.append(policy_ev)
        selected_evs.append(selected_ev)
        policy_minus_selected_evs.append(policy_minus_selected)
        oracle_evs.append(oracle_ev)
        oracle_gaps.append(gap)
        selected_actions.append(selected_action)
        oracle_actions.append(int(result.best_action))
        rows.append(
            {
                "root_idx": int(root_idx),
                "hero_cards": [int(card) for card in hero_cards],
                "policy_ev": policy_ev,
                "selected_action": selected_action,
                "selected_action_ev": selected_ev,
                "policy_minus_selected_ev": policy_minus_selected,
                "oracle_action": int(result.best_action),
                "oracle_ev": oracle_ev,
                "oracle_gap": gap,
                "policy": {
                    str(int(action)): float(policy[action])
                    for action in result.legal_actions
                },
                "action_values": {
                    str(int(action)): float(value)
                    for action, value in sorted(result.action_values.items())
                },
            }
        )

    selected_counts = Counter(selected_actions)
    oracle_counts = Counter(oracle_actions)
    policy_arr = np.asarray(policy_evs, dtype=np.float64)
    oracle_arr = np.asarray(oracle_evs, dtype=np.float64)
    selected_arr = np.asarray(selected_evs, dtype=np.float64)
    policy_minus_selected_arr = np.asarray(policy_minus_selected_evs, dtype=np.float64)
    gap_arr = np.asarray(oracle_gaps, dtype=np.float64)
    return {
        "algorithm": "neural_policy_iteration_public_world_gate",
        "checkpoint": str(cfg.checkpoint_path),
        "checkpoint_algorithm": str(payload.get("algorithm", "unknown")),
        "role": "heldout_public_world_root_decision_diagnostic",
        "uses_slumbot_training_data": False,
        "promotion": False,
        "n_roots": int(cfg.n_roots),
        "n_worlds": int(cfg.n_worlds),
        "continuation_policy": str(cfg.continuation_policy),
        "mean_policy_ev": float(policy_arr.mean()) if policy_arr.size else 0.0,
        "mean_selected_ev": float(selected_arr.mean()) if selected_arr.size else 0.0,
        "mean_policy_minus_selected_ev": (
            float(policy_minus_selected_arr.mean())
            if policy_minus_selected_arr.size
            else 0.0
        ),
        "mean_oracle_ev": float(oracle_arr.mean()) if oracle_arr.size else 0.0,
        "mean_oracle_gap": float(gap_arr.mean()) if gap_arr.size else 0.0,
        "max_oracle_gap": float(gap_arr.max()) if gap_arr.size else 0.0,
        "selected_action_counts": {
            str(int(action)): int(count) for action, count in sorted(selected_counts.items())
        },
        "oracle_action_counts": {
            str(int(action)): int(count) for action, count in sorted(oracle_counts.items())
        },
        "rows": rows,
        "elapsed_sec": float(time.perf_counter() - started),
        **device_info,
    }


def evaluate_neural_policy_iteration_head_to_head(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260526,
    min_lower95_candidate_payoff: float | None = None,
) -> dict:
    """Evaluate two neural policy-iteration checkpoints with duplicate-swapped seats."""
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_policy, candidate_payload = _load_policy_iteration_checkpoint(
        candidate_checkpoint,
        resolved_device,
    )
    baseline_policy, baseline_payload = _load_policy_iteration_checkpoint(
        baseline_checkpoint,
        resolved_device,
    )
    candidate_config = dict(candidate_payload.get("config", {}))
    initial_chips = int(candidate_config.get("initial_chips", 1000))
    max_steps_per_hand = int(candidate_config.get("max_steps_per_hand", 256))
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
                int(candidate_seat): candidate_policy,
                int(1 - candidate_seat): baseline_policy,
            }
            state = new_game(2, initial_chips=initial_chips)
            steps = 0
            while not state.is_terminal and steps < max_steps_per_hand:
                policy_net = policies[int(state.player_i)]
                features = state.to_feature_vector().astype(np.float32, copy=False)
                legal_mask = get_legal_mask(state)
                probs = _network_policy(policy_net, features, legal_mask, resolved_device)
                action_idx = select_action(probs, legal_mask, rng=rng)
                state = state.apply_action(INDEX_TO_ACTION[int(action_idx)])
                steps += 1
            total_steps += steps
            payoff = float(state.payout.get(int(candidate_seat), 0)) / float(initial_chips)
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
        passed = lower95 >= float(min_lower95_candidate_payoff)
    return {
        "algorithm": "neural_policy_iteration_h2h",
        "role": "duplicate_swapped_neural_policy_iteration_league_smoke",
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_algorithm": str(candidate_payload.get("algorithm", "")),
        "baseline_algorithm": str(baseline_payload.get("algorithm", "")),
        "uses_slumbot_training_data": False,
        "uses_resolver": False,
        "promotion": False,
        "passed": bool(passed),
        "n_games": int(n_games),
        "n_pairs": int(len(pair_payoffs)),
        "eval_seconds": float(elapsed),
        "eval_steps": int(total_steps),
        "eval_games_per_second": float(int(n_games) / max(elapsed, 1e-9)),
        "eval_steps_per_second": float(total_steps / max(elapsed, 1e-9)),
        "mean_candidate_payoff": mean,
        "std_candidate_payoff": std,
        "lower95_candidate_payoff": lower95,
        "upper95_candidate_payoff": upper95,
        "candidate_payoffs": [float(v) for v in candidate_payoffs],
        "candidate_pair_payoffs": [float(v) for v in pair_payoffs],
        **device_info,
    }


def _policy_distance_to_target(
    policy: np.ndarray,
    target: np.ndarray,
    legal_mask: np.ndarray,
) -> dict[str, float | int]:
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(N_ACTIONS)
    policy_arr = np.asarray(policy, dtype=np.float64).reshape(N_ACTIONS)
    target_arr = np.asarray(target, dtype=np.float64).reshape(N_ACTIONS)
    legal = np.flatnonzero(mask > 0)
    illegal_mass = float(np.sum(policy_arr[mask <= 0]))
    l1 = float(np.sum(np.abs(policy_arr[legal] - target_arr[legal])))
    eps = 1e-12
    kl = float(
        np.sum(
            target_arr[legal]
            * (np.log(np.maximum(target_arr[legal], eps))
               - np.log(np.maximum(policy_arr[legal], eps)))
        )
    )
    policy_top = int(legal[int(np.argmax(policy_arr[legal]))])
    target_top = int(legal[int(np.argmax(target_arr[legal]))])
    legal_policy = np.maximum(policy_arr[legal], eps)
    legal_target = np.maximum(target_arr[legal], eps)
    policy_entropy = float(-np.sum(policy_arr[legal] * np.log(legal_policy)))
    target_entropy = float(-np.sum(target_arr[legal] * np.log(legal_target)))
    return {
        "l1_to_cfr": l1,
        "kl_target_to_policy": kl,
        "illegal_mass": illegal_mass,
        "policy_top_action": policy_top,
        "cfr_top_action": target_top,
        "policy_mass_on_cfr_top_action": float(policy_arr[target_top]),
        "policy_entropy": policy_entropy,
        "cfr_target_entropy": target_entropy,
        "top_action_agreement": int(policy_top == target_top),
    }


def evaluate_neural_policy_iteration_cfr_gate(
    cfg: NeuralPolicyIterationCFRGateConfig,
) -> dict:
    """Root-disjoint public-belief CFR gate for policy-iteration checkpoints."""
    started = time.perf_counter()
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    policy_net, payload = _load_policy_iteration_checkpoint(cfg.checkpoint_path, device)
    state_source_path = cfg.state_source_checkpoint_path or cfg.checkpoint_path
    state_source_policy_net, _state_source_payload = _load_policy_iteration_checkpoint(
        state_source_path,
        device,
    )
    checkpoint_config = dict(payload.get("config", {}))
    hidden_dim = int(checkpoint_config.get("hidden_dim", 64))

    collection_cfg = NeuralPolicyIterationConfig(
        self_play_hands=1,
        max_self_play_hands=int(cfg.max_self_play_hands),
        min_searchable_self_play_states=int(cfg.n_roots),
        max_improvement_targets=int(cfg.n_roots),
        train_steps=0,
        hidden_dim=hidden_dim,
        parallel_self_play_hands=8,
        self_play_state_backend="full_deck",
        teacher_mode="public_belief_cfr",
        cfr_iterations=int(cfg.cfr_iterations),
        cfr_backend=str(cfg.cfr_backend),
        cfr_device=cfg.cfr_device,
        initial_chips=int(cfg.initial_chips),
        max_steps_per_hand=int(cfg.max_steps_per_hand),
        seed=int(cfg.seed),
        device=cfg.device,
    )
    samples, collection_stats = collect_stochastic_self_play_samples_with_stats(
        collection_cfg,
        policy_net=state_source_policy_net,
    )
    selected = _select_improvement_samples(
        samples,
        max_targets=int(cfg.n_roots),
        teacher_mode="public_belief_cfr",
    )
    teacher = PublicBeliefCFRPolicyImprovementTeacher(
        n_iterations=int(cfg.cfr_iterations),
        backend=str(cfg.cfr_backend),
        device=cfg.cfr_device,
    )
    rows: list[dict] = []
    l1_values: list[float] = []
    kl_values: list[float] = []
    illegal_masses: list[float] = []
    agreements: list[int] = []
    cfr_top_policy_masses: list[float] = []
    policy_entropies: list[float] = []
    cfr_target_entropies: list[float] = []
    skipped = 0
    for root_idx, sample in enumerate(selected):
        before_targets = int(teacher.last_stats.get("resolver_targets", 0))
        before_fallback = int(teacher.last_stats.get("fallback_targets", 0))
        target = teacher.improve_one(sample)
        after_targets = int(teacher.last_stats.get("resolver_targets", 0))
        after_fallback = int(teacher.last_stats.get("fallback_targets", 0))
        if after_targets <= before_targets or after_fallback > before_fallback:
            skipped += 1
            continue
        policy = _batched_network_policies(
            policy_net,
            np.asarray(sample.features, dtype=np.float32).reshape(1, -1),
            np.asarray(sample.legal_mask, dtype=np.float32).reshape(1, -1),
            device,
        )[0]
        distance = _policy_distance_to_target(policy, target, sample.legal_mask)
        l1_values.append(float(distance["l1_to_cfr"]))
        kl_values.append(float(distance["kl_target_to_policy"]))
        illegal_masses.append(float(distance["illegal_mass"]))
        agreements.append(int(distance["top_action_agreement"]))
        cfr_top_policy_masses.append(float(distance["policy_mass_on_cfr_top_action"]))
        policy_entropies.append(float(distance["policy_entropy"]))
        cfr_target_entropies.append(float(distance["cfr_target_entropy"]))
        state = sample.state
        rows.append(
            {
                "root_idx": int(root_idx),
                "street": str(getattr(state, "betting_stage", "unknown")),
                "l1_to_cfr": float(distance["l1_to_cfr"]),
                "kl_target_to_policy": float(distance["kl_target_to_policy"]),
                "illegal_mass": float(distance["illegal_mass"]),
                "policy_top_action": int(distance["policy_top_action"]),
                "cfr_top_action": int(distance["cfr_top_action"]),
                "policy_mass_on_cfr_top_action": float(
                    distance["policy_mass_on_cfr_top_action"]
                ),
                "policy_entropy": float(distance["policy_entropy"]),
                "cfr_target_entropy": float(distance["cfr_target_entropy"]),
                "top_action_agreement": bool(distance["top_action_agreement"]),
                "policy": {
                    str(action): float(policy[action])
                    for action in np.flatnonzero(np.asarray(sample.legal_mask) > 0)
                },
                "cfr_target": {
                    str(action): float(target[action])
                    for action in np.flatnonzero(np.asarray(sample.legal_mask) > 0)
                },
            }
        )

    l1_arr = np.asarray(l1_values, dtype=np.float64)
    kl_arr = np.asarray(kl_values, dtype=np.float64)
    illegal_arr = np.asarray(illegal_masses, dtype=np.float64)
    agreement_arr = np.asarray(agreements, dtype=np.float64)
    cfr_top_policy_mass_arr = np.asarray(cfr_top_policy_masses, dtype=np.float64)
    policy_entropy_arr = np.asarray(policy_entropies, dtype=np.float64)
    cfr_target_entropy_arr = np.asarray(cfr_target_entropies, dtype=np.float64)
    policy_top_counts = Counter(int(row["policy_top_action"]) for row in rows)
    cfr_top_counts = Counter(int(row["cfr_top_action"]) for row in rows)
    evaluated_roots = int(len(rows))
    max_policy_top_count = max(policy_top_counts.values(), default=0)
    return {
        "algorithm": "neural_policy_iteration_exact_cfr_gate",
        "checkpoint": str(cfg.checkpoint_path),
        "state_source_checkpoint": str(state_source_path),
        "checkpoint_algorithm": str(payload.get("algorithm", "unknown")),
        "role": "root_disjoint_public_belief_cfr_policy_gate",
        "uses_slumbot_training_data": False,
        "promotion": False,
        "n_roots": int(cfg.n_roots),
        "candidate_self_play_states": int(len(samples)),
        "searchable_self_play_states": int(_count_public_belief_cfr_eligible(samples)),
        "selected_roots": int(len(selected)),
        "evaluated_roots": evaluated_roots,
        "resolver_targets": int(teacher.last_stats.get("resolver_targets", 0)),
        "fallback_targets": int(teacher.last_stats.get("fallback_targets", 0)),
        "skipped_targets": int(skipped),
        "cfr_iterations": int(cfg.cfr_iterations),
        "cfr_backend": str(cfg.cfr_backend),
        "mean_l1_to_cfr": float(l1_arr.mean()) if l1_arr.size else 0.0,
        "mean_kl_target_to_policy": float(kl_arr.mean()) if kl_arr.size else 0.0,
        "top_action_agreement": float(agreement_arr.mean()) if agreement_arr.size else 0.0,
        "mean_policy_mass_on_cfr_top_action": (
            float(cfr_top_policy_mass_arr.mean())
            if cfr_top_policy_mass_arr.size
            else 0.0
        ),
        "mean_policy_entropy": (
            float(policy_entropy_arr.mean()) if policy_entropy_arr.size else 0.0
        ),
        "mean_cfr_target_entropy": (
            float(cfr_target_entropy_arr.mean())
            if cfr_target_entropy_arr.size
            else 0.0
        ),
        "max_illegal_mass": float(illegal_arr.max()) if illegal_arr.size else 0.0,
        "policy_top_action_counts": dict(sorted(policy_top_counts.items())),
        "cfr_top_action_counts": dict(sorted(cfr_top_counts.items())),
        "max_policy_top_action_fraction": (
            float(max_policy_top_count / evaluated_roots) if evaluated_roots else 0.0
        ),
        "rows": rows,
        "elapsed_sec": float(time.perf_counter() - started),
        **collection_stats,
        **device_info,
    }
