"""Isolated AlphaNLHoldem/RLCard compatibility benchmark helpers.

The code here is a clean benchmark bridge. It does not import the ignored
AlphaNLHoldem checkout and it does not use Slumbot data. It only implements the
observable RLCard-facing contract needed to evaluate a public reference
checkpoint under an isolated 50bb/5-action benchmark surface.
"""

from __future__ import annotations

import contextlib
import io
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F


_SUIT_TO_INDEX = dict(zip("CDHS", range(4), strict=True))
_RANK_TO_INDEX = dict(zip("23456789TJQKA", range(13), strict=True))


def _enum_value(value: Any) -> int:
    return int(getattr(value, "value", value))


def _card_indices(card: str) -> tuple[int, int]:
    if len(card) < 2:
        raise ValueError(f"invalid card string: {card!r}")
    return _SUIT_TO_INDEX[card[0]], _RANK_TO_INDEX[card[1]]


def encode_reference_observation(
    raw_obs: dict[str, Any],
    history: list[list[tuple[int, int, list[int]]]],
    *,
    current_player: int | None = None,
    action_num: int = 5,
) -> dict[str, np.ndarray]:
    """Encode an RLCard raw observation into the AlphaNLHoldem tensor contract."""

    card_info = np.zeros((4, 13, 6), dtype=np.float32)
    action_info = np.zeros((4, action_num, 4 * 6 + 1), dtype=np.float32)
    extra_info = np.zeros((2,), dtype=np.float32)
    legal_moves = np.zeros((action_num,), dtype=np.float32)

    hand = list(raw_obs.get("hand", []))
    public_cards = list(raw_obs.get("public_cards", []))
    for action in raw_obs.get("legal_actions", []):
        action_idx = _enum_value(action)
        if 0 <= action_idx < action_num:
            legal_moves[action_idx] = 1.0

    street_cards = [
        (hand, 0),
        (public_cards[:3], 1),
        (public_cards[3:4], 2),
        (public_cards[4:5], 3),
        (public_cards, 4),
        (public_cards + hand, 5),
    ]
    for cards, channel in street_cards:
        for card in cards:
            suit_idx, rank_idx = _card_indices(card)
            card_info[suit_idx, rank_idx, channel] = 1.0

    for street_idx, street_history in enumerate(history[:4]):
        for action_pos, (player_id, action_id, legal_actions) in enumerate(street_history[:6]):
            channel = street_idx * 6 + action_pos
            if 0 <= player_id < 2 and 0 <= action_id < action_num:
                action_info[player_id, action_id, channel] = 1.0
                action_info[2, action_id, channel] = 1.0
            for legal_action in legal_actions:
                if 0 <= legal_action < action_num:
                    action_info[3, legal_action, channel] = 1.0

    if current_player is None:
        current_player = int(raw_obs.get("current_player", 0))
    action_info[:, :, -1] = float(current_player)

    stakes = raw_obs.get("stakes", [0, 0])
    if len(stakes) >= 2:
        extra_info[0] = float(stakes[0])
        extra_info[1] = float(stakes[1])

    return {
        "card_info": card_info,
        "action_info": action_info,
        "extra_info": extra_info,
        "legal_moves": legal_moves,
    }


def _current_rlcard_state(observation: Any) -> dict[str, Any]:
    """Return RLCard's current state dict from reset/step output."""

    if isinstance(observation, (tuple, list)) and observation:
        state = observation[0]
    else:
        state = observation
    if not isinstance(state, dict):
        raise TypeError(f"expected RLCard state dict, got {type(state).__name__}")
    return state


@dataclass(frozen=True)
class PolicyDecision:
    action: int
    legal_actions: list[int]
    logits: np.ndarray
    masked_logits: np.ndarray
    policy_name: str


class AlphaNLHoldemPolicy(Protocol):
    policy_name: str

    def act(self, observation: dict[str, np.ndarray]) -> PolicyDecision:
        ...


class RLCardAgentPolicy:
    """Wrap an RLCard-trained agent for the RLCard AlphaNLHoldem H2H gate."""

    def __init__(self, agent: Any, *, policy_name: str = "rlcard_agent") -> None:
        self.agent = agent
        self.policy_name = policy_name

    @classmethod
    def from_nfsp_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "RLCardAgentPolicy":
        from rlcard.agents import NFSPAgent

        load_kwargs = {"map_location": torch.device(device)}
        try:
            checkpoint = torch.load(path, weights_only=False, **load_kwargs)
        except TypeError:
            checkpoint = torch.load(path, **load_kwargs)
        checkpoint = dict(checkpoint)
        checkpoint["device"] = device
        if isinstance(checkpoint.get("rl_agent"), dict):
            checkpoint["rl_agent"] = dict(checkpoint["rl_agent"])
            checkpoint["rl_agent"]["device"] = device
        with contextlib.redirect_stdout(io.StringIO()):
            agent = NFSPAgent.from_checkpoint(checkpoint)
        agent.evaluate_with = "average_policy"
        agent.set_device(device)
        return cls(agent, policy_name="rlcard_nfsp_average_policy")

    def act_rlcard_state(self, state: dict[str, Any]) -> PolicyDecision:
        action, info = self.agent.eval_step(state)
        action = int(action)
        legal_actions = sorted(int(idx) for idx in state["legal_actions"].keys())
        if not legal_actions:
            raise ValueError("RLCard state has no legal actions")

        probs = np.zeros((5,), dtype=np.float32)
        for raw_action, prob in info.get("probs", {}).items():
            action_idx = _enum_value(raw_action)
            if 0 <= action_idx < probs.shape[0]:
                probs[action_idx] = float(prob)
        legal_mask = np.zeros((5,), dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        probs *= legal_mask
        total = float(probs.sum())
        if total <= 0.0:
            probs[action] = 1.0
            probs *= legal_mask
            total = float(probs.sum())
        if total <= 0.0:
            probs = legal_mask / float(legal_mask.sum())
        else:
            probs /= total
        logits = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)
        masked = np.full_like(logits, -np.inf, dtype=np.float32)
        masked[legal_actions] = logits[legal_actions]
        if action not in legal_actions:
            action = int(np.argmax(masked))
        return PolicyDecision(
            action=action,
            legal_actions=legal_actions,
            logits=logits,
            masked_logits=masked,
            policy_name=self.policy_name,
        )


def _tf_same_pad_2d(x: torch.Tensor, *, kernel_size: int, stride: int) -> torch.Tensor:
    height = int(x.shape[-2])
    width = int(x.shape[-1])
    out_height = math.ceil(height / stride)
    out_width = math.ceil(width / stride)
    pad_height = max((out_height - 1) * stride + kernel_size - height, 0)
    pad_width = max((out_width - 1) * stride + kernel_size - width, 0)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    if pad_height == 0 and pad_width == 0:
        return x
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class AlphaNLHoldemTorchPolicy:
    """Torch inference wrapper for AlphaNLHoldem-format TensorFlow weights."""

    policy_name = "alphanlholdem_torch_checkpoint"

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.tensors = {name: tensor.to(self.device) for name, tensor in tensors.items()}

    @classmethod
    def from_pickle(cls, path: str | Path, *, device: str = "cpu") -> "AlphaNLHoldemTorchPolicy":
        with Path(path).open("rb") as handle:
            weights = pickle.load(handle)
        return cls.from_weights(weights, device=device)

    @classmethod
    def from_weights(
        cls,
        weights: dict[str, np.ndarray],
        *,
        device: str = "cpu",
    ) -> "AlphaNLHoldemTorchPolicy":
        tensors: dict[str, torch.Tensor] = {}
        for raw_name, value in weights.items():
            name = raw_name.removeprefix("oppo_policy/")
            array = np.asarray(value, dtype=np.float32)
            if name.endswith("/kernel"):
                if array.ndim == 4:
                    array = np.transpose(array, (3, 2, 0, 1))
                elif array.ndim == 2:
                    array = np.transpose(array, (1, 0))
            tensors[name] = torch.as_tensor(array, dtype=torch.float32)
        return cls(tensors, device=device)

    def _tensor(self, name: str) -> torch.Tensor:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"missing AlphaNLHoldem weight tensor: {name}") from exc

    def _conv(self, x: torch.Tensor, base: str, *, stride: int) -> torch.Tensor:
        weight = self._tensor(f"{base}/kernel")
        bias = self._tensor(f"{base}/bias")
        kernel_size = int(weight.shape[-1])
        x = _tf_same_pad_2d(x, kernel_size=kernel_size, stride=stride)
        return F.conv2d(x, weight, bias, stride=stride)

    def _linear(self, x: torch.Tensor, base: str) -> torch.Tensor:
        return F.linear(x, self._tensor(f"{base}/kernel"), self._tensor(f"{base}/bias"))

    def _branch(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        for stage, out_channels, stride, blocks in (
            (0, 16, 1, 1),
            (1, 32, 2, 2),
            (2, 64, 2, 2),
        ):
            del out_channels
            for block in range(blocks):
                base = f"{prefix}_conv{stage}b{block}"
                y = F.relu(self._conv(x, base, stride=stride))
                y = self._conv(y, f"{base}_2", stride=1)
                if stride > 1 and stage > 1:
                    x = self._conv(x, f"{base}_s", stride=stride)
                x = x + y if stage > 1 else y
                x = F.relu(x)
        return torch.flatten(x, start_dim=1)

    def logits_and_value(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        with torch.no_grad():
            card = torch.as_tensor(
                observation["card_info"], dtype=torch.float32, device=self.device
            ).permute(2, 0, 1).unsqueeze(0)
            action = torch.as_tensor(
                observation["action_info"], dtype=torch.float32, device=self.device
            ).permute(2, 0, 1).unsqueeze(0)
            card_feat = self._branch(card, "card")
            history_feat = self._branch(action, "history")
            # AlphaNLHoldem's published network applies extra_fc to the card
            # embedding, not to the two scalar extra_info inputs.
            extra_feat = F.relu(self._linear(card_feat, "extra_fc"))
            fused = torch.cat([card_feat, history_feat, extra_feat], dim=-1)
            x = F.relu(self._linear(fused, "fc_1"))
            x = F.relu(self._linear(x, "fc_2"))
            x = F.relu(self._linear(x, "fc_3"))
            logits = self._linear(x, "conv_fuse").squeeze(0).cpu().numpy()
            value = float(self._linear(x, "value_out").squeeze().cpu().item())
        return logits.astype(np.float32), value

    def act(self, observation: dict[str, np.ndarray]) -> PolicyDecision:
        logits, _value = self.logits_and_value(observation)
        legal_mask = np.asarray(observation["legal_moves"], dtype=np.float32)
        legal_actions = [int(idx) for idx in np.flatnonzero(legal_mask > 0)]
        if not legal_actions:
            raise ValueError("AlphaNLHoldem observation has no legal actions")
        masked = np.full_like(logits, -np.inf, dtype=np.float32)
        masked[legal_actions] = logits[legal_actions]
        action = int(np.argmax(masked))
        return PolicyDecision(
            action=action,
            legal_actions=legal_actions,
            logits=logits,
            masked_logits=masked,
            policy_name=self.policy_name,
        )


class RandomLegalAlphaNLHoldemPolicy:
    policy_name = "random_legal"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: dict[str, np.ndarray]) -> PolicyDecision:
        legal_actions = [int(idx) for idx in np.flatnonzero(observation["legal_moves"] > 0)]
        if not legal_actions:
            raise ValueError("AlphaNLHoldem observation has no legal actions")
        action = int(self.rng.choice(legal_actions))
        logits = np.zeros_like(observation["legal_moves"], dtype=np.float32)
        masked = np.full_like(logits, -np.inf, dtype=np.float32)
        masked[legal_actions] = 0.0
        return PolicyDecision(
            action=action,
            legal_actions=legal_actions,
            logits=logits,
            masked_logits=masked,
            policy_name=self.policy_name,
        )


def summarize_h2h_payoffs(
    payoffs: list[float],
    *,
    games_per_seat: int,
    min_lower95_candidate_payoff: float | None = None,
) -> dict[str, Any]:
    values = np.asarray(payoffs, dtype=np.float64)
    games = int(values.size)
    if games == 0:
        raise ValueError("cannot summarize zero games")
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / math.sqrt(games)) if games > 1 else 0.0
    margin = 1.96 * stderr
    lower95 = float(mean - margin)
    upper95 = float(mean + margin)
    passed = (
        None
        if min_lower95_candidate_payoff is None
        else bool(lower95 >= float(min_lower95_candidate_payoff))
    )
    metrics = {
        "games": games,
        "games_per_seat": int(games_per_seat),
        "mean_candidate_payoff": mean,
        "stderr_candidate_payoff": stderr,
        "lower95_candidate_payoff": lower95,
        "upper95_candidate_payoff": upper95,
        "promotion": False,
    }
    if min_lower95_candidate_payoff is not None:
        metrics["min_lower95_candidate_payoff"] = float(min_lower95_candidate_payoff)
        metrics["passed"] = passed
    return metrics


def _log_history(history: list[list[tuple[int, int, list[int]]]], raw_obs: dict[str, Any], action: int) -> None:
    stage = _enum_value(raw_obs.get("stage", 0))
    if not 0 <= stage < 4:
        return
    player = int(raw_obs.get("current_player", 0))
    legal = [_enum_value(legal_action) for legal_action in raw_obs.get("legal_actions", [])]
    history[stage].append((player, int(action), legal))


def _play_single_rlcard_game(
    *,
    candidate: AlphaNLHoldemPolicy,
    baseline: AlphaNLHoldemPolicy,
    candidate_seat: int,
    seed: int,
    max_steps: int = 256,
) -> float:
    import rlcard
    from rlcard.utils import set_seed

    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    env = rlcard.make("no-limit-holdem", config={"seed": seed})
    obs = env.reset()
    history: list[list[tuple[int, int, list[int]]]] = [[], [], [], []]
    steps = 0
    while not env.game.is_over():
        current_state = _current_rlcard_state(obs)
        raw_obs = current_state["raw_obs"]
        current_player = int(raw_obs["current_player"])
        policy = candidate if current_player == candidate_seat else baseline
        if hasattr(policy, "act_rlcard_state"):
            decision = policy.act_rlcard_state(current_state)  # type: ignore[attr-defined]
        else:
            encoded = encode_reference_observation(
                raw_obs,
                history,
                current_player=current_player,
                action_num=5,
            )
            decision = policy.act(encoded)
        action = decision.action
        _log_history(history, raw_obs, action)
        obs = env.step(action)
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"RLCard game exceeded max_steps={max_steps}")
    payoffs = env.get_payoffs()
    return float(payoffs[candidate_seat])


def evaluate_rlcard_reference_h2h(
    *,
    candidate: AlphaNLHoldemPolicy,
    baseline: AlphaNLHoldemPolicy,
    games_per_seat: int = 10,
    seed: int = 20260528,
    max_steps: int = 256,
    min_lower95_candidate_payoff: float | None = None,
) -> dict[str, Any]:
    payoffs: list[float] = []
    for index in range(int(games_per_seat)):
        game_seed = int(seed) + index
        payoffs.append(
            _play_single_rlcard_game(
                candidate=candidate,
                baseline=baseline,
                candidate_seat=0,
                seed=game_seed,
                max_steps=max_steps,
            )
        )
        payoffs.append(
            _play_single_rlcard_game(
                candidate=candidate,
                baseline=baseline,
                candidate_seat=1,
                seed=game_seed,
                max_steps=max_steps,
            )
        )
    summary = summarize_h2h_payoffs(
        payoffs,
        games_per_seat=int(games_per_seat),
        min_lower95_candidate_payoff=min_lower95_candidate_payoff,
    )
    summary.update(
        {
            "algorithm": "alphanlholdem_rlcard_reference_h2h",
            "environment": "rlcard:no-limit-holdem",
            "environment_contract": "AlphaNLHoldem 50bb 5-action reference surface",
            "candidate_policy": candidate.policy_name,
            "baseline_policy": baseline.policy_name,
            "seed": int(seed),
            "max_steps": int(max_steps),
            "payoffs": [float(x) for x in payoffs],
            "benchmark_surfaces": [
                "rlcard_alphanlholdem_reference",
                "native_9_action_local_league",
                "held_out_slumbot",
            ],
            "native_slumbot_evidence": False,
            "uses_slumbot_data": False,
            "uses_alphanlholdem_training_data": False,
        }
    )
    return summary
