"""Average-policy calibration utilities for learned range tracking."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.optim as optim

from poker_ai.deep_cfr.policy_targets import (
    PolicyTargetBuffer,
    masked_policy_cross_entropy,
)
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (  # noqa: E402
    action_to_slumbot,
    build_features,
    get_legal_mask_from_parsed,
    network_strategy,
    parse_action,
)


_SUITS = ("c", "d", "h", "s")
_RANKS = tuple("23456789TJQKA")


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _card_to_str(card: int) -> str:
    return _RANKS[card // 4] + _SUITS[card % 4]


def _sample_cards(rng: np.random.Generator, n: int) -> list[str]:
    cards = rng.choice(52, size=n, replace=False)
    return [_card_to_str(int(card)) for card in cards]


def _card_str_to_index(card: str) -> int:
    return _RANKS.index(card[0]) * 4 + _SUITS.index(card[1])


def _visible_board(board: list[str], street: int) -> list[str]:
    if street <= 0:
        return []
    if street == 1:
        return board[:3]
    if street == 2:
        return board[:4]
    return board[:5]


def _append_action(action_str: str, increment: str, previous_street: int) -> str:
    next_action = action_str + increment
    if increment.startswith("f"):
        return next_action
    parsed_next = parse_action(next_action)
    if "error" in parsed_next:
        return next_action
    if (
        int(parsed_next.get("st", previous_street)) > previous_street
        and int(parsed_next.get("pos", -1)) >= 0
        and not next_action.endswith("/")
    ):
        return next_action + "/"
    return next_action


def _sample_private_hands(
    board: Iterable[str],
    *,
    rng: np.random.Generator,
    max_hands: int | None,
) -> list[tuple[str, str]]:
    blocked = {_card_str_to_index(card) for card in board}
    available = [card for card in range(52) if card not in blocked]
    hands = list(itertools.combinations(available, 2))
    if max_hands is not None and max_hands < len(hands):
        selected = rng.choice(len(hands), size=max_hands, replace=False)
        hands = [hands[int(index)] for index in selected]
    return [(_card_to_str(hand[0]), _card_to_str(hand[1])) for hand in hands]


def _normalized_strategy(
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    *,
    target_temperature: float = 1.0,
) -> np.ndarray:
    target = np.asarray(strategy, dtype=np.float32) * (legal_mask > 0)
    total = float(target.sum())
    if total > 1e-8:
        normalized = target / total
    else:
        legal_total = float(legal_mask.sum())
        if legal_total <= 0:
            raise ValueError("cannot normalize a policy target without legal actions")
        normalized = legal_mask.astype(np.float32) / legal_total

    temperature = float(target_temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("target_temperature must be finite and positive")
    if abs(temperature - 1.0) < 1e-8:
        return normalized.astype(np.float32, copy=False)

    legal = legal_mask > 0
    softened = np.zeros_like(normalized, dtype=np.float32)
    powered = np.power(np.clip(normalized[legal], 1e-8, 1.0), 1.0 / temperature)
    powered_total = float(powered.sum())
    if powered_total <= 0.0:
        raise ValueError("cannot normalize a policy target without legal actions")
    softened[legal] = powered / powered_total
    return softened


def _target_summary(targets: np.ndarray) -> dict[str, float]:
    top_actions = np.argmax(targets, axis=1)
    entropies = []
    for target in targets:
        positive = target[target > 0]
        entropies.append(float(-(positive * np.log(positive)).sum()) if positive.size else 0.0)
    return {
        "target_allin_rate": round(float(np.mean(top_actions == 8)), 6),
        "mean_target_allin_prob": round(float(np.mean(targets[:, 8])), 6),
        "mean_target_entropy": round(float(np.mean(entropies)), 6),
    }


def sample_policy_calibration_targets(
    n_targets: int,
    *,
    checkpoint: str | Path,
    seed: int = 0,
    strategy_source: str = "regret",
    device: str | torch.device = "auto",
    max_hands: int | None = None,
    target_temperature: float = 1.0,
    stats: dict[str, Any] | None = None,
) -> tuple[PolicyTargetBuffer, dict[str, Any]]:
    """Collect supervised average-policy targets from learned self-play decisions."""
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, strategy_source)
    rng = np.random.default_rng(seed)
    max_hands = max_hands or max(100, n_targets * 20)

    features_out: list[np.ndarray] = []
    masks_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    weights_out: list[float] = []
    street_counts = {str(street): 0 for street in range(4)}
    attempted_hands = 0

    while len(features_out) < n_targets and attempted_hands < max_hands:
        attempted_hands += 1
        cards = _sample_cards(rng, 9)
        seat_holes = [cards[:2], cards[2:4]]
        board = cards[4:9]
        action_str = ""

        for _ in range(40):
            parsed = parse_action(action_str)
            if "error" in parsed:
                break
            street = int(parsed.get("st", -1))
            acting_pos = int(parsed.get("pos", -1))
            if acting_pos < 0 or street not in (0, 1, 2, 3):
                break
            visible_board = _visible_board(board, street)
            features = build_features(
                seat_holes[acting_pos],
                visible_board,
                action_str,
                acting_pos,
                parsed,
            )
            legal_mask = get_legal_mask_from_parsed(parsed, action_str, acting_pos)
            _, strategy = network_strategy(
                loaded.value_net,
                features,
                legal_mask,
                resolved_device,
                strategy_source=strategy_source,
            )
            target = _normalized_strategy(
                strategy,
                legal_mask,
                target_temperature=target_temperature,
            )
            features_out.append(features.astype(np.float32, copy=False))
            masks_out.append((legal_mask > 0).astype(np.float32, copy=False))
            targets_out.append(target)
            weights_out.append(float(max(int(loaded.metadata.get("iteration") or 1), 1)))
            street_counts[str(street)] += 1
            if len(features_out) >= n_targets:
                break

            legal_actions = np.flatnonzero(legal_mask > 0)
            probs = np.asarray([strategy[action] for action in legal_actions], dtype=np.float64)
            total = float(probs.sum())
            probs = probs / total if total > 0 else np.ones_like(probs) / len(probs)
            action_idx = int(rng.choice(legal_actions, p=probs))
            increment = action_to_slumbot(action_idx, parsed, action_str, acting_pos)
            action_str = _append_action(action_str, increment, street)
            if increment.startswith("f"):
                break

    if len(features_out) != n_targets:
        raise RuntimeError(
            f"generated {len(features_out)} policy calibration targets out of "
            f"requested {n_targets} after {attempted_hands} hands"
        )

    features_arr = np.asarray(features_out, dtype=np.float32).reshape(n_targets, N_FEATURES)
    masks_arr = np.asarray(masks_out, dtype=np.float32).reshape(n_targets, N_ACTIONS)
    targets_arr = np.asarray(targets_out, dtype=np.float32).reshape(n_targets, N_ACTIONS)
    weights_arr = np.asarray(weights_out, dtype=np.float32)
    buffer = PolicyTargetBuffer(features_arr, masks_arr, targets_arr, weights_arr)
    metadata: dict[str, Any] = {
        "mode": "policy_calibration_targets",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": loaded.metadata.get("checkpoint_iteration"),
        "strategy_source": strategy_source,
        "target_temperature": float(target_temperature),
        "device": str(resolved_device),
        "n_targets": int(n_targets),
        "attempted_hands": int(attempted_hands),
        "targets_per_hand": round(float(n_targets) / max(float(attempted_hands), 1.0), 6),
        "street_counts": street_counts,
        **_target_summary(buffer.target_probs),
    }
    if stats is not None:
        stats.update(metadata)
    return buffer, metadata


def save_policy_calibration_targets(
    output: str | Path,
    *,
    checkpoint: str | Path,
    n_targets: int,
    seed: int = 0,
    strategy_source: str = "regret",
    device: str | torch.device = "auto",
    max_hands: int | None = None,
    target_temperature: float = 1.0,
) -> dict[str, Any]:
    buffer, metadata = sample_policy_calibration_targets(
        n_targets,
        checkpoint=checkpoint,
        seed=seed,
        strategy_source=strategy_source,
        device=device,
        max_hands=max_hands,
        target_temperature=target_temperature,
    )
    output = Path(output)
    buffer.save_npz(output)
    meta_path = output.with_suffix(output.suffix + ".json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def sample_public_state_hand_sweep_targets(
    cases: Iterable[ResolverBenchmarkCase],
    *,
    checkpoint: str | Path,
    seed: int = 0,
    strategy_source: str = "regret",
    device: str | torch.device = "auto",
    hands_per_case: int | None = 256,
    target_temperature: float = 1.0,
) -> tuple[PolicyTargetBuffer, dict[str, Any]]:
    """Expand turn/river public states over compatible private hands."""
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, strategy_source)
    rng = np.random.default_rng(seed)
    selected_cases = list(cases)

    features_out: list[np.ndarray] = []
    masks_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    weights_out: list[float] = []
    records: list[dict[str, Any]] = []
    street_counts = {"2": 0, "3": 0}

    for case in selected_cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append({"label": case.label, "skipped": f"parse_error:{parsed['error']}"})
            continue
        street = int(parsed.get("st", -1))
        if street not in (2, 3):
            records.append({"label": case.label, "skipped": f"unsupported_street:{street}"})
            continue
        acting_pos = int(case.client_pos)
        n_board = 4 if street == 2 else 5
        board = list(case.board[:n_board])
        legal_mask = get_legal_mask_from_parsed(parsed, case.action_str, acting_pos)
        hands = _sample_private_hands(board, rng=rng, max_hands=hands_per_case)
        start = len(features_out)
        for hand in hands:
            features = build_features(
                list(hand),
                board,
                case.action_str,
                acting_pos,
                parsed,
            )
            _, strategy = network_strategy(
                loaded.value_net,
                features,
                legal_mask,
                resolved_device,
                strategy_source=strategy_source,
            )
            features_out.append(features.astype(np.float32, copy=False))
            masks_out.append((legal_mask > 0).astype(np.float32, copy=False))
            targets_out.append(
                _normalized_strategy(
                    strategy,
                    legal_mask,
                    target_temperature=target_temperature,
                )
            )
            weights_out.append(float(max(int(loaded.metadata.get("iteration") or 1), 1)))
        n_added = len(features_out) - start
        street_counts[str(street)] += n_added
        records.append(
            {
                "label": case.label,
                "street": street,
                "action_str": case.action_str,
                "n_hands": int(n_added),
                "n_legal_actions": int(legal_mask.sum()),
            }
        )

    if not features_out:
        raise RuntimeError("no public-state hand-sweep calibration targets generated")

    n_targets = len(features_out)
    features_arr = np.asarray(features_out, dtype=np.float32).reshape(n_targets, N_FEATURES)
    masks_arr = np.asarray(masks_out, dtype=np.float32).reshape(n_targets, N_ACTIONS)
    targets_arr = np.asarray(targets_out, dtype=np.float32).reshape(n_targets, N_ACTIONS)
    weights_arr = np.asarray(weights_out, dtype=np.float32)
    buffer = PolicyTargetBuffer(features_arr, masks_arr, targets_arr, weights_arr)
    metadata: dict[str, Any] = {
        "mode": "public_state_hand_sweep_policy_calibration_targets",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": loaded.metadata.get("checkpoint_iteration"),
        "strategy_source": strategy_source,
        "target_temperature": float(target_temperature),
        "device": str(resolved_device),
        "n_cases": int(len(selected_cases)),
        "n_targets": int(n_targets),
        "hands_per_case": int(hands_per_case) if hands_per_case is not None else "all",
        "street_counts": street_counts,
        "records": records,
        **_target_summary(buffer.target_probs),
    }
    return buffer, metadata


def save_public_state_hand_sweep_targets(
    output: str | Path,
    cases: Iterable[ResolverBenchmarkCase],
    *,
    checkpoint: str | Path,
    seed: int = 0,
    strategy_source: str = "regret",
    device: str | torch.device = "auto",
    hands_per_case: int | None = 256,
    target_temperature: float = 1.0,
) -> dict[str, Any]:
    buffer, metadata = sample_public_state_hand_sweep_targets(
        cases,
        checkpoint=checkpoint,
        seed=seed,
        strategy_source=strategy_source,
        device=device,
        hands_per_case=hands_per_case,
        target_temperature=target_temperature,
    )
    output = Path(output)
    buffer.save_npz(output)
    meta_path = output.with_suffix(output.suffix + ".json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def evaluate_policy_target_loss(
    value_net: torch.nn.Module,
    targets: PolicyTargetBuffer,
    device: str | torch.device = "auto",
    *,
    batch_size: int = 8192,
) -> float:
    resolved_device = _resolve_device(device)
    value_net.eval()
    weighted_sum = 0.0
    weight_total = 0.0
    with torch.no_grad():
        for start in range(0, targets.size, batch_size):
            end = min(start + batch_size, targets.size)
            features = torch.from_numpy(targets.features[start:end]).to(resolved_device)
            masks = torch.from_numpy(targets.legal_masks[start:end]).to(resolved_device)
            target_probs = torch.from_numpy(targets.target_probs[start:end]).to(resolved_device)
            weights = torch.from_numpy(targets.weights[start:end]).to(resolved_device)
            _, logits = value_net.forward_with_policy(features)
            legal = (masks > 0).to(dtype=logits.dtype)
            masked_logits = logits.masked_fill(legal <= 0, -1e4)
            per_sample = -(
                target_probs.to(dtype=logits.dtype)
                * torch.log_softmax(masked_logits, dim=1)
            ).sum(dim=1)
            weighted_sum += float((per_sample * weights).sum().cpu())
            weight_total += float(weights.sum().cpu())
    return weighted_sum / max(weight_total, 1e-8)


def train_policy_head_calibration(
    checkpoint: str | Path,
    targets: PolicyTargetBuffer,
    output: str | Path,
    *,
    n_steps: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Train only the policy head against supervised average-policy targets."""
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    value_net = loaded.value_net
    before_loss = evaluate_policy_target_loss(
        value_net,
        targets,
        resolved_device,
        batch_size=batch_size,
    )

    for param in value_net.parameters():
        param.requires_grad = False
    for param in value_net.policy_head.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(value_net.policy_head.parameters(), lr=lr, weight_decay=0.0)

    value_net.train()
    losses: list[float] = []
    for _ in range(int(n_steps)):
        batch = targets.sample_batch(batch_size, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        _, logits = value_net.forward_with_policy(batch.features)
        loss = masked_policy_cross_entropy(
            logits,
            batch.legal_masks,
            batch.target_probs,
            weights=batch.weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.policy_head.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    after_loss = evaluate_policy_target_loss(
        value_net,
        targets,
        resolved_device,
        batch_size=batch_size,
    )
    original = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    original["value_net"] = value_net.state_dict()
    original["policy_calibration"] = {
        "source_checkpoint": str(checkpoint),
        "target_size": int(targets.size),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "before_loss": round(float(before_loss), 6),
        "after_loss": round(float(after_loss), 6),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(original, output)
    return {
        "mode": "policy_head_calibration",
        "passed": output.exists() and after_loss <= before_loss,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "device": str(resolved_device),
        "target_size": int(targets.size),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "before_loss": round(float(before_loss), 6),
        "after_loss": round(float(after_loss), 6),
        "loss_delta": round(float(after_loss - before_loss), 6),
        "mean_train_loss": round(float(np.mean(losses)), 6) if losses else 0.0,
    }
