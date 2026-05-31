"""Early-street learned-reference policy-head calibration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.early_street_calibration import (
    STREET_NAMES,
    evaluate_early_street_calibration,
)
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.policy_calibration import (
    _append_action,
    _normalized_strategy,
    _per_street_target_diagnostics,
    _resolve_device,
    _sample_cards,
    _target_street_metadata,
    _target_summary,
    _visible_board,
    train_policy_head_calibration,
)


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


def _compact_diagnostic(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep calibration reports readable by dropping per-row strategy blobs."""
    return {key: value for key, value in metrics.items() if key != "rows"}


def collect_early_street_reference_targets(
    *,
    reference_checkpoint: str | Path,
    state_source_checkpoint: str | Path,
    n_states: int = 4096,
    seed: int = 0,
    reference_strategy_source: str = "regret",
    state_source_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    max_hands: int | None = None,
    target_temperature: float = 1.0,
) -> tuple[PolicyTargetBuffer, dict[str, Any]]:
    """Collect preflop/flop policy-head targets from a learned reference."""
    resolved_device = _resolve_device(device)
    reference = load_value_network_checkpoint(reference_checkpoint, resolved_device)
    state_source = load_value_network_checkpoint(state_source_checkpoint, resolved_device)
    assert_strategy_source_supported(reference, reference_strategy_source)
    assert_strategy_source_supported(state_source, state_source_strategy_source)

    rng = np.random.default_rng(seed)
    features_out: list[np.ndarray] = []
    masks_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    weights_out: list[float] = []
    street_counts = {"0": 0, "1": 0}
    attempted_hands = 0
    max_hands = int(max_hands or max(100, n_states * 5))

    while len(features_out) < int(n_states) and attempted_hands < max_hands:
        attempted_hands += 1
        cards = _sample_cards(rng, 9)
        seat_holes = [cards[:2], cards[2:4]]
        board = cards[4:9]
        action_str = ""
        for _step in range(24):
            parsed = parse_action(action_str)
            if "error" in parsed:
                break
            street = int(parsed.get("st", -1))
            acting_pos = int(parsed.get("pos", -1))
            if acting_pos < 0 or street > 1:
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
            _, reference_strategy = network_strategy(
                reference.value_net,
                features,
                legal_mask,
                resolved_device,
                strategy_source=reference_strategy_source,
            )
            target = _normalized_strategy(
                np.asarray(reference_strategy, dtype=np.float32),
                legal_mask,
                target_temperature=target_temperature,
            )
            features_out.append(features.astype(np.float32, copy=False))
            masks_out.append((legal_mask > 0).astype(np.float32, copy=False))
            targets_out.append(target)
            weights_out.append(float(max(int(reference.metadata.get("iteration") or 1), 1)))
            street_counts[str(street)] += 1
            if len(features_out) >= int(n_states):
                break

            _, source_strategy = network_strategy(
                state_source.value_net,
                features,
                legal_mask,
                resolved_device,
                strategy_source=state_source_strategy_source,
            )
            legal_actions = np.flatnonzero(legal_mask > 0)
            probs = np.asarray([source_strategy[action] for action in legal_actions], dtype=np.float64)
            total = float(probs.sum())
            probs = probs / total if total > 1e-12 else np.ones_like(probs) / len(probs)
            action_idx = int(rng.choice(legal_actions, p=probs))
            increment = action_to_slumbot(action_idx, parsed, action_str, acting_pos)
            action_str = _append_action(action_str, increment, street)
            if increment.startswith("f") or action_idx == 8:
                break

    if len(features_out) != int(n_states):
        raise RuntimeError(
            f"generated {len(features_out)} early-street reference targets out of "
            f"requested {n_states} after {attempted_hands} hands"
        )

    features_arr = np.asarray(features_out, dtype=np.float32).reshape(n_states, N_FEATURES)
    masks_arr = np.asarray(masks_out, dtype=np.float32).reshape(n_states, N_ACTIONS)
    targets_arr = np.asarray(targets_out, dtype=np.float32).reshape(n_states, N_ACTIONS)
    weights_arr = np.asarray(weights_out, dtype=np.float32)
    targets = PolicyTargetBuffer(features_arr, masks_arr, targets_arr, weights_arr)
    metadata: dict[str, Any] = {
        "mode": "early_street_learned_reference_targets",
        "reference_checkpoint": str(reference_checkpoint),
        "reference_checkpoint_iteration": reference.metadata.get("checkpoint_iteration"),
        "state_source_checkpoint": str(state_source_checkpoint),
        "state_source_checkpoint_iteration": state_source.metadata.get("checkpoint_iteration"),
        "reference_strategy_source": reference_strategy_source,
        "state_source_strategy_source": state_source_strategy_source,
        "target_temperature": float(target_temperature),
        "device": str(resolved_device),
        "n_states": int(n_states),
        "attempted_hands": int(attempted_hands),
        "targets_per_hand": round(float(n_states) / max(float(attempted_hands), 1.0), 6),
        "street_counts": street_counts,
        "street_names": {key: STREET_NAMES[int(key)] for key in street_counts},
        "per_street_target_diagnostics": _per_street_target_diagnostics(
            targets.target_probs,
            targets.legal_masks,
            targets.features,
        ),
        **_target_summary(targets.target_probs),
    }
    return targets, metadata


def calibrate_policy_head_from_early_targets(
    checkpoint: str | Path,
    targets: PolicyTargetBuffer,
    output: str | Path,
    *,
    reference_checkpoint: str | Path,
    state_source_checkpoint: str | Path | None = None,
    target_metadata: dict[str, Any] | None = None,
    n_steps: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str | torch.device = "auto",
    rank_loss_weight: float = 0.0,
    rank_margin: float = 0.25,
    rank_confidence_weighted: bool = False,
) -> dict[str, Any]:
    """Fit only the candidate policy head and record early-street provenance."""
    metrics = train_policy_head_calibration(
        checkpoint,
        targets,
        output,
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        device=device,
        rank_loss_weight=rank_loss_weight,
        rank_margin=rank_margin,
        rank_confidence_weighted=rank_confidence_weighted,
    )
    resolved_device = _resolve_device(device)
    output = Path(output)
    saved = torch.load(output, map_location=resolved_device, weights_only=False)
    calibration = dict(saved.get("policy_calibration") or {})
    calibration.update(
        {
            "mode": "early_street_learned_reference_policy_head_calibration",
            "source_checkpoint": str(checkpoint),
            "reference_checkpoint": str(reference_checkpoint),
            "state_source_checkpoint": str(state_source_checkpoint or checkpoint),
            "target_metadata": dict(target_metadata or {}),
            **_target_street_metadata(targets.features),
        }
    )
    saved["policy_calibration"] = calibration
    torch.save(saved, output)
    metrics.update(
        {
            "mode": "early_street_learned_reference_policy_head_calibration",
            "reference_checkpoint": str(reference_checkpoint),
            "state_source_checkpoint": str(state_source_checkpoint or checkpoint),
            "target_metadata": dict(target_metadata or {}),
        }
    )
    return metrics


def train_early_street_policy_calibration(
    *,
    candidate_checkpoint: str | Path,
    reference_checkpoint: str | Path,
    output_checkpoint: str | Path,
    state_source_checkpoint: str | Path | None = None,
    n_states: int = 4096,
    seed: int = 0,
    reference_strategy_source: str = "regret",
    state_source_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    max_hands: int | None = None,
    target_temperature: float = 1.0,
    n_steps: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    rank_loss_weight: float = 0.0,
    rank_margin: float = 0.25,
    rank_confidence_weighted: bool = False,
    diagnostic_states: int = 0,
    diagnostic_seed: int | None = None,
    max_candidate_preflop_allin_rate: float = 0.10,
    max_candidate_flop_allin_rate: float = 0.20,
    max_mean_l1_to_reference: float = 0.75,
) -> dict[str, Any]:
    """Collect learned-reference targets, fit the policy head, and optionally diagnose."""
    state_source_path = state_source_checkpoint or candidate_checkpoint
    targets, target_metadata = collect_early_street_reference_targets(
        reference_checkpoint=reference_checkpoint,
        state_source_checkpoint=state_source_path,
        n_states=n_states,
        seed=seed,
        reference_strategy_source=reference_strategy_source,
        state_source_strategy_source=state_source_strategy_source,
        device=device,
        max_hands=max_hands,
        target_temperature=target_temperature,
    )
    metrics = calibrate_policy_head_from_early_targets(
        candidate_checkpoint,
        targets,
        output_checkpoint,
        reference_checkpoint=reference_checkpoint,
        state_source_checkpoint=state_source_path,
        target_metadata=target_metadata,
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        device=device,
        rank_loss_weight=rank_loss_weight,
        rank_margin=rank_margin,
        rank_confidence_weighted=rank_confidence_weighted,
    )
    metrics["target_collection"] = target_metadata

    if diagnostic_states > 0:
        diag_seed = int(seed if diagnostic_seed is None else diagnostic_seed)
        before = evaluate_early_street_calibration(
            candidate_checkpoint=candidate_checkpoint,
            reference_checkpoint=reference_checkpoint,
            state_source_checkpoint=state_source_path,
            n_states=diagnostic_states,
            seed=diag_seed,
            candidate_strategy_source="regret",
            reference_strategy_source=reference_strategy_source,
            state_source_strategy_source=state_source_strategy_source,
            device=device,
            max_candidate_preflop_allin_rate=max_candidate_preflop_allin_rate,
            max_candidate_flop_allin_rate=max_candidate_flop_allin_rate,
            max_mean_l1_to_reference=max_mean_l1_to_reference,
        )
        after = evaluate_early_street_calibration(
            candidate_checkpoint=output_checkpoint,
            reference_checkpoint=reference_checkpoint,
            state_source_checkpoint=state_source_path,
            n_states=diagnostic_states,
            seed=diag_seed,
            candidate_strategy_source="policy-head",
            reference_strategy_source=reference_strategy_source,
            state_source_strategy_source=state_source_strategy_source,
            device=device,
            max_candidate_preflop_allin_rate=max_candidate_preflop_allin_rate,
            max_candidate_flop_allin_rate=max_candidate_flop_allin_rate,
            max_mean_l1_to_reference=max_mean_l1_to_reference,
        )
        metrics["before_diagnostic"] = _compact_diagnostic(before)
        metrics["after_diagnostic"] = _compact_diagnostic(after)
        metrics["passed"] = bool(metrics["passed"] and after["passed"])

    return metrics
