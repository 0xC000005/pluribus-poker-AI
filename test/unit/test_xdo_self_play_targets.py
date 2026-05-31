import numpy as np
import torch

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS
from poker_ai.research.mixed_policy_h2h import PolicyAdapter
from poker_ai.research.xdo_self_play_targets import collect_xdo_self_play_targets


def _call_adapter() -> PolicyAdapter:
    def _probs(_features, legal_mask, _device):
        legal = np.asarray(legal_mask, dtype=np.float32)
        probs = np.zeros(N_ACTIONS, dtype=np.float32)
        probs[1] = 1.0
        probs *= legal
        total = float(probs.sum())
        return probs / total if total > 0.0 else legal / float(legal.sum())

    return PolicyAdapter(
        kind="unit-call",
        checkpoint_path="unit-call",
        algorithm="unit_call_policy",
        action_probs_fn=_probs,
    )


def test_collect_xdo_self_play_targets_requires_requested_street_coverage(tmp_path):
    output_npz = tmp_path / "self_play_targets.npz"

    metrics = collect_xdo_self_play_targets(
        _call_adapter(),
        output_npz,
        n_states=4,
        n_worlds=1,
        target_streets=(0, 1),
        required_streets=(0, 1),
        min_rows_per_required_street=1,
        max_hands=4,
        max_steps_per_hand=16,
        device=torch.device("cpu"),
        seed=20260527,
    )

    assert metrics["passed"] is True
    assert metrics["coverage_gate"]["passed"] is True
    assert set(metrics["target_street_counts"]) >= {"0", "1"}
    buffer = PolicyTargetBuffer.from_npz(output_npz)
    assert buffer.size == metrics["target_size"]
    assert buffer.features.shape[1] == 126
    assert buffer.legal_masks.shape[1] == N_ACTIONS


def test_collect_xdo_self_play_targets_reports_generic_collection_exploration(tmp_path):
    output_npz = tmp_path / "explore_targets.npz"

    metrics = collect_xdo_self_play_targets(
        _call_adapter(),
        output_npz,
        n_states=2,
        n_worlds=1,
        target_streets=(0,),
        required_streets=(0,),
        max_hands=2,
        max_steps_per_hand=8,
        collection_exploration_epsilon=1.0,
        device=torch.device("cpu"),
        seed=20260528,
    )

    assert metrics["passed"] is True
    assert metrics["collection_exploration_epsilon"] == 1.0
    assert metrics["exploratory_actions"] > 0


def test_collect_xdo_self_play_targets_keeps_collecting_for_missing_required_streets(tmp_path):
    output_npz = tmp_path / "coverage_targets.npz"

    metrics = collect_xdo_self_play_targets(
        _call_adapter(),
        output_npz,
        n_states=1,
        n_worlds=1,
        target_streets=(0, 1),
        required_streets=(0, 1),
        min_rows_per_required_street=1,
        max_hands=4,
        max_steps_per_hand=16,
        device=torch.device("cpu"),
        seed=20260529,
    )

    assert metrics["passed"] is True
    assert metrics["target_size"] > 1
    assert set(metrics["target_street_counts"]) >= {"0", "1"}


def test_collect_xdo_self_play_targets_can_exceed_nominal_state_budget_for_min_rows(tmp_path):
    output_npz = tmp_path / "min_rows_targets.npz"

    metrics = collect_xdo_self_play_targets(
        _call_adapter(),
        output_npz,
        n_states=1,
        n_worlds=1,
        target_streets=(0,),
        required_streets=(0,),
        min_rows_per_required_street=3,
        max_hands=4,
        max_steps_per_hand=8,
        device=torch.device("cpu"),
        seed=20260530,
    )

    assert metrics["passed"] is True
    assert metrics["target_street_counts"]["0"] >= 3
