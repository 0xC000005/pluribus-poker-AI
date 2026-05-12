import numpy as np
import torch

from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.deep_cfr import train_value_network
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.deep_cfr.policy_targets import (
    PolicyTargetBuffer,
    masked_policy_cross_entropy,
)
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.search_target_eval import evaluate_search_targets
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase
from poker_ai.research.search_targets import (
    build_resolver_policy_targets,
    sample_resolver_cases,
)


def test_sample_resolver_cases_produces_valid_turn_river_states():
    cases = sample_resolver_cases(12, seed=20260512)

    assert len(cases) == 12
    seen_labels = {case.label for case in cases}
    assert len(seen_labels) == 12
    for case in cases:
        assert len(set(case.hole_cards + case.board)) == len(case.hole_cards + case.board)
        assert case.client_pos in (0, 1)
        assert len(case.board) in (4, 5)
        assert case.source == "sampled"


def test_policy_target_buffer_masks_and_normalizes_targets():
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    legal_masks = np.zeros((2, N_ACTIONS), dtype=np.float32)
    legal_masks[0, [1, 2]] = 1.0
    legal_masks[1, [3]] = 1.0
    target_probs = np.zeros((2, N_ACTIONS), dtype=np.float32)
    target_probs[0, 2] = 3.0
    target_probs[0, 8] = 7.0
    target_probs[1, 8] = 1.0

    buffer = PolicyTargetBuffer(features, legal_masks, target_probs)

    assert buffer.size == 2
    np.testing.assert_allclose(buffer.target_probs[0].sum(), 1.0)
    np.testing.assert_allclose(buffer.target_probs[0, 2], 1.0)
    np.testing.assert_allclose(buffer.target_probs[0, 8], 0.0)
    np.testing.assert_allclose(buffer.target_probs[1, 3], 1.0)


def test_masked_policy_cross_entropy_ignores_illegal_logits():
    logits = torch.zeros((1, N_ACTIONS), requires_grad=True)
    legal_masks = torch.zeros((1, N_ACTIONS))
    legal_masks[0, [0, 2]] = 1.0
    target_probs = torch.zeros((1, N_ACTIONS))
    target_probs[0, 2] = 1.0

    loss = masked_policy_cross_entropy(logits, legal_masks, target_probs)
    loss.backward()

    assert logits.grad is not None
    assert abs(float(logits.grad[0, 8])) < 1e-6
    assert float(logits.grad[0, 2]) < 0.0


def test_train_value_network_accepts_search_policy_targets():
    buffer = ReservoirBuffer(8)
    features = np.zeros(N_FEATURES, dtype=np.float32)
    advantages = np.zeros(N_ACTIONS, dtype=np.float32)
    advantages[1] = 0.1
    for _ in range(8):
        buffer.add(features, 1, advantages)

    legal_masks = np.zeros((2, N_ACTIONS), dtype=np.float32)
    legal_masks[:, [1, 2]] = 1.0
    target_probs = np.zeros((2, N_ACTIONS), dtype=np.float32)
    target_probs[:, 2] = 1.0
    search_targets = PolicyTargetBuffer(
        np.zeros((2, N_FEATURES), dtype=np.float32),
        legal_masks,
        target_probs,
    )

    net = train_value_network(
        buffer,
        hidden_dim=16,
        n_layers=1,
        n_epochs=2,
        batch_size=4,
        device=torch.device("cpu"),
        policy_target_buffer=search_targets,
        policy_target_weight=0.5,
    )

    with torch.no_grad():
        advantages_out, policy_logits = net.forward_with_policy(torch.zeros(N_FEATURES))
    assert torch.isfinite(advantages_out).all()
    assert torch.isfinite(policy_logits).all()


def test_evaluate_search_targets_reports_policy_fit(tmp_path):
    net = ValueNetwork(N_FEATURES, hidden_dim=16, output_dim=N_ACTIONS, n_layers=1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "value_net": net.state_dict(),
            "hidden_dim": 16,
            "n_layers": 1,
            "iteration": 1,
        },
        checkpoint,
    )

    legal_masks = np.zeros((2, N_ACTIONS), dtype=np.float32)
    legal_masks[:, [1, 2, 8]] = 1.0
    target_probs = np.zeros((2, N_ACTIONS), dtype=np.float32)
    target_probs[0, 2] = 1.0
    target_probs[1, 8] = 1.0
    targets = PolicyTargetBuffer(
        np.zeros((2, N_FEATURES), dtype=np.float32),
        legal_masks,
        target_probs,
    )
    target_path = tmp_path / "targets.npz"
    targets.save_npz(target_path)

    metrics = evaluate_search_targets(checkpoint, target_path, device="auto")

    assert metrics["passed"] is True
    assert metrics["mode"] == "search_target_eval"
    assert metrics["n_targets"] == 2
    assert metrics["strategy_source"] == "policy-head"
    assert metrics["has_policy_head"] is True
    assert np.isfinite(metrics["mean_l1"])
    assert np.isfinite(metrics["mean_kl"])


def test_belief_conditioned_policy_targets_record_range_diagnostics(tmp_path):
    net = ValueNetwork(N_FEATURES, hidden_dim=16, output_dim=N_ACTIONS, n_layers=1)
    checkpoint = tmp_path / "range_checkpoint.pt"
    torch.save(
        {
            "value_net": net.state_dict(),
            "hidden_dim": 16,
            "n_layers": 1,
            "iteration": 3,
        },
        checkpoint,
    )
    case = ResolverBenchmarkCase(
        label="belief-turn-open",
        hole_cards=("Ac", "Kd"),
        board=("2c", "7d", "Jh", "4s"),
        action_str="ck/kk/",
        client_pos=0,
        source="unit",
    )

    buffer, metadata = build_resolver_policy_targets(
        [case],
        solver_iterations=1,
        solver_backend="cpu",
        range_checkpoint=checkpoint,
        range_strategy_source="regret",
        range_device="cpu",
    )

    assert buffer.size == 1
    assert metadata["mode"] == "belief_conditioned_resolver_policy_targets"
    assert metadata["range_enabled"] is True
    assert metadata["range_checkpoint_iteration"] == 3
    assert metadata["target_allin_rate"] in (0.0, 1.0)
    record = metadata["records"][0]
    assert record["range_mode"] == "belief_conditioned"
    assert record["hero_range_support"] > 0
    assert record["villain_range_support"] > 0
    assert record["solver_n_hands"] <= record["solver_full_n_hands"]
