import numpy as np
import torch

from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.deep_cfr import train_value_network
from poker_ai.deep_cfr.policy_targets import (
    PolicyTargetBuffer,
    masked_policy_cross_entropy,
)
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES


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
