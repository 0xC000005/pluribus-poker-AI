import torch


def test_vtrace_returns_match_two_step_hand_calculation():
    from poker_ai.research.local_vtrace import vtrace_from_importance_weights

    rewards = torch.tensor([[1.0], [2.0]])
    discounts = torch.tensor([[0.9], [0.9]])
    values = torch.tensor([[0.5], [0.25]])
    bootstrap = torch.tensor([0.1])
    target_log_probs = torch.log(torch.tensor([[0.8], [0.4]]))
    behavior_log_probs = torch.log(torch.tensor([[0.4], [0.8]]))

    out = vtrace_from_importance_weights(
        rewards=rewards,
        discounts=discounts,
        values=values,
        bootstrap_value=bootstrap,
        target_action_log_probs=target_log_probs,
        behavior_action_log_probs=behavior_log_probs,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
        clip_c_threshold=1.0,
    )

    expected_v1 = 0.25 + 0.5 * (2.0 + 0.9 * 0.1 - 0.25)
    expected_v0 = 0.5 + 1.0 * (1.0 + 0.9 * 0.25 - 0.5) + 0.9 * 1.0 * (expected_v1 - 0.25)
    expected_pg1 = 0.5 * (2.0 + 0.9 * 0.1 - 0.25)
    expected_pg0 = 1.0 * (1.0 + 0.9 * expected_v1 - 0.5)

    assert torch.allclose(out.vs[:, 0], torch.tensor([expected_v0, expected_v1]))
    assert torch.allclose(out.pg_advantages[:, 0], torch.tensor([expected_pg0, expected_pg1]))


def test_masked_policy_loss_assigns_zero_probability_to_illegal_actions():
    from poker_ai.research.local_vtrace import masked_log_probs

    logits = torch.tensor([[2.0, 1.0, 10.0]])
    legal_mask = torch.tensor([[True, True, False]])

    log_probs = masked_log_probs(logits, legal_mask)

    assert torch.isneginf(log_probs[0, 2])
    assert torch.allclose(log_probs[0, :2].exp().sum(), torch.tensor(1.0))


def test_vtrace_loss_backward_updates_actor_and_value_terms():
    from poker_ai.research.local_vtrace import vtrace_policy_value_loss

    logits = torch.tensor(
        [
            [[1.0, 0.0, -1.0]],
            [[0.5, 0.25, -0.25]],
        ],
        requires_grad=True,
    )
    values = torch.tensor([[0.1], [0.2]], requires_grad=True)
    actions = torch.tensor([[0], [1]])
    legal_mask = torch.tensor(
        [
            [[True, True, False]],
            [[True, True, True]],
        ]
    )
    behavior_log_probs = torch.log(torch.tensor([[0.5], [0.5]]))
    rewards = torch.tensor([[0.0], [1.0]])
    discounts = torch.tensor([[0.99], [0.0]])
    bootstrap = torch.tensor([0.0])

    loss, stats = vtrace_policy_value_loss(
        logits=logits,
        values=values,
        actions=actions,
        legal_mask=legal_mask,
        behavior_action_log_probs=behavior_log_probs,
        rewards=rewards,
        discounts=discounts,
        bootstrap_value=bootstrap,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(values.grad).all()
    assert stats["illegal_action_probability"] == 0.0


def test_vtrace_loss_valid_mask_ignores_padded_steps():
    from poker_ai.research.local_vtrace import vtrace_policy_value_loss

    logits = torch.tensor(
        [
            [[0.5, 0.0]],
            [[0.0, 0.5]],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    actions = torch.tensor([[0], [1]])
    legal_mask = torch.ones_like(logits, dtype=torch.bool)
    behavior_log_probs = torch.log(torch.tensor([[0.5], [0.5]], dtype=torch.float32))
    rewards = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    discounts = torch.tensor([[0.99], [0.0]], dtype=torch.float32)
    bootstrap = torch.tensor([0.0], dtype=torch.float32)

    base_loss, base_stats = vtrace_policy_value_loss(
        logits=logits,
        values=values,
        actions=actions,
        legal_mask=legal_mask,
        behavior_action_log_probs=behavior_log_probs,
        rewards=rewards,
        discounts=discounts,
        bootstrap_value=bootstrap,
    )

    padded_loss, padded_stats = vtrace_policy_value_loss(
        logits=torch.cat([logits, torch.tensor([[[100.0, -100.0]]])], dim=0),
        values=torch.cat([values, torch.tensor([[999.0]])], dim=0),
        actions=torch.cat([actions, torch.tensor([[0]])], dim=0),
        legal_mask=torch.cat([legal_mask, torch.tensor([[[True, False]]])], dim=0),
        behavior_action_log_probs=torch.cat(
            [behavior_log_probs, torch.tensor([[0.0]])],
            dim=0,
        ),
        rewards=torch.cat([rewards, torch.tensor([[999.0]])], dim=0),
        discounts=torch.cat([discounts, torch.tensor([[1.0]])], dim=0),
        bootstrap_value=bootstrap,
        valid_mask=torch.tensor([[True], [True], [False]]),
    )

    assert torch.allclose(padded_loss, base_loss)
    assert padded_stats["valid_samples"] == base_stats["valid_samples"] == 2.0
