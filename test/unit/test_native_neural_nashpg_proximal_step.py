import copy

import torch


def test_masked_mean_policy_kl_ignores_illegal_actions():
    from poker_ai.research.local_vtrace import masked_log_probs
    from scripts.run_native_neural_nashpg_compiled_learner import _masked_mean_policy_kl

    legal = torch.tensor([[True, True, False]])
    old_log_probs = masked_log_probs(torch.tensor([[0.0, 0.0, 500.0]]), legal)
    old_probs = torch.exp(old_log_probs)
    illegal_only_new = masked_log_probs(torch.tensor([[0.0, 0.0, -500.0]]), legal)
    legal_move_new = masked_log_probs(torch.tensor([[0.5, -0.5, -500.0]]), legal)

    assert torch.isclose(
        _masked_mean_policy_kl(
            old_probs=old_probs,
            old_log_probs=old_log_probs,
            new_log_probs=illegal_only_new,
            legal_masks=legal,
        ),
        torch.tensor(0.0),
    )
    assert _masked_mean_policy_kl(
        old_probs=old_probs,
        old_log_probs=old_log_probs,
        new_log_probs=legal_move_new,
        legal_masks=legal,
    ) > 0.0


def test_policy_kl_control_accepts_or_skips_optimizer_step():
    from scripts.run_native_neural_nashpg_compiled_learner import (
        _optimizer_step_with_optional_policy_kl_control,
    )

    torch.manual_seed(7)
    policy = torch.nn.Linear(3, 2)
    value = torch.nn.Linear(3, 1)
    features = torch.tensor([[1.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True]])
    params = list(policy.parameters()) + list(value.parameters())

    optimizer = torch.optim.SGD(params, lr=1.0)
    loss = -policy(features)[0, 0] + 0.01 * value(features).square().mean()
    accepted = _optimizer_step_with_optional_policy_kl_control(
        policy_net=policy,
        value_net=value,
        optimizer=optimizer,
        loss=loss,
        features=features,
        legal_masks=legal,
        max_policy_kl=10.0,
        max_backtracks=0,
        backtrack_factor=0.5,
    )
    assert accepted["policy_kl_control_enabled"] is True
    assert accepted["policy_update_accepted"] is True
    assert accepted["policy_update_skipped"] is False
    assert accepted["policy_update_kl"] is not None

    policy_before = copy.deepcopy(policy.state_dict())
    value_before = copy.deepcopy(value.state_dict())
    optimizer = torch.optim.SGD(params, lr=1.0)
    loss = -policy(features)[0, 1] + 0.01 * value(features).square().mean()
    skipped = _optimizer_step_with_optional_policy_kl_control(
        policy_net=policy,
        value_net=value,
        optimizer=optimizer,
        loss=loss,
        features=features,
        legal_masks=legal,
        max_policy_kl=1e-12,
        max_backtracks=0,
        backtrack_factor=0.5,
    )
    assert skipped["policy_update_accepted"] is False
    assert skipped["policy_update_skipped"] is True
    assert skipped["policy_update_kl"] is None
    for key, tensor in policy.state_dict().items():
        assert torch.equal(tensor, policy_before[key])
    for key, tensor in value.state_dict().items():
        assert torch.equal(tensor, value_before[key])


def test_native_neural_nashpg_proximal_smoke_records_telemetry(tmp_path):
    from scripts.run_native_neural_nashpg_compiled_learner import run_learner

    metrics = run_learner(
        train_iterations=1,
        games_per_iteration=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        hidden_dim=16,
        reference_update_every=1,
        adaptive_policy_kl_target=1.0,
        adaptive_policy_kl_max_backtracks=2,
        adaptive_policy_kl_backtrack_factor=0.5,
        seed=20260686,
        device="cpu",
        checkpoint_out=tmp_path / "proximal.pt",
    )

    assert metrics["passed"] is True
    assert metrics["adaptive_policy_kl_enabled"] is True
    assert metrics["adaptive_policy_kl_target"] == 1.0
    assert metrics["policy_update_accepted_steps"] + metrics["policy_update_skipped_steps"] == 1
    assert metrics["policy_update_accepted_steps"] == 1
    assert metrics["max_policy_update_kl"] is not None
    assert metrics["max_policy_update_kl"] <= 1.0
