import numpy as np
import torch

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.early_street_policy_calibration import (
    calibrate_policy_head_from_early_targets,
)


def _write_value_checkpoint(path, *, seed: int = 0):
    torch.manual_seed(seed)
    net = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=2)
    torch.save(
        {
            "value_net": net.state_dict(),
            "hidden_dim": 32,
            "n_layers": 2,
            "n_players": 2,
            "initial_chips": 1000,
            "iteration": 17,
            "uses_betting_history": True,
        },
        path,
    )
    return net.state_dict()


def _early_street_targets() -> PolicyTargetBuffer:
    rng = np.random.default_rng(20260521)
    features = rng.normal(0.0, 0.25, size=(12, N_FEATURES)).astype(np.float32)
    features[:, 104:108] = 0.0
    features[:6, 104] = 1.0
    features[6:, 105] = 1.0
    legal_masks = np.zeros((12, N_ACTIONS), dtype=np.float32)
    legal_masks[:, [1, 2, 8]] = 1.0
    target_probs = np.zeros((12, N_ACTIONS), dtype=np.float32)
    target_probs[:6, 1] = 1.0
    target_probs[6:, 2] = 1.0
    return PolicyTargetBuffer(features, legal_masks, target_probs)


def test_calibrate_policy_head_from_early_targets_preserves_advantage_and_records_metadata(tmp_path):
    candidate = tmp_path / "candidate.pt"
    reference = tmp_path / "reference.pt"
    output = tmp_path / "calibrated.pt"
    original_state = _write_value_checkpoint(candidate, seed=3)
    _write_value_checkpoint(reference, seed=11)
    targets = _early_street_targets()

    metrics = calibrate_policy_head_from_early_targets(
        candidate,
        targets,
        output,
        reference_checkpoint=reference,
        state_source_checkpoint=candidate,
        target_metadata={"n_states": targets.size, "street_counts": {"0": 6, "1": 6}},
        n_steps=120,
        batch_size=12,
        lr=0.05,
        device="cpu",
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "early_street_learned_reference_policy_head_calibration"
    assert metrics["after_loss"] < metrics["before_loss"]

    saved = torch.load(output, map_location="cpu", weights_only=False)
    metadata = saved["policy_calibration"]
    assert metadata["mode"] == "early_street_learned_reference_policy_head_calibration"
    assert metadata["reference_checkpoint"] == str(reference)
    assert metadata["state_source_checkpoint"] == str(candidate)
    assert metadata["target_streets"] == [0, 1]
    assert metadata["target_metadata"]["street_counts"] == {"0": 6, "1": 6}
    assert torch.allclose(saved["value_net"]["adv_head.weight"], original_state["adv_head.weight"])
    assert not torch.allclose(
        saved["value_net"]["policy_head.weight"],
        original_state["policy_head.weight"],
    )
