import numpy as np
import torch
import sys
from pathlib import Path


def test_play_slumbot_loads_tiny_tianshou_rainbow_checkpoint(tmp_path):
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from scripts import play_slumbot

    device = torch.device("cpu")
    model = play_slumbot._RainbowDistributionNetForSlumbot(
        hidden_dim=8,
        num_atoms=3,
        device=device,
    )
    checkpoint = tmp_path / "rainbow.pt"
    torch.save(
        {
            "algorithm": "compiled_tianshou_rainbow_response_oracle",
            "num_actions": play_slumbot.N_ACTIONS,
            "num_features": play_slumbot.N_FEATURES,
            "hidden_dim": 8,
            "num_atoms": 3,
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    choice = play_slumbot._load_single_model_choice(
        checkpoint,
        device,
        strategy_source="regret",
        model_kind="auto",
    )

    assert choice.metadata["checkpoint_kind"] == "tianshou-rainbow"
    features = np.zeros(play_slumbot.N_FEATURES, dtype=np.float32)
    legal_mask = np.ones(play_slumbot.N_ACTIONS, dtype=np.float32)
    legal_mask[0] = 0.0
    advantages, strategy = play_slumbot.network_strategy(
        choice.value_net,
        features,
        legal_mask,
        device,
    )

    assert advantages.shape == (play_slumbot.N_ACTIONS,)
    assert strategy.shape == (play_slumbot.N_ACTIONS,)
    assert strategy[0] == 0.0
    assert np.isclose(strategy.sum(), 1.0)
