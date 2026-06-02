import numpy as np
import torch
import sys
from pathlib import Path
from types import SimpleNamespace


def _write_tiny_rainbow_checkpoint(play_slumbot, path, *, hidden_dim=8, num_atoms=3):
    device = torch.device("cpu")
    model = play_slumbot._RainbowDistributionNetForSlumbot(
        hidden_dim=hidden_dim,
        num_atoms=num_atoms,
        device=device,
    )
    torch.save(
        {
            "algorithm": "compiled_tianshou_rainbow_response_oracle",
            "num_actions": play_slumbot.N_ACTIONS,
            "num_features": play_slumbot.N_FEATURES,
            "hidden_dim": hidden_dim,
            "num_atoms": num_atoms,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def test_play_slumbot_loads_tiny_tianshou_rainbow_checkpoint(tmp_path):
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from scripts import play_slumbot

    device = torch.device("cpu")
    checkpoint = tmp_path / "rainbow.pt"
    _write_tiny_rainbow_checkpoint(play_slumbot, checkpoint)

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
    batched = choice.value_net(torch.zeros((2, play_slumbot.N_FEATURES), dtype=torch.float32))

    assert advantages.shape == (play_slumbot.N_ACTIONS,)
    assert batched.shape == (2, play_slumbot.N_ACTIONS)
    assert strategy.shape == (play_slumbot.N_ACTIONS,)
    assert strategy[0] == 0.0
    assert np.isclose(strategy.sum(), 1.0)


def test_play_slumbot_loads_tianshou_rainbow_checkpoint_mixture(tmp_path):
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from scripts import play_slumbot

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _write_tiny_rainbow_checkpoint(play_slumbot, first)
    _write_tiny_rainbow_checkpoint(play_slumbot, second)
    args = SimpleNamespace(
        model=None,
        model_kind="tianshou-rainbow",
        model_glob=[],
        model_checkpoint=[str(first), str(second)],
        model_mixture_weights="0.0,1.0",
        mixture_seed=20260899,
        strategy_source="regret",
    )

    selector = play_slumbot._load_model_selector(args, torch.device("cpu"))
    choice = selector.select_for_hand(hand_index=1)

    assert selector.size == 2
    assert choice.metadata["checkpoint"] == str(second)
    assert choice.metadata["checkpoint_kind"] == "tianshou-rainbow"
    assert choice.trace_context["mixture_index"] == 1
    assert choice.trace_context["mixture_weight"] == 1.0


def test_play_slumbot_loads_tianshou_rainbow_checkpoint_mixture_from_glob(tmp_path):
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from scripts import play_slumbot

    first = tmp_path / "rainbow_a.pt"
    second = tmp_path / "rainbow_b.pt"
    _write_tiny_rainbow_checkpoint(play_slumbot, first)
    _write_tiny_rainbow_checkpoint(play_slumbot, second)
    args = SimpleNamespace(
        model=None,
        model_kind="tianshou-rainbow",
        model_glob=[str(tmp_path / "rainbow_*.pt")],
        model_checkpoint=[],
        model_mixture_weights="0.25,0.75",
        mixture_seed=20260900,
        strategy_source="regret",
    )

    selector = play_slumbot._load_model_selector(args, torch.device("cpu"))

    assert selector.size == 2
    assert [choice.metadata["checkpoint"] for choice in selector.choices] == [
        str(first),
        str(second),
    ]
    assert [choice.trace_context["mixture_weight"] for choice in selector.choices] == [
        0.25,
        0.75,
    ]
