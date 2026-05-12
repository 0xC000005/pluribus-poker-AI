import torch

from poker_ai.deep_cfr.networks import PolicyNetwork, ValueNetwork
from poker_ai.games.full_deck.state import N_FEATURES, N_ACTIONS


def test_value_net_uses_betting_history_dims_until_sequence_encoder_exists():
    torch.manual_seed(0)
    net = ValueNetwork(N_FEATURES, 64, N_ACTIONS, n_layers=2)
    net.eval()

    x = torch.zeros(2, N_FEATURES)
    x2 = x.clone()
    x[:, 113] = 1.0
    x[:, 114:126] = torch.randn(2, 12)
    x2[:, 113] = 7.0
    x2[:, 114:126] = torch.randn(2, 12)

    with torch.no_grad():
        y1 = net(x)
        y2 = net(x2)

    assert not torch.allclose(y1, y2, atol=1e-6)


def test_value_net_can_preserve_legacy_masked_history_contract():
    torch.manual_seed(0)
    net = ValueNetwork(
        N_FEATURES,
        64,
        N_ACTIONS,
        n_layers=2,
        use_betting_history=False,
    )
    net.eval()

    x = torch.zeros(2, N_FEATURES)
    x2 = x.clone()
    x[:, 113] = 1.0
    x[:, 114:126] = torch.randn(2, 12)
    x2[:, 113] = 7.0
    x2[:, 114:126] = torch.randn(2, 12)

    with torch.no_grad():
        y1 = net(x)
        y2 = net(x2)

    assert torch.allclose(y1, y2, atol=1e-6)


def test_policy_net_uses_betting_history_dims_until_sequence_encoder_exists():
    torch.manual_seed(0)
    net = PolicyNetwork(N_FEATURES, 64, N_ACTIONS, n_layers=2)
    net.eval()

    x = torch.zeros(2, N_FEATURES)
    x2 = x.clone()
    x[:, 113] = 1.0
    x[:, 114:126] = torch.randn(2, 12)
    x2[:, 113] = 7.0
    x2[:, 114:126] = torch.randn(2, 12)

    with torch.no_grad():
        y1 = net(x)
        y2 = net(x2)

    assert not torch.allclose(y1, y2, atol=1e-6)
