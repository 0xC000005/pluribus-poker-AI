import torch

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import N_FEATURES, N_ACTIONS


def test_value_net_masks_engineered_dims_identically():
    torch.manual_seed(0)
    net = ValueNetwork(N_FEATURES, 64, N_ACTIONS, n_layers=2)
    net.eval()

    x = torch.zeros(2, N_FEATURES)
    x2 = x.clone()
    # Set engineered dims differently between x and x2.
    x[:, 113] = 1.0
    x[:, 114:126] = torch.randn(2, 12)
    # x2 has different engineered values.
    x2[:, 113] = 7.0
    x2[:, 114:126] = torch.randn(2, 12)

    with torch.no_grad():
        y1 = net(x)
        y2 = net(x2)

    # Because the network zeroes these dims internally, outputs match.
    assert torch.allclose(y1, y2, atol=1e-6)
