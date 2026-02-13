"""Value network for Single Deep CFR.

A small MLP that predicts advantage/regret values for each action given
a game state feature vector. The network is retrained from scratch each
CFR iteration (not fine-tuned), as this yields better convergence.
"""
import torch
import torch.nn as nn

from poker_ai.games.full_deck.state import N_FEATURES, N_ACTIONS


class ValueNetwork(nn.Module):
    """MLP that maps game state features to per-action advantage values.

    Architecture: input(N_FEATURES) -> 256 -> ReLU -> 256 -> ReLU -> N_ACTIONS
    ~100K parameters, <400KB on disk.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 256,
        output_dim: int = N_ACTIONS,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        advantages : Tensor of shape (batch, output_dim)
            Predicted advantage/regret for each action.
        """
        return self.net(x)

    def predict(
        self, features: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """Single-sample inference (no grad).

        Parameters
        ----------
        features : Tensor of shape (input_dim,)

        Returns
        -------
        advantages : Tensor of shape (output_dim,)
        """
        if device is not None:
            features = features.to(device)
        with torch.no_grad():
            return self.net(features.unsqueeze(0)).squeeze(0)
