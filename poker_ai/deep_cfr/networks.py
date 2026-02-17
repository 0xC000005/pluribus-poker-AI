"""Value and policy networks for Single Deep CFR.

Adds two improvements while keeping backward compatibility:
 1) Ignores engineered history features (n_raises and per-street counts)
    by zeroing those inputs internally.
 2) Provides an optional policy head that can be trained to predict a
    strategy distribution (e.g., using regret-matching targets).

An optional sequence encoder can summarize action sequences into a fixed
vector and inject it into the model by replacing the engineered feature
slots, but if no sequence is provided the network reduces to the classic
MLP over the (masked) 126-d feature vector.
"""
import torch
import torch.nn as nn

from poker_ai.games.full_deck.state import N_FEATURES, N_ACTIONS


class ValueNetwork(nn.Module):
    """MLP that maps features (+optional sequence) to per-action values.

    - Masks engineered history dims internally to reduce handcrafting.
    - Exposes a policy head alongside the advantage head.
    - If a sequence summary is provided, it is projected to 13 dims and
      injected into the feature vector slots [113:126].
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 256,
        output_dim: int = N_ACTIONS,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # Trunk before heads.
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)

        # Heads.
        self.adv_head = nn.Linear(hidden_dim, output_dim)
        self.policy_head = nn.Linear(hidden_dim, output_dim)

        # Optional sequence projection into engineered slots (13 dims).
        self.seq_proj = nn.Linear(hidden_dim, 13)

        # Engineered indices to mask: n_raises (113) and per-street counts 114..125.
        self._engineered_start = 113
        self._engineered_end = 126

    def forward(
        self,
        x: torch.Tensor,
        seq_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning advantages only (backward compatible).

        - Masks engineered dims regardless of seq input.
        - If seq_summary is given (batch, hidden_dim), projects to 13 dims
          and replaces slots [113:126] before encoding.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.clone()
        x[:, self._engineered_start:self._engineered_end] = 0.0
        if seq_summary is not None:
            inj = self.seq_proj(seq_summary)
            x[:, self._engineered_start:self._engineered_end] = inj

        h = self.trunk(x)
        adv = self.adv_head(h)
        return adv

    def forward_with_policy(
        self,
        x: torch.Tensor,
        seq_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (advantages, policy_logits)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.clone()
        x[:, self._engineered_start:self._engineered_end] = 0.0
        if seq_summary is not None:
            inj = self.seq_proj(seq_summary)
            x[:, self._engineered_start:self._engineered_end] = inj
        h = self.trunk(x)
        return self.adv_head(h), self.policy_head(h)

    def predict(
        self, features: torch.Tensor, device: torch.device | None = None,
        seq_summary: torch.Tensor | None = None,
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
            if seq_summary is not None:
                seq_summary = seq_summary.to(device)
        with torch.no_grad():
            return self.forward(features.unsqueeze(0), seq_summary).squeeze(0)
