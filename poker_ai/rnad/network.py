"""R-NaD policy/value network — PyTorch port of the haiku net (scripts/vendor/rnad.py:746-762).

Torso = MLP(policy_network_layers, activate_final=True); policy-logit head = Linear(.->A);
value head = Linear(.->1). forward() returns (pi, v, log_pi, logit), matching the reference
network(env_step) -> (pi, v, log_pi, logit). Game-agnostic: obs_dim/n_actions are constructor args.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from poker_ai.rnad.functional import legal_policy, legal_log_policy


class RNaDNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_layers=(256, 256)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_layers = tuple(int(h) for h in hidden_layers)

        torso = []
        prev = obs_dim
        for h in self.hidden_layers:
            torso.append(nn.Linear(prev, h))
            torso.append(nn.ReLU())  # haiku MLP(activate_final=True)
            prev = h
        self.torso = nn.Sequential(*torso)
        self.policy_head = nn.Linear(prev, n_actions)  # hk.nets.MLP([num_distinct_actions])
        self.value_head = nn.Linear(prev, 1)           # hk.nets.MLP([1])

    def forward(self, obs: torch.Tensor, legal: torch.Tensor):
        """obs: [..., obs_dim]; legal: [..., A] {0,1}. Returns (pi, v, log_pi, logit)."""
        h = self.torso(obs)
        logit = self.policy_head(h)
        v = self.value_head(h)
        pi = legal_policy(logit, legal)
        log_pi = legal_log_policy(logit, legal)
        return pi, v, log_pi, logit
