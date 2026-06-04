"""RNaDSolver — the game-agnostic R-NaD learner. PyTorch port of RNaDSolver (scripts/vendor/rnad.py:716).

Holds the FOUR network copies (learner / target-EMA / prev / prev_), Adam optimizer, EntropySchedule,
and the step() loop: collect trajectory -> compute loss -> backward+clip -> Adam -> EMA target ->
roll references forward on schedule boundaries. Consumes any collector implementing .collect(); the
collector is the only game-specific piece.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from poker_ai.rnad.functional import EntropySchedule, compute_rnad_loss
from poker_ai.rnad.network import RNaDNetwork


@dataclass
class RNaDConfig:
    """Mirrors rnad.RNaDConfig (rnad.py:613) defaults that affect the learner."""
    trajectory_max: int = 10
    policy_network_layers: Sequence[int] = (256, 256)
    batch_size: int = 256
    learning_rate: float = 0.00005
    adam_b1: float = 0.0
    adam_b2: float = 0.999
    adam_eps: float = 1e-7  # rnad uses 10e-8 == 1e-7
    clip_gradient: float = 10_000.0
    target_network_avg: float = 0.001
    entropy_schedule_repeats: Sequence[int] = (1,)
    entropy_schedule_size: Sequence[int] = (20_000,)
    eta_reward_transform: float = 0.2
    nerd_beta: float = 2.0
    nerd_clip: float = 10_000.0
    c_vtrace: float = 1.0
    cix_eta: float = 0.0  # NeuRD-CIX cap on importance weight 1/(mu+cix_eta); 0 = exact NeuRD
    encoder: str = "mlp"  # "mlp" (flat torso) or "cnn" (AlphaHoldem-style card/action encoder, M2a)
    encoder_out_dim: int = 256
    encoder_conv_channels: Sequence[int] = (32, 64)
    encoder_noncard_hidden: Sequence[int] = (128,)
    seed: int = 42


class RNaDSolver:
    def __init__(self, config: RNaDConfig, collector, device="cpu"):
        self.config = config
        self.collector = collector
        self.device = torch.device(device)
        self.num_players = collector.n_players
        self.n_actions = collector.n_actions
        self.obs_dim = collector.obs_dim

        torch.manual_seed(config.seed)
        self._rng = np.random.RandomState(config.seed)

        encoder = None
        if str(getattr(config, "encoder", "mlp")) == "cnn":
            from poker_ai.rnad.encoder import CardActionEncoder  # noqa: PLC0415
            encoder = CardActionEncoder(
                obs_dim=self.obs_dim,
                out_dim=int(config.encoder_out_dim),
                conv_channels=tuple(config.encoder_conv_channels),
                noncard_hidden=tuple(config.encoder_noncard_hidden),
            )
        self.net = RNaDNetwork(
            self.obs_dim, self.n_actions, config.policy_network_layers, encoder=encoder,
        ).to(self.device)
        # 4 copies init from the SAME params (rnad.py:774-778).
        self.net_target = copy.deepcopy(self.net).to(self.device)
        self.net_prev = copy.deepcopy(self.net).to(self.device)
        self.net_prev_ = copy.deepcopy(self.net).to(self.device)
        for m in (self.net_target, self.net_prev, self.net_prev_):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config.learning_rate,
            betas=(config.adam_b1, config.adam_b2), eps=config.adam_eps,
        )
        self._entropy_schedule = EntropySchedule(
            sizes=config.entropy_schedule_size, repeats=config.entropy_schedule_repeats,
        )
        self.learner_steps = 0

        # GPU collectors take a torch.Generator (device RNG) + a torch-native policy_fn (no numpy
        # round-trip). CPU collectors take a numpy RandomState + numpy policy_fn. Detect via .is_gpu.
        self._collector_is_gpu = bool(getattr(collector, "is_gpu", False))
        if self._collector_is_gpu:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(config.seed)

    # ----- behavior policy used by the collector (uses the LEARNER net, like rnad.py:1016) -----
    @torch.no_grad()
    def _policy_fn(self, obs_np, legal_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(legal_np, dtype=torch.float32, device=self.device)
        pi, _, _, _ = self.net(obs, legal)
        return pi.detach().cpu().numpy()

    @torch.no_grad()
    def _policy_fn_t(self, obs_t, legal_t):
        """Torch-native behavior policy for the GPU collector (stays on device, no host round-trip)."""
        pi, _, _, _ = self.net(obs_t, legal_t)
        return pi

    def _forward_all(self, obs, legal):
        net_out = self.net(obs, legal)
        with torch.no_grad():
            target_out = self.net_target(obs, legal)
            prev_out = self.net_prev(obs, legal)
            prev_out_ = self.net_prev_(obs, legal)
        return net_out, target_out, prev_out, prev_out_

    def loss_on_trajectory(self, traj, alpha: float) -> torch.Tensor:
        """Compute the R-NaD loss on a collected Trajectory (used by step() and parity tests)."""
        net_out, target_out, prev_out, prev_out_ = self._forward_all(traj.obs, traj.legal)
        return compute_rnad_loss(
            net_out, target_out, prev_out, prev_out_,
            valid=traj.valid, player_id=traj.player_id, legal=traj.legal,
            acting_policy=traj.policy, action_oh=traj.action_oh, rewards=traj.rewards,
            alpha=alpha, num_players=self.num_players,
            eta=self.config.eta_reward_transform, c_vtrace=self.config.c_vtrace,
            nerd_clip=self.config.nerd_clip, nerd_beta=self.config.nerd_beta,
            cix_eta=self.config.cix_eta,
        )

    @torch.no_grad()
    def _ema_target(self):
        """params_target -= avg * (params_target - params). rnad.py:867-868."""
        a = self.config.target_network_avg
        for pt, p in zip(self.net_target.parameters(), self.net.parameters()):
            pt.add_(-a * (pt - p))
        for bt, b in zip(self.net_target.buffers(), self.net.buffers()):
            bt.copy_(b)

    @torch.no_grad()
    def _roll_references(self):
        """prev_ <- prev ; prev <- target. rnad.py:872-875."""
        self.net_prev_.load_state_dict(self.net_prev.state_dict())
        self.net_prev.load_state_dict(self.net_target.state_dict())

    def step(self) -> dict:
        """One R-NaD learner step. Mirrors RNaDSolver.step (rnad.py:932)."""
        if self._collector_is_gpu:
            traj = self.collector.collect(
                self._policy_fn_t, self.config.batch_size, self.config.trajectory_max, self._gen,
            )
        else:
            traj = self.collector.collect(
                self._policy_fn, self.config.batch_size, self.config.trajectory_max, self._rng,
            )
            traj = traj.to(self.device)
        alpha, update_target_net = self._entropy_schedule(self.learner_steps)

        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_on_trajectory(traj, alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
        self.optimizer.step()

        self._ema_target()
        if update_target_net:
            self._roll_references()

        self.learner_steps += 1
        return {"loss": float(loss.detach().cpu()), "alpha": alpha,
                "update_target_net": bool(update_target_net), "learner_steps": self.learner_steps}

    @torch.no_grad()
    def action_probabilities(self, obs_np, legal_np):
        """Eval policy from the TARGET net (rnad.py:994 uses params_target)."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(legal_np, dtype=torch.float32, device=self.device)
        pi, _, _, _ = self.net_target(obs, legal)
        return pi.detach().cpu().numpy()
