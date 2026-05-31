"""Arm A of the GO/NO-GO: a net-only PPO learner sharing the R-NaD collector + network.

Controlled A/B design (governance bundle 20260531T000000Z-ppo-vs-rnad-smallnlhe-gonogo, Option A): both
arms run on the SAME small-NLHE OpenSpiel game via the SAME collector and the SAME RNaDNetwork (policy +
value heads); they differ ONLY in the learner loss. Arm B = R-NaD (NeuRD + mixed-player V-trace + reference
roll-forward). Arm A = PPO here: clipped surrogate + value MSE + entropy bonus, with AlphaHoldem-style
Trinal-Clip (dual-clip for negative advantages). Behavior policy at collection = the current learner net
(self-play). The K-best historical opponent pool (anti-collapse) is added on top later (needs a seat-aware
collector; tracked separately) — base self-play first.

Advantages: small-NLHE rewards are sparse terminal payoffs. For each valid decision (t,b) by player p, the
Monte-Carlo return G[b,p] = sum_t rewards[t,b,p] (the game's terminal payoff for p); advantage = G - v(obs),
the 'value' advantage mode (matches native_ppo_policy.py's default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from poker_ai.rnad.network import RNaDNetwork


@dataclass
class PPOConfig:
    trajectory_max: int = 12
    policy_network_layers: Sequence[int] = (64, 64)
    batch_size: int = 256
    learning_rate: float = 0.005
    clip_epsilon: float = 0.2
    trinal_clip: float = 3.0          # AlphaHoldem dual-clip constant for negative advantages
    value_coef: float = 0.5
    entropy_coef: float = 0.10        # 2502.08938 "correct entropy" band 0.05-0.2
    ppo_epochs: int = 2
    clip_gradient: float = 1.0
    seed: int = 1


def compute_ppo_loss(
    pi: torch.Tensor,            # [T,B,A] current policy (legal)
    log_pi: torch.Tensor,       # [T,B,A] current log policy (legal)
    v: torch.Tensor,            # [T,B,1] current value
    *,
    valid: torch.Tensor,        # [T,B]
    player_id: torch.Tensor,    # [T,B]
    legal: torch.Tensor,        # [T,B,A]
    action_oh: torch.Tensor,    # [T,B,A]
    behavior_policy: torch.Tensor,  # [T,B,A] mu at collection
    rewards: torch.Tensor,      # [T,B,P]
    num_players: int,
    clip_epsilon: float,
    trinal_clip: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict]:
    """Net-only PPO loss over valid decisions (clipped surrogate + value MSE + entropy). Returns (loss, logs)."""
    eps = 1e-9
    T, B, A = pi.shape
    valid_f = valid.to(pi.dtype)                                   # [T,B]
    v_flat = v.squeeze(-1)                                         # [T,B]

    # Monte-Carlo per-player game return G[b,p]. The collectors REPEAT the terminal payoff on every
    # post-terminal padding step (rewards[t]==env.rewards of the next state; terminals stay terminal),
    # so summing over T inflates the return (measured ~7x here). Take the terminal step's return once;
    # T == max_game_length+1, so every game has terminated by the last row.
    G_bp = rewards[-1]                                            # [B,P]
    pid_long = player_id.long().clamp(min=0)                       # [T,B]
    G_actor = torch.gather(G_bp.unsqueeze(0).expand(T, B, num_players), 2, pid_long.unsqueeze(-1)).squeeze(-1)
    G_actor = G_actor * valid_f                                    # [T,B]

    advantage = (G_actor - v_flat).detach()                       # [T,B]
    m = valid_f.sum().clamp_min(1.0)
    adv_mean = (advantage * valid_f).sum() / m
    adv_var = (((advantage - adv_mean) ** 2) * valid_f).sum() / m
    advantage = (advantage - adv_mean) / (adv_var.sqrt() + 1e-5)

    # Defense-in-depth: the caller (loss_on_trajectory) feeds the net a SAFE legal mask so pi/log_pi
    # are finite on every row. The nan_to_num guards below remain for direct callers that may pass a
    # fully-masked row (which would make pi/log_pi nan); invalid rows are masked out by valid_f anyway.
    logp_new = torch.nan_to_num((action_oh * log_pi).sum(-1))      # [T,B]
    mu_a = (action_oh * behavior_policy).sum(-1).clamp_min(eps)    # [T,B]
    logp_old = torch.log(mu_a)
    ratio = torch.exp((logp_new - logp_old).clamp(-20.0, 20.0))   # [T,B]

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    surrogate = torch.minimum(unclipped, clipped)                 # [T,B]
    if trinal_clip and trinal_clip > 0:                           # AlphaHoldem dual-clip on negative adv
        dual = trinal_clip * advantage
        surrogate = torch.where(advantage < 0.0, torch.maximum(surrogate, dual), surrogate)
    policy_loss = -(torch.nan_to_num(surrogate) * valid_f).sum() / m

    value_loss = (((v_flat - G_actor) ** 2) * valid_f).sum() / m

    pi_log = torch.nan_to_num(pi.clamp_min(eps).log())            # guard log on all-illegal rows
    entropy = -(pi_log * pi * legal.to(pi.dtype)).sum(-1)         # [T,B]
    entropy = torch.nan_to_num(entropy)
    entropy_term = (entropy * valid_f).sum() / m

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_term
    logs = {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_term.detach().cpu()),
        "mean_ratio": float(((ratio * valid_f).sum() / m).detach().cpu()),
    }
    return loss, logs


class PPOSolver:
    """PPO learner with the SAME collector + network contract as RNaDSolver (controlled A/B)."""

    def __init__(self, config: PPOConfig, collector, device="cpu"):
        self.config = config
        self.collector = collector
        self.device = torch.device(device)
        self.num_players = collector.n_players
        self.n_actions = collector.n_actions
        self.obs_dim = collector.obs_dim
        torch.manual_seed(config.seed)
        self._rng = np.random.RandomState(config.seed)
        self.net = RNaDNetwork(self.obs_dim, self.n_actions, config.policy_network_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.learner_steps = 0
        self._collector_is_gpu = bool(getattr(collector, "is_gpu", False))
        if self._collector_is_gpu:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(config.seed)

    @torch.no_grad()
    def _policy_fn(self, obs_np, legal_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(legal_np, dtype=torch.float32, device=self.device)
        pi, _, _, _ = self.net(obs, legal)
        return pi.detach().cpu().numpy()

    @torch.no_grad()
    def _policy_fn_t(self, obs_t, legal_t):
        pi, _, _, _ = self.net(obs_t, legal_t)
        return pi

    def loss_on_trajectory(self, traj):
        # Dead rows (opponent/terminal padding from the seat-aware collector) can carry an all-illegal
        # mask; the network's legal_policy/legal_log_policy return NaN on a fully-masked row. Those rows
        # are masked out of every loss TERM, but the NaN still propagates through autograd (0 * NaN =
        # NaN) and poisons every parameter on the first optimizer step. Feed the net a SAFE mask
        # (uniform on fully-masked rows; valid rows always have >=1 legal action, so they are unchanged).
        # This keeps the shared R-NaD loss stack (functional.py) and network untouched — the fix is
        # confined to the PPO arm, and the value head does not depend on the legal mask.
        safe_legal = torch.where(traj.legal.sum(-1, keepdim=True) > 0, traj.legal,
                                 torch.ones_like(traj.legal))
        pi, v, log_pi, _ = self.net(traj.obs, safe_legal)
        return compute_ppo_loss(
            pi, log_pi, v,
            valid=traj.valid, player_id=traj.player_id, legal=safe_legal,
            action_oh=traj.action_oh, behavior_policy=traj.policy, rewards=traj.rewards,
            num_players=self.num_players, clip_epsilon=self.config.clip_epsilon,
            trinal_clip=self.config.trinal_clip, value_coef=self.config.value_coef,
            entropy_coef=self.config.entropy_coef,
        )

    def step(self) -> dict:
        if self._collector_is_gpu:
            traj = self.collector.collect(self._policy_fn_t, self.config.batch_size,
                                          self.config.trajectory_max, self._gen)
        else:
            traj = self.collector.collect(self._policy_fn, self.config.batch_size,
                                          self.config.trajectory_max, self._rng).to(self.device)
        logs = {}
        for _ in range(self.config.ppo_epochs):  # reuse the on-policy batch a few epochs (standard PPO)
            self.optimizer.zero_grad(set_to_none=True)
            loss, logs = self.loss_on_trajectory(traj)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
            self.optimizer.step()
        self.learner_steps += 1
        logs.update(loss=float(loss.detach().cpu()), learner_steps=self.learner_steps)
        return logs

    @torch.no_grad()
    def action_probabilities(self, obs_np, legal_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(legal_np, dtype=torch.float32, device=self.device)
        pi, _, _, _ = self.net(obs, legal)
        return pi.detach().cpu().numpy()
