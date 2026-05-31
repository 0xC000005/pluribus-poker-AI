"""Pure-functional R-NaD loss stack in PyTorch — 1:1 port of scripts/vendor/rnad.py.

Every function here mirrors a canonical DeepMind R-NaD function (jax) with the SAME math,
so the port can be numerically parity-tested against the reference (test/unit/test_rnad_torch.py).
Reference line anchors are given per function. Nothing here is game-specific: all tensors are
abstract batched trajectory tensors of shape [T, B, ...] (T=time, B=batch, A=actions, P=players).

Conventions matched to the reference:
- legal masks are {0,1} float/int; logits over A actions.
- v-trace: eta=0.2, c=1.0, rho=inf, lambda=1.0, gamma=1.0 (rnad.py:415,823-826).
- nerd loss: clip=10000, threshold(beta)=2.0 (rnad.py callsite :843-844).
- target EMA speed 0.001; reference roll-forward on EntropySchedule boundaries.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Policy / log-policy on legal actions  (rnad.py:262-295)
# --------------------------------------------------------------------------------------
def legal_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """Softmax that respects legal_actions. Mirrors _legal_policy (rnad.py:262)."""
    legal = legal_actions.to(logits.dtype)
    l_min = logits.min(dim=-1, keepdim=True).values
    logits = torch.where(legal > 0, logits, l_min)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = logits * legal
    exp_logits = torch.where(legal > 0, torch.exp(logits), torch.zeros_like(logits))
    exp_logits_sum = exp_logits.sum(dim=-1, keepdim=True)
    return exp_logits / exp_logits_sum


def legal_log_policy(logits: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
    """Log of the policy on legal actions, 0 on illegal. Mirrors legal_log_policy (rnad.py:276)."""
    legal = legal_actions.to(logits.dtype)
    # logits_masked has illegal actions set to -inf via + log(legal) (log(0) = -inf).
    logits_masked = logits + torch.log(legal)
    max_legal_logit = logits_masked.max(dim=-1, keepdim=True).values
    logits_masked = logits_masked - max_legal_logit
    exp_logits_masked = torch.exp(logits_masked)  # 0 for illegal
    baseline = torch.log(exp_logits_masked.sum(dim=-1, keepdim=True))
    # multiply by legal to avoid 0 * -inf = nan on illegal actions.
    log_policy = legal * (logits - max_legal_logit - baseline)
    return log_policy


# --------------------------------------------------------------------------------------
# v-trace helpers  (rnad.py:298-387)
# --------------------------------------------------------------------------------------
def player_others(player_ids: torch.Tensor, valid: torch.Tensor, player: int) -> torch.Tensor:
    """+1 for current player, -1 for others, *valid; trailing dim added. (rnad.py:298)."""
    current = (player_ids == player).to(torch.int32)
    res = 2 * current - 1
    res = res * valid.to(res.dtype)
    return res.unsqueeze(-1)


def policy_ratio(pi: torch.Tensor, mu: torch.Tensor, actions_oh: torch.Tensor,
                 valid: torch.Tensor) -> torch.Tensor:
    """pi/mu on the chosen action (1 on invalid states). Mirrors _policy_ratio (rnad.py:318)."""
    valid_f = valid.to(pi.dtype)

    def _select(p):
        return (actions_oh * p).sum(dim=-1) * valid_f + (1.0 - valid_f)

    return _select(pi) / _select(mu)


def has_played(valid: torch.Tensor, player_id: torch.Tensor, player: int) -> torch.Tensor:
    """Reverse scan mask: states that have a future state for `player`. Mirrors _has_played (rnad.py:358).

    valid, player_id: [T, B]. Returns [T, B]. Carry is per-batch [B].
    """
    T = valid.shape[0]
    carry = torch.zeros_like(player_id[-1])  # [B]
    out = [None] * T
    for t in range(T - 1, -1, -1):  # reverse=True
        v = valid[t]
        pid = player_id[t]
        is_player = (pid == player)
        # our_res = ones, opp_res = carry, reset_res = 0
        our_res = torch.ones_like(pid)
        opp_res = carry
        res = torch.where(v > 0, torch.where(is_player, our_res, opp_res), torch.zeros_like(carry))
        # carry update: our_carry=carry, opp_carry=carry, reset_carry=0
        carry = torch.where(v > 0, carry, torch.zeros_like(carry))
        out[t] = res
    return torch.stack(out, dim=0)


# --------------------------------------------------------------------------------------
# Mixed-player V-trace  (rnad.py:397-508) — the genuinely tricky piece.
# Reverse scan with a 3-way (our / opp / reset) carry, run ONCE PER PLAYER.
# --------------------------------------------------------------------------------------
def v_trace(
    v: torch.Tensor,              # [T, B, 1]  target-net value
    valid: torch.Tensor,          # [T, B]
    player_id: torch.Tensor,      # [T, B]
    acting_policy: torch.Tensor,  # [T, B, A]  mu (behavior policy)
    merged_policy: torch.Tensor,  # [T, B, A]  pi (current, finetuned)
    merged_log_policy: torch.Tensor,  # [T, B, A]  log_policy_reg (reward transform)
    player_others_t: torch.Tensor,    # [T, B, 1]
    actions_oh: torch.Tensor,     # [T, B, A]
    reward: torch.Tensor,         # [T, B]  reward for `player`
    player: int,
    *,
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom V-trace for mixed-player trajectories. Mirrors v_trace (rnad.py:397)."""
    gamma = 1.0
    hp = has_played(valid, player_id, player)  # [T, B]

    pr = policy_ratio(merged_policy, acting_policy, actions_oh, valid)   # [T, B]
    inv_mu = policy_ratio(torch.ones_like(merged_policy), acting_policy, actions_oh, valid)  # [T, B]

    # eta_reg_entropy: [T, B]   (rnad.py:423-425)
    eta_reg_entropy = (-eta
                       * (merged_policy * merged_log_policy).sum(dim=-1)
                       * player_others_t.squeeze(-1))
    # eta_log_policy: [T, B, A]  (rnad.py:426)
    eta_log_policy = -eta * merged_log_policy * player_others_t

    T = v.shape[0]
    # Carry components (all per-batch, matching the reference LoopVTraceCarry):
    c_reward = torch.zeros_like(reward[-1])                 # [B]
    c_reward_uncorrected = torch.zeros_like(reward[-1])     # [B]
    c_next_value = torch.zeros_like(v[-1])                  # [B, 1]
    c_next_v_target = torch.zeros_like(v[-1])               # [B, 1]
    c_importance_sampling = torch.ones_like(pr[-1])         # [B]

    v_target_out = [None] * T
    learning_output_out = [None] * T

    for t in range(T - 1, -1, -1):  # reverse=True
        cs = pr[t]                          # [B]
        pid = player_id[t]                  # [B]
        v_t = v[t]                          # [B, 1]
        rew = reward[t]                     # [B]
        ere = eta_reg_entropy[t]            # [B]
        val = valid[t]                      # [B]
        imu = inv_mu[t]                     # [B]
        aoh = actions_oh[t]                 # [B, A]
        elp = eta_log_policy[t]            # [B, A]

        reward_uncorrected = rew + gamma * c_reward_uncorrected + ere   # [B]
        discounted_reward = rew + gamma * c_reward                      # [B]

        cs_is = cs * c_importance_sampling                              # [B]
        min_rho = torch.clamp(cs_is, max=rho).unsqueeze(-1)            # [B,1]
        min_c = torch.clamp(cs_is, max=c).unsqueeze(-1)               # [B,1]

        # V-target (our): [B, 1]   (rnad.py:455-461)
        our_v_target = (
            v_t
            + min_rho * (reward_uncorrected.unsqueeze(-1) + gamma * c_next_value - v_t)
            + lambda_ * min_c * gamma * (c_next_v_target - c_next_value)
        )

        # Learning output (our): [B, A]   (rnad.py:467-472)
        our_learning_output = (
            v_t
            + elp
            + aoh * imu.unsqueeze(-1)
            * (discounted_reward.unsqueeze(-1)
               + gamma * c_importance_sampling.unsqueeze(-1) * c_next_v_target - v_t)
        )

        is_player = (pid == player)
        valid_b = (val > 0)

        # Broadcast helpers for selecting carries / outputs.
        vb1 = valid_b.unsqueeze(-1)        # [B,1]
        ip1 = is_player.unsqueeze(-1)      # [B,1]

        zeros_t1 = torch.zeros_like(our_v_target)

        # ---- outputs (v_target, learning_output) ----
        v_target_sel = torch.where(vb1, torch.where(ip1, our_v_target, zeros_t1), zeros_t1)
        lo_sel = torch.where(vb1, torch.where(ip1, our_learning_output, torch.zeros_like(our_learning_output)),
                             torch.zeros_like(our_learning_output))
        v_target_out[t] = v_target_sel
        learning_output_out[t] = lo_sel

        # ---- carry update ----
        # our_carry
        our_reward = torch.zeros_like(c_reward)
        our_reward_unc = torch.zeros_like(c_reward_uncorrected)
        our_next_value = v_t
        our_next_v_target = our_v_target
        our_is = torch.ones_like(c_importance_sampling)
        # opp_carry
        opp_reward = ere + cs * discounted_reward
        opp_reward_unc = reward_uncorrected
        opp_next_value = gamma * c_next_value
        opp_next_v_target = gamma * c_next_v_target
        opp_is = cs * c_importance_sampling
        # reset_carry = init state
        reset_reward = torch.zeros_like(c_reward)
        reset_reward_unc = torch.zeros_like(c_reward_uncorrected)
        reset_next_value = torch.zeros_like(c_next_value)
        reset_next_v_target = torch.zeros_like(c_next_v_target)
        reset_is = torch.ones_like(c_importance_sampling)

        # select per component: valid ? (is_player ? our : opp) : reset
        def sel(our, opp, reset, two_d):
            pred_v = vb1 if two_d else valid_b
            pred_p = ip1 if two_d else is_player
            return torch.where(pred_v, torch.where(pred_p, our, opp), reset)

        c_reward = sel(our_reward, opp_reward, reset_reward, False)
        c_reward_uncorrected = sel(our_reward_unc, opp_reward_unc, reset_reward_unc, False)
        c_next_value = sel(our_next_value, opp_next_value, reset_next_value, True)
        c_next_v_target = sel(our_next_v_target, opp_next_v_target, reset_next_v_target, True)
        c_importance_sampling = sel(our_is, opp_is, reset_is, False)

    v_target = torch.stack(v_target_out, dim=0)
    learning_output = torch.stack(learning_output_out, dim=0)
    return v_target, hp, learning_output


# --------------------------------------------------------------------------------------
# Losses  (rnad.py:511-589)
# --------------------------------------------------------------------------------------
def get_loss_v(v_list: Sequence[torch.Tensor], v_target_list: Sequence[torch.Tensor],
               mask_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Critic MSE, per-player masked. Mirrors get_loss_v (rnad.py:511)."""
    total = None
    for v_n, v_target, mask in zip(v_list, v_target_list, mask_list):
        loss_v = mask.unsqueeze(-1) * (v_n - v_target.detach()) ** 2
        normalization = mask.sum()
        loss_v = loss_v.sum() / (normalization + (normalization == 0).to(normalization.dtype))
        total = loss_v if total is None else total + loss_v
    return total


def apply_force_with_threshold(decision_outputs: torch.Tensor, force: torch.Tensor,
                               threshold: float, threshold_center: torch.Tensor) -> torch.Tensor:
    """NeuRD thresholded force. Mirrors apply_force_with_threshold (rnad.py:532)."""
    can_decrease = (decision_outputs - threshold_center) > -threshold
    can_increase = (decision_outputs - threshold_center) < threshold
    force_negative = torch.clamp(force, max=0.0)
    force_positive = torch.clamp(force, min=0.0)
    clipped_force = can_decrease.to(force.dtype) * force_negative + can_increase.to(force.dtype) * force_positive
    return decision_outputs * clipped_force.detach()


def renormalize(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum(loss*mask)/sum(mask). Mirrors renormalize (rnad.py:545)."""
    loss = (loss * mask).sum()
    normalization = mask.sum()
    return loss / (normalization + (normalization == 0).to(normalization.dtype))


def get_loss_nerd(logit_list: Sequence[torch.Tensor], policy_list: Sequence[torch.Tensor],
                  q_vr_list: Sequence[torch.Tensor], valid: torch.Tensor,
                  player_ids: torch.Tensor, legal_actions: torch.Tensor,
                  importance_sampling_correction: Sequence[torch.Tensor],
                  clip: float = 100.0, threshold: float = 2.0) -> torch.Tensor:
    """NeuRD policy loss. Mirrors get_loss_nerd (rnad.py:553)."""
    legal = legal_actions.to(logit_list[0].dtype)
    num_valid_actions = legal.sum(dim=-1, keepdim=True)
    total = None
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
            zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)):
        adv_pi = q_vr - (pi * q_vr).sum(dim=-1, keepdim=True)
        adv_pi = is_c * adv_pi
        adv_pi = torch.clamp(adv_pi, min=-clip, max=clip)
        adv_pi = adv_pi.detach()

        valid_logit_sum = (logit_pi * legal).sum(dim=-1, keepdim=True)
        mean_logit = valid_logit_sum / num_valid_actions
        logits = logit_pi - mean_logit

        threshold_center = torch.zeros_like(logits)
        nerd_loss = (legal * apply_force_with_threshold(logits, adv_pi, threshold, threshold_center)).sum(dim=-1)
        nerd_loss = -renormalize(nerd_loss, valid.to(nerd_loss.dtype) * (player_ids == k).to(nerd_loss.dtype))
        total = nerd_loss if total is None else total + nerd_loss
    return total


# --------------------------------------------------------------------------------------
# Entropy schedule  (rnad.py:40-133) — numpy port (control logic, not autodiff).
# --------------------------------------------------------------------------------------
class EntropySchedule:
    """Schedule of (alpha, update_target_net) over learner steps. Mirrors EntropySchedule (rnad.py:40)."""

    def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
        if len(repeats) != len(sizes):
            raise ValueError("`repeats` must be parallel to `sizes`.")
        if not sizes:
            raise ValueError("`sizes` and `repeats` must not be empty.")
        if any(r <= 0 for r in repeats):
            raise ValueError("All repeat values must be strictly positive")
        if repeats[-1] != 1:
            raise ValueError("The last value in `repeats` must be equal to 1.")
        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])
        self.schedule = np.array(schedule, dtype=np.int64)

    def __call__(self, learner_step: int) -> tuple[float, bool]:
        sched = self.schedule
        last_size = int(sched[-1] - sched[-2])
        last_start = int(sched[-1] + (learner_step - sched[-1]) // last_size * last_size)
        # within-schedule case
        le = sched[sched <= learner_step]
        start = int(le.max()) if le.size else 0
        gt = sched[learner_step < sched]
        finish = int(gt.min()) if gt.size else int(sched[-1])
        size = finish - start
        beyond = sched[-1] <= learner_step
        iteration_start = last_start if beyond else start
        iteration_size = last_size if beyond else size
        update_target_net = bool(learner_step > 0 and (learner_step == iteration_start + iteration_size - 1))
        alpha = min((2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)
        return float(alpha), update_target_net


# --------------------------------------------------------------------------------------
# Full R-NaD loss  (rnad.py:792-845) — assembles the above over a TimeStep batch.
# --------------------------------------------------------------------------------------
def compute_rnad_loss(
    net_out, target_out, prev_out, prev_out_,
    *,
    valid: torch.Tensor,          # [T, B]
    player_id: torch.Tensor,      # [T, B]
    legal: torch.Tensor,          # [T, B, A]
    acting_policy: torch.Tensor,  # [T, B, A]  mu sampled at collection
    action_oh: torch.Tensor,      # [T, B, A]
    rewards: torch.Tensor,        # [T, B, P]
    alpha: float,
    num_players: int,
    eta: float,
    c_vtrace: float,
    nerd_clip: float,
    nerd_beta: float,
) -> torch.Tensor:
    """Total R-NaD loss = loss_v + loss_nerd. Mirrors RNaDSolver.loss (rnad.py:792).

    *_out are tuples (pi, v, log_pi, logit) from each of the 4 network copies, evaluated on the
    same [T, B, obs] observations. `pi`/`log_pi`/`logit`: [T,B,A]; `v`: [T,B,1].
    """
    pi, v, log_pi, logit = net_out
    _, v_target, _, _ = target_out
    _, _, log_pi_prev, _ = prev_out
    _, _, log_pi_prev_, _ = prev_out_

    # Reward transform (convex mix of two references by alpha). rnad.py:807
    log_policy_reg = log_pi - (alpha * log_pi_prev + (1.0 - alpha) * log_pi_prev_)

    v_target_list, has_played_list, policy_target_list = [], [], []
    for player in range(num_players):
        reward_p = rewards[:, :, player]  # [T, B]
        v_target_, hp, policy_target_ = v_trace(
            v_target, valid, player_id, acting_policy, pi, log_policy_reg,
            player_others(player_id, valid, player), action_oh, reward_p, player,
            eta=eta, lambda_=1.0, c=c_vtrace, rho=float("inf"),
        )
        v_target_list.append(v_target_)
        has_played_list.append(hp)
        policy_target_list.append(policy_target_)

    loss_v = get_loss_v([v] * num_players, v_target_list, has_played_list)

    is_vector = torch.ones_like(valid).unsqueeze(-1)  # [T,B,1]
    is_correction = [is_vector] * num_players
    loss_nerd = get_loss_nerd(
        [logit] * num_players, [pi] * num_players, policy_target_list,
        valid, player_id, legal, is_correction,
        clip=nerd_clip, threshold=nerd_beta,
    )
    return loss_v + loss_nerd
