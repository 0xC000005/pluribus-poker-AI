"""Seat-aware pyspiel collector for the small-NLHE GO/NO-GO (frozen-opponent anti-collapse).

The shared fast tree collector (collector.py) is pure self-play: every decision is the learner's, both
seats train the one net. AlphaHoldem-style PPO needs anti-collapse — one seat plays a FROZEN past snapshot
(opponent pool) so the learner does not chase a moving copy of itself. To avoid PARITY-GUARD-risky surgery
on the shared collector, this is a SEPARATE, small collector that walks pyspiel states directly (small-NLHE
is 637 nodes / max-len 7, so per-state stepping at batch 256 is trivially fast on CPU).

It produces the SAME Trajectory contract as collector.py, with one difference used by the loss: decisions
made by the FROZEN OPPONENT are marked valid=0, so only the LEARNER's seat decisions train the net. Rewards
are the full per-player terminal returns (the learner's advantage uses its own seat's return).

Pool: a FIFO of recent learner snapshots (a standard fictitious-play-style anti-collapse). The governance
bundle said "K-best"; FIFO-recent is a justified simplification (no cheap per-snapshot fitness on a toy
game) and is recorded as such. Empty pool => self-play (opponent = current learner), matching the base case.
"""
from __future__ import annotations

import numpy as np
import torch

from poker_ai.rnad.collector import Trajectory


class SeatAwarePyspielCollector:
    is_gpu = False

    def __init__(self, game, state_representation: str = "info_set", device="cpu"):
        import pyspiel
        self.game = pyspiel.load_game(game) if isinstance(game, str) else game
        self.device = device
        self.n_players = self.game.num_players()
        self.n_actions = self.game.num_distinct_actions()
        self.use_obs = state_representation == "observation"
        self.obs_dim = (self.game.observation_tensor_size() if self.use_obs
                        else self.game.information_state_tensor_size())

    def _obs_legal(self, state):
        obs = (state.observation_tensor() if self.use_obs else state.information_state_tensor())
        return np.asarray(obs, np.float32), np.asarray(state.legal_actions_mask(), np.float32)

    def collect(self, learner_policy_fn, batch_size, trajectory_max, rng, *, opponent_policy_fn=None):
        """learner_policy_fn / opponent_policy_fn: (obs[K,D], legal[K,A]) -> pi[K,A] numpy.

        If opponent_policy_fn is None, both seats use the learner (self-play). Each game b has a fixed
        learner_seat = b % 2. Records only the LEARNER's decisions as valid (opponent steps -> valid=0).
        """
        B, A, P, D, T = batch_size, self.n_actions, self.n_players, self.obs_dim, trajectory_max
        opp = opponent_policy_fn or learner_policy_fn
        learner_seat = np.arange(B) % P

        obs_buf = np.zeros((T, B, D), np.float32)
        legal_buf = np.zeros((T, B, A), np.float32)
        pid_buf = np.zeros((T, B), np.float32)
        valid_buf = np.zeros((T, B), np.float32)
        rew_buf = np.zeros((T, B, P), np.float32)
        aoh_buf = np.zeros((T, B, A), np.float32)
        pol_buf = np.zeros((T, B, A), np.float32)

        states = [self.game.new_initial_state() for _ in range(B)]
        # resolve initial chance
        for s in states:
            while s.is_chance_node():
                outs, ps = zip(*s.chance_outcomes())
                s.apply_action(int(rng.choice(outs, p=np.asarray(ps) / np.sum(ps))))

        for t in range(T):
            # partition live decision states by which net acts (learner seat vs opponent seat)
            obs_t = np.zeros((B, D), np.float32)
            legal_t = np.zeros((B, A), np.float32)
            pid_t = np.full(B, -1.0, np.float32)
            live = np.zeros(B, bool)
            is_learner = np.zeros(B, bool)
            for b, s in enumerate(states):
                if s.is_terminal():
                    rew_buf[t, b] = np.asarray(s.returns(), np.float32)
                    continue
                live[b] = True
                cur = s.current_player()
                pid_t[b] = cur
                o, l = self._obs_legal(s)
                obs_t[b] = o; legal_t[b] = l
                is_learner[b] = (cur == learner_seat[b])

            if not live.any():
                # carry terminal rewards forward (already set), nothing to act. valid stays 0; but write a
                # FINITE uniform policy (not the all-zero init) so no row sums to 0 / NaNs downstream.
                obs_buf[t] = obs_t; legal_buf[t] = legal_t; pid_buf[t] = pid_t
                pol_buf[t] = 1.0 / A
                continue

            # query both nets on the full batch (cheap), then pick per row.
            # Initialize EVERY row to a FINITE uniform-over-(its legal, or all-actions for terminal/dummy)
            # so no all-zero policy row ever exists -> downstream log/ratio math can't NaN on masked rows.
            denom = legal_t.sum(-1, keepdims=True)
            uni = np.where(denom > 0, legal_t / np.maximum(denom, 1.0), 1.0 / A).astype(np.float64)
            pi = uni.copy()
            lrows = np.flatnonzero(live & is_learner)
            orows = np.flatnonzero(live & ~is_learner)
            if lrows.size:
                pl = np.asarray(learner_policy_fn(obs_t[lrows], legal_t[lrows]), np.float64)
                pi[lrows] = pl / pl.sum(-1, keepdims=True)
            if orows.size:
                po = np.asarray(opp(obs_t[orows], legal_t[orows]), np.float64)
                pi[orows] = po / po.sum(-1, keepdims=True)

            # sample + apply per live row, intersecting with the state's ACTUAL legal_actions() so the
            # ACPC engine never rejects an action (a masked-legal id can still be invalid at that state).
            aoh = np.zeros((B, A), np.float32)
            obs_buf[t] = obs_t
            legal_buf[t] = legal_t
            pid_buf[t] = pid_t
            valid_buf[t] = (live & is_learner).astype(np.float32)  # ONLY learner decisions train
            pol_buf[t] = pi.astype(np.float32)
            for b in np.flatnonzero(live):
                s = states[b]
                legal_ids = s.legal_actions()
                p = pi[b, legal_ids].astype(np.float64)
                tot = p.sum()
                p = (p / tot) if tot > 1e-12 else np.full(len(legal_ids), 1.0 / len(legal_ids))
                cdf = np.cumsum(p); cdf[-1] = 1.0
                a_id = int(legal_ids[int((rng.random() < cdf).argmax())])
                aoh[b, a_id] = 1.0
                s.apply_action(a_id)
                while s.is_chance_node():
                    outs, ps = zip(*s.chance_outcomes())
                    s.apply_action(int(rng.choice(outs, p=np.asarray(ps) / np.sum(ps))))
                if s.is_terminal():
                    rew_buf[t, b] = np.asarray(s.returns(), np.float32)
            aoh_buf[t] = aoh

        dev = self.device
        tt = lambda x: torch.from_numpy(x).to(dev)
        return Trajectory(obs=tt(obs_buf), legal=tt(legal_buf), player_id=tt(pid_buf),
                          valid=tt(valid_buf), rewards=tt(rew_buf), action_oh=tt(aoh_buf), policy=tt(pol_buf))
