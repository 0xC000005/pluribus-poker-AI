"""Trajectory collectors for R-NaD — the pluggable, game-specific layer.

`Trajectory` is the abstract batched [T, B, ...] tensor bundle the game-agnostic learner consumes.
`LeducTreeCollector` is the first adapter: a precomputed-tree sampler for small OpenSpiel games
(leduc_poker / kuhn_poker). It builds the game tree ONCE into flat numpy arrays, then samples B
trajectories per step with FULLY VECTORIZED numpy gathers — NO per-state pyspiel cloning and NO
per-batch-element Python loop in the hot path. That vectorization is the genuine collection speedup
over the vendored solver's per-state python collector (which clones pyspiel states every step).

The cuda-NLHE collector (later) implements the same `collect()` contract over the numba GameBatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Trajectory:
    """Batched rollout tensors, all leading dims [T, B, ...]. Mirrors the reference TimeStep."""
    obs: torch.Tensor          # [T, B, obs_dim]
    legal: torch.Tensor        # [T, B, A]   {0,1}
    player_id: torch.Tensor    # [T, B]      float (current player; dummy when invalid)
    valid: torch.Tensor        # [T, B]      {0,1}
    rewards: torch.Tensor      # [T, B, P]   reward visible AFTER the step (== env.rewards of next state)
    action_oh: torch.Tensor    # [T, B, A]   sampled action one-hot
    policy: torch.Tensor       # [T, B, A]   behavior policy mu at sampling time

    def to(self, device):
        return Trajectory(**{k: v.to(device) for k, v in self.__dict__.items()})


# Node kinds.
_TERM, _CHANCE, _DEC = 0, 1, 2


class _LeducTree:
    """Precomputed OpenSpiel game tree built once into python lists, then arrayified by the collector."""

    def __init__(self, game, state_representation: str = "info_set"):
        self.game = game
        self.n_actions = game.num_distinct_actions()
        self.n_players = game.num_players()
        self.use_obs = state_representation == "observation"
        self.obs_dim = (game.observation_tensor_size() if self.use_obs
                        else game.information_state_tensor_size())

        self.kind: list[int] = []
        self.returns: list = []          # TERM: np.array [P]
        self.chance_children: list = []  # CHANCE: list[(prob, child_id)]
        self.dec_player: list = []       # DEC: current player
        self.dec_obs: list = []          # DEC: obs vector
        self.dec_legal: list = []        # DEC: legal mask [A]
        self.dec_actions: list = []      # DEC: list[(action_id, child_id)]
        self.root = self._build(game.new_initial_state())

    def _new_node(self, kind):
        nid = len(self.kind)
        self.kind.append(kind)
        self.returns.append(None)
        self.chance_children.append(None)
        self.dec_player.append(-1)
        self.dec_obs.append(None)
        self.dec_legal.append(None)
        self.dec_actions.append(None)
        return nid

    def _build(self, state) -> int:
        if state.is_terminal():
            nid = self._new_node(_TERM)
            self.returns[nid] = np.asarray(state.returns(), dtype=np.float64)
            return nid
        if state.is_chance_node():
            nid = self._new_node(_CHANCE)
            kids = []
            for a, p in state.chance_outcomes():
                c = state.clone(); c.apply_action(a)
                kids.append((float(p), self._build(c)))
            self.chance_children[nid] = kids
            return nid
        nid = self._new_node(_DEC)
        self.dec_player[nid] = state.current_player()
        obs = state.observation_tensor() if self.use_obs else state.information_state_tensor()
        self.dec_obs[nid] = np.asarray(obs, dtype=np.float64)
        self.dec_legal[nid] = np.asarray(state.legal_actions_mask(), dtype=np.float64)
        kids = []
        for a in state.legal_actions():
            c = state.clone(); c.apply_action(a)
            kids.append((int(a), self._build(c)))
        self.dec_actions[nid] = kids
        return nid


class LeducTreeCollector:
    """Vectorized trajectory sampler for small OpenSpiel games via a precomputed flat-array tree."""

    def __init__(self, game, state_representation: str = "info_set", device="cpu"):
        # `game` may be an OpenSpiel game NAME (str) or a pre-built pyspiel game object. The latter is
        # needed for parametrized games (e.g. small-NLHE universal_poker) whose mixed-type params cannot
        # be expressed as a single load_game(name) string.
        import pyspiel
        self.game = pyspiel.load_game(game) if isinstance(game, str) else game
        tree = _LeducTree(self.game, state_representation)
        self.n_actions = A = tree.n_actions
        self.n_players = P = tree.n_players
        self.obs_dim = D = tree.obs_dim
        self.device = device
        N = len(tree.kind)

        # ---- flat numpy node tables (built once) ----
        self.kind_arr = np.asarray(tree.kind, dtype=np.int8)                  # [N]
        self.obs_arr = np.zeros((N, D), dtype=np.float32)                     # [N, D]
        self.legal_arr = np.zeros((N, A), dtype=np.float32)                   # [N, A]
        self.player_arr = np.full(N, -1, dtype=np.float32)                    # [N]
        self.term_returns = np.zeros((N, P), dtype=np.float32)               # [N, P]
        self.dec_child = np.full((N, A), -1, dtype=np.int64)                  # [N, A] decision child
        # chance padded table
        max_outcomes = 1
        for n in range(N):
            if tree.kind[n] == _CHANCE:
                max_outcomes = max(max_outcomes, len(tree.chance_children[n]))
        self.chance_child = np.full((N, max_outcomes), -1, dtype=np.int64)    # [N, Kc]
        self.chance_prob = np.zeros((N, max_outcomes), dtype=np.float64)      # [N, Kc]

        for n in range(N):
            k = tree.kind[n]
            if k == _TERM:
                self.term_returns[n] = tree.returns[n]
            elif k == _CHANCE:
                for j, (p, cid) in enumerate(tree.chance_children[n]):
                    self.chance_child[n, j] = cid
                    self.chance_prob[n, j] = p
            else:  # DEC
                self.obs_arr[n] = tree.dec_obs[n]
                self.legal_arr[n] = tree.dec_legal[n]
                self.player_arr[n] = tree.dec_player[n]
                for a_id, cid in tree.dec_actions[n]:
                    self.dec_child[n, a_id] = cid

        # A fixed decision node used as the dummy/padding state for terminals (reference _ex_state).
        ex = tree.root
        while self.kind_arr[ex] != _DEC:
            ex = self.chance_child[ex, 0] if self.kind_arr[ex] == _CHANCE else ex
            if ex < 0:
                break
        self._ex_dec = int(ex)
        self.root = int(tree.root)

    def _resolve_chance_vec(self, nodes: np.ndarray, rng) -> np.ndarray:
        """Vectorized: for any node currently at a chance node, sample one outcome; repeat to fixpoint."""
        nodes = nodes.copy()
        while True:
            is_chance = self.kind_arr[nodes] == _CHANCE
            if not is_chance.any():
                return nodes
            idx = np.nonzero(is_chance)[0]
            cn = nodes[idx]                              # chance node ids
            probs = self.chance_prob[cn]                 # [m, Kc]
            probs = probs / probs.sum(axis=1, keepdims=True)
            cdf = np.cumsum(probs, axis=1)
            cdf[:, -1] = 1.0
            r = rng.random((len(idx), 1))
            choice = (r < cdf).argmax(axis=1)            # [m]
            nodes[idx] = self.chance_child[cn, choice]

    def collect(self, policy_fn, batch_size: int, trajectory_max: int, rng) -> Trajectory:
        """policy_fn(obs[B,D], legal[B,A]) -> pi[B,A] numpy. Returns a Trajectory on self.device."""
        B = batch_size
        A = self.n_actions
        P = self.n_players
        D = self.obs_dim
        T = trajectory_max

        nodes = self._resolve_chance_vec(np.full(B, self.root, dtype=np.int64), rng)

        obs_buf = np.zeros((T, B, D), dtype=np.float32)
        legal_buf = np.zeros((T, B, A), dtype=np.float32)
        pid_buf = np.zeros((T, B), dtype=np.float32)
        valid_buf = np.zeros((T, B), dtype=np.float32)
        rew_buf = np.zeros((T, B, P), dtype=np.float32)
        aoh_buf = np.zeros((T, B, A), dtype=np.float32)
        pol_buf = np.zeros((T, B, A), dtype=np.float32)

        ar = np.arange(B)
        for t in range(T):
            is_term = self.kind_arr[nodes] == _TERM           # [B]
            valid = ~is_term
            # safe index: terminals read the dummy decision node (ignored via valid=0), as in the reference.
            safe = np.where(valid, nodes, self._ex_dec)
            obs_t = self.obs_arr[safe]                         # [B, D]
            legal_t = self.legal_arr[safe]                     # [B, A]
            pid_t = self.player_arr[safe]                      # [B]

            pi = np.asarray(policy_fn(obs_t, legal_t), dtype=np.float64)
            pi = pi / pi.sum(axis=-1, keepdims=True)

            # vectorized categorical sampling (mirrors actor_step rnad.py:1023-1026)
            cdf = np.cumsum(pi, axis=-1)
            cdf[:, -1] = 1.0
            r = rng.random((B, 1))
            actions = (r < cdf).argmax(axis=-1)               # [B]
            aoh = np.zeros((B, A), dtype=np.float32)
            aoh[ar, actions] = 1.0

            # apply actions: decision child for valid nodes; terminals stay; then resolve chance
            raw_child = self.dec_child[safe, actions]          # [B]
            # guard against an (illegal) -1 child: fall back to current node
            raw_child = np.where(raw_child < 0, safe, raw_child)
            next_nodes = np.where(valid, raw_child, nodes)     # terminals stay put
            next_nodes = self._resolve_chance_vec(next_nodes, rng)

            next_is_term = self.kind_arr[next_nodes] == _TERM
            next_rewards = np.where(next_is_term[:, None], self.term_returns[next_nodes],
                                    np.zeros((B, P), dtype=np.float32))

            obs_buf[t] = obs_t
            legal_buf[t] = legal_t
            pid_buf[t] = pid_t
            valid_buf[t] = valid.astype(np.float32)
            rew_buf[t] = next_rewards
            aoh_buf[t] = aoh
            pol_buf[t] = pi.astype(np.float32)

            nodes = next_nodes

        dev = self.device
        return Trajectory(
            obs=torch.from_numpy(obs_buf).to(dev),
            legal=torch.from_numpy(legal_buf).to(dev),
            player_id=torch.from_numpy(pid_buf).to(dev),
            valid=torch.from_numpy(valid_buf).to(dev),
            rewards=torch.from_numpy(rew_buf).to(dev),
            action_oh=torch.from_numpy(aoh_buf).to(dev),
            policy=torch.from_numpy(pol_buf).to(dev),
        )


class GPUTreeCollector:
    """GPU-RESIDENT precomputed-tree collector — the whole rollout runs on `device` with NO host sync.

    Stores the flat tree tables (built once by `_LeducTree`) as DEVICE tensors and advances B cursors
    with batched torch gathers. Action + chance sampling use inverse-CDF with `torch.rand` (launch-light
    and CUDA-graph-capturable; avoids torch.multinomial host-sync). Chance is resolved with a FIXED
    2-iteration unroll (measured max consecutive chance depth = 2 for kuhn AND leduc) so there is NO
    data-dependent `.any()`/`.item()` host sync. Returns a Trajectory whose tensors are already on device.

    Semantics match `LeducTreeCollector` EXACTLY (so v-trace targets are correct): at timestep t we record
    the prev env_step (obs/legal/player/valid) + the sampled action/policy + the NEXT state's rewards
    (mirrors rnad.py:1054-1055 / collector.py:211-213). RNG differs from the numpy collector, so GPU vs
    CPU trajectories are DISTRIBUTIONALLY equal, not bitwise.

    `collect(policy_fn_t, batch_size, trajectory_max, gen)` where `policy_fn_t(obs_t[B,D], legal_t[B,A])
    -> pi_t[B,A]` is a TORCH-NATIVE callable on `device` (no numpy round-trip), and `gen` is a
    torch.Generator on `device`.
    """

    is_gpu = True

    def __init__(self, game, state_representation: str = "info_set", device="cuda"):
        # `game` may be an OpenSpiel game NAME (str) or a pre-built pyspiel game object (needed for
        # parametrized games like small-NLHE universal_poker; see LeducTreeCollector for rationale).
        import pyspiel
        self.game = pyspiel.load_game(game) if isinstance(game, str) else game
        tree = _LeducTree(self.game, state_representation)
        self.n_actions = A = tree.n_actions
        self.n_players = P = tree.n_players
        self.obs_dim = D = tree.obs_dim
        self.device = torch.device(device)
        N = len(tree.kind)

        kind = np.asarray(tree.kind, dtype=np.int64)
        obs = np.zeros((N, D), dtype=np.float32)
        legal = np.zeros((N, A), dtype=np.float32)
        player = np.full(N, -1, dtype=np.float32)
        term_returns = np.zeros((N, P), dtype=np.float32)
        dec_child = np.full((N, A), -1, dtype=np.int64)
        max_outcomes = 1
        for n in range(N):
            if tree.kind[n] == _CHANCE:
                max_outcomes = max(max_outcomes, len(tree.chance_children[n]))
        chance_child = np.full((N, max_outcomes), -1, dtype=np.int64)
        chance_cdf = np.zeros((N, max_outcomes), dtype=np.float32)  # cumulative probs, last forced to 1
        for n in range(N):
            k = tree.kind[n]
            if k == _TERM:
                term_returns[n] = tree.returns[n]
            elif k == _CHANCE:
                kids = tree.chance_children[n]
                probs = np.array([p for p, _ in kids], dtype=np.float64)
                probs = probs / probs.sum()
                cdf = np.cumsum(probs)
                for j, (_, cid) in enumerate(kids):
                    chance_child[n, j] = cid
                    chance_cdf[n, j] = cdf[j]
                chance_cdf[n, len(kids) - 1:] = 1.0  # pad tail to 1 so inverse-CDF always lands
                # for unused padded slots, point child at the last real child (never selected: cdf already 1)
                for j in range(len(kids), max_outcomes):
                    chance_child[n, j] = chance_child[n, len(kids) - 1]
            else:
                obs[n] = tree.dec_obs[n]
                legal[n] = tree.dec_legal[n]
                player[n] = tree.dec_player[n]
                for a_id, cid in tree.dec_actions[n]:
                    dec_child[n, a_id] = cid

        ex = tree.root
        while kind[ex] != _DEC:
            ex = int(chance_child[ex, 0]) if kind[ex] == _CHANCE else ex
            if ex < 0:
                break
        self._ex_dec = int(ex)
        self.root = int(tree.root)

        # Derive the true max consecutive-chance run length (DFS over chained _CHANCE nodes between
        # decisions) and use it as the fixed unroll bound — NOT a hard-coded magic number. A debug
        # assert (verify_no_residual_chance) guarantees no chance node survives the unroll.
        memo: dict[int, int] = {}

        def _chance_run(nid: int) -> int:
            if nid in memo:
                return memo[nid]
            if tree.kind[nid] != _CHANCE:
                memo[nid] = 0
                return 0
            memo[nid] = 1 + max(_chance_run(c) for _, c in tree.chance_children[nid])
            return memo[nid]

        self._max_chance_depth = max((_chance_run(n) for n in range(N)), default=0)

        dev = self.device
        self.kind = torch.as_tensor(kind, device=dev)
        self.obs = torch.as_tensor(obs, device=dev)
        self.legal = torch.as_tensor(legal, device=dev)
        self.player = torch.as_tensor(player, device=dev)
        self.term_returns = torch.as_tensor(term_returns, device=dev)
        self.dec_child = torch.as_tensor(dec_child, device=dev)
        self.chance_child = torch.as_tensor(chance_child, device=dev)
        self.chance_cdf = torch.as_tensor(chance_cdf, device=dev)
        self._TERM_t = int(_TERM)
        self._CHANCE_t = int(_CHANCE)

    def _resolve_chance(self, nodes: torch.Tensor, gen) -> torch.Tensor:
        """Fixed-unroll on-device chance resolution (no data-dependent sync).

        Unroll count = the tree's true max consecutive-chance depth (derived at build time), so no
        chance node can survive. `verify_no_residual_chance()` asserts this in tests (host sync, off
        the hot path).
        """
        for _ in range(self._max_chance_depth):
            is_chance = self.kind[nodes] == self._CHANCE_t           # [B] bool
            cdf = self.chance_cdf[nodes]                              # [B, Kc] (per-row)
            child_rows = self.chance_child[nodes]                     # [B, Kc] (per-row)
            r = torch.rand(nodes.shape[0], 1, device=self.device, generator=gen)
            choice = torch.searchsorted(cdf, r).squeeze(1)           # [B] first j with r<=cdf[j]
            choice = choice.clamp_(max=cdf.shape[1] - 1)
            cand = child_rows.gather(1, choice.unsqueeze(1)).squeeze(1)  # [B]
            nodes = torch.where(is_chance, cand, nodes)
        return nodes

    @torch.no_grad()
    def collect(self, policy_fn_t, batch_size: int, trajectory_max: int, gen) -> Trajectory:
        B, A, P, D, T = batch_size, self.n_actions, self.n_players, self.obs_dim, trajectory_max
        dev = self.device
        nodes = torch.full((B,), self.root, dtype=torch.int64, device=dev)
        nodes = self._resolve_chance(nodes, gen)

        obs_buf = torch.empty((T, B, D), device=dev)
        legal_buf = torch.empty((T, B, A), device=dev)
        pid_buf = torch.empty((T, B), device=dev)
        valid_buf = torch.empty((T, B), device=dev)
        rew_buf = torch.empty((T, B, P), device=dev)
        aoh_buf = torch.zeros((T, B, A), device=dev)
        pol_buf = torch.empty((T, B, A), device=dev)
        ar = torch.arange(B, device=dev)

        for t in range(T):
            is_term = self.kind[nodes] == self._TERM_t      # [B] bool
            valid = ~is_term
            safe = torch.where(valid, nodes, torch.full_like(nodes, self._ex_dec))
            obs_t = self.obs[safe]                           # [B, D]
            legal_t = self.legal[safe]                       # [B, A]
            pid_t = self.player[safe]                        # [B]

            pi = policy_fn_t(obs_t, legal_t)                 # [B, A] on device, sums to 1 over legal
            pi = pi / pi.sum(dim=-1, keepdim=True)

            # inverse-CDF action sampling on GPU (graph-capturable, no host sync).
            # Use the reference's exact convention (rnad.py:1023-1026): action = first j with r < cdf[j]
            # == (r < cdf).argmax. searchsorted(side='left') differs at the lower boundary and was found
            # to truncate long trajectories by one step vs the jax/CPU ground truth.
            cdf = torch.cumsum(pi, dim=-1)
            cdf[:, -1] = 1.0
            r = torch.rand(B, 1, device=dev, generator=gen)
            actions = (r < cdf).to(torch.int8).argmax(dim=-1)  # [B] first j with r<cdf[j]
            aoh = aoh_buf[t]
            aoh.zero_()
            aoh[ar, actions] = 1.0

            raw_child = self.dec_child[safe, actions]        # [B]
            raw_child = torch.where(raw_child < 0, safe, raw_child)
            nxt = torch.where(valid, raw_child, nodes)
            nxt = self._resolve_chance(nxt, gen)

            next_is_term = self.kind[nxt] == self._TERM_t
            next_rew = torch.where(next_is_term.unsqueeze(1), self.term_returns[nxt],
                                   torch.zeros((B, P), device=dev))

            obs_buf[t] = obs_t
            legal_buf[t] = legal_t
            pid_buf[t] = pid_t
            valid_buf[t] = valid.to(obs_t.dtype)
            rew_buf[t] = next_rew
            pol_buf[t] = pi
            nodes = nxt

        return Trajectory(obs=obs_buf, legal=legal_buf, player_id=pid_buf, valid=valid_buf,
                          rewards=rew_buf, action_oh=aoh_buf, policy=pol_buf)

    def verify_no_residual_chance(self, batch_size: int = 4096, seed: int = 0) -> None:
        """Debug check (host sync, OFF the hot path): the fixed unroll fully resolves chance.

        Samples many start states + random transitions and asserts no _CHANCE node survives
        _resolve_chance. Guards against the unroll bound being too small for the game's tree.
        """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        nodes = torch.full((batch_size,), self.root, dtype=torch.int64, device=self.device)
        nodes = self._resolve_chance(nodes, gen)
        assert not bool((self.kind[nodes] == self._CHANCE_t).any()), \
            "chance node survived the unroll at root — _max_chance_depth too small"
        # one decision step + chance resolve, repeated, to exercise interior chance nodes
        A = self.n_actions
        ar = torch.arange(batch_size, device=self.device)
        for _ in range(8):
            is_term = self.kind[nodes] == self._TERM_t
            safe = torch.where(~is_term, nodes, torch.full_like(nodes, self._ex_dec))
            acts = torch.randint(0, A, (batch_size,), device=self.device, generator=gen)
            child = self.dec_child[safe, acts]
            child = torch.where(child < 0, safe, child)
            nxt = torch.where(~is_term, child, nodes)
            nxt = self._resolve_chance(nxt, gen)
            assert not bool((self.kind[nxt] == self._CHANCE_t).any()), \
                "chance node survived the unroll mid-rollout — _max_chance_depth too small"
            nodes = nxt
