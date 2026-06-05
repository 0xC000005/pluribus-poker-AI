"""Generic OpenSpiel-backed public tree + CFR+ for ANY 2-player zero-sum imperfect-info game.

The SAME enumeration + solver runs on Kuhn, Leduc, Liar's Dice, ... -- the "same computation solves
any IIG" foundation for the general (game-agnostic) ReBeL method. Game logic comes ENTIRELY from
OpenSpiel (pyspiel); this module only enumerates the game into a (terminal / chance / decision) tree,
runs CFR+, and reports EXACT NashConv via OpenSpiel's exploitability. It is the game-agnostic
generalization of the trusted Leduc substrate (``poker_ai/rebel/leduc.py`` ``LeducTree.cfr_plus``):
that module tags the private deals (``deal``) and the public-card depth-limit cut (``board``)
separately for poker; here both are a single ``chance`` node type (mathematically identical for
full-game CFR). Validated against ``LeducTree`` by a parity gate. The depth-limited PBS net leaf
(the actual efficiency method) builds on this generic tree in a later step.

Node encoding:
  ("term", returns[2])
  ("chance", [(prob, child), ...])
  ("dec", player, iset_id, [(action, child), ...])
"""
from __future__ import annotations

import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability


class PublicTree:
    """A precomputed, game-agnostic tree enumerated once from an OpenSpiel game (turn-based, 2p0s)."""

    def __init__(self, game_name, params=None):
        self.game_name = game_name
        self.game = pyspiel.load_game(game_name, params) if params else pyspiel.load_game(game_name)
        if self.game.num_players() != 2:
            raise ValueError("PublicTree supports 2-player games only")
        self.infosets: dict[str, int] = {}
        self.iset_player: list[int] = []
        self.iset_actions: list[list[int]] = []
        self.iset_key: list[str] = []
        self.root = self._build(self.game.new_initial_state())

    def _iset_id(self, key, player, acts):
        if key not in self.infosets:
            self.infosets[key] = len(self.iset_player)
            self.iset_player.append(player)
            self.iset_actions.append(list(acts))
            self.iset_key.append(key)
        return self.infosets[key]

    def _build(self, state):
        if state.is_terminal():
            return ("term", np.asarray(state.returns(), dtype=np.float64))
        if state.is_chance_node():
            kids = []
            for a, p in state.chance_outcomes():
                s = state.clone(); s.apply_action(a)
                kids.append((float(p), self._build(s)))
            return ("chance", kids)
        if state.is_simultaneous_node():
            raise NotImplementedError("simultaneous-move games not supported yet (sequential only)")
        pl = state.current_player()
        acts = state.legal_actions()
        iid = self._iset_id(state.information_state_string(), pl, acts)
        kids = []
        for a in acts:
            s = state.clone(); s.apply_action(a)
            kids.append((a, self._build(s)))
        return ("dec", pl, iid, kids)

    @property
    def n_iset(self) -> int:
        return len(self.iset_player)

    @property
    def n_nodes(self) -> int:
        def count(node):
            t = node[0]
            if t == "term":
                return 1
            if t == "chance":
                return 1 + sum(count(ch) for _p, ch in node[1])
            return 1 + sum(count(ch) for _a, ch in node[3])
        return count(self.root)

    def uniform_policy(self):
        return [np.ones(len(a)) / len(a) for a in self.iset_actions]

    def to_tabular(self, pol):
        tp = policy_lib.TabularPolicy(self.game)
        for key, iid in self.infosets.items():
            if key in tp.state_lookup:
                row = tp.action_probability_array[tp.state_lookup[key]]
                row[:] = 0.0
                for i, a in enumerate(self.iset_actions[iid]):
                    row[a] = pol[iid][i]
        return tp

    def nash_conv(self, pol) -> float:
        return float(exploitability.nash_conv(self.game, self.to_tabular(pol)))

    @staticmethod
    def _avg(stratsum):
        out = []
        for ss in stratsum:
            s = ss.sum()
            out.append(ss / s if s > 1e-15 else np.ones_like(ss) / len(ss))
        return out

    def cfr_plus(self, iters, eval_every=0, averaging="linear"):
        """CFR+ over the full tree with ALTERNATING updates; strategy averaged by OWN reach with
        per-iteration weight w_t (``linear`` -> w_t=t+1; ``uniform`` -> w_t=1). The game-agnostic
        twin of ``LeducTree.cfr_plus`` -- identical math, with private deals and the public cut both
        treated as ``chance``. Returns (avg_policy, history=[(t, nashconv), ...])."""
        regret = [np.zeros(len(a)) for a in self.iset_actions]
        stratsum = [np.zeros(len(a)) for a in self.iset_actions]
        hist = []

        def sigma_of():
            sig = []
            for r in regret:
                pos = np.maximum(r, 0.0); s = pos.sum()
                sig.append(pos / s if s > 1e-15 else np.ones_like(r) / len(r))
            return sig

        for t in range(iters):
            upd = t % 2
            sig = sigma_of()
            cfvnum = [np.zeros(len(a)) for a in self.iset_actions]
            ownreach = [0.0] * self.n_iset

            def walk(node, r0, r1, rc):
                ty = node[0]
                if ty == "term":
                    return node[1]
                if ty == "chance":
                    ev = np.zeros(2)
                    for p, ch in node[1]:
                        ev += p * walk(ch, r0, r1, rc * p)
                    return ev
                _, pl, iid, kids = node
                s = sig[iid]
                ownreach[iid] = r0 if pl == 0 else r1
                ev = np.zeros(2)
                cf = (r1 * rc) if pl == 0 else (r0 * rc)
                row = cfvnum[iid]
                for i, (a, ch) in enumerate(kids):
                    cv = walk(ch, r0 * s[i], r1, rc) if pl == 0 else walk(ch, r0, r1 * s[i], rc)
                    ev += s[i] * cv
                    row[i] += cf * cv[pl]
                return ev

            walk(self.root, 1.0, 1.0, 1.0)
            w = float(t + 1) if averaging == "linear" else 1.0
            for iid in range(self.n_iset):
                if self.iset_player[iid] != upd:
                    continue
                s = sig[iid]
                v = float(np.dot(s, cfvnum[iid]))
                regret[iid] = np.maximum(regret[iid] + (cfvnum[iid] - v), 0.0)
                stratsum[iid] += w * ownreach[iid] * s

            if eval_every and ((t + 1) % eval_every == 0 or t == iters - 1):
                hist.append((t + 1, self.nash_conv(self._avg(stratsum))))

        avg = self._avg(stratsum)
        if not hist:
            hist.append((iters, self.nash_conv(avg)))
        return avg, hist
