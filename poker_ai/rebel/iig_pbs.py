"""Generic public-state / PBS structure over an OpenSpiel game -- the game-agnostic generalization of
``leduc.py``'s ``cut_instances``/``cut_reaches`` and the keystone for the general depth-limited PBS
method (the net leaf + safe-resolving gadget + self-play all key on public states).

A PUBLIC STATE groups histories by their PUBLIC observation (OpenSpiel's public observer; or a per-game
``public_key_fn`` where OpenSpiel exposes no public observer, e.g. Liar's Dice). At a depth-limit CUT
(``is_cut_fn`` -- the one game-specific hook from the design pass), each player's PRIVATE state is
``information_state_string(player)`` (OpenSpiel reports it even at the chance cut node), and the
per-private reach mass (including chance) is the Public Belief State. The reach convention matches
``LeducTree.cut_reaches`` exactly (per public/private-instance reach incl. deal chance), so the generic
structure is validated BIT-IDENTICALLY against the trusted Leduc substrate.
"""
from __future__ import annotations

import numpy as np
import pyspiel
from open_spiel.python.observation import make_observation


def make_public_key_fn(game):
    """OpenSpiel public observer -> ``public_state_key(state)`` (player-agnostic), or None if the game
    exposes no public observer."""
    try:
        obs = make_observation(game, pyspiel.IIGObservationType(
            public_info=True, perfect_recall=True, private_info=pyspiel.PrivateInfoType.NONE))
    except Exception:
        obs = None
    if obs is None:
        return None

    def fn(state):
        obs.set_from(state, 0)
        return obs.string_from(state, 0)

    return fn


def uniform_policy_fn(_info_state, legal_actions):
    return np.ones(len(legal_actions)) / len(legal_actions)


# --- per-game depth-limit cut predicates (the ONE game-specific hook) -------------------------------
def leduc_is_cut(state):
    """Leduc's PBS cut = the public-card deal (the chance node after the two private deals), matching
    LeducTree's 'board' cut: public state = round-1 betting, private state = each player's card. The
    two private deals are move 0 and 1; the board deal is the only later chance node."""
    return state.is_chance_node() and state.move_number() >= 2


def first_decision_is_cut(state):
    """Generic cut for games with no public chance after the initial deal (e.g. Liar's Dice): the
    first decision node -- the root PBS = belief over each player's private state."""
    return (not state.is_terminal()) and (not state.is_chance_node())


class PBSStructure:
    """Generic public-state structure: enumerate the depth-limit cuts of an OpenSpiel game and read the
    public belief state (per-player per-private reach) under a policy."""

    def __init__(self, game, is_cut_fn, public_key_fn=None):
        self.game = game
        self.is_cut = is_cut_fn
        self.public_key_fn = public_key_fn or make_public_key_fn(game)
        if self.public_key_fn is None:
            raise ValueError("no OpenSpiel public observer for this game; pass public_key_fn")

    def cut_reaches(self, policy_fn=uniform_policy_fn):
        """Walk the game under ``policy_fn`` (info_state_str, legal_actions -> prob array). Returns
        ``{public_key: ({priv0_key: reach0}, {priv1_key: reach1})}``, reach INCL. chance. Per
        (public, player, private) the reach is well-defined (own reach depends only on own private +
        public history), so it is SET (overwrite), matching LeducTree.cut_reaches."""
        cuts: dict = {}

        def rec(state, r0, r1, rc):
            if state.is_terminal():
                return
            if self.is_cut(state):
                key = self.public_key_fn(state)
                d0, d1 = cuts.setdefault(key, ({}, {}))
                d0[state.information_state_string(0)] = r0 * rc
                d1[state.information_state_string(1)] = r1 * rc
                return  # depth limit: do not recurse below the cut
            if state.is_chance_node():
                for a, p in state.chance_outcomes():
                    s = state.clone(); s.apply_action(a)
                    rec(s, r0, r1, rc * p)
                return
            pl = state.current_player()
            acts = state.legal_actions()
            probs = policy_fn(state.information_state_string(pl), acts)
            for i, a in enumerate(acts):
                s = state.clone(); s.apply_action(a)
                if pl == 0:
                    rec(s, r0 * probs[i], r1, rc)
                else:
                    rec(s, r0, r1 * probs[i], rc)

        rec(self.game.new_initial_state(), 1.0, 1.0, 1.0)
        return cuts

    def belief_dims(self, policy_fn=uniform_policy_fn):
        """Report PBS structure: n_public_states and the per-player private-state dimension (the belief
        vector size -- the scaling quantity that grows from ~few on small games to >1000 on HUNL)."""
        cuts = self.cut_reaches(policy_fn)
        n_pub = len(cuts)
        d0 = [len(c[0]) for c in cuts.values()]
        d1 = [len(c[1]) for c in cuts.values()]
        return {"n_public_states": n_pub,
                "priv_dim_p0_max": max(d0) if d0 else 0, "priv_dim_p1_max": max(d1) if d1 else 0,
                "priv_dim_p0_mean": float(np.mean(d0)) if d0 else 0.0}
