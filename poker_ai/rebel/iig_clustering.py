"""Game-agnostic infoset ABSTRACTION for the generic depth-limited substrate -- the single-consumer-GPU
SCALE lever (the unabstracted Python tree-walk caps at ~few-k infosets; abstraction shrinks the effective
game, LAMIR-style).

Clusters infosets that share (player, legal-action signature) by their OpenSpiel INFORMATION-STATE TENSOR
(a domain-independent feature -- no hand-crafted poker buckets, so it stays tabula-rasa) into K
representatives. Merging clustered infosets (they share one strategy) shrinks the game.

THE LOAD-BEARING PRECONDITION (this module's gate, before any abstract solver is built): feature-clustered
infosets must be STRATEGICALLY COHERENT -- small within-cluster equilibrium-strategy deviation. If a cheap
feature clustering is coherent, abstraction can preserve soundness; if not, a value-based signal is needed
(which is circular on big games). ``strategy_incoherence`` measures this on a game we can solve exactly.
Slumbot held-out; diagnostic only.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def collect_features(game, dlg):
    """One walk of the game -> {iid: information_state_tensor} (first occurrence per infoset). O(tree)
    once, not per-iteration."""
    feats = {}

    def rec(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _p in state.chance_outcomes():
                s = state.clone(); s.apply_action(a); rec(s)
            return
        pl = state.current_player()
        iid = dlg.infosets.get(state.information_state_string(pl))
        if iid is not None and iid not in feats:
            feats[iid] = np.asarray(state.information_state_tensor(pl), np.float32)
        for a in state.legal_actions():
            s = state.clone(); s.apply_action(a); rec(s)

    rec(game.new_initial_state())
    return feats


def cluster_infosets(dlg, feats, ratio, seed=0):
    """Cluster infosets sharing (player, legal-action signature) by their feature tensor, targeting ~``ratio``x
    compression (clusters_in_group = max(1, round(group_size/ratio))). Returns ({iid: cluster_id}, n_clusters)."""
    from sklearn.cluster import KMeans

    groups = defaultdict(list)
    for iid in feats:
        groups[(dlg.iset_player[iid], tuple(dlg.iset_actions[iid]))].append(iid)

    cluster_of = {}
    cid = 0
    for _key, iids in groups.items():
        if len(iids) == 1:
            cluster_of[iids[0]] = cid; cid += 1
            continue
        F = np.array([feats[i] for i in iids], np.float32)
        std = F.std(0); std[std < 1e-9] = 1.0
        Fn = (F - F.mean(0)) / std                      # standardize features
        k = max(1, min(len(iids), round(len(iids) / ratio)))
        if k == 1:
            for i in iids:
                cluster_of[i] = cid
            cid += 1
            continue
        labels = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(Fn).labels_
        for i, lab in zip(iids, labels):
            cluster_of[i] = cid + int(lab)
        cid += k
    return cluster_of, cid


def strategy_incoherence(dlg, pol, cluster_of, reach=None):
    """Mean within-cluster equilibrium-strategy deviation from the cluster mean (L1 over the action
    simplex). reach (optional {iid: weight}) reach-weights infosets (low-reach ones matter less). Low
    incoherence at high compression => feature clustering is a sound abstraction signal."""
    members = defaultdict(list)
    for iid, c in cluster_of.items():
        members[c].append(iid)
    devs, wts = [], []
    for _c, iids in members.items():
        if len(iids) == 1:
            continue
        S = np.array([pol[i] for i in iids])            # (m, n_actions); same action set within cluster
        w = np.array([reach.get(i, 1.0) for i in iids]) if reach else np.ones(len(iids))
        if w.sum() <= 1e-12:
            continue
        mean = (w[:, None] * S).sum(0) / w.sum()
        dev = np.abs(S - mean).sum(1)                    # per-infoset L1 from cluster mean
        devs.append(float((w * dev).sum() / w.sum())); wts.append(float(w.sum()))
    if not devs:
        return 0.0
    devs, wts = np.array(devs), np.array(wts)
    return float((devs * wts).sum() / wts.sum())
