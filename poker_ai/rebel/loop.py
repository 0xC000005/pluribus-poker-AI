"""ReBeL de-risk, Stage 1: depth-limited solving on Leduc + the resolving-safety findings.

Depth-limited solving = run CFR on the TRUNK (round 1); at the depth-limit cut (the public card),
do not recurse -- query a leaf value function for the continuation counterfactual values given the
trunk-induced ranges.

A KEY CORRECTNESS FINDING this module records (see ``scripts/run_rebel_leduc_stage1.py``): the leaf
value function must NOT be a per-iteration *re-solved subgame equilibrium*. Doing that lets the
opponent re-adapt inside the leaf value, which biases the trunk's CFR regrets (CFR needs the
continuation value under the FIXED current continuation strategy, not a re-optimized equilibrium).
Empirically the isolated-oracle trunk fails to recover the equilibrium round-1 strategy
(``depth_limited_solve_isolated_oracle`` below: round-1 L1 ~0.15), and a frozen isolated subgame
strategy is off-path exploitable (assembled NashConv ~0.21 vs ~5e-4 true). Two separate phenomena:
biased trunk regrets, and subgame-resolving-safety.

The CORRECT mechanics (what scale-up must use):
  * Exact-oracle control = full CFR over the public tree with co-evolving subgame iterations
    (CFR-D); ``LeducTree.cfr_plus`` is exactly this and reaches NashConv ~5e-4.
  * At scale, the subgame is replaced by a value net that is FIXED during each solve (a function of
    the PBS), and PLAY does continual re-solving (re-solve the subgame for the actually-reached
    range). The leaf VALUES the oracle produces are accurate at all positive-reach entries (good for
    training the net); only the frozen STRATEGY is unsafe.

``solve_round2_equilibrium`` (a correct, reusable subgame solver -- needed for play-time re-solving
and for generating value-net targets in Stage 2) is the production-useful piece here.
"""
from __future__ import annotations

import numpy as np

from poker_ai.rebel.leduc import LeducTree, NCARDS
from poker_ai.rebel.leaf_eval import ExactLeafOracle


def _rm_plus(regret):
    pos = np.maximum(regret, 0.0)
    s = pos.sum()
    return pos / s if s > 1e-15 else np.ones_like(regret) / len(regret)


def _round2_iids(tree, instances):
    """Round-2 infoset ids reachable from a cut key's board nodes (each belongs to one key)."""
    iids = set()

    def disc(node):
        t = node[0]
        if t == "term":
            return
        if t == "board":
            for _a, _p, ch in node[4]:
                disc(ch)
        else:  # dec
            iids.add(node[2])
            for _a, ch in node[7]:
                disc(ch)

    for _c0, _c1, bn in instances:
        disc(bn)
    return list(iids)


def solve_round2_equilibrium(tree, key, range0, range1, iters, averaging="linear"):
    """CFR+ over the round-2 subgame at public cut ``key``, seeded with entry per-card ranges
    (incl. deal chance). Returns the equilibrium round-2 strategy as a dict {iid: prob_array} over
    the touched round-2 infosets. Board chance (1/4 per remaining card) is inside the board nodes."""
    instances = tree.cut_instances()[key]
    iids = _round2_iids(tree, instances)
    regret = {i: np.zeros(len(tree.iset_actions[i])) for i in iids}
    stratsum = {i: np.zeros(len(tree.iset_actions[i])) for i in iids}

    for t in range(iters):
        upd = t % 2
        sig = {i: _rm_plus(regret[i]) for i in iids}
        cfvnum = {i: np.zeros(len(tree.iset_actions[i])) for i in iids}
        ownreach = {i: 0.0 for i in iids}

        def walk(node, r0, r1, rc):
            ty = node[0]
            if ty == "term":
                return node[1]
            if ty == "board":
                ev = np.zeros(2)
                for _a, p, ch in node[4]:
                    ev += p * walk(ch, r0, r1, rc * p)
                return ev
            _, pl, iid, _rnd, _c0, _c1, _b, kids = node
            s = sig[iid]
            ownreach[iid] = r0 if pl == 0 else r1
            ev = np.zeros(2)
            cf = (r1 * rc) if pl == 0 else (r0 * rc)
            row = cfvnum[iid]
            for i, (_a, ch) in enumerate(kids):
                cv = walk(ch, r0 * s[i], r1, rc) if pl == 0 else walk(ch, r0, r1 * s[i], rc)
                ev += s[i] * cv
                row[i] += cf * cv[pl]
            return ev

        for c0, c1, bn in instances:
            walk(bn, range0[c0], range1[c1], 1.0)

        w = float(t + 1) if averaging == "linear" else 1.0
        for i in iids:
            if tree.iset_player[i] != upd:
                continue
            s = sig[i]
            v = float(np.dot(s, cfvnum[i]))
            regret[i] = np.maximum(regret[i] + (cfvnum[i] - v), 0.0)
            stratsum[i] += w * ownreach[i] * s

    avg = {}
    for i in iids:
        ss = stratsum[i].sum()
        avg[i] = stratsum[i] / ss if ss > 1e-15 else np.ones(len(stratsum[i])) / len(stratsum[i])
    return avg


def _full_pol_with_round2(tree, round1_pol, round2_by_key):
    """Build a length-n_iset policy: round-1 from ``round1_pol`` (list aligned to round-1 iids by
    iid), round-2 from the per-key equilibria; remaining (unused) infosets uniform."""
    pol = tree.uniform_policy()
    for iid, pr in round1_pol.items():
        pol[iid] = pr
    for _key, avg in round2_by_key.items():
        for iid, pr in avg.items():
            pol[iid] = pr
    return pol


def _trunk_cfv(tree, sig_round1, leaf_v):
    """One CFR value pass over the trunk (round 1), cutting at the board with leaf_v. Returns
    (cfvnum {iid: array}, ownreach {iid: float}) for round-1 infosets."""
    cfvnum = {}
    ownreach = {}

    def walk(node, r0, r1, rc):
        ty = node[0]
        if ty == "term":
            return node[1]
        if ty == "deal":
            ev = np.zeros(2)
            for _a, p, ch in node[1]:
                ev += p * walk(ch, r0, r1, rc * p)
            return ev
        if ty == "board":
            _, k, c0, c1, _kids = node
            v0, v1 = leaf_v[k]
            return np.array([v0[c0], v1[c1]], dtype=np.float64)
        _, pl, iid, _rnd, _c0, _c1, _b, kids = node
        s = sig_round1[iid]
        ownreach[iid] = r0 if pl == 0 else r1
        if iid not in cfvnum:
            cfvnum[iid] = np.zeros(len(kids))
        cf = (r1 * rc) if pl == 0 else (r0 * rc)
        row = cfvnum[iid]
        ev = np.zeros(2)
        for i, (_a, ch) in enumerate(kids):
            cv = walk(ch, r0 * s[i], r1, rc) if pl == 0 else walk(ch, r0, r1 * s[i], rc)
            ev += s[i] * cv
            row[i] += cf * cv[pl]
        return ev

    walk(tree.root, 1.0, 1.0, 1.0)
    return cfvnum, ownreach


def trunk_solve(tree, leaf_fn, trunk_iters=400, averaging="linear"):
    """Depth-limited CFR+ on the round-1 trunk with a FIXED leaf value FUNCTION (the correct ReBeL
    usage: the leaf function does not change within the solve; it is queried at the current iterate's
    ranges). ``leaf_fn(cut_key, range0, range1) -> (v0[NCARDS], v1[NCARDS])`` in the pinned
    normalized convention. Returns the round-1 average strategy as {iid: prob_array}. Both the exact
    leaf oracle and a learned value net plug in through ``leaf_fn`` -- this is the Stage-2 substrate.
    """
    r1_iids = [i for i in range(tree.n_iset) if tree.iset_round[i] == 1]
    regret = {i: np.zeros(len(tree.iset_actions[i])) for i in r1_iids}
    stratsum = {i: np.zeros(len(tree.iset_actions[i])) for i in r1_iids}
    keys = sorted(tree.cut_keys())

    for t in range(trunk_iters):
        upd = t % 2
        sig = {i: _rm_plus(regret[i]) for i in r1_iids}
        reaches = tree.cut_reaches(_listify(tree, sig))
        leaf_v = {k: leaf_fn(k, reaches[k][0], reaches[k][1]) for k in keys}
        cfvnum, ownreach = _trunk_cfv(tree, sig, leaf_v)
        w = float(t + 1) if averaging == "linear" else 1.0
        for i in r1_iids:
            if tree.iset_player[i] != upd:
                continue
            s = sig[i]
            v = float(np.dot(s, cfvnum[i]))
            regret[i] = np.maximum(regret[i] + (cfvnum[i] - v), 0.0)
            stratsum[i] += w * ownreach[i] * s
    return _avg_dict(stratsum)


def exact_leaf_fn(tree, round2_iters=2000, round2_averaging="linear"):
    """Build an exact ``leaf_fn`` for ``trunk_solve``: solves the round-2 subgame for the queried
    ranges and returns the normalized per-card CFVs. This is the per-solve fixed-function exact leaf
    (the control the learned net is compared against in Stage 2)."""
    oracle = ExactLeafOracle(tree, normalize=True)

    def fn(key, range0, range1):
        r2 = solve_round2_equilibrium(tree, key, range0, range1, round2_iters, round2_averaging)
        pol = _listify(tree, {}, round2={key: r2})
        return oracle.evaluate(key, range0, range1, pol)

    return fn


def _round2_q(tree, key, range0, range1, pol):
    """Belief-weighted action values q[iid][a] for the round-2 infosets at cut ``key``, seeded with
    entry ranges (incl. deal chance). cf = opponent reach * chance; q = reach-weighted then
    normalized (same convention as ``LeducTree.values``). Returns {iid: q_array}."""
    instances = tree.cut_instances()[key]
    iids = _round2_iids(tree, instances)
    qnum = {i: np.zeros(len(tree.iset_actions[i])) for i in iids}
    qden = {i: 0.0 for i in iids}

    def rec(node, r0, r1, rc):
        ty = node[0]
        if ty == "term":
            return node[1]
        if ty == "board":
            ev = np.zeros(2)
            for _a, p, ch in node[4]:
                ev += p * rec(ch, r0, r1, rc * p)
            return ev
        _, pl, iid, _rnd, _c0, _c1, _b, kids = node
        s = pol[iid]
        cf = (r1 * rc) if pl == 0 else (r0 * rc)
        qden[iid] += cf
        ev = np.zeros(2)
        row = qnum[iid]
        for i, (_a, ch) in enumerate(kids):
            cv = rec(ch, r0 * s[i], r1, rc) if pl == 0 else rec(ch, r0, r1 * s[i], rc)
            ev += s[i] * cv
            row[i] += cf * cv[pl]
        return ev

    for c0, c1, bn in instances:
        rec(bn, range0[c0], range1[c1], 1.0)
    return {i: (qnum[i] / qden[i] if qden[i] > 1e-15 else np.zeros(len(qnum[i]))) for i in iids}


def solve_round2_qre(tree, key, range0, range1, tau=0.1, eta=0.5, iters=400):
    """Regularized OMD (MMD, uniform magnet) last-iterate solve of the round-2 subgame for the given
    ranges -> the UNIQUE quantal-response equilibrium (QRE). Unlike a plain Nash re-solve, the QRE is
    unique -> consistent CFVs across ranges (learnable targets) AND near-optimal for small tau (a
    good trunk leaf). Returns {iid: strategy} over round-2 infosets (last iterate)."""
    instances = tree.cut_instances()[key]
    iids = _round2_iids(tree, instances)
    pol = tree.uniform_policy()  # full list; we only update round-2 iids
    ae = tau * eta
    for _ in range(iters):
        q = _round2_q(tree, key, range0, range1, pol)
        for i in iids:
            acts = tree.iset_actions[i]
            n = len(acts)
            rho = 1.0 / n
            logu = (np.log(np.maximum(pol[i], 1e-12)) + ae * np.log(rho) + eta * q[i]) / (1.0 + ae)
            m = logu.max()
            e = np.exp(logu - m)
            pol[i] = e / e.sum()
    return {i: pol[i] for i in iids}


def gadget_resolve(tree, key, resolver, range_resolver, range_opp, opp_cfv,
                   iters=2000, averaging="linear"):
    """DeepStack-style SAFE re-solving of the round-2 subgame for ``resolver`` (player 0 or 1),
    constraining the opponent to its blueprint counterfactual values ``opp_cfv`` (per opponent card,
    normalized convention -- e.g. the leaf values the trunk used). The opponent gets a per-hand
    FOLLOW/TERMINATE gadget decision (terminate -> its blueprint CFV), so the resolver's strategy is
    guaranteed not to let the opponent exceed those CFVs -> safe (not off-path exploitable).

    Returns the resolver's average round-2 strategy {iid: prob_array}. ``range_resolver`` /
    ``range_opp`` are per-card entry reaches (incl. deal chance) at the cut.
    """
    opp = 1 - resolver
    instances = tree.cut_instances()[key]
    r2_iids = _round2_iids(tree, instances)
    res_iids = [i for i in r2_iids if tree.iset_player[i] == resolver]
    opp_iids = [i for i in r2_iids if tree.iset_player[i] == opp]
    regret = {i: np.zeros(len(tree.iset_actions[i])) for i in r2_iids}
    stratsum = {i: np.zeros(len(tree.iset_actions[i])) for i in res_iids}
    # opponent per-hand gadget regret over {follow, terminate}
    g_reg = np.zeros((NCARDS, 2))

    def g_follow():
        out = np.zeros(NCARDS)
        for c in range(NCARDS):
            pos = np.maximum(g_reg[c], 0.0); s = pos.sum()
            out[c] = (pos[0] / s) if s > 1e-15 else 0.5  # P(follow)
        return out

    for t in range(iters):
        sig = {i: _rm_plus(regret[i]) for i in r2_iids}
        gf = g_follow()
        cfvnum = {i: np.zeros(len(tree.iset_actions[i])) for i in r2_iids}
        # opponent root subgame value per hand (normalized): num/den
        opp_num = np.zeros(NCARDS); opp_den = np.zeros(NCARDS)

        def walk(node, r0, r1, rc):
            ty = node[0]
            if ty == "term":
                return node[1]
            if ty == "board":
                ev = np.zeros(2)
                for _a, p, ch in node[4]:
                    ev += p * walk(ch, r0, r1, rc * p)
                return ev
            _, pl, iid, _rnd, _c0, _c1, _b, kids = node
            s = sig[iid]
            ev = np.zeros(2)
            cf = (r1 * rc) if pl == 0 else (r0 * rc)
            row = cfvnum[iid]
            for i, (_a, ch) in enumerate(kids):
                cv = walk(ch, r0 * s[i], r1, rc) if pl == 0 else walk(ch, r0, r1 * s[i], rc)
                ev += s[i] * cv
                row[i] += cf * cv[pl]
            return ev

        for c0, c1, bn in instances:
            res_card, opp_card = (c0, c1) if resolver == 0 else (c1, c0)
            rr = range_resolver[res_card]
            ro = range_opp[opp_card] * gf[opp_card]
            r0, r1 = (rr, ro) if resolver == 0 else (ro, rr)
            ev = walk(bn, r0, r1, 1.0)
            # opponent's conditional value of FOLLOWING with opp_card (weighted by resolver range)
            opp_num[opp_card] += rr * ev[opp]
            opp_den[opp_card] += rr

        # regret updates (simultaneous CFR+)
        w = float(t + 1) if averaging == "linear" else 1.0
        for i in res_iids:
            s = sig[i]; v = float(np.dot(s, cfvnum[i]))
            regret[i] = np.maximum(regret[i] + (cfvnum[i] - v), 0.0)
            stratsum[i] += w * s
        for i in opp_iids:
            s = sig[i]; v = float(np.dot(s, cfvnum[i]))
            regret[i] = np.maximum(regret[i] + (cfvnum[i] - v), 0.0)
        # opponent gadget regret: follow value = normalized subgame value; terminate = blueprint CFV
        for c in range(NCARDS):
            if opp_den[c] <= 1e-15:
                continue
            fval = opp_num[c] / opp_den[c]
            tval = opp_cfv[c]
            ev_g = gf[c] * fval + (1.0 - gf[c]) * tval
            g_reg[c, 0] = max(g_reg[c, 0] + (fval - ev_g), 0.0)
            g_reg[c, 1] = max(g_reg[c, 1] + (tval - ev_g), 0.0)

    avg = {}
    for i in res_iids:
        ss = stratsum[i].sum()
        avg[i] = stratsum[i] / ss if ss > 1e-15 else np.ones(len(stratsum[i])) / len(stratsum[i])
    return avg


def qre_leaf_fn(tree, tau=0.1, eta=0.5, iters=400):
    """Exact ``leaf_fn`` whose value is the QRE continuation value for the queried ranges: unique
    (consistent across ranges) AND near-optimal (small tau). Resolves the consistency<->quality
    tradeoff that plain Nash re-solve (under-determined) and blueprint (fixed-strategy, crude) hit."""
    oracle = ExactLeafOracle(tree, normalize=True)

    def fn(key, range0, range1):
        qre = solve_round2_qre(tree, key, range0, range1, tau=tau, eta=eta, iters=iters)
        pol = _listify(tree, {}, round2={key: qre})
        return oracle.evaluate(key, range0, range1, pol)

    return fn


def blueprint_leaf_fn(tree, ref_strategy):
    """Build a CONSISTENT exact ``leaf_fn`` for ``trunk_solve``: the leaf value is the CFV of
    continuing with a single fixed near-equilibrium round-2 strategy ``ref_strategy`` (a full policy
    list). Unlike ``exact_leaf_fn`` (isolated per-range re-solve, under-determined across ranges),
    this is a smooth, consistent value function -- cheap (no per-range solve) and the matching exact
    control for a net trained on the same blueprint-continuation targets."""
    oracle = ExactLeafOracle(tree, normalize=True)

    def fn(key, range0, range1):
        return oracle.evaluate(key, range0, range1, ref_strategy)

    return fn


def depth_limited_solve_isolated_oracle(tree, trunk_iters=200, round2_iters=300, averaging="uniform",
                                        round2_averaging="linear", eval_every=0, verbose=False):
    """PITFALL DEMO (do not use for production solving). Depth-limited CFR+ on the round-1 trunk
    where the leaf value is the round-2 subgame RE-SOLVED TO EQUILIBRIUM for the current ranges each
    trunk iteration. This biases the trunk regrets (the opponent re-adapts inside the leaf value), so
    the trunk does NOT recover the equilibrium round-1 strategy. Returns (full_policy_assembled,
    history). Used by the Stage-1 diagnostic to quantify the bias; the correct exact-oracle control
    is ``tree.cfr_plus`` (co-evolving CFR-D)."""
    r1_iids = [i for i in range(tree.n_iset) if tree.iset_round[i] == 1]
    regret = {i: np.zeros(len(tree.iset_actions[i])) for i in r1_iids}
    stratsum = {i: np.zeros(len(tree.iset_actions[i])) for i in r1_iids}
    oracle = ExactLeafOracle(tree, normalize=True)
    keys = sorted(tree.cut_keys())
    hist = []

    def assemble(round1_pol):
        reaches = tree.cut_reaches(_listify(tree, round1_pol))
        round2_by_key = {}
        for k in keys:
            r0, r1 = reaches[k]
            round2_by_key[k] = solve_round2_equilibrium(tree, k, r0, r1, round2_iters, round2_averaging)
        return _full_pol_with_round2(tree, round1_pol, round2_by_key)

    for t in range(trunk_iters):
        upd = t % 2
        sig = {i: _rm_plus(regret[i]) for i in r1_iids}
        reaches = tree.cut_reaches(_listify(tree, sig))
        leaf_v = {}
        for k in keys:
            r0, r1 = reaches[k]
            r2 = solve_round2_equilibrium(tree, k, r0, r1, round2_iters, round2_averaging)
            pol_list = _listify(tree, sig, round2={k: r2})
            leaf_v[k] = oracle.evaluate(k, r0, r1, pol_list)
        cfvnum, ownreach = _trunk_cfv(tree, sig, leaf_v)
        w = float(t + 1) if averaging == "linear" else 1.0
        for i in r1_iids:
            if tree.iset_player[i] != upd:
                continue
            s = sig[i]
            v = float(np.dot(s, cfvnum[i]))
            regret[i] = np.maximum(regret[i] + (cfvnum[i] - v), 0.0)
            stratsum[i] += w * ownreach[i] * s

        if eval_every and ((t + 1) % eval_every == 0 or t == trunk_iters - 1):
            avg1 = _avg_dict(stratsum)
            nc = tree.nash_conv(assemble(avg1))
            hist.append((t + 1, nc))
            if verbose:
                print(f"  trunk iter {t+1}: NashConv={nc:.5f}")

    avg1 = _avg_dict(stratsum)
    full = assemble(avg1)
    if not hist:
        hist.append((trunk_iters, tree.nash_conv(full)))
    return full, hist


def _avg_dict(stratsum):
    out = {}
    for i, ss in stratsum.items():
        s = ss.sum()
        out[i] = ss / s if s > 1e-15 else np.ones(len(ss)) / len(ss)
    return out


def _listify(tree, round1, round2=None):
    """Build a length-n_iset policy list: round-1 from dict ``round1``, round-2 from optional dict
    of {key: {iid: arr}} (uniform elsewhere). Used to feed tree recursions that index by iid."""
    pol = tree.uniform_policy()
    for iid, pr in round1.items():
        pol[iid] = pr
    if round2:
        for _k, avg in round2.items():
            for iid, pr in avg.items():
                pol[iid] = pr
    return pol
