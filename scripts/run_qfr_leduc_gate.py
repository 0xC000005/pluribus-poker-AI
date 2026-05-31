#!/usr/bin/env python3
"""B Stage-1 gate: does OPTIMISM (QFR) beat non-optimistic OMD (MMD) on EXACT exploitability?

Both arms = the SAME belief-weighted-Q regularized OMD update (behavioral form, Sokota MMD Eq-10/12),
differing ONLY in the gradient: MMD uses g=Q_t; QFR uses the optimistic g=2*Q_t - Q_{t-1}. Matched
tau (QRE regularization toward a uniform magnet) and eta (stepsize). This isolates optimism alone.

Metric: EXACT last-iterate NashConv (OpenSpiel) on Kuhn then Leduc, + the last-vs-best-iterate gap.
PASS (optimism is a real lever) = QFR last-iterate clearly below matched-tau MMD by a pre-declared margin,
small last-vs-best gap. FAIL/NULL -> fall back to MMD. Diagnostic only; no GPU; no protected-surface edit.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability


def _infosets(game):
    """Walk the tree once; return {infostate_str: sorted_legal_action_ids} and the owning player."""
    sets, owner = {}, {}
    stack = [game.new_initial_state()]
    seen = set()
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                c = s.clone(); c.apply_action(a); stack.append(c)
            continue
        I = s.information_state_string()
        if I not in sets:
            sets[I] = list(s.legal_actions()); owner[I] = s.current_player()
        key = (s.history_str(),)
        for a in s.legal_actions():
            c = s.clone(); c.apply_action(a)
            if c.history_str() not in seen:
                seen.add(c.history_str()); stack.append(c)
    return sets, owner


def _belief_weighted_q(game, pol):
    """Exact belief-weighted local action values Q_I(a) for the current profile `pol`
    (pol[I][a] = prob). Returns {I: {a: Q}} (O(1) scale: opponent-reach-weighted returns)."""
    qnum, qden = {}, {}

    def rec(state, r0, r1, rc):
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float64)
        if state.is_chance_node():
            ev = np.zeros(2)
            for a, p in state.chance_outcomes():
                c = state.clone(); c.apply_action(a)
                ev += p * rec(c, r0, r1, rc * p)
            return ev
        pl = state.current_player(); I = state.information_state_string()
        pi = pol[I]
        cf = (r1 * rc) if pl == 0 else (r0 * rc)  # opponent+chance reach = belief weight
        qden[I] = qden.get(I, 0.0) + cf
        row = qnum.setdefault(I, {})
        ev = np.zeros(2)
        for a in state.legal_actions():
            c = state.clone(); c.apply_action(a)
            cv = rec(c, r0 * pi[a], r1, rc) if pl == 0 else rec(c, r0, r1 * pi[a], rc)
            ev += pi[a] * cv
            row[a] = row.get(a, 0.0) + cf * cv[pl]
        return ev

    rec(game.new_initial_state(), 1.0, 1.0, 1.0)
    return {I: {a: row[a] / qden[I] for a in row} if qden.get(I, 0) > 0 else {a: 0.0 for a in row}
            for I, row in qnum.items()}


def _to_tabular(game, pol, sets):
    tp = policy_lib.TabularPolicy(game)
    for I, row in tp.state_lookup.items():
        if I in pol:
            arr = np.zeros_like(tp.action_probability_array[row])
            for a, p in pol[I].items():
                arr[a] = p
            tp.action_probability_array[row] = arr
    return tp


def run(game_name, optimistic, tau, eta, iters, seed=0):
    game = pyspiel.load_game(game_name)
    sets, owner = _infosets(game)
    pol = {I: {a: 1.0 / len(acts) for a in acts} for I, acts in sets.items()}  # uniform init
    qprev = {I: {a: 0.0 for a in acts} for I, acts in sets.items()}
    ae = tau * eta
    best = float("inf"); last = None
    for it in range(iters):
        Q = _belief_weighted_q(game, pol)
        for I, acts in sets.items():
            qI = Q.get(I, {a: 0.0 for a in acts})
            logu = {}
            for a in acts:
                g = (2.0 * qI[a] - qprev[I][a]) if optimistic else qI[a]
                rho = 1.0 / len(acts)  # uniform magnet
                logu[a] = (math.log(max(pol[I][a], 1e-12)) + ae * math.log(rho) + eta * g) / (1.0 + ae)
            m = max(logu.values()); z = sum(math.exp(logu[a] - m) for a in acts)
            pol[I] = {a: math.exp(logu[a] - m) / z for a in acts}
            qprev[I] = qI
        if (it + 1) % max(1, iters // 20) == 0 or it == iters - 1:
            nc = exploitability.nash_conv(game, _to_tabular(game, pol, sets))
            best = min(best, nc); last = nc
    return {"last_iterate": float(last), "best_iterate": float(best), "gap": float(last - best)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["kuhn_poker", "leduc_poker"])
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--margin", type=float, default=0.05, help="QFR must beat MMD last-iterate by this RELATIVE fraction to PASS")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    out = {"tau": args.tau, "eta": args.eta, "iters": args.iters, "games": {}}
    for gname in args.games:
        mmd = run(gname, False, args.tau, args.eta, args.iters)
        qfr = run(gname, True, args.tau, args.eta, args.iters)
        rel = (mmd["last_iterate"] - qfr["last_iterate"]) / max(mmd["last_iterate"], 1e-9)
        out["games"][gname] = {"mmd": mmd, "qfr": qfr, "qfr_rel_improvement": rel,
                               "optimism_helps": bool(rel >= args.margin)}
        print(f"{gname:14s}: MMD last={mmd['last_iterate']:.4f}(best {mmd['best_iterate']:.4f}) | "
              f"QFR last={qfr['last_iterate']:.4f}(best {qfr['best_iterate']:.4f}) | "
              f"QFR rel-improve={rel:+.1%} gap(mmd/qfr)={mmd['gap']:.4f}/{qfr['gap']:.4f}")
    leduc = out["games"].get("leduc_poker", {})
    verdict = "PASS (optimism helps -> QFR)" if leduc.get("optimism_helps") else "NULL/FAIL (fall back to MMD)"
    out["verdict"] = verdict
    print(f"\nLEDUC GATE VERDICT: {verdict}  (pre-declared margin {args.margin:.0%} relative last-iterate)")
    if args.output_json:
        import pathlib; pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
