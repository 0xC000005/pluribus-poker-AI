#!/usr/bin/env python3
"""B Stage-1 gate (CHEAP): does the R-NaD MECHANISM reach LOW exact exploitability?

Tabular proper R-NaD on a PRECOMPUTED game tree (no per-step state cloning -> CFR-class fast,
thousands of iters in seconds). Tests the mechanism: NeuRD inner update + reward-transform toward
a reference + reference-reset outer loop. Exact NashConv via OpenSpiel. The neural/FA path is the
port's job on the fast cuda substrate; this isolates "is the algorithm sound (converges to low)?".

PASS = NashConv -> low on Kuhn AND Leduc. FAIL = cycles / plateaus high -> mechanism-limited.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability


class Tree:
    """Precomputed game tree (built once; all iterations reuse it, no cloning)."""
    def __init__(self, game):
        self.infosets = {}          # infoset_str -> id
        self.iset_player = []        # id -> player
        self.iset_actions = []       # id -> [action ids]
        self.nodes = []              # built bottom-up via recursion; root last
        self.root = self._build(game.new_initial_state())

    def _iset_id(self, key, player, actions):
        if key not in self.infosets:
            self.infosets[key] = len(self.iset_player)
            self.iset_player.append(player)
            self.iset_actions.append(list(actions))
        return self.infosets[key]

    def _build(self, state):
        if state.is_terminal():
            return ("term", np.asarray(state.returns(), dtype=np.float64))
        if state.is_chance_node():
            kids = []
            for a, p in state.chance_outcomes():
                c = state.clone(); c.apply_action(a); kids.append((p, self._build(c)))
            return ("chance", kids)
        pl = state.current_player()
        acts = state.legal_actions()
        iid = self._iset_id(state.information_state_string(), pl, acts)
        kids = []
        for a in acts:
            c = state.clone(); c.apply_action(a); kids.append((a, self._build(c)))
        return ("dec", pl, iid, kids)

    @property
    def n_iset(self):
        return len(self.iset_player)


def values(tree, pi):
    """One exact pass: returns belief-weighted action values q[iid][a] (dict) and the players' EV.
    cf_reach (opponent+chance) is accumulated per infoset; q is reach-weighted then normalized."""
    qnum = [dict() for _ in range(tree.n_iset)]
    qden = [0.0] * tree.n_iset

    def rec(node, r0, r1, rc):
        t = node[0]
        if t == "term":
            return node[1]
        if t == "chance":
            ev = np.zeros(2)
            for p, ch in node[1]:
                ev += p * rec(ch, r0, r1, rc * p)
            return ev
        _, pl, iid, kids = node
        pol = pi[iid]
        cf = (r1 * rc) if pl == 0 else (r0 * rc)
        qden[iid] += cf
        ev = np.zeros(2)
        row = qnum[iid]
        for a, ch in kids:
            cv = rec(ch, r0 * pol[a], r1, rc) if pl == 0 else rec(ch, r0, r1 * pol[a], rc)
            ev += pol[a] * cv
            row[a] = row.get(a, 0.0) + cf * cv[pl]
        return ev

    rec(tree.root, 1.0, 1.0, 1.0)
    q = []
    for iid in range(tree.n_iset):
        d = qden[iid]
        q.append({a: (qnum[iid][a] / d if d > 1e-15 else 0.0) for a in tree.iset_actions[iid]})
    return q


def to_tabular(game, tree, pi):
    tp = policy_lib.TabularPolicy(game)
    for key, iid in tree.infosets.items():
        if key in tp.state_lookup:
            row = tp.action_probability_array[tp.state_lookup[key]]
            row[:] = 0.0
            for a, p in pi[iid].items():
                row[a] = p
    return tp


def run(game_name, eta, lr, reset_every, rounds, seed=0, eval_every=1, log=None):
    game = pyspiel.load_game(game_name)
    tree = Tree(game)
    pi = [{a: 1.0 / len(acts) for a in acts} for acts in tree.iset_actions]
    y = [{a: 0.0 for a in acts} for acts in tree.iset_actions]
    pi_reg = [dict(p) for p in pi]
    hist = []
    step = 0
    for rnd in range(rounds):
        for _ in range(reset_every):
            q = values(tree, pi)
            for iid in range(tree.n_iset):
                acts = tree.iset_actions[iid]
                V = sum(pi[iid][a] * q[iid][a] for a in acts)
                for a in acts:
                    reg = -eta * (math.log(max(pi[iid][a], 1e-12)) - math.log(max(pi_reg[iid][a], 1e-12)))
                    y[iid][a] += lr * ((q[iid][a] - V) + reg)
                m = max(y[iid].values()); z = sum(math.exp(y[iid][a] - m) for a in acts)
                pi[iid] = {a: math.exp(y[iid][a] - m) / z for a in acts}
            step += 1
        pi_reg = [dict(p) for p in pi]  # R-NaD reference reset (event = end of inner block)
        if rnd % eval_every == 0 or rnd == rounds - 1:
            nc = exploitability.nash_conv(game, to_tabular(game, tree, pi))
            hist.append((step, float(nc)))
            if log:
                print(f"{game_name}: step {step} (round {rnd+1})  NashConv={nc:.4f}")
    return {"game": game_name, "history": hist, "last": hist[-1][1], "best": min(h[1] for h in hist),
            "n_iset": tree.n_iset}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["kuhn_poker", "leduc_poker"])
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--reset-every", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    import time
    out = {"eta": args.eta, "lr": args.lr, "reset_every": args.reset_every, "rounds": args.rounds, "games": {}}
    for g in args.games:
        t0 = time.time()
        r = run(g, args.eta, args.lr, args.reset_every, args.rounds, eval_every=max(1, args.rounds // 15), log=True)
        r["seconds"] = round(time.time() - t0, 1)
        out["games"][g] = r
        print(f"  => {g}: last {r['last']:.4f} (best {r['best']:.4f}), {r['n_iset']} infosets, {r['seconds']}s\n")
    if args.output_json:
        import pathlib; pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output_json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
