#!/usr/bin/env python3
"""ED (#40) v1 figures from the persisted JSONs (see ed_writeup_evidence_pack.md):
F1 controlled keysweep (speedup vs #co-solved subgames, G4+G5) -- the regime-model picture.
F2 e2e wall-clock per arm (stacked inner/train/measure), 5 seeds, Amdahl-ceiling overlay.
F3 band distributions (fused vs sequential finals, G4) with per-seed paired lines.
F4 exploitability-vs-wallclock overlay: fused / sequential / NFSP (log-x).
Pure read of autoresearch-session/rebel/*.json -> autoresearch-session/rebel/figures/*.{png,pdf}.
"""
from __future__ import annotations

import json
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path("autoresearch-session/rebel")
OUT = R / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _j(name):
    return json.loads((R / name).read_text())


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.pdf")


def f1_keysweep():
    d = _j("costpersolve_bakeoff.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for game, marker, color in (("goofspiel4", "o", "#1f77b4"), ("goofspiel5", "s", "#d62728")):
        ks = d["keysweeps"][game]
        n = [r["n_subgames"] for r in ks]
        sp = [r["speedup_gpu"] for r in ks]
        label = f"{'G4 (small subgames, launch-bound)' if game == 'goofspiel4' else 'G5 (large subgames, saturated)'}"
        ax.plot(n, sp, marker=marker, color=color, label=label)
    nmax = max(r["n_subgames"] for r in d["keysweeps"]["goofspiel4"])
    ax.plot([1, nmax], [1, nmax], ls="--", c="gray", lw=1, label="ideal: speedup = N")
    ax.set_xlabel("# co-solved subgames N (fixed per-subgame size)")
    ax.set_ylabel("cost-per-solve speedup (GPU time)")
    ax.set_title("Population fusion: near-linear when launch-bound, saturating when GPU-bound")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    _save(fig, "f1_keysweep")


def _g4_runs():
    runs = {}
    for m in ("fused", "sequential"):
        for s in range(5):
            runs[(m, s)] = _j(f"b0_e2e_g4_{m}_seed{s}.json")
    return runs


def f2_wall_bars():
    runs = _g4_runs()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    width = 0.38
    for i, m in enumerate(("fused", "sequential")):
        xs = [s + (i - 0.5) * width for s in range(5)]
        inner = [runs[(m, s)]["inner_resolve_wall_s_total"] for s in range(5)]
        train = [runs[(m, s)]["train_wall_s_total"] for s in range(5)]
        other = [max(runs[(m, s)]["total_wall_s"] - runs[(m, s)]["inner_resolve_wall_s_total"]
                     - runs[(m, s)]["train_wall_s_total"], 0.0) for s in range(5)]
        c = "#1f77b4" if m == "fused" else "#d62728"
        ax.bar(xs, inner, width, color=c, label=f"{m}: inner re-solve")
        ax.bar(xs, train, width, bottom=inner, color=c, alpha=0.55,
               label=f"{m}: net train" if i == 0 else None)
        ax.bar(xs, other, width, bottom=[a + b for a, b in zip(inner, train)], color=c, alpha=0.25,
               label=f"{m}: other (measure/gadget)" if i == 0 else None)
    med_f = statistics.median([runs[("fused", s)]["total_wall_s"] for s in range(5)])
    s_tot = statistics.median([runs[("sequential", s)]["total_wall_s"] / runs[("fused", s)]["total_wall_s"]
                               for s in range(5)])
    ax.axhline(med_f, ls=":", c="#1f77b4", lw=1)
    ax.set_xlabel("seed"); ax.set_ylabel("training wall-clock (s)")
    ax.set_title(f"End-to-end training wall per arm (G4, 5 seeds): total-wall speedup {s_tot:.2f}x")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    _save(fig, "f2_e2e_wall")


def f3_band():
    runs = _g4_runs()
    f = [runs[("fused", s)]["nashconv_on_policy"] for s in range(5)]
    q = [runs[("sequential", s)]["nashconv_on_policy"] for s in range(5)]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for s in range(5):
        ax.plot([0, 1], [f[s], q[s]], c="gray", lw=0.8, alpha=0.6)
    ax.scatter([0] * 5, f, c="#1f77b4", zorder=3, label="fused")
    ax.scatter([1] * 5, q, c="#d62728", zorder=3, label="sequential")
    for x, vals in ((0, f), (1, q)):
        med = statistics.median(vals)
        ax.hlines(med, x - 0.15, x + 0.15, color="k", lw=2, zorder=4)
    ax.set_xticks([0, 1], ["fused", "sequential"])
    ax.set_ylabel("final exact NashConv (on-policy)")
    ax.set_title("Same band, no systematic bias\n(per-seed scatter = compounding CUDA atomics; CPU exact)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    _save(fig, "f3_band")


def f4_exploit_vs_wall():
    runs = _g4_runs()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for m, c in (("fused", "#1f77b4"), ("sequential", "#d62728")):
        for s in range(5):
            d = runs[(m, s)]
            t, e = 0.0, []
            xs, ys = [], []
            for h in d["history"]:
                t += h["inner_resolve_wall_ms"] / 1e3 + h["train_wall_ms"] / 1e3
                if "exploit_onpolicy" in h:
                    xs.append(t); ys.append(h["exploit_onpolicy"])
            ax.plot(xs, ys, c=c, alpha=0.5, lw=1, marker=".", ms=4,
                    label=m if s == 0 else None)
    for s in (0, 1):
        d = _j(f"searchfree_nfsp_g4_seed{s}.json")
        xs = [p[0] for p in d["curve_seconds_vs_nashconv"]]
        ys = [p[1] for p in d["curve_seconds_vs_nashconv"]]
        ax.plot(xs, ys, c="#2ca02c", lw=1.2, ls="--", marker="x", ms=4,
                label="NFSP (search-free, avg-policy)" if s == 0 else None)
    ax.axhline(1.41667, ls=":", c="gray", lw=1)
    ax.text(0.5, 1.36, "uniform", fontsize=7, color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training wall-clock (s, log)"); ax.set_ylabel("exact NashConv (log)")
    ax.set_title("Exploitability vs wall-clock (G4): resolving arms vs search-free boundary")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    _save(fig, "f4_exploit_vs_wall")


if __name__ == "__main__":
    f1_keysweep(); f2_wall_bars(); f3_band(); f4_exploit_vs_wall()
    print("done.")
