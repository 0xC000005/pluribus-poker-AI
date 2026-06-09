#!/usr/bin/env python
"""HUNL trunk-reachable street-entry config census + topology-bucket estimator.

p1_design.json STEP 2 (estimation_harness). Two stages:

Stage 1 (census; CPU, minutes; the default): enumerate the trunk-reachable
street-entry config set EXACTLY by walking the betting abstraction's chip
arithmetic (scripts/solver.py StreetSolver._build_node semantics, reused via
poker_ai.rebel.hunl.tree_builder -- protected surfaces imported, never edited)
from 20000/20000 stacks, blinds 50/100: preflop -> flop -> turn -> river.
Street-end 'showdown' terminals' (pot, stacks) are the next street's entry
configs. Per config: build_street_tree, record topology_key / n_nodes /
n_actions / depth / terminal counts. Outputs per street: #configs, #topology
classes, class-size histogram (the POPULATION SIZES = the go/no-go datum),
n_nodes percentiles, and the B_max table from the design memory model
(default 6.5 GB budget, float64 dense v1; river H=1081, turn H=1128;
worst-case H=1326; v2 edge-segmented/O(H)-terminal variant as secondary).

Stage 2 (--gpu-probe; GPU; NOT run by default -- the regime GO/NO-GO datum):
for sampled real classes (p50/p95 n_nodes), run the EXISTING
fast_cfr.solve_cfr_levelsync_torch_batched_same_topology at H=1081 with
B in {1, 4, 16, B_max} and report fused-vs-sequential wall-time scaling.
Pre-registered thresholds (recorded in the JSON before any GPU run):
GO / CONDITIONAL-GO / NO-GO as defined in STAGE2_THRESHOLDS below.

JSON results -> autoresearch-session/rebel/hunl_topology_census.json.
No solver construction, no showdown matrices from real boards, no nets,
no Slumbot.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.rebel.hunl import tree_builder as tb  # noqa: E402

fast_cfr = tb.fast_cfr

# ---------------------------------------------------------------------------
# Protocol constants (the trunk being walked)
# ---------------------------------------------------------------------------
STACK = 20000
SMALL_BLIND = 50
BIG_BLIND = 100
PREFLOP_POT = SMALL_BLIND + BIG_BLIND
# Blinds consume two of the street's three raise slots in the abstraction's
# preflop trunk (the design-prototype convention: SB post + BB post), leaving
# one raise bucket + all-in preflop. This is what reproduces the prototype's
# 6 flop-entry configs.
DEFAULT_PREFLOP_N_RAISES = 2
# Player 0 = SB/button (acts FIRST preflop, LAST postflop); player 1 = BB.
PREFLOP_FIRST_TO_ACT = 0
POSTFLOP_FIRST_TO_ACT = 1

STREET_ORDER = ("flop", "turn", "river")
# Street-local private-hand widths (design memory_model): the kernel runs at
# local width, NOT the 1326 global interface index.
H_BY_STREET = {"turn": 1128, "river": 1081}
H_WORST_CASE = 1326

STAGE2_THRESHOLDS = {
    "go": (
        "fused speedup at B=16 >= 4x sequential AND dense-kernel B_max >= 16 "
        "for the p95 class"
    ),
    "conditional_go": (
        "speedup holds but memory binds (dense B_max < 16) -> escalate the "
        "edge-segmented/O(H)-terminal kernel into the critical path before "
        "schedule commit"
    ),
    "no_go": "fused speedup < 2x at B=16 -> pivot economics re-review",
}

# ---------------------------------------------------------------------------
# Memory model (design memory_model section; per batch element, float64 v1)
# ---------------------------------------------------------------------------
F64 = 8
F32 = 4


def dense_bytes_per_elem(nn, nn_dec, n_actions, h):
    """v1 dense layout per element: regret_sum+strategy_sum [nn,A,H] f64 x2,
    hr/vr/hvals/vvals [nn,H] f64 x4, retained per-level strategies
    ~[nn_dec,A,H] f64 x1."""
    return (2 * nn * n_actions * h + 4 * nn * h + nn_dec * n_actions * h) * F64


def dense_terminal_bytes_per_elem(h):
    """W/L/T/valid + contiguous transposes: [H,H] float32 x8."""
    return 8 * h * h * F32


def v2_bytes_per_elem(nn, n_edges, h):
    """v2 edge-segmented regrets [E,H] f64 (regret+strategy+retained) + node
    vectors [nn,H] f64 x4; O(H) terminal decomposition -> no [H,H] tensors."""
    return (3 * n_edges * h + 4 * nn * h) * F64


def memory_row(nn, nn_dec, n_edges, n_actions, h, budget_bytes):
    dense_elem = dense_bytes_per_elem(nn, nn_dec, n_actions, h)
    dense_term = dense_terminal_bytes_per_elem(h)
    v2_elem = v2_bytes_per_elem(nn, n_edges, h)
    return {
        "H": int(h),
        "dense_mb_per_elem": round(dense_elem / 1e6, 1),
        "dense_terminal_mb_per_elem": round(dense_term / 1e6, 1),
        "B_max_dense": int(budget_bytes // (dense_elem + dense_term)),
        "v2_mb_per_elem": round(v2_elem / 1e6, 1),
        "B_max_v2_segmented": int(budget_bytes // v2_elem),
    }


# ---------------------------------------------------------------------------
# Stage 1: exact trunk-reachable config enumeration + per-street census
# ---------------------------------------------------------------------------

def collect_street_end_entries(tree):
    """'showdown' street-end terminals' (pot, s0, s1) = next-street entries."""
    entries = set()
    pot = tree["pot"]
    s_h = tree["stacks_h"]
    s_v = tree["stacks_v"]
    for idx in tree["showdown_idx"]:
        entries.add((int(pot[idx]), int(s_h[idx]), int(s_v[idx])))
    return entries


def config_record(pot, s0, s1, street):
    tree = tb.build_street_tree(pot, s0, s1, POSTFLOP_FIRST_TO_ACT, street)
    n_nodes = int(tree["n_nodes"])
    depths = fast_cfr._tree_depths(tree["parent_idx"])
    n_showdown = int(len(tree["showdown_idx"]))
    n_folds = int(len(tree["hero_fold_idx"]) + len(tree["villain_fold_idx"]))
    record = {
        "pot": int(pot),
        "s0": int(s0),
        "s1": int(s1),
        "live": bool(s0 > 0 and s1 > 0),
        "topology_key": tb.topology_key(tree),
        "n_nodes": n_nodes,
        "n_actions": int(tree["n_actions"]),
        "n_decision": int(len(tree["decision_idx"])),
        "n_edges": int((tree["children"] >= 0).sum()),
        "depth": int(depths.max()) if depths.size else 0,
        "n_showdown_terminals": n_showdown,
        "n_fold_terminals": n_folds,
        "n_terminals": n_showdown + n_folds,
    }
    return record, tree


def enumerate_and_census(preflop_n_raises):
    """Walk preflop -> flop -> turn -> river; return per-street record lists."""
    preflop_tree = tb.build_betting_tree(
        PREFLOP_POT,
        STACK - SMALL_BLIND,  # s0 = SB (acts first preflop)
        STACK - BIG_BLIND,    # s1 = BB
        first_to_act=PREFLOP_FIRST_TO_ACT,
        n_raises=preflop_n_raises,
        to_call=BIG_BLIND - SMALL_BLIND,  # SB completes 50 to the BB's 100
    )
    current_entries = collect_street_end_entries(preflop_tree)

    street_records = {}
    timings = {}
    for street_idx, street in enumerate(STREET_ORDER):
        t0 = time.perf_counter()
        records = []
        next_entries = set()
        is_last = street_idx == len(STREET_ORDER) - 1
        for pot, s0, s1 in sorted(current_entries):
            record, tree = config_record(pot, s0, s1, street)
            records.append(record)
            if not is_last:
                next_entries |= collect_street_end_entries(tree)
        street_records[street] = records
        timings[street] = round(time.perf_counter() - t0, 2)
        # Bound memory: cached Node-object trees are only needed within a street.
        tb.street_tree_cache_clear()
        current_entries = next_entries
    return street_records, timings


def _percentile_nn(records, q):
    nn = np.array([r["n_nodes"] for r in records], dtype=np.int64)
    return int(np.percentile(nn, q, method="nearest"))


def _closest_record(records, nn_target):
    return min(records, key=lambda r: (abs(r["n_nodes"] - nn_target), r["pot"]))


def street_summary(records, street, budget_bytes):
    live = [r for r in records if r["live"]]
    dead = [r for r in records if not r["live"]]
    classes = defaultdict(list)
    for r in records:
        classes[r["topology_key"]].append(r)
    live_classes = {k: v for k, v in classes.items() if v[0]["live"]}

    class_sizes = sorted((len(v) for v in classes.values()), reverse=True)
    size_histogram = defaultdict(int)
    for size in class_sizes:
        size_histogram[size] += 1

    summary = {
        "n_configs": len(records),
        "n_configs_live": len(live),
        "n_configs_allin_dead": len(dead),
        "n_topology_classes": len(classes),
        "n_topology_classes_live": len(live_classes),
        "collapse_x": round(len(records) / max(len(classes), 1), 2),
        "class_sizes_top10": class_sizes[:10],
        "class_size_histogram": {str(k): v for k, v in sorted(size_histogram.items())},
        "class_size_median": float(np.median(class_sizes)) if class_sizes else 0.0,
        "n_classes_ge_16": sum(1 for s in class_sizes if s >= 16),
        "n_classes_ge_32": sum(1 for s in class_sizes if s >= 32),
        "n_classes_ge_64": sum(1 for s in class_sizes if s >= 64),
        "frac_configs_in_classes_ge_16": round(
            sum(s for s in class_sizes if s >= 16) / max(len(records), 1), 4
        ),
        "frac_configs_in_classes_ge_32": round(
            sum(s for s in class_sizes if s >= 32) / max(len(records), 1), 4
        ),
        "frac_size1_classes": round(
            sum(1 for s in class_sizes if s == 1) / max(len(classes), 1), 4
        ),
    }
    if live:
        summary["n_nodes_live"] = {
            "p50": _percentile_nn(live, 50),
            "p90": _percentile_nn(live, 90),
            "p95": _percentile_nn(live, 95),
            "p99": _percentile_nn(live, 99),
            "max": max(r["n_nodes"] for r in live),
        }
        summary["depth_live"] = {
            "p50": int(np.percentile([r["depth"] for r in live], 50, method="nearest")),
            "max": max(r["depth"] for r in live),
        }
        summary["asymmetric_live_entries"] = sum(1 for r in live if r["s0"] != r["s1"])

    b_max_table = None
    if street in H_BY_STREET and live:
        h_street = H_BY_STREET[street]
        rows = {}
        for label, q in (("p50", 50), ("p95", 95), ("max", 100)):
            target = (
                max(r["n_nodes"] for r in live) if label == "max" else _percentile_nn(live, q)
            )
            rec = _closest_record(live, target)
            rows[label] = {
                "n_nodes": rec["n_nodes"],
                "n_decision": rec["n_decision"],
                "n_edges": rec["n_edges"],
                "n_actions": rec["n_actions"],
                "example_config": [rec["pot"], rec["s0"], rec["s1"]],
                "H_street": memory_row(
                    rec["n_nodes"], rec["n_decision"], rec["n_edges"],
                    rec["n_actions"], h_street, budget_bytes,
                ),
                "H_worstcase_1326": memory_row(
                    rec["n_nodes"], rec["n_decision"], rec["n_edges"],
                    rec["n_actions"], H_WORST_CASE, budget_bytes,
                ),
            }
        b_max_table = {"H_street": H_BY_STREET[street], "rows": rows}
    return summary, b_max_table


# ---------------------------------------------------------------------------
# Stage 2: GPU fused-vs-sequential timing probe (IMPLEMENTED; not run by
# default -- this is the pre-registered regime GO/NO-GO measurement)
# ---------------------------------------------------------------------------

def _synthetic_showdown_matrices(h, rng):
    """Shape-faithful synthetic W/L/T/valid at width h (timing-only stand-ins;
    real per-board matrices have identical shapes/dtypes, which is all the
    wall-time measurement depends on)."""
    ranks = rng.integers(0, 7462, size=h)
    valid = np.ones((h, h), dtype=np.float32)
    np.fill_diagonal(valid, 0.0)
    ri = ranks[:, None]
    rj = ranks[None, :]
    win = ((rj > ri) * valid).astype(np.float32)
    lose = ((rj < ri) * valid).astype(np.float32)
    tie = ((rj == ri) * valid).astype(np.float32)
    return win, lose, tie, valid


def _median_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_gpu_probe(records, street, budget_bytes, n_iterations, repeats, device, seed=0):
    """Fused-vs-sequential scaling on sampled REAL topology classes (p50/p95
    n_nodes) at street-local H, B in {1, 4, 16, B_max}, via the EXISTING
    fast_cfr.solve_cfr_levelsync_torch_batched_same_topology."""
    import torch  # noqa: F401  (already imported transitively; explicit for clarity)

    h = H_BY_STREET[street]
    live = [r for r in records if r["live"]]
    classes = defaultdict(list)
    for r in live:
        classes[r["topology_key"]].append(r)

    chosen = []
    seen = set()
    for label, q in (("p50", 50), ("p95", 95)):
        target = _percentile_nn(live, q)
        key = min(
            classes,
            key=lambda k: (abs(classes[k][0]["n_nodes"] - target), -len(classes[k])),
        )
        if key not in seen:
            chosen.append((label, key))
            seen.add(key)

    rng = np.random.default_rng(seed)
    solve = fast_cfr.solve_cfr_levelsync_torch_batched_same_topology
    probe_results = []
    for label, key in chosen:
        members = classes[key]
        rep = members[0]
        mem = memory_row(
            rep["n_nodes"], rep["n_decision"], rep["n_edges"], rep["n_actions"],
            h, budget_bytes,
        )
        b_max = max(1, mem["B_max_dense"])
        b_list = sorted({1, 4, 16, b_max})
        b_top = max(b_list)

        configs = [members[i % len(members)] for i in range(b_top)]
        trees = [
            tb.build_street_tree(c["pot"], c["s0"], c["s1"], POSTFLOP_FIRST_TO_ACT, street)
            for c in configs
        ]
        # Pre-compile level edge groups so timing measures the pure solve
        # (matches the committed bake-off methodology).
        for tree in trees:
            tree.setdefault("_level_edge_groups", fast_cfr._level_edge_groups(tree))
        matrices = [_synthetic_showdown_matrices(h, rng) for _ in range(b_top)]
        pots = [float(c["pot"]) for c in configs]
        s0s = [float(c["s0"]) for c in configs]
        s1s = [float(c["s1"]) for c in configs]

        def _solve_slice(lo, hi, iters):
            wins, loses, ties, valids = zip(*matrices[lo:hi])
            return solve(
                trees[lo:hi], h, list(wins), list(loses), list(ties), list(valids),
                pots[lo:hi], s0s[lo:hi], s1s[lo:hi],
                n_iterations=iters, device=device,
            )

        # Warmup (CUDA context / kernel compile) outside the timer.
        _solve_slice(0, 1, 2)

        class_result = {
            "class_label": label,
            "topology_key": key,
            "class_size_in_census": len(members),
            "n_nodes": rep["n_nodes"],
            "n_actions": rep["n_actions"],
            "B_max_dense_f64_model": mem["B_max_dense"],
            "H": h,
            "n_iterations": n_iterations,
            "points": [],
        }
        for b in b_list:
            point = {"B": b}
            try:
                point["fused_s"] = _median_time(
                    lambda b=b: _solve_slice(0, b, n_iterations), repeats
                )
                point["sequential_s"] = _median_time(
                    lambda b=b: [
                        _solve_slice(i, i + 1, n_iterations) for i in range(b)
                    ],
                    repeats,
                )
                point["speedup_fused_vs_sequential"] = round(
                    point["sequential_s"] / max(point["fused_s"], 1e-12), 3
                )
                point["status"] = "ok"
            except torch.cuda.OutOfMemoryError as exc:
                point["status"] = "oom"
                point["error"] = str(exc).split("\n")[0]
                torch.cuda.empty_cache()
            class_result["points"].append(point)
        probe_results.append(class_result)

    return {
        "status": "ran",
        "device": device,
        "street": street,
        "repeats": repeats,
        "seed": seed,
        "matrix_source": "synthetic rank-comparison W/L/T (shape-faithful; timing-only)",
        "pre_registered_thresholds": STAGE2_THRESHOLDS,
        "classes": probe_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(result):
    print("\n" + "=" * 76)
    print("HUNL trunk-reachable street-entry census (Stage 1)")
    print("=" * 76)
    proto = result["protocol"]
    print(
        f"stacks {proto['stack']}/{proto['stack']}, blinds "
        f"{proto['small_blind']}/{proto['big_blind']}, preflop n_raises="
        f"{proto['preflop_n_raises']}, budget {proto['budget_bytes'] / 1e9:.1f} GB "
        f"(float64 dense v1)"
    )
    for street in STREET_ORDER:
        s = result["streets"][street]["summary"]
        print(f"\n--- {street.upper()} entry ---")
        print(
            f"  configs: {s['n_configs']} ({s['n_configs_live']} live, "
            f"{s['n_configs_allin_dead']} all-in/dead)"
        )
        print(
            f"  topology classes: {s['n_topology_classes']} "
            f"(collapse {s['collapse_x']}x; size-1 classes "
            f"{s['frac_size1_classes'] * 100:.1f}%)"
        )
        print(f"  top-10 class sizes: {s['class_sizes_top10']}")
        print(
            f"  classes >=16/>=32/>=64 members: {s['n_classes_ge_16']}/"
            f"{s['n_classes_ge_32']}/{s['n_classes_ge_64']}  "
            f"(configs covered by >=16-classes: "
            f"{s['frac_configs_in_classes_ge_16'] * 100:.1f}%)"
        )
        if "n_nodes_live" in s:
            nn = s["n_nodes_live"]
            print(
                f"  n_nodes (live): p50={nn['p50']} p90={nn['p90']} "
                f"p95={nn['p95']} p99={nn['p99']} max={nn['max']}"
            )
        table = result["streets"][street].get("b_max_table")
        if table:
            print(
                f"  B_max table (H_street={table['H_street']}, budget "
                f"{proto['budget_bytes']:.2g} B):"
            )
            for label, row in table["rows"].items():
                hs = row["H_street"]
                wc = row["H_worstcase_1326"]
                print(
                    f"    {label:>4} nn={row['n_nodes']:>5} (dec={row['n_decision']}, "
                    f"edges={row['n_edges']}): dense {hs['dense_mb_per_elem']:.0f}MB/elem"
                    f" -> B_max={hs['B_max_dense']:>4} | v2 {hs['v2_mb_per_elem']:.0f}MB"
                    f" -> B_max={hs['B_max_v2_segmented']:>4} | H=1326 dense"
                    f" B_max={wc['B_max_dense']}"
                )
    probe = result["stage2_gpu_probe"]
    print(f"\nStage 2 GPU probe: {probe['status']}")
    if probe["status"] == "ran":
        for cls in probe["classes"]:
            print(
                f"  class {cls['class_label']} nn={cls['n_nodes']} "
                f"(census size {cls['class_size_in_census']}):"
            )
            for point in cls["points"]:
                if point["status"] == "ok":
                    print(
                        f"    B={point['B']:>3}: fused {point['fused_s']:.3f}s vs "
                        f"sequential {point['sequential_s']:.3f}s -> "
                        f"{point['speedup_fused_vs_sequential']}x"
                    )
                else:
                    print(f"    B={point['B']:>3}: {point['status']}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "autoresearch-session/rebel/hunl_topology_census.json"),
    )
    parser.add_argument(
        "--budget-gb", type=float, default=6.5,
        help="usable GPU memory budget in decimal GB (design memory model)",
    )
    parser.add_argument(
        "--preflop-nr", type=int, default=DEFAULT_PREFLOP_N_RAISES,
        help="raise slots consumed by the blinds in the preflop trunk",
    )
    parser.add_argument(
        "--gpu-probe", action="store_true",
        help="Stage 2: run the fused-vs-sequential GPU timing probe "
        "(the regime GO/NO-GO datum). NOT run by default.",
    )
    parser.add_argument("--probe-street", default="river", choices=("river", "turn"))
    parser.add_argument("--probe-iters", type=int, default=100)
    parser.add_argument("--probe-repeats", type=int, default=3)
    parser.add_argument("--probe-device", default="cuda")
    parser.add_argument("--probe-seed", type=int, default=0)
    args = parser.parse_args()

    budget_bytes = int(args.budget_gb * 1e9)
    t_start = time.perf_counter()
    street_records, timings = enumerate_and_census(args.preflop_nr)

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "stack": STACK,
            "small_blind": SMALL_BLIND,
            "big_blind": BIG_BLIND,
            "preflop_n_raises": args.preflop_nr,
            "preflop_root": {
                "pot": PREFLOP_POT,
                "s0_sb": STACK - SMALL_BLIND,
                "s1_bb": STACK - BIG_BLIND,
                "first_to_act": "player0 (SB/button)",
                "to_call": SMALL_BLIND,
            },
            "postflop_first_to_act": "player1 (BB)",
            "abstraction": (
                "scripts/solver.py _build_node: BET_FRACS "
                "(0.25,0.5,0.75,1.0,1.5,2.0)*pot + all-in, 3-raise cap, "
                "min_raise_contribution legality, int() bet rounding"
            ),
            "budget_bytes": budget_bytes,
            "dtype": "float64 dense v1 (terminal matrices float32)",
            "memory_model": {
                "dense_per_elem": "(2*nn*A*H + 4*nn*H + nn_dec*A*H) * 8",
                "dense_terminal_per_elem": "8 * H*H * 4 (W/L/T/valid + transposes)",
                "v2_per_elem": "(3*E*H + 4*nn*H) * 8 (edge-segmented, O(H) terminals)",
                "B_max": "floor(budget / (per_elem + terminal_per_elem))",
            },
            "H_by_street": H_BY_STREET,
            "H_worst_case": H_WORST_CASE,
            "design_prototype_config_counts": {"flop": 6, "turn": 426, "river": "6k+"},
        },
        "stage2_pre_registered_thresholds": STAGE2_THRESHOLDS,
        "streets": {},
        "census_seconds_per_street": timings,
    }

    for street in STREET_ORDER:
        records = street_records[street]
        summary, b_max_table = street_summary(records, street, budget_bytes)
        result["streets"][street] = {
            "summary": summary,
            "b_max_table": b_max_table,
            "configs": [
                [r["pot"], r["s0"], r["s1"], r["topology_key"], r["n_nodes"],
                 r["depth"], r["n_showdown_terminals"]]
                for r in records
            ],
            "configs_schema": [
                "pot", "s0", "s1", "topology_key", "n_nodes", "depth",
                "n_showdown_terminals",
            ],
        }

    if args.gpu_probe:
        result["stage2_gpu_probe"] = run_gpu_probe(
            street_records[args.probe_street],
            args.probe_street,
            budget_bytes,
            args.probe_iters,
            args.probe_repeats,
            args.probe_device,
            args.probe_seed,
        )
    else:
        result["stage2_gpu_probe"] = {
            "status": "not_run",
            "note": (
                "Stage 2 is the pre-registered regime GO/NO-GO measurement; "
                "run with --gpu-probe on the 3070 Ti when the GPU is free."
            ),
            "pre_registered_thresholds": STAGE2_THRESHOLDS,
        }

    result["census_total_seconds"] = round(time.perf_counter() - t_start, 2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {output_path}")
    print_summary(result)


if __name__ == "__main__":
    main()
