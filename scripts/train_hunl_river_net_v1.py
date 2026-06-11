"""River PBS net v1: suit-isomorphism canonicalization (the ONE v1 change).

Extends the v0 trainer (``scripts/train_hunl_river_net.py``, imported and
reused -- net, loss, optimizer, epochs, seeds, queue, census distribution all
IDENTICAL) with the lossless suit symmetry (``poker_ai/rebel/hunl/suit_iso``,
exact-tested in test/unit/test_hunl_suit_iso.py):

  * every (board, ranges) input is mapped to its canonical suit relabeling at
    TARGET-GENERATION time (boards are canonicalized before belief sampling --
    Dirichlet beliefs are coordinate-exchangeable, so sampling directly on the
    canonical board is distributionally identical to sampling raw and
    relabeling) and at INFERENCE time (canonicalize input, de-canonicalize
    output);
  * the v0 target data is REUSED: its 16.5k solved rows are canonicalized
    post-hoc (a pure index relabeling -- bit-exactly invertible, verified row
    by row with the round-trip check), then ~14.5k fresh canonical-board specs
    are generated to reach ~30k train rows within the 5 GPU-hour budget;
  * the board-disjoint validation discipline is enforced in CANONICAL space:
    reused train rows whose canonical board collides with a canonical val
    board are DROPPED, new train boards exclude canonical val boards, and
    disjointness is asserted on canonical boards (no leakage via suit
    relabeling).

Outputs the v1 board-generalization curve at the v0-matched sample counts
(2.5k/5k/10k/15.5k) plus the ~30k point, the 5-spot sanity check through the
full canonicalize -> net -> de-canonicalize inference path against fresh
direct solves, models/hunl_river_net_v1.pt, and
autoresearch-session/rebel/river_net_boardgen_v1.json.

Usage:
    .venv/bin/python scripts/train_hunl_river_net_v1.py \
        --target-total-specs 30000 --seed 0 --max-gen-hours 4.6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_hunl_river_net as v0                              # noqa: E402

from poker_ai.rebel.hunl import population_solver as pop      # noqa: E402
from poker_ai.rebel.hunl import subgame_spec as sgs           # noqa: E402
from poker_ai.rebel.hunl import suit_iso as siso              # noqa: E402
from poker_ai.rebel.hunl import targets as tgt                # noqa: E402

REPO = v0.REPO
SCALE = v0.SCALE
N_BOARD = v0.N_BOARD
N_SC = v0.N_SC
NG = v0.NG


# ---------------------------------------------------------------------------
# Row-level canonicalization (post-hoc relabeling of stored v0 rows)
# ---------------------------------------------------------------------------

def permute_row(x, y, w, perm):
    """Apply one suit perm to a (x, y, w) training row (pure relabeling)."""
    xp, yp, wp = x.copy(), y.copy(), w.copy()
    board = tuple(np.flatnonzero(x[:N_BOARD]).tolist())
    xp[:N_BOARD] = 0.0
    xp[list(siso.permute_board(board, perm))] = 1.0
    xp[N_BOARD + N_SC:N_BOARD + N_SC + NG] = siso.permute_global(
        x[N_BOARD + N_SC:N_BOARD + N_SC + NG], perm)
    xp[N_BOARD + N_SC + NG:] = siso.permute_global(
        x[N_BOARD + N_SC + NG:], perm)
    yp[:NG] = siso.permute_global(y[:NG], perm)
    yp[NG:] = siso.permute_global(y[NG:], perm)
    wp[:NG] = siso.permute_global(w[:NG], perm)
    wp[NG:] = siso.permute_global(w[NG:], perm)
    return xp, yp, wp


def canonicalize_rows(X, Y, W, label, verify_roundtrip=True):
    """Suit-canonicalize stored training rows; verify losslessness row-by-row.

    Returns (Xc, Yc, Wc, stats). The round-trip check de-canonicalizes every
    relabeled row with the inverse perm and asserts BIT-IDENTITY with the
    original (the exactness contract of test_hunl_suit_iso.py applied to the
    actual reused data).
    """
    Xc, Yc, Wc = X.copy(), Y.copy(), W.copy()
    n_identity = 0
    t0 = time.perf_counter()
    for i in range(X.shape[0]):
        board = tuple(np.flatnonzero(X[i, :N_BOARD]).tolist())
        r0 = X[i, N_BOARD + N_SC:N_BOARD + N_SC + NG]
        r1 = X[i, N_BOARD + N_SC + NG:]
        canon, perm = siso.canonicalize(board, r0, r1)
        if perm == siso.IDENTITY_PERM:
            n_identity += 1
            continue
        Xc[i], Yc[i], Wc[i] = permute_row(X[i], Y[i], W[i], perm)
        if verify_roundtrip:
            inv = siso.invert_perm(perm)
            xb, yb, wb = permute_row(Xc[i], Yc[i], Wc[i], inv)
            if not (np.array_equal(xb, X[i]) and np.array_equal(yb, Y[i])
                    and np.array_equal(wb, W[i])):
                raise AssertionError(
                    f"round-trip failure canonicalizing {label} row {i}")
    stats = {
        "n_rows": int(X.shape[0]),
        "n_identity_perm": int(n_identity),
        "n_relabeled": int(X.shape[0] - n_identity),
        "roundtrip_verified_bit_identical": bool(verify_roundtrip),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    return Xc, Yc, Wc, stats


def sample_new_canonical_boards(n_boards, seed, exclude):
    """Fresh canonical river boards: raw-uniform draws pushed through the
    canonical map (= orbit-size-weighted, the exact push-forward of v0's raw
    board distribution), deduped and excluding ``exclude`` (canonical)."""
    rng = np.random.default_rng(seed)
    out, seen = [], set(exclude)
    while len(out) < n_boards:
        b = tuple(sorted(rng.choice(52, size=5, replace=False).tolist()))
        cb = siso.canonical_board(b)
        if cb in seen:
            continue
        seen.add(cb)
        out.append(cb)
    return out


# ---------------------------------------------------------------------------
# Soundness re-measurement (fresh numbers into the report, not transcribed)
# ---------------------------------------------------------------------------

def measure_soundness(device):
    """Run the unit-test soundness cases end-to-end and record the numbers."""
    import importlib.util
    path = os.path.join(REPO, "test", "unit", "test_hunl_suit_iso.py")
    spec = importlib.util.spec_from_file_location("test_hunl_suit_iso", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {"cpu_f64": mod._run_soundness("cpu"),
           "tol_cpu_chips": mod.TOL_SOUNDNESS_CHIPS,
           "tol_cuda_chips": mod.TOL_CUDA_CHIPS,
           "n_iterations": mod.N_ITERATIONS}
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        out["cuda_f64"] = mod._run_soundness("cuda")
    for key in ("cpu_f64", "cuda_f64"):
        for d in out.get(key, []):
            d["board"] = list(d["board"])
            d["perm"] = list(d["perm"])
    return out


# ---------------------------------------------------------------------------
# Sanity gate through the FULL canonicalize -> net -> de-canonicalize path
# ---------------------------------------------------------------------------

def sanity_gate_v1(net, Xva, Yva_chips, pots, n_iterations, device,
                   n_spots=5, seed=123):
    """v0's 5-spot gate, but each held-out spot is first pushed OUT of
    canonical space by a random non-identity suit perm (a raw-space input),
    fresh-solved directly, and compared against the net run through the v1
    inference contract: canonicalize input -> net -> de-canonicalize output.
    Also asserts the wrapper recovers the stored canonical row bit-exactly.
    """
    order = np.argsort(pots)
    picks = [order[int(q * (len(order) - 1))]
             for q in np.linspace(0.0, 1.0, n_spots)]
    rng = np.random.default_rng(seed)
    raw_specs, applied = [], []
    for i in picks:
        perm = siso.SUIT_PERMS[int(rng.integers(1, len(siso.SUIT_PERMS)))]
        xr, _, _ = permute_row(Xva[i], Yva_chips[i], Yva_chips[i], perm)
        raw_specs.append(v0.rebuild_spec_from_row(xr))
        applied.append((perm, xr))
    t0 = time.perf_counter()
    results = pop.solve_population(
        raw_specs, n_iterations=n_iterations, dtype=torch.float64,
        device=device, b_max=1, compute_value_pass=True)
    solve_secs = time.perf_counter() - t0

    dev = torch.device(device)
    net_dev = net.to(dev)
    spots = []
    for j, (i, raw, res) in enumerate(zip(picks, raw_specs, results)):
        perm_applied, xr = applied[j]
        # ---- the inference wrapper: canonicalize -> net -> de-canonicalize
        board_raw = tuple(np.flatnonzero(xr[:N_BOARD]).tolist())
        r0r = xr[N_BOARD + N_SC:N_BOARD + N_SC + NG]
        r1r = xr[N_BOARD + N_SC + NG:]
        canon, perm = siso.canonicalize(board_raw, r0r, r1r)
        x_canon, _, _ = permute_row(xr, np.zeros_like(Yva_chips[i]),
                                    np.zeros_like(Yva_chips[i]), perm)
        assert np.array_equal(x_canon, Xva[i]), \
            "wrapper failed to recover the stored canonical row bit-exactly"
        with torch.no_grad():
            pred_canon = net_dev(
                torch.as_tensor(x_canon[None, :], device=dev)
            ).cpu().numpy()[0]
        inv = siso.invert_perm(perm)
        pred0 = siso.permute_global(pred_canon[:NG], inv)
        pred1 = siso.permute_global(pred_canon[NG:], inv)
        # ---- compare in the RAW spec's local space (chips)
        l2g = sgs.local_to_global(raw.board)
        pot = float(raw.pot)
        v_solve = np.concatenate([res.v0, res.v1])
        v_net = np.concatenate([pred0[l2g], pred1[l2g]]) * pot
        # stored canonical target de-canonicalized to raw space
        y0_raw = siso.permute_global(Yva_chips[i, :NG], inv)
        y1_raw = siso.permute_global(Yva_chips[i, NG:], inv)
        v_row = np.concatenate([y0_raw[l2g], y1_raw[l2g]])
        w = np.concatenate([raw.local_ranges()[0], raw.local_ranges()[1]])
        net_err = np.abs(v_net - v_solve)
        spots.append({
            "spot": j, "board_raw": list(raw.board),
            "suit_perm_applied": list(perm_applied),
            "pot": raw.pot, "eff_stack": min(raw.stack0, raw.stack1),
            "net_vs_solve_linf_chips": round(float(net_err.max()), 4),
            "net_vs_solve_linf_over_pot": round(float(net_err.max()) / pot, 6),
            "net_vs_solve_reach_wmae_chips": round(
                float((w * net_err).sum() / (w.sum() + 1e-9)), 4),
            "net_vs_solve_reach_wmae_over_pot": round(
                float((w * net_err).sum() / (w.sum() + 1e-9)) / pot, 6),
            "storedrow_vs_fresh_solve_linf_chips": round(
                float(np.abs(v_row - v_solve).max()), 6),
        })
    net.cpu()
    return {"protocol": {
                "n_spots": n_spots,
                "selection": "held-out val rows at pot quantiles 0..1, each "
                             "pushed out of canonical space by a random "
                             "non-identity suit perm (raw-space input)",
                "inference_path": "canonicalize input -> net (canonical "
                                  "space) -> de-canonicalize output "
                                  "(suit_iso contract); canonical-row "
                                  "recovery asserted bit-exact",
                "solver": f"solve_population float64 {device} B=1 "
                          f"{n_iterations} iters (fresh, end-to-end, on the "
                          "RAW non-canonical spec)"},
            "solve_wall_seconds": round(solve_secs, 2),
            "spots": spots}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-total-specs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-beliefs-per-board", type=int, default=8)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--zs-lambda", type=float, default=1.0)
    ap.add_argument("--budget-bytes", type=float, default=5.2e9)
    ap.add_argument("--boards-per-chunk", type=int, default=24)
    ap.add_argument("--max-gen-hours", type=float, default=4.6,
                    help="wall budget for NEW target generation (the reused "
                         "v0 rows cost zero new GPU time)")
    ap.add_argument("--curve-cuts", default="2500,5000,10000,15552,30000")
    ap.add_argument("--v0-data", default=os.path.join(
        REPO, "autoresearch-session", "rebel", "hunl_river_net_data_seed0.npz"))
    ap.add_argument("--v0-report", default=os.path.join(
        REPO, "autoresearch-session", "rebel", "river_net_boardgen.json"))
    ap.add_argument("--data-out", default=None)
    ap.add_argument("--json-out", default=os.path.join(
        REPO, "autoresearch-session", "rebel", "river_net_boardgen_v1.json"))
    ap.add_argument("--ckpt-out", default=os.path.join(
        REPO, "models", "hunl_river_net_v1.pt"))
    ap.add_argument("--skip-gen", action="store_true",
                    help="reuse --data-out npz from a previous v1 run")
    ap.add_argument("--resume-partial", action="store_true",
                    help="resume new-spec generation from *_partial.npz / "
                         "*_seg*.npz checkpoints of a killed run (the board "
                         "stream is deterministic; resumed beliefs use a "
                         "fresh documented rng stream)")
    args = ap.parse_args()

    data_out = args.data_out or os.path.join(
        REPO, "autoresearch-session", "rebel",
        f"hunl_river_net_data_v1_seed{args.seed}.npz")
    t_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    live_configs, by_pot_eff = v0.load_census_configs()
    sampler = v0.make_config_sampler(live_configs)
    print(f"census: {len(live_configs)} live river-entry configs", flush=True)

    # ---- measure suit-iso soundness fresh (the v1 mechanism's gate)
    soundness = measure_soundness(args.device)
    for key in ("cpu_f64", "cuda_f64"):
        for d in soundness.get(key, []):
            print(f"[soundness {key}] nn={d['n_nodes']} pot={d['pot']}: "
                  f"L-inf {d['linf_chips']:.3e} chips", flush=True)

    chunk_log = []
    if args.skip_gen and os.path.exists(data_out):
        d = np.load(data_out, allow_pickle=True)
        Xtr, Ytr, Wtr = d["X_train"], d["Y_train"], d["W_train"]
        Xva, Yva, Wva = d["X_val"], d["Y_val"], d["W_val"]
        gen_info = json.loads(str(d["gen_info"]))
        canon_stats = gen_info["canonicalization"]
        print(f"loaded {data_out}: {Xtr.shape[0]} train / {Xva.shape[0]} val",
              flush=True)
    else:
        # ---- 1) canonicalize the reused v0 rows (lossless, verified)
        d0 = np.load(args.v0_data, allow_pickle=True)
        Xva, Yva, Wva, va_stats = canonicalize_rows(
            d0["X_val"], d0["Y_val"], d0["W_val"], "val")
        Xr, Yr, Wr, tr_stats = canonicalize_rows(
            d0["X_train"], d0["Y_train"], d0["W_train"], "train")
        print(f"canonicalized v0 rows: val {va_stats}, train {tr_stats}",
              flush=True)

        # ---- 2) canonical-space board-disjointness: drop colliding train rows
        val_canon_boards = set(v0.boards_of_rows(Xva))
        train_canon_boards = v0.boards_of_rows(Xr)
        keep = np.array([b not in val_canon_boards
                         for b in train_canon_boards])
        n_dropped = int((~keep).sum())
        if n_dropped:
            dropped_boards = sorted(
                {train_canon_boards[i] for i in np.flatnonzero(~keep)})
            print(f"dropped {n_dropped} reused train rows on "
                  f"{len(dropped_boards)} canonical boards colliding with "
                  f"val: {dropped_boards}", flush=True)
        Xr, Yr, Wr = Xr[keep], Yr[keep], Wr[keep]

        # ---- 3) fresh canonical boards -> new specs up to the target total
        # Resume support: a killed run leaves *_partial.npz (rows emitted up
        # to its last 15-chunk save). It is converted to a *_seg{i}.npz
        # segment (rows + boards consumed off the DETERMINISTIC board
        # stream); specs that were pending in the deferred queue at the kill
        # are lost and compensated by extending the board stream to reach the
        # target total. Resumed beliefs use a fresh documented rng stream
        # (beliefs are i.i.d. per board, so this is distribution-identical).
        partial_path = data_out.replace(".npz", "_partial.npz")
        seg_prefix = data_out.replace(".npz", "_seg")
        Xsegs, Ysegs, Wsegs = [], [], []
        boards_done = 0
        if args.resume_partial:
            import glob
            segs = sorted(glob.glob(seg_prefix + "*.npz"))
            if os.path.exists(partial_path):
                dp = np.load(partial_path)
                seg_path = f"{seg_prefix}{len(segs)}.npz"
                np.savez(seg_path, X=dp["X"], Y=dp["Y"], W=dp["W"],
                         boards_done=(int(dp["chunks_done"])
                                      * args.boards_per_chunk))
                os.remove(partial_path)
                segs.append(seg_path)
            for sp in segs:
                ds = np.load(sp)
                Xsegs.append(ds["X"])
                Ysegs.append(ds["Y"])
                Wsegs.append(ds["W"])
                boards_done += int(ds["boards_done"])
            if segs:
                print(f"resume: {sum(x.shape[0] for x in Xsegs)} rows from "
                      f"{boards_done} boards in {len(segs)} segment(s)",
                      flush=True)
        n_resumed = int(sum(x.shape[0] for x in Xsegs))
        n_needed = max(
            0, args.target_total_specs - Xr.shape[0] - n_resumed)
        n_todo_boards = (n_needed + args.n_beliefs_per_board - 1) \
            // args.n_beliefs_per_board
        exclude = val_canon_boards | set(v0.boards_of_rows(Xr))
        new_boards = sample_new_canonical_boards(
            boards_done + n_todo_boards, seed=args.seed + 999,
            exclude=exclude)[boards_done:]
        print(f"reused {Xr.shape[0]} canonical rows (+{n_resumed} resumed); "
              f"generating {n_needed} new specs on {len(new_boards)} fresh "
              f"canonical boards", flush=True)
        rng_new = np.random.default_rng(
            [args.seed, 3] if boards_done == 0
            else [args.seed, 4, boards_done])
        q_new = v0.EmpiricalBMaxQueue(
            n_iterations=args.iters, dtype=torch.float64, device=args.device,
            budget_bytes=args.budget_bytes, deferred=True)
        Xn, Yn, Wn, new_info = v0.generate_rows(
            new_boards, args.n_beliefs_per_board, sampler, rng_new, q_new,
            "v1-new", boards_per_chunk=args.boards_per_chunk,
            max_seconds=args.max_gen_hours * 3600.0, chunk_log=chunk_log,
            partial_path=partial_path)

        Xtr = np.concatenate([Xr] + Xsegs + [Xn], axis=0)
        Ytr = np.concatenate([Yr] + Ysegs + [Yn], axis=0)
        Wtr = np.concatenate([Wr] + Wsegs + [Wn], axis=0)
        canon_stats = {
            "val": va_stats, "train_reused": tr_stats,
            "n_train_rows_dropped_canonical_val_collision": n_dropped,
            "n_reused_rows": int(Xr.shape[0]),
            "n_new_rows": int(n_resumed + Xn.shape[0]),
        }
        gen_info = {"new": new_info, "canonicalization": canon_stats,
                    "n_new_boards_planned": len(new_boards),
                    "n_resumed_rows": n_resumed,
                    "n_resumed_boards": boards_done,
                    "resumed_belief_stream": (
                        None if boards_done == 0
                        else [args.seed, 4, boards_done]),
                    "v0_data_reused": args.v0_data}
        np.savez_compressed(
            data_out, X_train=Xtr, Y_train=Ytr, W_train=Wtr,
            X_val=Xva, Y_val=Yva, W_val=Wva, gen_info=json.dumps(gen_info))
        print(f"saved dataset -> {data_out}", flush=True)

    # ---- metadata + canonical-space integrity checks
    pots_tr, effs_tr, nns_tr = v0.row_meta(Xtr, by_pot_eff)
    pots_va, effs_va, nns_va = v0.row_meta(Xva, by_pot_eff)
    train_boards = v0.boards_of_rows(Xtr)
    val_boards = v0.boards_of_rows(Xva)
    train_board_set, val_board_set = set(train_boards), set(val_boards)
    assert not (train_board_set & val_board_set), \
        "canonical-space board split leakage!"
    for b in list(train_board_set)[:64] + list(val_board_set):
        assert siso.canonical_board(b) == b, f"non-canonical board {b}"
    print(f"canonical disjointness OK: {len(train_board_set)} train vs "
          f"{len(val_board_set)} val canonical boards", flush=True)
    Yn_tr = (Ytr / pots_tr[:, None]).astype(np.float32)
    Yn_va = (Yva / pots_va[:, None]).astype(np.float32)
    valid_g = sgs.global_valid_matrix()
    k = min(64, Xva.shape[0])
    r0 = Xva[:k, N_BOARD + N_SC:N_BOARD + N_SC + NG].astype(np.float64)
    r1 = Xva[:k, N_BOARD + N_SC + NG:].astype(np.float64)
    lhs = (Wva[:k].astype(np.float64) * Yn_va[:k].astype(np.float64)).sum(1)
    rhs = np.einsum("bi,ij,bj->b", r0, valid_g.astype(np.float64), r1)
    zs_target_resid = float(np.abs(lhs - rhs).max())
    print(f"target zero-sum identity residual (max over {k} val rows, "
          f"pot units): {zs_target_resid:.2e}", flush=True)

    # ---- the v1 board-generalization curve (v0 protocol, matched cuts)
    cuts = sorted({min(int(c), Xtr.shape[0])
                   for c in args.curve_cuts.split(",")})
    curve = []
    final_net = None
    for cut in cuts:
        t0 = time.perf_counter()
        net, m = v0.train_net(
            Xtr[:cut], Yn_tr[:cut], Wtr[:cut], Xva, Yn_va, Wva,
            hidden=args.hidden, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            zs_lambda=args.zs_lambda, seed=args.seed, device=args.device)
        m["train_wall_seconds"] = round(time.perf_counter() - t0, 1)
        m["n_train_boards"] = len(set(train_boards[:cut]))
        curve.append(m)
        print(f"[curve v1] n_train={cut}: val wMAE "
              f"{m['val_reach_weighted_mae_potfrac']:.4f} pot-frac "
              f"(train {m['train_reach_weighted_mae_potfrac']:.4f}, "
              f"best ep {m['best_epoch']})", flush=True)
        if cut == cuts[-1]:
            final_net = net

    # ---- v0 vs v1 at matched n
    with open(args.v0_report) as f:
        v0_curve = json.load(f)["boardgen_curve"]
    v0_by_n = {c["n_train"]: c["val_reach_weighted_mae_potfrac"]
               for c in v0_curve}
    comparison = []
    for m in curve:
        n = m["n_train"]
        v1_mae = m["val_reach_weighted_mae_potfrac"]
        v0_mae = v0_by_n.get(n)
        comparison.append({
            "n_train": n,
            "v0_val_wmae_potfrac": v0_mae,
            "v1_val_wmae_potfrac": v1_mae,
            "abs_improvement": (round(v0_mae - v1_mae, 6)
                                if v0_mae is not None else None),
            "rel_improvement_pct": (round(100.0 * (v0_mae - v1_mae) / v0_mae, 2)
                                    if v0_mae is not None else None),
        })
        print(f"[v0-vs-v1] n={n}: v0 {v0_mae} -> v1 {v1_mae}", flush=True)

    # ---- per-config breakdown + sanity gate (canonical inference path)
    seen_configs = {(int(p), int(e)) for p, e in zip(pots_tr, effs_tr)}
    breakdown = v0.per_config_breakdown(
        final_net, Xva, Yn_va, Wva, pots_va, effs_va, nns_va, seen_configs,
        args.device)
    sanity = sanity_gate_v1(final_net, Xva, Yva, pots_va, args.iters,
                            args.device, n_spots=5)

    peak_vram = (int(torch.cuda.max_memory_allocated())
                 if torch.cuda.is_available() else None)
    wall = time.perf_counter() - t_start

    # ---- checkpoint
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({
        "state_dict": final_net.state_dict(),
        "arch": {"class": "BoardGeneralRiverNet", "d_in": tgt.X_DIM,
                 "d_out": tgt.Y_DIM, "hidden": args.hidden},
        "conventions": {
            "x": "board 52-hot | pot/2e4 | eff_stack/2e4 | r0n[1326] | r1n[1326]",
            "y": "v0|v1 per-hand V* CFVs normalized by POT (chips/pot), "
                 "global 1326 index, zero off-board",
            "value_convention": "net-from-river-start counterfactual, "
                                "opponent-reach-weighted (population_solver "
                                "value pass)",
            "denormalize": "chips = net(x) * pot",
            "suit_canonicalization": "REQUIRED at inference: canonicalize the "
                                     "(board, r0, r1) input with poker_ai.rebel"
                                     ".hunl.suit_iso.canonicalize, run the net "
                                     "in canonical space, map outputs back "
                                     "with permute_global(y, invert_perm(perm))"
                                     " -- the net only ever sees canonical "
                                     "inputs",
        },
        "training": {"seed": args.seed, "epochs": args.epochs,
                     "batch_size": args.batch_size, "lr": args.lr,
                     "zs_lambda": args.zs_lambda,
                     "n_train": int(Xtr.shape[0]),
                     "target_iters": args.iters, "target_dtype": "float64"},
        "val_metrics": curve[-1],
    }, args.ckpt_out)

    # ---- verdict
    matched = [c for c in comparison if c["abs_improvement"] is not None]
    final_v1 = curve[-1]["val_reach_weighted_mae_potfrac"]
    v0_final = v0_by_n.get(15552)
    verdict = {
        "matched_n_mean_rel_improvement_pct": round(
            float(np.mean([c["rel_improvement_pct"] for c in matched])), 2)
        if matched else None,
        "matched_n_range_rel_improvement_pct": (
            [min(c["rel_improvement_pct"] for c in matched),
             max(c["rel_improvement_pct"] for c in matched)]
            if matched else None),
        "v1_final_n": curve[-1]["n_train"],
        "v1_final_val_wmae_potfrac": final_v1,
        "v0_final_val_wmae_potfrac_at_15552": v0_final,
        "total_improvement_v0_15552_to_v1_final_pct": (
            round(100.0 * (v0_final - final_v1) / v0_final, 2)
            if v0_final else None),
    }

    n_params = sum(p.numel() for p in final_net.parameters())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": "river PBS net v1: suit-isomorphism canonicalization "
                     "(lossless game symmetry; the ONE v1 change)",
        "protocol": {
            "v0_protocol": "inherited verbatim from river_net_boardgen.json "
                           "(same targets chain, census config distribution, "
                           "Dirichlet beliefs, net 2706->512x2->2652 "
                           f"({n_params} params), loss, optimizer, epochs, "
                           "model selection, seed) -- imported from "
                           "scripts/train_hunl_river_net.py",
            "v1_change": "suit-isomorphism canonicalization (suit_iso.py, "
                         "exact-tested): canonical suit relabeling applied at "
                         "target-generation time (canonical boards, beliefs "
                         "sampled directly in canonical space -- Dirichlet "
                         "exchangeability) and at inference time "
                         "(canonicalize input / de-canonicalize output)",
            "data_reuse": "v0's 16.5k solved rows canonicalized post-hoc "
                          "(pure index relabeling, round-trip verified "
                          "bit-identical per row); new specs solved with the "
                          "IDENTICAL parity-gated chain",
            "board_split": "board-disjoint validation enforced in CANONICAL "
                           "space: colliding reused train rows dropped, new "
                           "boards exclude val, disjointness asserted "
                           "(no leakage via suit relabeling)",
            "n_iterations": args.iters,
            "seed": args.seed,
            "governance": "ONE principled change, no sweeps; float64 targets; "
                          "no Slumbot anything; fixed seeds; no commits",
        },
        "suit_iso_soundness": soundness,
        "throughput": {
            "gen_info": gen_info,
            "chunk_log_tail": chunk_log[-8:],
            "v0_reference_seconds_per_spec": 1.12,
        },
        "dataset": {
            "n_train_rows": int(Xtr.shape[0]),
            "n_train_boards_canonical": len(train_board_set),
            "n_val_rows": int(Xva.shape[0]),
            "n_val_boards_canonical": len(val_board_set),
            "n_train_configs_distinct": len(seen_configs),
            "canonicalization": canon_stats,
            "target_zero_sum_identity_residual_max_potfrac": zs_target_resid,
        },
        "boardgen_curve_v1": curve,
        "v0_vs_v1_at_matched_n": comparison,
        "verdict": verdict,
        "per_config_breakdown_unseen_boards": breakdown,
        "sanity_gate_5_spots": sanity,
        "wall_clock_seconds_total": round(wall, 1),
        "peak_vram_bytes": peak_vram,
        "files": {"checkpoint": args.ckpt_out, "dataset": data_out,
                  "report": args.json_out,
                  "suit_iso": os.path.join(
                      REPO, "poker_ai", "rebel", "hunl", "suit_iso.py"),
                  "tests": os.path.join(
                      REPO, "test", "unit", "test_hunl_suit_iso.py")},
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.json_out}\nwrote {args.ckpt_out}\n"
          f"total wall {wall / 3600.0:.2f} h, peak VRAM "
          f"{(peak_vram or 0) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
