"""R-NaD 8GB VRAM feasibility probe (allocation-only; NO training, NO Slumbot, NO commit).

Branch B's biggest unknown: does R-NaD fit on one RTX 3070 Ti (8GB)? R-NaD needs FOUR
ValueNetwork copies (learner + EMA target + two frozen references) co-resident with the
cuda GameBatch collector AND Adam optimizer state. This probe instantiates the REAL objects
and reads torch.cuda.max_memory_allocated() (ground truth, incl. activation + allocator
overhead), for several (hidden_dim, n_layers, n_games) configs.

Run:
  LD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/nvidia/cuda_nvcc/nvvm/lib64:$LD_LIBRARY_PATH \
  .venv/bin/python scripts/probe_rnad_vram.py
"""
import argparse
import json
import sys

import numpy as np
import torch

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import N_FEATURES, N_ACTIONS

GIB = 1024 ** 3


def total_vram_bytes() -> int:
    props = torch.cuda.get_device_properties(0)
    return int(props.total_memory)


def gamebatch_bytes(n_games: int, n_players: int = 2) -> int:
    """Exact byte count of GameBatch device arrays (mirrors cuda/game_state.py:103-126)."""
    i32, i16, i8 = 4, 2, 1
    per = 0
    per += n_games * n_players * i32  # chips
    per += n_games * n_players * i32  # bets
    per += n_games * n_players * i8   # active
    per += n_games * n_players * 2 * i8  # hole_cards
    per += n_games * n_players * i32  # payout
    per += n_games * 5 * i8           # community
    per += n_games * 52 * i8          # deck
    per += n_games * i32              # deck_cursor
    per += n_games * i8               # stage
    per += n_games * i8               # n_raises
    per += n_games * i8               # player_i_index
    per += n_games * i16              # n_actions
    per += n_games * i32              # pot_total
    per += n_games * i8               # n_players_started_round
    per += n_games * 4 * 3 * i8       # history
    per += n_games * i8               # is_done
    return per


def probe_one(hidden_dim: int, n_layers: int, n_games: int, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    # 4 R-NaD param copies. Only the learner needs grad + optimizer state.
    learner = ValueNetwork(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    target = ValueNetwork(hidden_dim=hidden_dim, n_layers=n_layers).to(device).eval()
    prev = ValueNetwork(hidden_dim=hidden_dim, n_layers=n_layers).to(device).eval()
    prev_ = ValueNetwork(hidden_dim=hidden_dim, n_layers=n_layers).to(device).eval()
    for net in (target, prev, prev_):
        for p in net.parameters():
            p.requires_grad_(False)

    n_params = sum(p.numel() for p in learner.parameters())

    # Adam optimizer state (2 moment buffers) materializes on first step.
    opt = torch.optim.Adam(learner.parameters(), lr=5e-5, betas=(0.0, 0.999), eps=1e-7)

    nets_only = torch.cuda.memory_allocated() - base

    # One forward+backward on a realistic learner minibatch to capture activation peak.
    # R-NaD forwards learner + 3 references on the same feature batch (reward transform).
    feats = torch.randn(n_games, N_FEATURES, device=device)
    adv, pol = learner.forward_with_policy(feats)
    with torch.no_grad():
        _ = target.forward_with_policy(feats)
        _ = prev.forward_with_policy(feats)
        _ = prev_.forward_with_policy(feats)
    loss = adv.pow(2).mean() + pol.pow(2).mean()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    nets_peak = torch.cuda.max_memory_allocated() - base

    # The cuda GameBatch lives in numba's allocator (separate from torch's pool) — add it.
    gb = gamebatch_bytes(n_games)
    total_peak = nets_peak + gb
    total = total_vram_bytes()

    del learner, target, prev, prev_, opt, feats, adv, pol, loss
    torch.cuda.empty_cache()

    return {
        "config": f"{n_layers}x{hidden_dim}",
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_games": n_games,
        "params_per_net": n_params,
        "params_x4": n_params * 4,
        "nets4_plus_adam_bytes": int(nets_only),
        "nets4_plus_adam_GiB": round(nets_only / GIB, 3),
        "torch_peak_with_activations_GiB": round(nets_peak / GIB, 3),
        "gamebatch_GiB": round(gb / GIB, 3),
        "estimated_total_peak_GiB": round(total_peak / GIB, 3),
        "gpu_total_GiB": round(total / GIB, 3),
        "headroom_GiB": round((total - total_peak) / GIB, 3),
        "fits_8gb": bool(total_peak < total * 0.92),  # 8% reserve for fragmentation/driver
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="autoresearch-session/rnad_vram_probe.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — cannot probe real VRAM.", file=sys.stderr)
        return 2

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={total_vram_bytes()/GIB:.2f} GiB\n")

    # Configs: the default (2x256 ~106K), the prior best Slumbot net (4x512 ~857K),
    # and a larger 4x1024 'use the card' option. n_games spans the collector's range.
    configs = [
        (256, 2, 65536),
        (512, 4, 65536),
        (512, 4, 131072),
        (1024, 4, 131072),
        (1024, 4, 196608),
    ]
    results = []
    for hd, nl, ng in configs:
        try:
            r = probe_one(hd, nl, ng, device)
        except RuntimeError as e:
            r = {"config": f"{nl}x{hd}", "n_games": ng, "error": str(e)[:200], "fits_8gb": False}
        results.append(r)
        tag = "FITS" if r.get("fits_8gb") else "TIGHT/NO"
        if "error" in r:
            print(f"  {r['config']:>8s}  n_games={ng:>7d}  ERROR: {r['error']}")
        else:
            print(f"  {r['config']:>8s}  n_games={ng:>7d}  "
                  f"4nets+adam={r['nets4_plus_adam_GiB']:.2f}  "
                  f"peak+game={r['estimated_total_peak_GiB']:.2f}/{r['gpu_total_GiB']:.2f} GiB  "
                  f"headroom={r['headroom_GiB']:.2f}  [{tag}]")

    payload = {
        "probe": "rnad_8gb_vram_feasibility",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_GiB": round(total_vram_bytes() / GIB, 3),
        "note": "allocation+1-step only; no training, no Slumbot, no commit",
        "n_features": N_FEATURES,
        "n_actions": N_ACTIONS,
        "results": results,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
