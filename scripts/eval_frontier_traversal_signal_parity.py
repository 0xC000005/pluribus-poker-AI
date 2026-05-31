#!/usr/bin/env python3
"""Evaluate distributional signal parity for frontier-indexed CUDA traversal."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.deep_cfr.buffer import ReservoirBuffer  # noqa: E402
from poker_ai.deep_cfr.cuda.gpu_trainer import (  # noqa: E402
    GPUDeepCFRTrainer,
    _GPUTraverseWorkspace,
    gpu_traverse_for_player,
)
from poker_ai.research.frontier_traversal_signal_parity import (  # noqa: E402
    compare_signal_groups,
    summarize_signal_samples,
)


def _collect_once(
    *,
    value_net,
    device: torch.device,
    use_frontier_indexing: bool,
    n_traversals: int,
    traversal_seed: int,
    hidden_capacity: int,
    pool_max_slots: int,
    slots_per_traversal: int,
    initial_chips: int,
) -> tuple[dict[str, object], dict[str, object]]:
    buffer = ReservoirBuffer(capacity=hidden_capacity)
    workspace = _GPUTraverseWorkspace(
        max_traversals=n_traversals,
        n_players=2,
        initial_chips=initial_chips,
        pool_max_slots=pool_max_slots,
        slots_per_traversal=slots_per_traversal,
        traversal_seed=traversal_seed,
    )
    stats = gpu_traverse_for_player(
        traverser=0,
        n_traversals=n_traversals,
        value_net=value_net,
        buffer=buffer,
        iteration=1,
        n_players=2,
        device=device,
        initial_chips=initial_chips,
        workspace=workspace,
        discard_on_pool_exhaustion=True,
        use_frontier_indexing=use_frontier_indexing,
    )
    summary = summarize_signal_samples(
        buffer.features[: buffer.size],
        buffer.advantages[: buffer.size],
    )
    summary["stats"] = stats
    return summary, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare frontier-indexed traversal signal against baseline."
    )
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--n-traversals", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--pool-max-slots", type=int, default=1_000_000)
    parser.add_argument("--slots-per-traversal", type=int, default=12000)
    parser.add_argument("--initial-chips", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--max-count-rel-gap", type=float, default=0.05)
    parser.add_argument("--max-regret-l1-ratio", type=float, default=2.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    json_stdout = sys.stdout
    logging.getLogger("numba").setLevel(logging.ERROR)
    logging.getLogger("numba.cuda").setLevel(logging.ERROR)
    logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    with contextlib.redirect_stdout(sys.stderr):
        from cuda_env import configure_numba_cuda_env

        configure_numba_cuda_env()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = GPUDeepCFRTrainer(
            n_players=2,
            initial_chips=args.initial_chips,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            device=device,
            traversal_pool_max_slots=args.pool_max_slots,
            traversal_slots_per_traversal=args.slots_per_traversal,
            traversal_seed=args.seed,
        )
        value_net = trainer.value_net

        baseline: list[dict[str, object]] = []
        frontier: list[dict[str, object]] = []
        for repeat in range(args.n_repeats):
            traversal_seed = int(args.seed + repeat)
            baseline_summary, _ = _collect_once(
                value_net=value_net,
                device=device,
                use_frontier_indexing=False,
                n_traversals=args.n_traversals,
                traversal_seed=traversal_seed,
                hidden_capacity=max(100_000, args.pool_max_slots),
                pool_max_slots=args.pool_max_slots,
                slots_per_traversal=args.slots_per_traversal,
                initial_chips=args.initial_chips,
            )
            frontier_summary, _ = _collect_once(
                value_net=value_net,
                device=device,
                use_frontier_indexing=True,
                n_traversals=args.n_traversals,
                traversal_seed=traversal_seed,
                hidden_capacity=max(100_000, args.pool_max_slots),
                pool_max_slots=args.pool_max_slots,
                slots_per_traversal=args.slots_per_traversal,
                initial_chips=args.initial_chips,
            )
            baseline.append(baseline_summary)
            frontier.append(frontier_summary)

    report = compare_signal_groups(
        baseline,
        frontier,
        max_count_rel_gap=args.max_count_rel_gap,
        max_regret_l1_ratio=args.max_regret_l1_ratio,
    )
    report.update(
        {
            "n_repeats": int(args.n_repeats),
            "n_traversals": int(args.n_traversals),
            "seed": int(args.seed),
            "device": str(device),
            "baseline": baseline,
            "frontier": frontier,
        }
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text, file=json_stdout)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
