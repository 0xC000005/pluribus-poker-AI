"""Duplicate-swapped league helpers for native policy checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from poker_ai.research.native_ppo_policy import evaluate_native_ppo_policy_head_to_head


def load_native_policy_league_records(record_json_paths: list[str]) -> list[dict]:
    """Load existing pairwise native-policy H2H JSON artifacts."""
    records: list[dict] = []
    for path_str in record_json_paths:
        path = Path(path_str)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("algorithm") != "native_policy_h2h":
            raise ValueError(f"{path} is not a native_policy_h2h artifact")
        for required in (
            "candidate_checkpoint",
            "baseline_checkpoint",
            "mean_candidate_payoff",
            "lower95_candidate_payoff",
        ):
            if required not in payload:
                raise ValueError(f"{path} missing required field: {required}")
        records.append(payload)
    return records


def summarize_native_policy_league(pairwise_records: list[dict]) -> dict:
    """Summarize pairwise native-policy H2H records by worst lower95 per candidate."""
    if not pairwise_records:
        return {
            "algorithm": "native_policy_league",
            "passed": False,
            "best_checkpoint": None,
            "errors": ["no_pairwise_records"],
            "candidate_summaries": {},
        }

    by_candidate: dict[str, list[dict]] = {}
    for record in pairwise_records:
        candidate = str(record["candidate_checkpoint"])
        by_candidate.setdefault(candidate, []).append(record)

    candidate_summaries: dict[str, dict] = {}
    for candidate, records in by_candidate.items():
        lower95_values = [float(r["lower95_candidate_payoff"]) for r in records]
        mean_values = [float(r["mean_candidate_payoff"]) for r in records]
        candidate_summaries[candidate] = {
            "n_matchups": len(records),
            "worst_lower95": min(lower95_values),
            "mean_lower95": sum(lower95_values) / float(len(lower95_values)),
            "mean_payoff": sum(mean_values) / float(len(mean_values)),
            "passes_all_positive_lower95": all(value > 0.0 for value in lower95_values),
            "baselines": [str(r["baseline_checkpoint"]) for r in records],
        }

    best_checkpoint = max(
        candidate_summaries,
        key=lambda path: (
            candidate_summaries[path]["worst_lower95"],
            candidate_summaries[path]["mean_lower95"],
            candidate_summaries[path]["mean_payoff"],
        ),
    )
    return {
        "algorithm": "native_policy_league",
        "passed": True,
        "best_checkpoint": best_checkpoint,
        "best_worst_lower95": candidate_summaries[best_checkpoint]["worst_lower95"],
        "all_candidates_positive": all(
            item["passes_all_positive_lower95"]
            for item in candidate_summaries.values()
        ),
        "candidate_summaries": candidate_summaries,
        "pairwise_records": pairwise_records,
        "errors": [],
    }


def evaluate_native_policy_league(
    candidate_checkpoints: list[str],
    baseline_checkpoints: list[str],
    *,
    n_games: int = 1000,
    device: str = "auto",
    seed: int = 20260515,
    strategy_source: str = "auto",
) -> dict:
    """Run a native-policy checkpoint league and summarize best iterate."""
    records: list[dict] = []
    for candidate_idx, candidate in enumerate(candidate_checkpoints):
        for baseline_idx, baseline in enumerate(baseline_checkpoints):
            if Path(candidate).resolve() == Path(baseline).resolve():
                continue
            result = evaluate_native_ppo_policy_head_to_head(
                candidate,
                baseline,
                n_games=int(n_games),
                device=device,
                seed=int(seed) + candidate_idx * 10_000 + baseline_idx,
                strategy_source=strategy_source,
            )
            records.append(result)
    summary = summarize_native_policy_league(records)
    summary.update(
        {
            "candidate_checkpoints": [str(path) for path in candidate_checkpoints],
            "baseline_checkpoints": [str(path) for path in baseline_checkpoints],
            "n_games_per_matchup": int(n_games),
            "device": str(device),
            "seed": int(seed),
            "strategy_source": str(strategy_source),
        }
    )
    return summary
