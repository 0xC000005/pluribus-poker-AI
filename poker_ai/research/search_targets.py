"""Build search-consistency policy targets for Deep CFR."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.resolver_benchmark import (
    ResolverBenchmarkCase,
    _solver_decision,
    default_benchmark_cases,
    load_cases_json,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (  # noqa: E402
    build_features,
    get_legal_mask_from_parsed,
    parse_action,
)


_SUITS = ("c", "d", "h", "s")
_RANKS = tuple("23456789TJQKA")
_BET_SIZES = (200, 300, 500, 800, 1200, 2000, 4000)


def _card_to_str(card: int) -> str:
    return _RANKS[card // 4] + _SUITS[card % 4]


def _sample_cards(rng: np.random.Generator, n: int) -> list[str]:
    cards = rng.choice(52, size=n, replace=False)
    return [_card_to_str(int(card)) for card in cards]


def _sample_action_template(rng: np.random.Generator) -> tuple[int, str, int]:
    flop_bet = int(rng.choice(_BET_SIZES))
    turn_bet = int(rng.choice(_BET_SIZES))
    river_bet = int(rng.choice(_BET_SIZES))
    templates = (
        (2, "ck/kk/", 0),
        (2, f"ck/kk/b{turn_bet}", 1),
        (2, f"ck/b{flop_bet}c/", 0),
        (2, f"ck/b{flop_bet}c/b{turn_bet}", 1),
        (3, "ck/kk/kk/", 0),
        (3, f"ck/kk/kk/b{river_bet}", 1),
        (3, f"ck/b{flop_bet}c/kk/", 0),
        (3, f"ck/b{flop_bet}c/kk/b{river_bet}", 1),
        (3, f"ck/b{flop_bet}c/b{turn_bet}c/", 0),
    )
    return templates[int(rng.integers(0, len(templates)))]


def sample_resolver_cases(
    n_cases: int,
    *,
    seed: int = 0,
    source: str = "sampled",
) -> list[ResolverBenchmarkCase]:
    """Sample valid turn/river public states for resolver-target generation."""
    rng = np.random.default_rng(seed)
    cases: list[ResolverBenchmarkCase] = []
    attempts = 0
    while len(cases) < n_cases and attempts < max(100, n_cases * 20):
        attempts += 1
        street, action_str, client_pos = _sample_action_template(rng)
        cards = _sample_cards(rng, 7)
        hole_cards = tuple(cards[:2])
        board = tuple(cards[2:6] if street == 2 else cards[2:7])
        parsed = parse_action(action_str)
        if "error" in parsed:
            continue
        if int(parsed.get("st", -1)) != street or int(parsed.get("pos", -1)) != client_pos:
            continue
        label = f"{source}-{len(cases):04d}-street{street}"
        cases.append(
            ResolverBenchmarkCase(
                label=label,
                hole_cards=hole_cards,
                board=board,
                action_str=action_str,
                client_pos=client_pos,
                source=source,
            )
        )
    if len(cases) != n_cases:
        raise RuntimeError(f"generated {len(cases)} valid cases out of requested {n_cases}")
    return cases


def save_cases_json(cases: Iterable[ResolverBenchmarkCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cases": [
            {
                "label": case.label,
                "hole_cards": list(case.hole_cards),
                "board": list(case.board),
                "action_str": case.action_str,
                "client_pos": int(case.client_pos),
                "source": case.source,
            }
            for case in cases
        ]
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalized_legal_target(
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    fallback_action: int,
) -> np.ndarray:
    target = np.asarray(strategy, dtype=np.float32) * (legal_mask > 0)
    total = float(target.sum())
    if total > 1e-8:
        return target / total
    fallback = np.zeros_like(target, dtype=np.float32)
    if 0 <= fallback_action < len(fallback) and legal_mask[fallback_action] > 0:
        fallback[fallback_action] = 1.0
        return fallback
    legal_total = float(legal_mask.sum())
    if legal_total <= 0:
        raise ValueError("cannot build target without legal actions")
    return legal_mask.astype(np.float32) / legal_total


def build_resolver_policy_targets(
    cases: Iterable[ResolverBenchmarkCase] | None = None,
    *,
    solver_iterations: int = 25,
    solver_backend: str = "auto",
) -> tuple[PolicyTargetBuffer, dict]:
    """Build a small policy-target dataset from deterministic resolver cases."""
    cases = list(default_benchmark_cases() if cases is None else cases)
    features: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    target_probs: list[np.ndarray] = []
    records: list[dict] = []

    for case in cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append(
                {
                    "label": case.label,
                    "skipped": f"parse_error:{parsed['error']}",
                }
            )
            continue
        feature_vec = build_features(
            list(case.hole_cards),
            list(case.board),
            case.action_str,
            case.client_pos,
            parsed,
        ).astype(np.float32)
        legal_mask = get_legal_mask_from_parsed(
            parsed,
            case.action_str,
            case.client_pos,
        ).astype(np.float32)
        solver = _solver_decision(
            case,
            parsed,
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
        )
        if solver is None:
            records.append({"label": case.label, "skipped": "unsupported_street"})
            continue

        target = _normalized_legal_target(
            solver.strategy,
            legal_mask,
            fallback_action=solver.action,
        )
        features.append(feature_vec)
        legal_masks.append(legal_mask)
        target_probs.append(target)
        records.append(
            {
                "label": case.label,
                "street": int(parsed["st"]),
                "solver_action": int(solver.action),
                "solver_increment": solver.increment,
                "solver_latency_ms": round(float(solver.latency_ms), 3),
                "target_entropy": round(
                    float(-(target[target > 0] * np.log(target[target > 0])).sum()),
                    6,
                ),
            }
        )

    if not features:
        raise RuntimeError("no resolver policy targets were generated")

    buffer = PolicyTargetBuffer(
        np.stack(features, axis=0),
        np.stack(legal_masks, axis=0),
        np.stack(target_probs, axis=0),
    )
    metadata = {
        "mode": "resolver_policy_targets",
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "n_cases": len(cases),
        "n_targets": int(buffer.size),
        "records": records,
    }
    return buffer, metadata


def save_resolver_policy_targets(
    output: str | Path,
    *,
    cases_json: str | Path | None = None,
    sampled_cases: int = 0,
    seed: int = 0,
    solver_iterations: int = 25,
    solver_backend: str = "auto",
) -> dict:
    if cases_json and sampled_cases:
        raise ValueError("choose either cases_json or sampled_cases, not both")
    if cases_json:
        cases = load_cases_json(cases_json)
        case_source = str(cases_json)
    elif sampled_cases:
        cases = sample_resolver_cases(sampled_cases, seed=seed)
        case_source = "sampled"
    else:
        cases = default_benchmark_cases()
        case_source = "fixed_default"
    buffer, metadata = build_resolver_policy_targets(
        cases,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
    )
    output = Path(output)
    buffer.save_npz(output)
    metadata_path = output.with_suffix(".json")
    cases_path = output.with_suffix(".cases.json")
    save_cases_json(cases, cases_path)
    metadata["output"] = str(output)
    metadata["metadata"] = str(metadata_path)
    metadata["cases_json"] = str(cases_path)
    metadata["case_source"] = case_source
    metadata["seed"] = int(seed) if sampled_cases else None
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata
