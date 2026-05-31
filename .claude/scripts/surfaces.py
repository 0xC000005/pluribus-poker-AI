"""Canonical protected-surface registry (single source of truth for the guard hook).

A SUPERSET of the legacy engine's list (``poker_ai/research/autoresearch.py``):
golden-tested to include every engine surface/pattern (no dropped protection), plus
self-protection for the live Claude-native governance core (``.claude/scripts/*.py``,
``.claude/hooks/*``, and the parity test) so it cannot be silently weakened.
Stdlib-only and free of sibling imports, so the guard hook can load this file by path.

`is_protected` mirrors the engine's ``_is_protected_path``: explicit surfaces match by
exact path or directory prefix; patterns match via fnmatch.
"""
from __future__ import annotations

from collections.abc import Iterable
from fnmatch import fnmatch

PROTECTED_EVAL_SURFACES: tuple[str, ...] = (
    "scripts/poker_autoresearch.py",
    "scripts/poker_autoresearch_eval.py",
    "scripts/poker_autoresearch_slumbot.py",
    "scripts/run_neural_policy_iteration_loop.py",
    "scripts/eval_mixed_policy_h2h.py",
    "scripts/poker_resolver_benchmark.py",
    "scripts/play_slumbot.py",
    "scripts/solver.py",
    "scripts/fast_cfr.py",
    "scripts/eval_cfr_budget_frontier.py",
    "scripts/eval_solver_budget_profiles.py",
    "scripts/eval_solver_budget_selective_escalation.py",
    "scripts/eval_solver_budget_boundary_predictor.py",
    "scripts/eval_slumbot_response_range_ev_gate.py",
    "scripts/analyze_cfr_trace_sequence_predictor.py",
    "scripts/run_frozen_best_response.py",
    "scripts/eval_restricted_action_values.py",
    "scripts/poker_objective_audit.py",
    "poker_ai/research/autoresearch.py",
    "poker_ai/research/mixed_policy_h2h.py",
    "poker_ai/research/promotion.py",
    "test/unit/test_network_mask.py",
    "test/unit/test_slumbot_mapping.py",
    "test/unit/test_legal_mask_parity.py",
    "test/unit/test_poker_autoresearch_eval.py",
    "test/unit/test_governance_port.py",  # the Claude-native governance parity proof itself
)

PROTECTED_EVAL_SURFACE_PATTERNS: tuple[str, ...] = (
    "scripts/eval_*.py",
    "scripts/analyze_*trace*.py",
    "scripts/build_*trace*.py",
    "scripts/*slumbot*.py",
    "scripts/*resolver*.py",
    "scripts/*objective_audit*.py",
    "scripts/*methodology_review*.py",
    "scripts/*synthesis_review*.py",
    "poker_ai/research/*trace*.py",
    "poker_ai/research/*slumbot*.py",
    "poker_ai/research/*resolver*.py",
    "poker_ai/research/*promotion*.py",
    "test/unit/test_*slumbot*.py",
    "test/unit/test_*resolver*.py",
    "test/unit/test_*parity*.py",
    "test/unit/test_*autoresearch*.py",
    # Self-protection: the live Claude-native governance core + hooks (auditor finding).
    ".claude/scripts/*.py",
    ".claude/hooks/*",
)


def is_protected(
    path: str,
    protected_surfaces: Iterable[str] = PROTECTED_EVAL_SURFACES,
    protected_patterns: Iterable[str] | None = PROTECTED_EVAL_SURFACE_PATTERNS,
) -> bool:
    """Faithful port of autoresearch._is_protected_path."""
    normalized = path.replace("\\", "/").lstrip("./")
    for protected in protected_surfaces:
        prefix = protected.replace("\\", "/").rstrip("/")
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            return True
    for pattern in protected_patterns or ():
        if fnmatch(normalized, pattern.replace("\\", "/").lstrip("./")):
            return True
    return False
