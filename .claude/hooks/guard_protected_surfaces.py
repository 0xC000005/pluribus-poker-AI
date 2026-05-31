#!/usr/bin/env python3
"""PreToolUse guard: block edits to protected autoresearch evaluation surfaces.

Sources the protected-surface definition from the single source of truth,
``.claude/scripts/surfaces.py`` (loaded standalone by path), so the hook and the
governance checks can never drift. Denies the edit
(exit code 2) when the target matches, unless an explicit override is present.

Override (deliberate, human-gated):
  - sentinel file  autoresearch-session/ALLOW_PROTECTED_EDIT, or
  - environment    POKER_AI_ALLOW_PROTECTED=1

Fail-open: any unexpected error — including inability to load the registry —
allows the edit. A guard bug must never wedge the session; the deterministic
objective-drift audit at commit time is the authoritative backstop.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SURFACES = REPO / ".claude" / "scripts" / "surfaces.py"


def _load_is_protected():
    try:
        spec = importlib.util.spec_from_file_location("_gov_surfaces", _SURFACES)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.is_protected
    except Exception:
        return None


def _overridden() -> bool:
    if os.environ.get("POKER_AI_ALLOW_PROTECTED") == "1":
        return True
    return (REPO / "autoresearch-session" / "ALLOW_PROTECTED_EDIT").exists()


def _repo_relative(raw: str) -> str:
    try:
        return Path(raw).resolve().relative_to(REPO).as_posix()
    except Exception:
        return raw.lstrip("./")


def main() -> int:
    if _overridden():
        return 0
    is_protected = _load_is_protected()
    if is_protected is None:
        return 0  # fail-open: registry unavailable
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("path") or ""
    if not raw:
        return 0
    rel = _repo_relative(raw)
    if is_protected(rel):
        sys.stderr.write(
            f"BLOCKED: '{rel}' is a protected evaluation surface.\n"
            "These are immutable during ordinary experiments. Changing them requires "
            "a completed methodology review + objective-drift audit.\n"
            "Deliberate override: `touch autoresearch-session/ALLOW_PROTECTED_EDIT` "
            "or set POKER_AI_ALLOW_PROTECTED=1.\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
