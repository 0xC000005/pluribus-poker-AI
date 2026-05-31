#!/usr/bin/env python3
"""PostToolUse reminder: the three Deep-CFR state implementations must stay in sync.

When an edit touches one of the slow / fast / GPU state files, inject a reminder
that player ordering, the 126-d feature vector, the 9-action space, and card
indexing must be mirrored across ALL three, and that the parity suites should be
run. Non-blocking: emits additionalContext (seen by the model) and exits 0.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STATE_FILES = {
    "poker_ai/games/full_deck/state.py",   # slow (reference)
    "poker_ai/deep_cfr/fast_state.py",     # fast (CPU)
    "poker_ai/deep_cfr/cuda/game_state.py",  # GPU (Numba CUDA)
}

REMINDER = (
    "Deep-CFR parity: you edited one of the three state implementations "
    "(slow games/full_deck/state.py, fast deep_cfr/fast_state.py, GPU "
    "deep_cfr/cuda/game_state.py). Mirror any change to player ordering, the "
    "126-d feature vector, the 9-action space, or card indexing across ALL three, "
    "then run: TESTING_SUITE=1 pytest test/unit/test_fast_vs_slow.py "
    "test/unit/test_gpu_optimizations.py test/unit/test_legal_mask_parity.py"
)


def _rel(raw: str) -> str:
    try:
        return Path(raw).resolve().relative_to(REPO).as_posix()
    except Exception:
        return raw.lstrip("./")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("path") or ""
    if raw and _rel(raw) in STATE_FILES:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": REMINDER,
            }
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
