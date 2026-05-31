#!/usr/bin/env bash
# PreToolUse(Bash) guard: block autonomous `git commit` during an active /goal run.
#
# Hot-path optimized: this fires on EVERY Bash call, so the common case (no active
# continuous/goal run) must be cheap. If the CONTINUOUS_ACTIVE marker is absent we
# exit immediately without parsing stdin. Only while a run is marked active do we
# inspect the command. Override: POKER_AI_ALLOW_COMMIT=1. (No `set -e`: a false
# `[ ]` test must not abort the hook with a nonzero status.)

[ "${POKER_AI_ALLOW_COMMIT:-}" = "1" ] && exit 0
marker="${CLAUDE_PROJECT_DIR:-.}/autoresearch-session/CONTINUOUS_ACTIVE"
[ -e "$marker" ] || exit 0

cmd="$(jq -r '.tool_input.command // ""' 2>/dev/null)"
case "$cmd" in
  *git*commit*)
    echo "BLOCKED: autonomous 'git commit' is not allowed during an active autoresearch continuous/goal run (autoresearch-session/CONTINUOUS_ACTIVE present). Continuous mode must not commit autonomously — a human/supervising agent reviews the batch first. Override: set POKER_AI_ALLOW_COMMIT=1, or remove the CONTINUOUS_ACTIVE marker once paused for review." >&2
    exit 2
    ;;
esac
exit 0
