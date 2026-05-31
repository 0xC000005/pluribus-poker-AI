---
name: autoresearch-status
description: Report autoresearch readiness, objective-drift status, and failure-synthesis-due status (read-only). Use when checking the autoresearch workflow state before or between cycles.
---

Report the current autoresearch workflow state. Run the governance dispatcher (read-only — it never mutates state) from the repo root:

```bash
.venv/bin/python .claude/scripts/gov.py status
.venv/bin/python .claude/scripts/gov.py drift-status
.venv/bin/python .claude/scripts/gov.py synthesis-status
```

Then present a concise summary:
- **Readiness** — `ready`, any `missing` files, `stop_requested` (if true, say so prominently and stop).
- **Objective drift** — `blocked` + `matched_family` + `recent_failed_same_family_count`; if `blocked`, surface every entry in `blockers` verbatim.
- **Synthesis** — `due` + `experiments_since_synthesis` / `interval`.

This is read-only; do not change any files.
