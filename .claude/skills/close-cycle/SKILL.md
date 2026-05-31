---
name: close-cycle
description: Close the active autoresearch cycle and append the research log (writes poker_state.json). Human entry point — the autonomous driver calls gov.py directly.
disable-model-invocation: true
---

Close the active research cycle via the dispatcher (deterministic, tested write). From the repo root:

```bash
.venv/bin/python .claude/scripts/gov.py close-cycle \
  --run-id "<run_id from /new-cycle>" \
  --outcome "<passed|failed|blocked>" \
  --failure-class "<none|eval_invalid|strategy_quality|...>" \
  --summary "<one-line result>"
  # optional: --metrics-path <run_dir>/metrics.json
```

It moves `active_cycle` → `history`, sets `last_metrics`, updates `updated_at`, and appends an entry to `RESEARCH_LOG.md`. Raises if the active cycle's `run_id` does not match the one given.
