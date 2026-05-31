---
name: run-gate
description: Run a configured autoresearch gate (executes its commands — may launch GPU training or live eval; can be expensive). Human entry point.
disable-model-invocation: true
---

Run a gate defined in `autoresearch-session/poker_goal.json` (e.g. `tier0`, `eval-local`, `eval-local-confidence`, `eval-local-multiseed`, `slumbot-smoke`). From the repo root:

```bash
.venv/bin/python .claude/scripts/gov.py run-gate <gate-name> [--run-dir <dir>]
```

A gate **passes iff every command returns 0**. It emits metrics JSON (and writes `metrics.json` to `--run-dir` if given); command stdout that is valid JSON is captured as `stdout_json`.

Gates may launch GPU training or live evaluation — confirm the compute cost before running heavy gates. Never loosen a gate's commands to make it pass (benchmark-hacking); gate scripts are protected eval surfaces.
