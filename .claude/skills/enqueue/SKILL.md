---
name: enqueue
description: Queue a research cycle onto the autoresearch hypothesis_queue (writes poker_state.json) without opening it. Human entry point.
disable-model-invocation: true
---

Append a cycle to the hypothesis queue (it is NOT opened; `/new-cycle` opens the next one). From the repo root:

```bash
.venv/bin/python .claude/scripts/gov.py enqueue \
  --hypothesis "<one falsifiable claim>" \
  --cycle-type "<experiment|methodology_review|synthesis|...>" \
  --failure-class "<eval_invalid|strategy_quality|...>" \
  --gate "<configured gate name>"
```

Appends the item to `hypothesis_queue` in `autoresearch-session/poker_state.json` and prints it.
