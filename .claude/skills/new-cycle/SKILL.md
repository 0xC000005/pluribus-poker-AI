---
name: new-cycle
description: Open a new autoresearch cycle (writes poker_state.json). Human entry point — the autonomous driver calls gov.py directly instead.
disable-model-invocation: true
---

Open a new research cycle via the governance dispatcher (it performs the deterministic, tested state write). From the repo root:

```bash
.venv/bin/python .claude/scripts/gov.py new-cycle \
  --hypothesis "<one falsifiable claim>" \
  --cycle-type "<experiment|methodology_review|synthesis|innovation_review|...>" \
  --failure-class "<eval_invalid|strategy_quality|mechanism_transfer|...>" \
  --gate "<configured gate name, e.g. tier0>"
```

It writes `active_cycle` into `autoresearch-session/poker_state.json` and a `cycle.json` under the run dir, then prints the cycle JSON. **Note the `run_id`** — you need it for `/close-cycle`. It raises if a cycle is already active; close that one first.

Per the workflow contract, state one falsifiable claim, the expected metric movement, and the failure class being targeted.
