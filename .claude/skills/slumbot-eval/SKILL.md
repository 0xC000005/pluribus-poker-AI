---
name: slumbot-eval
description: Run the held-out Slumbot smoke gate (a few live hands) as an integration / catastrophic-transfer check. Human entry point.
disable-model-invocation: true
---

Run the Slumbot smoke gate from the repo root:

```bash
.venv/bin/python .claude/scripts/gov.py run-gate slumbot-smoke
```

**Slumbot is strictly held-out evaluation.** It must never supply training data, targets, replay starts, opponent priors, checkpoint selectors, or hyperparameter signals. Treat the result as an integration / catastrophic-transfer check only: report chips/hand + CI and action/parse diagnostics, and do **not** feed any of it back into training or promotion before the self-play league gates pass.
