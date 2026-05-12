# Poker Autoresearch Session State

This directory holds local resumability state for `scripts/poker_autoresearch.py`.
Generated JSON, TSV, run logs, and Slumbot transcripts are ignored by git.

Tracked state belongs in `docs/research_protocols/` and `RESEARCH_LOG.md`.
Use `python scripts/poker_autoresearch.py init` to create the local files.

Important ignored files:

- `poker_goal.json`: gates, hard stops, commit policy, review policy, and knob
  policy.
- `poker_state.json`: incumbent, active cycle, queue, history, and last
  metrics.
- `poker_knobs.tsv`: governed knob ledger with mechanism and removal criteria.
- `poker_reviews/`: methodology-review bundles containing `review.md`,
  `related_work.md`, `benchmark_audit.md`, `team_review.md`, and
  `decision.json`.
- `poker_runs/`: per-cycle metrics and raw command output.

Use `enqueue-review` before method, evaluation-protocol, promotion, or
persistent-knob changes. Validate review bundles with
`scripts/poker_methodology_review.py --require-complete`.
Run `scripts/poker_objective_audit.py --base-ref HEAD` before keeping
candidates that touched protected evaluation surfaces.
Use `python scripts/poker_autoresearch.py enqueue-falsification --candidate
<path> --mechanism "<mechanism>"` before spending Slumbot confirmation hands on
a candidate.
