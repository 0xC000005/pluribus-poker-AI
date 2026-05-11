# Contributing

Contributions and focused experiments are welcome. The current development
target is the full-deck Deep CFR and Slumbot pipeline; the old short-deck
tabular MCCFR path is retained for compatibility only.

## Branch Workflow

Work from `develop` and keep changes focused:

```bash
git fetch origin -p
git checkout develop
git rebase origin/develop
git checkout -b feature/short-description
```

Use concise imperative commit subjects, for example:

```bash
git commit -m "Add Slumbot legal-mask parity test"
```

Do not commit every small file edit. For autoresearch work, batch commits by
one research objective: implementation, tests, documentation, and the relevant
log update should land together. Do not commit from continuous autoresearch
mode. Include the objective, files changed, tests or gates run, key result, and
review or related-work status in the commit body when the change affects the
research workflow or training methodology.

## Autoresearch Governance

Method, evaluation-protocol, checkpoint-promotion, and persistent-knob changes
require a methodology review before they become mainline:

```bash
python scripts/poker_autoresearch.py enqueue-review \
  --subject "New search objective" \
  --trigger method_change \
  --claim "The objective should improve Slumbot transfer."
```

Complete `review.md`, `related_work.md`, and `decision.json`, then validate:
The review bundle also includes `benchmark_audit.md` and `team_review.md`.
Use separate sub-agents for independent verification, literature review, and
benchmark-hacking audit when available; the files are the durable record.

```bash
python scripts/poker_methodology_review.py \
  --review-dir autoresearch-session/poker_reviews/<review_id> \
  --require-complete
```

Protected evaluation surfaces are immutable during ordinary experiments:
evaluation scripts, Slumbot adapters, solver benchmarks, promotion logic,
parsers, seed lists, and parity tests. Audit protected-surface changes before
keeping a candidate or committing methodology changes:

```bash
python scripts/poker_objective_audit.py --base-ref HEAD
python scripts/poker_autoresearch.py objective-audit \
  --changed-path scripts/play_slumbot.py \
  --review-dir autoresearch-session/poker_reviews/<review_id>
```

Register persistent knobs through the workflow so they have a mechanism and a
removal criterion:

```bash
python scripts/poker_autoresearch.py add-knob \
  --name search_target_mix \
  --default 0.0 \
  --failure-class search_quality \
  --mechanism "Test whether search-distilled targets reduce live transfer loss." \
  --rationale "One variable isolates the target mechanism." \
  --removal-criterion "Retire if Slumbot transfer remains negative after confirmation."
```

Avoid broad hyperparameter sweeps. Prefer one falsifiable mechanism and one
primary variable per cycle.

## Testing Expectations

Run the smallest test set that covers your change. For action-space, feature,
or Slumbot mapping work, run:

```bash
pytest -q test/unit/test_network_mask.py test/unit/test_slumbot_mapping.py test/unit/test_legal_mask_parity.py
python scripts/test_feature_encoding.py
```

For CUDA trainer/kernel work, also run:

```bash
pytest -q test/unit/test_gpu_optimizations.py
```

For performance work, include the exact command plus `iters/hour`,
`samples/sec`, and `train sec/iter`.

For autoresearch workflow changes, also run:

```bash
pytest -q test/unit/test_poker_autoresearch.py test/unit/test_poker_autoresearch_eval.py
python -m compileall -q poker_ai/research/autoresearch.py scripts/poker_autoresearch.py scripts/poker_methodology_review.py
python -m compileall -q scripts/poker_objective_audit.py
```

## Pull Requests

Open pull requests against `develop`. Include:

- What changed and why.
- Exact tests or benchmarks run.
- Any checkpoint, CUDA, or environment assumptions.
- Screenshots only when changing visual or terminal UI behavior.

Do not commit generated model checkpoints, lookup tables, or large experiment
artifacts.
