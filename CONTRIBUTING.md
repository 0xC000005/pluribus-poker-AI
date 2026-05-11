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

## Pull Requests

Open pull requests against `develop`. Include:

- What changed and why.
- Exact tests or benchmarks run.
- Any checkpoint, CUDA, or environment assumptions.
- Screenshots only when changing visual or terminal UI behavior.

Do not commit generated model checkpoints, lookup tables, or large experiment
artifacts.
