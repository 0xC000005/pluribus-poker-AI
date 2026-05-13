# Poker Review Manifests

This directory stores small tracked digests for ignored methodology-review
bundles under `autoresearch-session/poker_reviews/`.

Generate a manifest after a review is used for a methodology decision:

```bash
python scripts/poker_autoresearch.py write-review-manifest \
  --review-dir autoresearch-session/poker_reviews/<review_id>
```

The manifest records the review decision, sources, file names, sizes, and
SHA-256 digests without committing local raw artifacts.
