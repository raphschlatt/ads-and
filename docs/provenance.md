# Provenance

This repository is the maintained package form of a larger research code lineage around neural author name disambiguation.

## What Changed

- The package namespace is now `author_name_disambiguation`.
- The public surface was reduced to four CLI commands and one source-based inference API.
- Notebook-specific runtime assumptions and repo-root heuristics were removed from the public package path.
- The vendored snapshot directory `neural_name_dismabiguator-main/` was removed from the repository tree.

## What Stays in Git History

- prior reconstruction steps
- research-driven refactors
- earlier experimental layouts and interfaces

The current package does not ship the old vendored directory as part of the installable product. Provenance is tracked through repository history and this documentation instead.

## Canonical Inference Reference

For the ADS full-source inference line, provenance is now anchored by two retained artifact runs:

- historical full reference:
  - `artifacts/exports/infer_ads_full_20260305_full_20260310T134713Z`
- current operational baseline:
  - `artifacts/exports/bench_full_v21_fix4`

The contract for this retained pair is documented in:

- `docs/baselines/infer_ads_full_run_20260305_v21_fix4.json`

That baseline manifest records:

- the historical source run
- the current baseline run
- accepted drift between them
- the artifact keep-set used for workspace cleanup

Intermediate optimization runs are not part of long-term provenance once the baseline manifest and compare report exist.
