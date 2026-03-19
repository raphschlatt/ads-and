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
  - `artifacts/exports/bench_full_v22_fix2`

The contract for this retained pair is documented in:

- `docs/baselines/infer_ads_full_run_20260305_v22_fix2.json`
- `docs/baselines/infer_ads_active.json`
- `docs/baselines/infer_ads_full_run_20260305_v22_fix2_transition.md`

That baseline manifest records:

- the promoted ADS baseline run
- the model bundle and runtime metadata used for inference
- the artifact keep-set used for workspace cleanup
- optional compare metadata against the previous baseline

Operational ADS candidates are expected to carry a local decision record before they are promoted or pruned:

- `98_infer_baseline_decision.json`
- `98_infer_baseline_decision.md`

Those records bind a candidate run to:

- the compare report used for the decision
- the policy gates used for promotion vs. keep-baseline
- the final decision outcome and, when promoted, the resulting versioned manifest path

The active ADS baseline is also exposed as a small machine-readable pointer in `docs/baselines/infer_ads_active.json`.
That file is intended for tooling and should be updated together with every future infer baseline promotion.

Intermediate optimization runs are not part of long-term provenance once the compare report exists and failed ADS candidates have been pruned to the JSON-only retention set.
