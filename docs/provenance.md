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

## Current Optimization Record

The latest accepted cold-run package optimization session is documented separately in:

- `docs/experiments/infer_cold_path_wave_20260408.md`

Important distinction:

- the formal active ADS baseline manifest is still the historical promoted baseline in `docs/baselines/infer_ads_active.json`
- the 2026-04-08 optimization record documents the currently accepted package-side cold-run improvements that were validated against the operational CPU reference but not yet promoted as a new historical ADS baseline manifest

## Current Product Decisions

The current product/runtime stance is intentionally narrower than the full set of experimental paths still present in the codebase.

- `chars2vec` GPU is not a production `infer_sources` path; product inference uses CPU-only `chars2vec`
- `numba` remains optional and is not part of the standard runtime contract
- `cuml_gpu` remains special/explicit and is not the standard `auto` clustering path
- the historical ADS baseline manifest remains unchanged; the faster 2026-04-08 package state is documented operationally rather than promoted here as a new historical provenance anchor

That split is deliberate:

- provenance stays stable and historically readable
- the operational package can still improve and document accepted runtime policy changes
- special hardware paths remain available without redefining the standard install/runtime expectations for all users
