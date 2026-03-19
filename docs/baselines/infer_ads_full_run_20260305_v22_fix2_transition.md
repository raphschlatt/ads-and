# ADS Baseline Transition: v21_fix4 -> v22_fix2

This note closes the historical audit gap for the promotion from `bench_full_v21_fix4` to `bench_full_v22_fix2`.

## Status

- current active baseline: `bench_full_v22_fix2`
- previous baseline id: `bench_full_v21_fix4`
- current manifest: `docs/baselines/infer_ads_full_run_20260305_v22_fix2.json`
- active baseline marker: `docs/baselines/infer_ads_active.json`

## Limitation

The repository no longer retains the full `artifacts/exports/bench_full_v21_fix4` run directory in the current workspace snapshot.
Because that candidate directory is not present, the modern infer freeze workflow cannot reconstruct a full machine-generated compare report for the original promotion.

This means:

- the promotion predates the current `freeze_infer_baseline.py` workflow
- the old and new manifests remain the authoritative historical evidence
- the transition is documented explicitly rather than re-created from incomplete artifacts

## Practical Rule

For all promotions after this point, the expected audit chain is:

1. `99_compare_infer_to_baseline.json`
2. `98_infer_baseline_decision.json`
3. versioned manifest under `docs/baselines/`
4. `docs/baselines/infer_ads_active.json`

The `v21_fix4 -> v22_fix2` transition is the last known promotion that does not meet that full chain.