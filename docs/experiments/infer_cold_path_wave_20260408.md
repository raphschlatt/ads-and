# ADS Cold-Run Optimization Wave 2026-04-08

## Goal

Improve the real one-shot `run-infer-sources` user path for ADS full inference:

- keep public outputs and cluster semantics unchanged
- avoid hardware-specific breakage and keep CPU fallback valid
- make the cold path materially faster on the current host
- stop once the remaining gains become too small or too risky

This document records the accepted and rejected changes from the 2026-04-07 to 2026-04-08 optimization session.

## Kept Changes

### 1. `chars2vec` runtime repair and CPU default

- restored progress behavior and TensorFlow runtime diagnostics
- repaired the repo `.venv` so TensorFlow GPU detection is explicit again
- kept `chars2vec` on CPU for production ADS inference because the GPU path drifted at block level

Operational reference:

- ADS compare reference:
  - `artifacts/exports/ads_full_chars2vec_cpu_ab_20260407`
- LSPO quality reference:
  - `artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__chars_cpu_20260407_v1.json`

### 2. Exact-Graph fast path

- compact global state replaced eager per-block setup
- callback and finalize telemetry were reconciled and then optimized
- callback-side union work was removed from the hot path

Validated run:

- `artifacts/exports/ads_full_exact_graph_fastpath_20260408_v1`
- `artifacts/exports/ads_full_exact_graph_callback_20260408_v1`
- `artifacts/exports/ads_full_exact_graph_finalize_20260408_v1`

### 3. Arrow fast path

- `infer_sources` pair scoring now uses the Arrow/numeric helper fast path instead of materializing public string columns for the normal callback/scoring path
- this removed the last large pair-input bottleneck

Validated run:

- `artifacts/exports/ads_full_arrow_fastpath_20260408_v1`

Key effect:

- `pair_scoring.arrow_column_extract_seconds` dropped from about `81.3s` to about `0.03s`
- `pair_inference.wall_seconds` dropped to about `192.7s`

### 4. Export frame reuse

- export now reuses the already loaded `publications` and `references` frames instead of rereading the source parquets
- one redundant parquet-path frame copy was removed

Validated run:

- `artifacts/exports/ads_full_export_reuse_20260408_v1`

Isolated A/B on real ADS data:

- `artifacts/exports/ads_full_export_reuse_20260408_v1/99_compare_infer_to_export_ab_summary.json`
- `parquet_reread`: `35.78s`
- `parquet_frame_reuse`: `25.15s`

The full validation run stayed output-identical to the CPU reference and the run artifact was pruned to `json-only`.

## Rejected Change

### SPECTER auto-batch heuristic on A6000

- tried a higher GPU auto-batch heuristic for large cards
- SPECTER itself improved only slightly
- the full ADS run got slower overall
- output drifted relative to the current CPU reference

Rejected candidate:

- `artifacts/exports/ads_full_specter_batch_20260408_v1`

Reason for rejection:

- not output-identical
- no convincing cold-run ROI

## Quality Outcome

For the kept changes:

- ADS comparisons to `ads_full_chars2vec_cpu_ab_20260407` remained exact for the accepted validation runs
- LSPO quality stayed exact to the operational LSPO reference after the Arrow wave:
  - `artifacts/metrics/full_20260218T111506Z_cli02681429/06_clustering_test_report__arrow_fastpath_20260408_v1.json`

The historical March ADS baseline and the historical `srcd52...` LSPO/train state still remain separate historical references. They were not re-established as the current operational package reference in this session.

## Performance Outcome

The main accepted cold-run gain is still anchored by the Arrow validation run relative to the earlier package-auto reference:

- earlier package-auto reference:
  - `artifacts/exports/ads_full_package_auto_20260407`
- accepted Arrow fast-path run:
  - `artifacts/exports/ads_full_arrow_fastpath_20260408_v1`

Observed improvement across the major stages on the same host:

- about `1178.8s -> 852.9s`
- about `-325.9s`
- about `27.6%` faster overall

The export reuse wave is a smaller final polish on top:

- strong isolated export-only win
- kept because it is simple, safe, and output-identical
- not used as evidence for a new large end-to-end gain by itself

## Recommendation

Stop performance work here unless a genuinely new lever with a plausible `>=20s` cold-run gain appears first.

What the repository should treat as settled after this wave:

- `chars2vec` GPU stays experimental
- SPECTER auto-batch uplift is rejected
- Arrow fast path is kept
- export frame reuse is kept
- future work should focus on baseline policy and hardware robustness before chasing more micro-optimizations
