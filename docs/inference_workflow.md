# Inference Workflow

Public inference is source-based and bundle-based.

## Required Inputs

- curated `publications`
- optional curated `references`
- exported `model_bundle`
- explicit `output_root`
- explicit `dataset_id`

## Minimal Command

```bash
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/smoke_20260309T120000Z_cli12345678/bundle_v1
```

## Optional Controls

- `--infer-stage smoke|mini|mid|full`
- `--cluster-config <yaml>`
- `--gates-config <yaml>`
- `--device auto|cpu|cuda`
- `--precision-mode fp32|amp_bf16`
- `--cluster-backend auto|sklearn_cpu|cuml_gpu`
- `--uid-scope dataset|local|registry`
- `--uid-namespace <name>`

## Output Contract

Each successful run writes:

- `publications_disambiguated.{parquet|jsonl}`
- optional `references_disambiguated.{parquet|jsonl}`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`

The source-mirrored outputs preserve all input fields and add:

- `AuthorUID`
- `AuthorDisplayName`

## Notes

- The public inference path does not accept `model_run_id`.
- The package does not expect repo-relative workspace discovery at runtime.
- Packaged defaults are used when `cluster_config` and `gates_config` are omitted.
