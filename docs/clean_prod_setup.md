# Clean Production Setup

## Keep-Set

Für einen produktiven Infer-Lauf brauchst du nur:

- ein exportiertes Model-Bundle
- die kuratierten Source-Dateien (`publications`, optional `references`)
- ein leeres oder bestehendes `output_root`

## Empfohlener Ablauf

Bundle exportieren:

```bash
author-name-disambiguation export-model-bundle \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml
```

Source-Inferenz starten:

```bash
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/ads_prod_current/publications.parquet \
  --references-path data/raw/ads/ads_prod_current/references.parquet \
  --output-root artifacts/exports/ads_prod_current \
  --dataset-id ads_prod_current \
  --model-bundle artifacts/models/full_20260218T111506Z_cli02681429/bundle_v1 \
  --infer-stage full \
  --device auto \
  --cluster-backend auto
```

## Erwartete Outputs

- `publications_disambiguated.*`
- optional `references_disambiguated.*`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`
