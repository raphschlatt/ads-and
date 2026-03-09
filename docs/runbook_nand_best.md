# Runbook: Train and Source Inference

## Goal

1. LSPO-basiert trainieren.
2. finalen LSPO-Clustering-Report erzeugen.
3. Model-Bundle exportieren.
4. Source-Datensätze mit `run-infer-sources` disambiguieren.

## Commands

Train:

```bash
author-name-disambiguation run-train-stage \
  --run-stage smoke \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Clustering-Report:

```bash
author-name-disambiguation run-cluster-test-report \
  --model-run-id smoke_2026... \
  --paths-config configs/paths.local.yaml \
  --device auto \
  --precision-mode fp32
```

Bundle exportieren:

```bash
author-name-disambiguation export-model-bundle \
  --model-run-id smoke_2026... \
  --paths-config configs/paths.local.yaml
```

Source-Inferenz:

```bash
author-name-disambiguation run-infer-sources \
  --publications-path data/raw/ads/my_ads_2026/publications.parquet \
  --references-path data/raw/ads/my_ads_2026/references.parquet \
  --output-root artifacts/exports/my_ads_2026 \
  --dataset-id my_ads_2026 \
  --model-bundle artifacts/models/smoke_2026.../bundle_v1 \
  --infer-stage full \
  --device auto \
  --cluster-backend auto
```

## Gate Expectations

Train:

- Schema-, Determinismus- und Run-ID-Konsistenz
- Split-Feasibility und Negative-Coverage
- test-basiertes `lspo_pairwise_f1`
- `eps`-Diagnostik

Infer:

- Mention-Coverage
- UID-Konsistenz
- Cluster-Qualitätsraten
- `eps`-Diagnostik
- vollständige source-mirrored Exporte

## Notes

- `run-stage` ist nur noch ein deprecated Alias für den Train-Pfad.
- Die öffentliche Infer-Schnittstelle ist ausschließlich source-basiert.
