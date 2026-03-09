# Author Name Disambiguation

Dieses Repo hat jetzt zwei saubere Pfade:

1. `run-train-stage`: Trainieren und benchmarken auf LSPO.
2. `run-infer-sources`: kuratierte Source-Datensätze rein, disambiguierte Source-Datensätze plus Autor-Artefakte raus.

`run-stage` bleibt nur als deprecated Train-Alias erhalten. Der öffentliche Infer-Pfad ist rein source-basiert und läuft über das installierte Package.

## Install

```bash
python -m pip install -e .
```

## Quickstart

CLI-Hilfe:

```bash
author-name-disambiguation -h
```

Train:

```bash
author-name-disambiguation run-train-stage \
  --run-stage smoke \
  --paths-config configs/paths.local.yaml \
  --device auto
```

Finalen LSPO-Clustering-Report schreiben:

```bash
author-name-disambiguation run-cluster-test-report \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml \
  --device auto \
  --precision-mode fp32
```

Model-Bundle exportieren:

```bash
author-name-disambiguation export-model-bundle \
  --model-run-id full_20260218T111506Z_cli02681429 \
  --paths-config configs/paths.local.yaml
```

Source-Inferenz mit Bundle:

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

## Programmatic API

```python
from author_name_disambiguation import InferSourcesRequest, run_infer_sources

result = run_infer_sources(
    InferSourcesRequest(
        publications_path="data/raw/ads/ads_prod_current/publications.parquet",
        references_path="data/raw/ads/ads_prod_current/references.parquet",
        output_root="artifacts/exports/ads_prod_current",
        dataset_id="ads_prod_current",
        model_bundle="artifacts/models/full_20260218T111506Z_cli02681429/bundle_v1",
        infer_stage="full",
        uid_scope="dataset",
        progress=False,
    )
)

print(result.run_id, result.go, result.publications_disambiguated_path)
```

## Public Infer Contract

Input pro Record:

- Pflicht: `Bibcode`, `Author`, `Year`
- Pflicht: `Title_en` oder `Title`
- Pflicht: `Abstract_en` oder `Abstract`
- Optional: `Affiliation`
- Optional: `embedding` oder `precomputed_embedding`

Outputs unter `output_root`:

- `publications_disambiguated.{parquet|jsonl}`
- optional `references_disambiguated.{parquet|jsonl}`
- `source_author_assignments.parquet`
- `author_entities.parquet`
- `mention_clusters.parquet`
- `05_stage_metrics_infer_sources.json`
- `05_go_no_go_infer_sources.json`

Die source-mirrored Outputs behalten alle Inputspalten und ergänzen:

- `AuthorUID`
- `AuthorDisplayName`

## Runtime Notes

- `run-infer-sources` akzeptiert nur `model_bundle`, keinen `model_run_id`.
- `cluster_config` und `gates_config` sind optional; ohne Angabe werden Package-Defaults geladen.
- `uid_scope` unterstützt `dataset`, `local` und `registry`.
- `author_entities.parquet` und `source_author_assignments.parquet` sind verpflichtende Infer-Artefakte.

## Utility Commands

Cache prüfen:

```bash
author-name-disambiguation cache doctor --paths-config configs/paths.local.yaml
```

Cache gezielt bereinigen:

```bash
author-name-disambiguation cache purge --paths-config configs/paths.local.yaml --target stale-subsets
```
